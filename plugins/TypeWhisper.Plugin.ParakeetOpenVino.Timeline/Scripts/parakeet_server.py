#!/usr/bin/env python3

"""Parakeet TDT 0.6B v3 inference sidecar for TypeWhisper (OpenVINO NPU/GPU/CPU).



Protocol: one JSON object per line on stdin/stdout.

Commands: check, setup (progress lines), serve (load/transcribe/quit).

"""



from __future__ import annotations



import argparse

import base64

import io

import json

import logging

import os

import re

import sys

import tempfile

import time

import wave

from pathlib import Path


def _sidecar_log(message: str) -> None:
    path = os.environ.get("PARAKEET_SIDECAR_LOG")
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass


def _configure_stdio_utf8() -> None:
    """Windows default (cp1252) cannot emit Slovak diacritics on stdout/stderr."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
            elif hasattr(stream, "buffer"):
                wrapper = io.TextIOWrapper(
                    stream.buffer,
                    encoding="utf-8",
                    errors="replace",
                    line_buffering=True,
                )
                if stream is sys.stdout:
                    sys.stdout = wrapper
                else:
                    sys.stderr = wrapper
        except Exception:
            pass


_configure_stdio_utf8()

MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

DEFAULT_CACHE = os.environ.get("OV_CACHE_DIR", "")

EAGER_BUCKETS_SEC = [5.0]

LAZY_BUCKETS_SEC = [20.0]



logging.basicConfig(level=logging.INFO, format="[ParakeetOV] %(message)s")

_LOGGER = logging.getLogger(__name__)





def _weights_dir(model_dir: Path) -> Path:

    return model_dir / "weights"





def _static_decoder_dir(model_dir: Path) -> Path:

    return model_dir / "static_decoder"





def _configure_windows_hf() -> None:

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    try:

        import huggingface_hub.file_download as fd



        fd._are_symlinks_supported_in_dir.clear()



        def _symlinks_disabled(_cache_dir: str | Path | None = None) -> bool:

            return False



        fd.are_symlinks_supported = _symlinks_disabled  # type: ignore[method-assign]

    except ImportError:

        pass





def _load_onnx_model(model_dir: Path):

    _configure_windows_hf()

    import onnx_asr



    return onnx_asr.load_model(

        MODEL_ID,

        path=str(_weights_dir(model_dir)),

        quantization="int8",

        providers=["CPUExecutionProvider"],

    )





def respond(data: dict) -> None:
    line = json.dumps(data, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(line.encode("utf-8"))
    sys.stdout.buffer.flush()





def wav_duration_sec(wav_bytes: bytes) -> float:

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:

        return w.getnframes() / float(w.getframerate())


def _sanitize_text(text: str) -> str:
    stripped = re.sub(r"<pad>\s*", "", text, flags=re.IGNORECASE).strip()
    if not stripped and "<pad>" in text.lower():
        return ""
    if text.lower().count("<pad>") >= 3 and len(stripped) < 8:
        return ""
    return stripped


def _validate_wav(wav_bytes: bytes) -> None:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        rate = w.getframerate()
        frames = w.getnframes()
        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels")
        if rate != 16000:
            raise ValueError(f"Expected 16 kHz WAV, got {rate} Hz")
        if frames < 1600:
            raise ValueError("Audio too short (< 0.1 s)")


def _ov_cache_dir(model_dir: Path, ov_cache: str | None) -> str:

    if ov_cache:

        return ov_cache

    return str(model_dir / "ov_cache")





def _apply_openvino_shims(model, model_dir: Path, device: str, ov_cache: str | None) -> str:

    import openvino as ov



    from bootstrap_npu import ensure_npu_assets

    from ov_shims import OpenVinoDecoderShim, OpenVinoEncoderShim



    available = ov.Core().available_devices

    if device not in available:

        raise RuntimeError(

            f"OpenVINO device '{device}' not available (have: {', '.join(available)})"

        )



    weights = _weights_dir(model_dir)

    decoder_dir = _static_decoder_dir(model_dir)

    ensure_npu_assets(weights, decoder_dir)



    encoder_onnx = weights / "encoder-model.onnx"

    decoder_ir = decoder_dir / "decoder-static.xml"

    cache = _ov_cache_dir(model_dir, ov_cache)



    _LOGGER.info("Installing OpenVINO shims on %s (cache=%s)", device, cache)

    model.asr._encoder = OpenVinoEncoderShim(

        onnx_path=str(encoder_onnx),

        device=device,

        cache_dir=cache,

        eager_seconds=EAGER_BUCKETS_SEC,

        lazy_seconds=LAZY_BUCKETS_SEC,

    )

    model.asr._decoder_joint = OpenVinoDecoderShim(
        ir_path=str(decoder_ir),
        device="CPU" if device == "NPU" else device,
        cache_dir=cache,
    )
    return f"openvino-{device.lower()}"





class ParakeetEngine:

    def __init__(self, model_dir: Path, device: str, ov_cache: str | None) -> None:

        self.model_dir = model_dir

        self.device = (device or "CPU").upper()

        self.ov_cache = ov_cache or ""

        self._model = None

        self._backend = "onnx-asr-cpu"



    def load(self, on_progress=None) -> str:

        def prog(frac: float, msg: str) -> None:
            if on_progress:
                on_progress(frac, msg)

        os.environ.setdefault("HF_HOME", str(self.model_dir / "hf-cache"))

        if self.ov_cache:

            os.environ["OPENVINO_CACHE_DIR"] = self.ov_cache

        prog(0.1, "Loading INT8 model...")
        self._model = _load_onnx_model(self.model_dir)

        if self.device in ("NPU", "GPU"):

            prog(0.35, f"Applying OpenVINO shims ({self.device})...")
            self._backend = _apply_openvino_shims(

                self._model, self.model_dir, self.device, self.ov_cache or None

            )

            encoder = self._model.asr._encoder
            if hasattr(encoder, "warm_all_buckets"):
                prog(0.55, f"Pre-compiling {self.device} encoder buckets (~1 min first run)...")
                encoder.warm_all_buckets()
                prog(0.95, "Encoder buckets ready")

        else:

            self._backend = "onnx-asr-int8-cpu"

        prog(1.0, "Model ready")
        return self._backend

    def transcribe(self, wav_bytes: bytes) -> tuple[str, float]:

        if self._model is None:

            raise RuntimeError("Model not loaded")

        _validate_wav(wav_bytes)

        duration = wav_duration_sec(wav_bytes)

        # Partial live-poll buffers (<1s) often decode to <pad> garbage on NPU.
        if duration < 1.0:
            return ("", duration)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:

            tmp.write(wav_bytes)

            tmp_path = tmp.name



        try:

            text = self._model.recognize(tmp_path)

            return (_sanitize_text(str(text).strip()), duration)

        except Exception as exc:
            raise RuntimeError(f"Parakeet recognize failed ({duration:.1f}s audio): {exc}") from exc

        finally:

            try:

                os.unlink(tmp_path)

            except OSError:

                pass





def cmd_check(model_dir: Path) -> None:

    issues: list[str] = []

    for mod in ("numpy", "soundfile", "onnx_asr", "openvino"):

        try:

            __import__(mod.replace("-", "_") if mod == "onnx_asr" else mod)

        except ImportError:

            issues.append(f"{mod} not installed")



    marker = model_dir / ".setup-complete"

    if not marker.is_file():

        issues.append("model not set up (run download in TypeWhisper)")



    try:

        import openvino as ov



        devices = ov.Core().available_devices

        if not any(device in devices for device in ("NPU", "GPU", "CPU")):

            issues.append("No supported OpenVINO device visible")

    except Exception as exc:

        issues.append(f"openvino check failed: {exc}")



    respond({"ready": not issues, "issues": issues})





def cmd_setup(model_dir: Path) -> None:

    model_dir.mkdir(parents=True, exist_ok=True)

    respond({"progress": 0.05, "phase": "deps"})



    try:

        import onnx_asr  # noqa: F401

    except ImportError as exc:

        respond({"error": f"onnx-asr not installed: {exc}"})

        sys.exit(1)



    respond({"progress": 0.15, "phase": "int8-model"})

    try:

        _ = _load_onnx_model(model_dir)

    except Exception as exc:

        respond({"error": f"Model download failed: {exc}"})

        sys.exit(1)



    respond({"progress": 0.35, "phase": "openvino-assets"})

    try:

        from bootstrap_npu import ensure_npu_assets



        def on_progress(frac: float, msg: str) -> None:

            respond({"progress": 0.35 + frac * 0.6, "phase": msg})



        ensure_npu_assets(

            _weights_dir(model_dir),

            _static_decoder_dir(model_dir),

            progress=on_progress,

        )

    except Exception as exc:

        respond({"error": f"OpenVINO asset download failed: {exc}"})

        sys.exit(1)



    (model_dir / ".setup-complete").write_text(

        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8"

    )

    respond({"progress": 1.0, "phase": "done"})





def cmd_serve(model_dir: Path, default_device: str, ov_cache: str | None) -> None:

    engine: ParakeetEngine | None = None
    _sidecar_log(f"serve start pid={os.getpid()} model_dir={model_dir} device={default_device} cache={ov_cache}")



    for line in sys.stdin:

        line = line.strip().lstrip("\ufeff")
        _sidecar_log(f"stdin line: {line[:500]}")

        if not line:

            continue



        try:

            req = json.loads(line)

        except json.JSONDecodeError as exc:

            respond({"error": f"invalid json: {exc}"})

            continue



        req_id = req.get("req_id")

        cmd = req.get("cmd")



        def wrap(payload: dict) -> None:

            if req_id is not None:

                payload["req_id"] = req_id

            respond(payload)



        if cmd == "quit":

            wrap({"ok": True})

            break



        if cmd == "load":

            device = str(req.get("device") or default_device or "NPU").upper()
            _sidecar_log(f"load start device={device}")

            try:

                engine = ParakeetEngine(model_dir, device, ov_cache)

                def on_progress(frac: float, phase: str) -> None:
                    wrap({"pending": True, "progress": frac, "phase": phase})

                backend = engine.load(on_progress=on_progress)
                _sidecar_log(f"load ok backend={backend}")

                wrap({"ok": True, "device": device, "backend": backend})

            except Exception as exc:
                _sidecar_log(f"load error: {exc}")

                wrap({"error": str(exc)})

            continue



        if cmd == "transcribe":

            if engine is None:

                wrap({"error": "Model not loaded"})

                continue

            b64 = req.get("audio_base64")

            if not b64:

                wrap({"error": "missing audio_base64"})

                continue

            try:

                wav_bytes = base64.b64decode(b64)

                text, duration = engine.transcribe(wav_bytes)

                wrap({"text": text, "duration": duration})

            except Exception as exc:

                wrap({"error": str(exc)})

            continue



        wrap({"error": f"unknown cmd: {cmd}"})





def main() -> None:

    _configure_windows_hf()



    parser = argparse.ArgumentParser()

    parser.add_argument("--check", action="store_true")

    parser.add_argument("--setup", action="store_true")

    parser.add_argument("--serve", action="store_true")

    parser.add_argument("--model-dir", default="")

    parser.add_argument("--device", default="NPU")

    parser.add_argument("--ov-cache", default=DEFAULT_CACHE)

    args = parser.parse_args()



    model_dir = Path(args.model_dir) if args.model_dir else Path.cwd() / "models"



    if args.check:

        cmd_check(model_dir)

    elif args.setup:

        cmd_setup(model_dir)

    elif args.serve:

        cmd_serve(model_dir, args.device, args.ov_cache or None)

    else:

        parser.print_help()

        sys.exit(2)





if __name__ == "__main__":

    main()


