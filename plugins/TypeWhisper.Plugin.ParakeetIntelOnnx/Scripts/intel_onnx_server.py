#!/usr/bin/env python3
"""Parakeet TDT 0.6B v3 via ONNX Runtime (DirectML / CPU) for TypeWhisper.

One JSON object per line on stdin/stdout.
Commands: check, setup (progress), serve (load/transcribe/quit).
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
    path = os.environ.get("PARAKEET_INTEL_ONNX_LOG")
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass


def _configure_stdio_utf8() -> None:
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

logging.basicConfig(level=logging.INFO, format="[ParakeetIntelOnnx] %(message)s")
_LOGGER = logging.getLogger(__name__)


def _weights_dir(model_dir: Path) -> Path:
    return model_dir / "weights"


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


def _available_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def _directml_available() -> bool:
    return "DmlExecutionProvider" in _available_providers()


def _providers_for_device(device: str) -> list[str]:
    normalized = (device or "CPU").upper()
    if normalized == "CPU":
        return ["CPUExecutionProvider"]
    if _directml_available():
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _backend_id_for_device(device: str) -> str:
    normalized = (device or "CPU").upper()
    if normalized == "DML" and _directml_available():
        return "directml"
    return "cpu"


def _load_onnx_model(model_dir: Path, device: str):
    _configure_windows_hf()
    import onnx_asr

    providers = _providers_for_device(device)
    _LOGGER.info("Loading model with providers: %s", providers)
    return onnx_asr.load_model(
        MODEL_ID,
        path=str(_weights_dir(model_dir)),
        quantization="int8",
        providers=providers,
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


class ParakeetEngine:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.model_dir = model_dir
        self.device = (device or "CPU").upper()
        self._model = None
        self._backend = "cpu"

    def load(self, on_progress=None) -> str:
        def prog(frac: float, msg: str) -> None:
            if on_progress:
                on_progress(frac, msg)

        os.environ.setdefault("HF_HOME", str(self.model_dir / "hf-cache"))
        prog(0.1, "Loading INT8 model...")
        self._model = _load_onnx_model(self.model_dir, self.device)
        self._backend = _backend_id_for_device(self.device)
        prog(1.0, "Model ready")
        return self._backend

    def transcribe(self, wav_bytes: bytes, language: str | None) -> tuple[str, float]:
        if self._model is None:
            raise RuntimeError("Model not loaded")

        _validate_wav(wav_bytes)
        duration = wav_duration_sec(wav_bytes)
        if duration < 1.0:
            return ("", duration)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            if language:
                text = self._model.recognize(tmp_path, language=language)
            else:
                text = self._model.recognize(tmp_path)
            return (_sanitize_text(str(text).strip()), duration)
        except TypeError:
            text = self._model.recognize(tmp_path)
            return (_sanitize_text(str(text).strip()), duration)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def cmd_check(model_dir: Path) -> None:
    issues: list[str] = []
    for mod in ("numpy", "soundfile", "onnx_asr", "onnxruntime"):
        try:
            __import__(mod.replace("-", "_") if mod == "onnx_asr" else mod)
        except ImportError:
            issues.append(f"{mod} not installed")

    marker = model_dir / ".setup-complete"
    if not marker.is_file():
        issues.append("model not set up (run download in TypeWhisper)")

    respond(
        {
            "ready": not issues,
            "issues": issues,
            "directml_available": _directml_available(),
            "providers": _available_providers(),
        }
    )


def cmd_setup(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    respond({"progress": 0.05, "phase": "deps"})

    try:
        import onnx_asr  # noqa: F401
    except ImportError as exc:
        respond({"error": f"onnx-asr not installed: {exc}"})
        sys.exit(1)

    respond({"progress": 0.2, "phase": "int8-model"})
    try:
        _ = _load_onnx_model(model_dir, "CPU")
    except Exception as exc:
        respond({"error": f"Model download failed: {exc}"})
        sys.exit(1)

    (model_dir / ".setup-complete").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8"
    )
    respond({"progress": 1.0, "phase": "complete"})


def cmd_serve(model_dir: Path, device: str) -> None:
    engine: ParakeetEngine | None = None

    for raw in sys.stdin:
        raw = raw.lstrip("\ufeff").strip()
        if not raw:
            continue

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            respond({"error": f"Invalid JSON: {exc}"})
            continue

        req_id = msg.get("req_id")
        cmd = msg.get("cmd")

        def reply(data: dict) -> None:
            if req_id is not None:
                data["req_id"] = req_id
            respond(data)

        try:
            if cmd == "load":
                target = (msg.get("device") or device or "CPU").upper()
                engine = ParakeetEngine(model_dir, target)

                def on_progress(frac: float, phase: str) -> None:
                    reply({"pending": True, "phase": phase, "progress": frac})

                backend = engine.load(on_progress=on_progress)
                reply(
                    {
                        "ok": True,
                        "device": target,
                        "backend": backend,
                        "directml_available": _directml_available(),
                    }
                )
            elif cmd == "transcribe":
                if engine is None:
                    reply({"error": "Model not loaded"})
                    continue
                audio_b64 = msg.get("audio_base64")
                if not audio_b64:
                    reply({"error": "Missing audio_base64"})
                    continue
                wav_bytes = base64.b64decode(audio_b64)
                language = msg.get("language")
                if language in ("auto", ""):
                    language = None
                text, duration = engine.transcribe(wav_bytes, language)
                reply({"text": text, "duration": duration})
            elif cmd == "quit":
                reply({"ok": True})
                break
            else:
                reply({"error": f"Unknown command: {cmd}"})
        except Exception as exc:
            _LOGGER.exception("Command failed")
            reply({"error": str(exc)})

    _sidecar_log("serve loop ended")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()

    if args.check:
        cmd_check(args.model_dir)
    elif args.setup:
        cmd_setup(args.model_dir)
    elif args.serve:
        cmd_serve(args.model_dir, args.device)
    else:
        parser.error("Specify --check, --setup, or --serve")


if __name__ == "__main__":
    main()
