#!/usr/bin/env python3
"""Phase 0 of docs/plans/native-openvino-engine.md: dump golden reference data.

Runs the same pipeline as the Python sidecar (onnx-asr NumPy preprocessor, Parakeet
encoder, TDT greedy decoding) and writes every intermediate result as raw little-endian
binaries plus a meta.json per clip, so each C# stage can be checked on its own.

The Python sidecar was removed when the plugin moved to the C# engine. Extract it from the last
Python release first:  git archive parakeet-openvino-v0.5.0 plugins/TypeWhisper.Plugin.ParakeetOpenVino/Scripts | tar -x -C <dir>
(requires the openvino and onnx-asr packages).

Usage:
    python dump_reference.py --model-dir <PluginData>/com.whispernpu.parakeet-openvino
                             --scripts <extracted Scripts dir> --out <dir> clip.wav [...]
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
NPU_MAX_SECONDS = 20.0


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == SAMPLE_RATE and w.getsampwidth() == 2, path
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def save(out: Path, name: str, array: np.ndarray, meta: dict) -> None:
    array = np.ascontiguousarray(array)
    array.astype(array.dtype.newbyteorder("<"), copy=False).tofile(out / f"{name}.bin")
    meta[name] = {"dtype": str(array.dtype), "shape": list(array.shape)}


def preprocess_stages(pre, waveform: np.ndarray) -> dict[str, np.ndarray]:
    """Re-run NemoPreprocessorNumpy step by step (same code, intermediates kept)."""
    x = waveform[None, :].copy()
    x = x - pre._preemph * np.pad(x, ((0, 0), (1, 0)))[:, :-1]
    preemph = x[0].copy()
    x = np.pad(x, ((0, 0), (pre._n_fft // 2, pre._n_fft // 2)))
    frames = np.lib.stride_tricks.sliding_window_view(x, pre._n_fft, axis=1)[:, :: pre._hop_length]
    window = np.pad(
        np.hanning(pre._win_length),
        ((pre._n_fft - pre._win_length) // 2, (pre._n_fft - pre._win_length) // 2),
    )
    power = np.abs(np.fft.rfft(frames * window, pre._n_fft)).astype(np.float32) ** 2
    mel = np.matmul(power, pre._melscale_fbanks)
    log_mel = np.log(mel + pre._log_zero_guard_value)
    return {
        "stage_preemph": preemph.astype(np.float32),
        "stage_window": window.astype(np.float32),
        "stage_power": power[0].astype(np.float32),
        "stage_mel": mel[0].astype(np.float32),
        "stage_logmel": log_mel[0].astype(np.float32),
    }


def decode_traced(asr, encoder_out: np.ndarray, encoder_len: int) -> tuple[list[int], list[dict]]:
    """onnx-asr TDT greedy loop with a per-iteration trace (t, token, step)."""
    state = asr._create_state()
    tokens: list[int] = []
    trace: list[dict] = []
    t = 0
    emitted = 0
    while t < encoder_len:
        logits, step, new_state = asr._decode(tokens, state, encoder_out[t])
        token = int(logits.argmax())
        trace.append({"t": t, "token": token, "step": step})
        if token != asr._blank_idx:
            state = new_state
            tokens.append(token)
            emitted += 1
        if step > 0:
            t += step
            emitted = 0
        elif token == asr._blank_idx or emitted == asr._max_tokens_per_step:
            t += 1
            emitted = 0
    return tokens, trace


def load_asr(engine_cls, model_dir: Path, device: str, cache: str):
    if device == "OVCPU":
        # FP32 model on OpenVINO CPU (dynamic encoder) -- what the C# engine runs on CPU first.
        import parakeet_server as ps

        engine = engine_cls(model_dir, "CPU", cache)
        engine._model = ps._load_onnx_model(model_dir)
        ps._apply_openvino_shims(engine._model, model_dir, "CPU", cache)
    else:
        engine = engine_cls(model_dir, device, cache)
        engine.load()
    return engine._model.asr


def run_clip(asr, waveform: np.ndarray):
    features, features_len = asr._preprocessor(waveform[None, :], np.array([len(waveform)], dtype=np.int64))
    enc, enc_len = asr._encode(features, features_len)
    tokens, trace = decode_traced(asr, enc[0], int(enc_len[0]))
    text = asr._decode_tokens(tokens, None, None).text
    return asr, features, features_len, enc[0], int(enc_len[0]), tokens, trace, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--scripts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache", default="")
    ap.add_argument("--devices", default="OVCPU,CPU,GPU,NPU")
    ap.add_argument("--synthetic-from", type=Path, help="wav used to build 0.5/3/19.9/45 s and silence clips")
    ap.add_argument("clips", nargs="*", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(args.scripts))
    import parakeet_server as ps  # noqa: E402

    clips: dict[str, np.ndarray] = {p.stem: read_wav(p) for p in args.clips}
    if args.synthetic_from:
        base = read_wav(args.synthetic_from)
        gap = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
        long = np.concatenate([base, gap, base, gap, base, gap, base])
        clips["synthetic_0p5s"] = base[: SAMPLE_RATE // 2]
        clips["synthetic_3s"] = base[: 3 * SAMPLE_RATE]
        clips["synthetic_19p9s"] = long[: int(19.9 * SAMPLE_RATE)]
        clips["synthetic_45s"] = long
        clips["synthetic_silence_2s"] = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    devices = [d.strip().upper() for d in args.devices.split(",") if d.strip()]
    summary: dict[str, dict] = {}

    # Shared constants for the C# port.
    shared: dict = {}
    from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy

    pre = NemoPreprocessorNumpy("nemo128")
    save(args.out, "nemo128_fbanks", pre._melscale_fbanks.astype(np.float32), shared)
    (args.out / "shared.json").write_text(json.dumps(shared, indent=2), encoding="utf-8")

    for device in devices:
        asr_model = load_asr(ps.ParakeetEngine, args.model_dir, device, args.cache or None)
        for name, waveform in clips.items():
            input_wave = waveform
            if device == "NPU" and len(waveform) > NPU_MAX_SECONDS * SAMPLE_RATE:
                input_wave = waveform[: int(NPU_MAX_SECONDS * SAMPLE_RATE)]

            clip_dir = args.out / name
            clip_dir.mkdir(exist_ok=True)
            meta_path = clip_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

            asr, features, features_len, enc, enc_len, tokens, trace, text = run_clip(asr_model, input_wave)
            d = device.lower()
            if device == devices[0] or "waveform" not in meta:
                save(clip_dir, "waveform", waveform, meta)
                meta["seconds"] = len(waveform) / SAMPLE_RATE
                save(clip_dir, "features", features[0], meta)  # [128, frames]
                meta["features_len"] = int(features_len[0])
                if name == "jfk_16k":
                    for stage, arr in preprocess_stages(pre, waveform).items():
                        save(clip_dir, stage, arr, meta)
            save(clip_dir, f"encoder_{d}", enc, meta)  # [T', 1024]
            meta[f"encoder_len_{d}"] = enc_len
            meta[f"input_seconds_{d}"] = len(input_wave) / SAMPLE_RATE
            meta[f"tokens_{d}"] = tokens
            meta[f"text_{d}"] = text
            (clip_dir / f"trace_{d}.json").write_text(json.dumps(trace), encoding="utf-8")
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            summary.setdefault(name, {})[device] = text
            print(f"[{device}] {name}: {len(tokens)} tokens :: {text[:80]!r}", flush=True)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
