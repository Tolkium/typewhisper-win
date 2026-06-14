"""OpenVINO shims replacing onnx_asr ORT encoder/decoder for Intel NPU/GPU.

Adapted from wyoming-parakeet-on-intel-npu (MIT).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import openvino as ov

_LOGGER = logging.getLogger(__name__)


def _build_compile_cfg(device: str, cache_dir: str) -> dict:
    cfg = {"CACHE_DIR": cache_dir, "PERFORMANCE_HINT": "LATENCY"}
    if device == "NPU":
        cfg["NPU_TURBO"] = "YES"
    return cfg


class _Spec:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = list(shape)


class _SingleEncoder:
    def __init__(self, model: ov.Model, device: str, cache_dir: str, t_fixed: int) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        core = ov.Core()
        self._compiled = core.compile_model(
            model, device, config=_build_compile_cfg(device, cache_dir)
        )
        self._req = self._compiled.create_infer_request()
        self.T = t_fixed
        self.MEL = 128
        self.B = 1
        self._inputs = list(self._compiled.inputs)
        self._outputs = list(self._compiled.outputs)

    def infer(self, audio: np.ndarray, real_length: int) -> dict:
        _b, _m, t = audio.shape
        if t < self.T:
            padded = np.zeros((self.B, self.MEL, self.T), dtype=np.float32)
            padded[:, :, :t] = audio
        elif t > self.T:
            padded = audio[:, :, : self.T].astype(np.float32, copy=False)
        else:
            padded = audio.astype(np.float32, copy=False)
        clamped = np.array([min(real_length, self.T)], dtype=np.int64)
        try:
            result = self._req.infer({"audio_signal": padded, "length": clamped})
        except Exception as exc:
            raise RuntimeError(
                f"NPU encoder infer failed (T={self.T}, length={int(clamped[0])}, "
                f"audio_shape={audio.shape}): {exc}"
            ) from exc
        return {k.any_name: np.array(v) for k, v in result.items()}


class OpenVinoEncoderShim:
    def __init__(
        self,
        *,
        onnx_path: str,
        device: str,
        cache_dir: str,
        eager_seconds: list[float],
        lazy_seconds: Optional[list[float]] = None,
        fps: int = 100,
    ) -> None:
        if not eager_seconds:
            raise ValueError("At least one eager bucket is required")
        self._onnx_path = onnx_path
        self._device = device
        self._cache_dir = cache_dir
        self._buckets: dict[int, Optional[_SingleEncoder]] = {}

        for sec in eager_seconds:
            t = int(sec * fps)
            self._buckets[t] = self._build(t)
        for sec in lazy_seconds or []:
            t = int(sec * fps)
            self._buckets.setdefault(t, None)

        self._sorted_ts = sorted(self._buckets.keys())
        first = next(b for b in self._buckets.values() if b is not None)
        self._inputs_meta = first._inputs
        self._outputs_meta = first._outputs

    def _build(self, t: int) -> _SingleEncoder:
        core = ov.Core()
        model = core.read_model(self._onnx_path)
        model.reshape({"audio_signal": [1, 128, t], "length": [1]})
        _LOGGER.info("[ParakeetOV] Compiling encoder bucket T=%d on %s", t, self._device)
        return _SingleEncoder(model, self._device, self._cache_dir, t)

    def get_inputs(self) -> list[_Spec]:
        return [_Spec(i.any_name, list(i.shape)) for i in self._inputs_meta]

    def get_outputs(self) -> list[_Spec]:
        return [_Spec(o.any_name, list(o.shape)) for o in self._outputs_meta]

    def warm_all_buckets(self) -> None:
        for t in self._sorted_ts:
            if self._buckets[t] is None:
                self._buckets[t] = self._build(t)

    def _pick(self, real_t: int) -> _SingleEncoder:
        target_t = next((t for t in self._sorted_ts if real_t <= t), None)
        if target_t is None:
            target_t = self._sorted_ts[-1]
            _LOGGER.warning(
                "[ParakeetOV] audio %d frames > largest bucket %d — truncating",
                real_t,
                target_t,
            )
        if self._buckets[target_t] is None:
            _LOGGER.info("[ParakeetOV] Lazy-loading encoder bucket T=%d", target_t)
            self._buckets[target_t] = self._build(target_t)
        return self._buckets[target_t]

    def run(self, output_names: list[str], feed: dict) -> list[np.ndarray]:
        audio = feed["audio_signal"]
        length = feed["length"]
        real_t = int(length[0]) if length is not None else int(audio.shape[2])
        bucket = self._pick(real_t)
        out = bucket.infer(audio, int(length[0]))
        res: list[np.ndarray] = []
        for name in output_names:
            if name in out:
                res.append(out[name])
            elif name == "outputs":
                res.append(next(a for a in out.values() if a.ndim == 3))
            elif name == "encoded_lengths":
                res.append(next(a for a in out.values() if a.ndim == 1))
            else:
                raise KeyError(name)
        return res


class OpenVinoDecoderShim:
    def __init__(self, ir_path: str, device: str, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        core = ov.Core()
        model = core.read_model(ir_path)
        _LOGGER.info("[ParakeetOV] Compiling decoder on %s", device)
        self._compiled = core.compile_model(
            model, device, config=_build_compile_cfg(device, cache_dir)
        )
        self._req = self._compiled.create_infer_request()

    def get_inputs(self) -> list[_Spec]:
        return [_Spec(i.any_name, list(i.shape)) for i in self._compiled.inputs]

    def get_outputs(self) -> list[_Spec]:
        return [_Spec(o.any_name, list(o.shape)) for o in self._compiled.outputs]

    def run(self, output_names: list[str], feed: dict) -> list[np.ndarray]:
        feed_typed = {
            "encoder_outputs": np.asarray(feed["encoder_outputs"], dtype=np.float32),
            "targets": np.asarray(feed["targets"], dtype=np.int32),
            "target_length": np.asarray(feed["target_length"], dtype=np.int32),
            "input_states_1": np.asarray(feed["input_states_1"], dtype=np.float32),
            "input_states_2": np.asarray(feed["input_states_2"], dtype=np.float32),
        }
        try:
            result = self._req.infer(feed_typed)
        except Exception as exc:
            enc_shape = getattr(feed_typed["encoder_outputs"], "shape", None)
            raise RuntimeError(
                f"OpenVINO decoder infer failed (encoder_outputs shape={enc_shape}): {exc}"
            ) from exc
        out = {k.any_name: np.array(v) for k, v in result.items()}
        res: list[np.ndarray] = []
        for name in output_names:
            if name in out:
                res.append(out[name])
            elif name == "outputs":
                res.append(next(a for a in out.values() if a.ndim == 4))
            elif name == "output_states_1":
                cands = sorted(
                    [(nm, a) for nm, a in out.items() if a.ndim == 3 and a.shape[0] == 2]
                )
                res.append(cands[0][1])
            elif name == "output_states_2":
                cands = sorted(
                    [(nm, a) for nm, a in out.items() if a.ndim == 3 and a.shape[0] == 2]
                )
                res.append(cands[-1][1])
            else:
                raise KeyError(name)
        return res
