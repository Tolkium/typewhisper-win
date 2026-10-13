"""OpenVINO shims replacing onnx_asr ORT encoder/decoder for Intel GPU.

Adapted from wyoming-parakeet-on-intel-npu (MIT).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import openvino as ov

_LOGGER = logging.getLogger(__name__)


def _build_compile_cfg(cache_dir: str) -> dict:
    return {"CACHE_DIR": cache_dir, "PERFORMANCE_HINT": "LATENCY"}


class _Spec:
    def __init__(self, name: str, shape: list[int]) -> None:
        self.name = name
        self.shape = list(shape)


class OpenVinoEncoderShim:
    """Encoder compiled with a dynamic time axis, so any recording length is encoded in one pass."""

    def __init__(self, *, onnx_path: str, device: str, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        core = ov.Core()
        model = core.read_model(onnx_path)
        model.reshape({"audio_signal": [1, 128, -1], "length": [1]})
        _LOGGER.info("[ParakeetOV] Compiling dynamic encoder on %s", device)
        self._compiled = core.compile_model(model, device, config=_build_compile_cfg(cache_dir))
        self._req = self._compiled.create_infer_request()

    def get_inputs(self) -> list[_Spec]:
        return [_Spec(i.any_name, list(i.get_partial_shape().get_min_shape())) for i in self._compiled.inputs]

    def get_outputs(self) -> list[_Spec]:
        return [_Spec(o.any_name, list(o.get_partial_shape().get_min_shape())) for o in self._compiled.outputs]

    def warm_up(self, seconds: float = 5.0, fps: int = 100) -> None:
        """Runs one silent inference so GPU kernels are built before the first dictation."""
        frames = int(seconds * fps)
        self._infer(np.zeros((1, 128, frames), dtype=np.float32), frames)

    def _infer(self, audio: np.ndarray, real_length: int) -> dict:
        feed = {
            "audio_signal": np.ascontiguousarray(audio, dtype=np.float32),
            "length": np.array([real_length], dtype=np.int64),
        }
        try:
            result = self._req.infer(feed)
        except Exception as exc:
            raise RuntimeError(
                f"OpenVINO encoder infer failed (length={real_length}, audio_shape={audio.shape}): {exc}"
            ) from exc
        return {k.any_name: np.array(v) for k, v in result.items()}

    def run(self, output_names: list[str], feed: dict) -> list[np.ndarray]:
        audio = feed["audio_signal"]
        length = feed["length"]
        real_t = int(length[0]) if length is not None else int(audio.shape[2])
        out = self._infer(audio, real_t)
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
            model, device, config=_build_compile_cfg(cache_dir)
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
