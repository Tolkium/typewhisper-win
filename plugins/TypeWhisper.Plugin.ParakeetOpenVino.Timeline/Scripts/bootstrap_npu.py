"""Download FP32 Parakeet assets and build static OpenVINO decoder IR for NPU/GPU."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"

REQUIRED_FILES = [
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
]

ProgressFn = Callable[[float, str], None]


def _configure_hf() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import huggingface_hub.file_download as fd

        fd._are_symlinks_supported_in_dir.clear()

        def _no_symlinks(_cache_dir: str | Path | None = None) -> bool:
            return False

        fd.are_symlinks_supported = _no_symlinks  # type: ignore[method-assign]
    except ImportError:
        pass


def ensure_npu_assets(
    weights_dir: Path,
    static_decoder_dir: Path,
    progress: ProgressFn | None = None,
) -> None:
    """Idempotent: fetch FP32 ONNX + build decoder IR for OpenVINO shims."""
    _configure_hf()
    weights_dir.mkdir(parents=True, exist_ok=True)
    static_decoder_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    total = len(REQUIRED_FILES) + 1
    done = 0

    for fname in REQUIRED_FILES:
        target = weights_dir / fname
        if target.exists() and target.stat().st_size > 0:
            done += 1
            progress and progress(done / total, f"present: {fname}")
            continue
        progress and progress(done / total, f"downloading {fname}")
        t0 = time.perf_counter()
        hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=str(weights_dir))
        done += 1
        progress and progress(
            done / total,
            f"downloaded {fname} ({(time.perf_counter() - t0):.1f}s)",
        )

    out_xml = static_decoder_dir / "decoder-static.xml"
    if out_xml.exists() and out_xml.stat().st_size > 0:
        progress and progress(1.0, "decoder IR ready")
        return

    progress and progress(done / total, "building static decoder IR")
    import openvino as ov

    src = weights_dir / "decoder_joint-model.onnx"
    if not src.is_file():
        raise FileNotFoundError(f"Missing decoder ONNX: {src}")

    core = ov.Core()
    model = core.read_model(str(src))
    model.reshape(
        {
            "encoder_outputs": [1, 1024, 1],
            "targets": [1, 1],
            "target_length": [1],
            "input_states_1": [2, 1, 640],
            "input_states_2": [2, 1, 640],
        }
    )
    ov.save_model(model, str(out_xml))
    progress and progress(1.0, "decoder IR built")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--static-decoder-dir", required=True)
    args = ap.parse_args()
    try:
        ensure_npu_assets(Path(args.weights_dir), Path(args.static_decoder_dir))
    except Exception as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        sys.exit(1)
