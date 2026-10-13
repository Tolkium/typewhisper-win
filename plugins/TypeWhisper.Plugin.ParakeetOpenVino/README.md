# Parakeet OpenVINO (Intel GPU/NPU)

TypeWhisper transcription plugin for **Parakeet TDT 0.6B v3** on **Intel integrated GPUs** (Arc, Iris Xe) or the **Intel NPU** via **OpenVINO**, with CPU fallback. No Python required.

- **Plugin id:** `com.whispernpu.parakeet-openvino`
- **Devices:** Intel GPU (default), Intel NPU, CPU

## Architecture

```
TypeWhisper → ParakeetOpenVinoPlugin (.NET, in-process)
           → Engine/  NemoFeatureExtractor → encoder → TdtGreedyDecoder → ParakeetVocabulary
           → OpenVINO C API (openvino_c.dll, P/Invoke)
```

- **Features:** 128-bin log-mel, a C# port of the onnx-asr NumPy preprocessor.
- **Encoder:** FP32 ONNX model. GPU and CPU compile it with a dynamic time axis, so any recording length is encoded in one pass.
- **Decoder:** greedy TDT decoding, run on CPU (measured faster than GPU for this per-token loop).

## Setup (model download in TypeWhisper)

Downloading the model in TypeWhisper fetches, into the plugin data folder:

| What | Source | Size |
|---|---|---|
| Parakeet ONNX model | Hugging Face `istupakov/parakeet-tdt-0.6b-v3-onnx`, pinned revision | ~2.6 GB |
| OpenVINO 2026.2.0 runtime DLLs | Intel's official `openvino` wheel on PyPI (only `openvino/libs/*.dll` is kept) | 73 MB download, 194 MB on disk |

Downloads resume after interruptions. The OpenVINO wheel and the model weights are checked by SHA-256; the small `config.json` and `vocab.txt` only by size. Compiled models are cached in `ovcache/`; the first NPU load compiles for about a minute.

## Known limitation: NPU records at most 20 s

The NPU only runs statically shaped models, so the encoder is compiled for a fixed 20 s window (2000 mel frames at exactly 100 frames/s). The plugin **transcribes only the first 20 s of each recording on NPU** and logs a warning when it cuts. At about 19.4 s the plugin stops the dictation like the stop shortcut and shows the reason in TypeWhisper's notification area, so the audio (plus TypeWhisper's 0.3 s tail padding) fits the 20 s window. The plugin SDK has no stop API, so this uses TypeWhisper internals found at runtime (tested with 0.8.3 and 1.0.1); if they are not found, the plugin shows its own red banner instead. GPU and CPU have no limit.

Splitting long NPU recordings at sentence boundaries is planned for the next version.

## TypeWhisper settings integration

| Host setting (Local model acceleration) | Behavior |
|---|---|
| **Auto** | Device chosen in plugin settings (default: Intel GPU) |
| **CPU** | OpenVINO CPU |
| **NVIDIA CUDA / AMD Vulkan / AMD ROCm** | Device chosen in plugin settings; status explains the remap |

Reload the model after changing the device.

## Requirements

- Windows 10/11 x64
- TypeWhisper ≥ 0.8.4
- Up-to-date Intel graphics driver (GPU) / Intel NPU driver (NPU)
- .NET 10 SDK (to build)

## Performance (Intel Core Ultra 7 258V, Arc 140V)

| 11 s recording | First time at a new length | Repeat |
|---|---|---|
| Intel NPU | ~180 ms | ~180 ms |
| Intel GPU | ~300–650 ms | ~110 ms |
| CPU (FP32) | ~1.3 s | ~1.3 s |

The GPU's first call at a new recording length includes OpenVINO preparing kernels for that shape.

## Licences

- Plugin: GPL-3.0 (same as TypeWhisper)

User-facing texts live in `Localization/en.json` and `de.json` (same convention as the official plugins); TypeWhisper picks the app language and falls back to English.
- Feature extraction and decoding are ported from [onnx-asr](https://github.com/istupakov/onnx-asr) (MIT); `Engine/nemo128_fbanks.bin` is its mel filter bank
- OpenVINO: Apache-2.0
- Parakeet TDT 0.6B v3 model: CC-BY-4.0 (NVIDIA)
