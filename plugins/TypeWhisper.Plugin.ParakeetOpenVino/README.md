# Parakeet OpenVINO (Intel GPU)

TypeWhisper transcription plugin for **Parakeet TDT 0.6B v3** on **Intel integrated GPUs** (Arc, Iris Xe) via **OpenVINO**, with CPU fallback.

- **Plugin id:** `com.whispernpu.parakeet-openvino`
- **Devices:** Intel GPU (OpenVINO), CPU (onnx-asr INT8)

## Architecture

```
TypeWhisper → ParakeetOpenVinoPlugin (.NET)
           → parakeet_server.py (Python sidecar, JSON lines over stdin/stdout)
           → onnx-asr + OpenVINO
```

On GPU the FP32 encoder is compiled with a dynamic time axis, so a recording of any length is encoded in one pass — no chunking or stitching in the host. The decoder runs as a static OpenVINO IR on the same device.

## TypeWhisper settings integration

| Host setting (Local model acceleration) | Behavior |
|---|---|
| **Auto** | Intel GPU |
| **CPU** | CPU INT8 |
| **NVIDIA CUDA / AMD Vulkan / AMD ROCm** | Intel GPU — status explains the remap |

The plugin settings panel can override the device (GPU / CPU) and set the Python path and OpenVINO cache directory. Reload the model after changing the device.

## Requirements

- Windows 10/11 x64
- TypeWhisper ≥ 0.8.4
- .NET 10 SDK (to build)
- Python 3.11+ with `openvino` and `onnx-asr` (installed by the plugin on first model download)
- Up-to-date Intel graphics driver

## Build and install

```powershell
dotnet build plugins\TypeWhisper.Plugin.ParakeetOpenVino\TypeWhisper.Plugin.ParakeetOpenVino.csproj -c Release
```

The build copies the plugin to `%LOCALAPPDATA%\TypeWhisper\Plugins\com.whispernpu.parakeet-openvino\`. Restart TypeWhisper → enable the plugin → download the model.

The first GPU load compiles kernels (~15–20 s); later loads use the OpenVINO cache.

## Measured (Intel Arc 140V, Lunar Lake)

| Audio | CPU INT8 | Intel GPU |
|---|---|---|
| 11 s | 0.79 s | 0.09 s |
| 45 s | 2.00 s | 0.53 s |

## Tests

```powershell
dotnet test tests\TypeWhisper.PluginSystem.Tests\TypeWhisper.PluginSystem.Tests.csproj --filter ParakeetOpenVino
```

## License

GPL-3.0 (compatible with TypeWhisper).
