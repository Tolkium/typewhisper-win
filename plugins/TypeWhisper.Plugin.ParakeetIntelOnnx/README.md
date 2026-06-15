# Parakeet Intel iGPU (ONNX DirectML)

TypeWhisper transcription plugin for **Parakeet TDT 0.6B v3** on **Intel iGPU** via ONNX Runtime **DirectML**, with **CPU** fallback.

- **Plugin id:** `com.tolkium.parakeet-intel-onnx`
- **Devices:** Intel iGPU (DirectML), CPU
- **Not supported:** NVIDIA CUDA, AMD Vulkan/ROCm (host selections are remapped to DirectML with a clear status message)

## TypeWhisper settings integration

| Host setting | Behavior |
|--------------|----------|
| **Local model acceleration → Auto** | DirectML (iGPU) when available, else CPU |
| **CPU** | CPU INT8 ONNX only |
| **NVIDIA CUDA** | Intel iGPU (DirectML) — status explains CUDA is not used |
| **AMD Vulkan / ROCm** | Intel iGPU (DirectML) — same remap |
| **Spoken language** | Passed to Parakeet (`auto` = multilingual detect) |
| **Target language / translation** | Handled by TypeWhisper **after** transcription (this engine does not translate) |

The host acceleration dropdown cannot be replaced by plugins; this plugin maps foreign GPU choices to DirectML and shows the active backend under **Engine**.

## Requirements

- Windows 10/11 x64
- TypeWhisper ≥ 0.8.4
- Python 3.10+ with `onnx-asr` and `onnxruntime-directml` (installed on first model download)
- Intel iGPU with up-to-date graphics drivers for DirectML

## Install (manual)

```powershell
dotnet build plugins\TypeWhisper.Plugin.ParakeetIntelOnnx\TypeWhisper.Plugin.ParakeetIntelOnnx.csproj -c Release
```

Copy `bin\Release\net10.0-windows\*` to:

`%LOCALAPPDATA%\TypeWhisper\Plugins\com.tolkium.parakeet-intel-onnx\`

Restart TypeWhisper → enable plugin → download model.

## Tests

```powershell
dotnet test tests\TypeWhisper.PluginSystem.Tests\TypeWhisper.PluginSystem.Tests.csproj --filter ParakeetIntelOnnx
```

## License

GPL-3.0 (compatible with TypeWhisper host). See [typewhisper-win](https://github.com/TypeWhisper/typewhisper-win) for upstream plugin contribution.

The archived [typewhisper-plugins](https://github.com/TypeWhisper/typewhisper-plugins) catalog is retired; Windows plugins belong in **typewhisper-win**.
