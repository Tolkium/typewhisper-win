# Parakeet OpenVINO plugin (TypeWhisper)

Lokalny engine **Parakeet TDT 0.6B v3** pre TypeWhisper s podporou **Intel NPU / GPU / CPU**
a prepínaním zariadenia **NPU / GPU / CPU** cez plugin settings.

## Architektura

```
TypeWhisper → ParakeetOpenVinoPlugin (.NET)
           → parakeet_server.py (Python sidecar)
           → onnx-asr + OpenVINO
```

## Poziadavky

- TypeWhisper Windows (0.8.4+)
- .NET 10 SDK (na build)
- `C:\WhisperNPU\venv` s Python 3.11+ (alebo vlastna cesta v nastaveniach pluginu)
- Intel NPU driver (pre NPU)

## Build a instalacia

```powershell
cd C:\WhisperNPU
.\install-parakeet-plugin.ps1
```

Alebo manualne:

```powershell
cd C:\WhisperNPU\typewhisper-win
dotnet build plugins\TypeWhisper.Plugin.ParakeetOpenVino\TypeWhisper.Plugin.ParakeetOpenVino.csproj -c Release
```

Plugin sa skopiruje do:

`%LOCALAPPDATA%\TypeWhisper\Plugins\com.whispernpu.parakeet-openvino\`

Reštartuj TypeWhisper → Settings → Engines → **Parakeet OpenVINO (NPU/GPU)**.

## Nastavenia pluginu

| Pole | Vyznam |
|------|--------|
| **Zariadenie** | NPU / GPU / CPU |
| **Python** | cesta k `python.exe` (default `C:\WhisperNPU\venv\Scripts\python.exe`) |
| **OV cache** | OpenVINO compile cache (default `C:\WhisperNPU\ovcache`) |

Po zmene zariadenia znova stiahni/nacitaj model v TypeWhisper.

## Zariadenie (v0.3.0+)

Plugin settings obsahujú voľbu **NPU / GPU / CPU**:

- **NPU**: encoder beží cez OpenVINO NPU, decoder cez CPU (stabilnejšie)
- **GPU**: encoder aj decoder bežia cez OpenVINO GPU
- **CPU**: fallback cez INT8 CPU

Po zmene zariadenia treba model znovu načítať. Status pluginu ukáže, ak je stále
nahratý predchádzajúci device.

## Live preview (v0.2.5+)

Parakeet OpenVINO je **batch sidecar** — live polling počas nahrávania je vypnutý
(`ProviderId = parakeet-openvino`). Finálna transkripcia ide až po uvoľnení klávesy.

Ak rebuildneš TypeWhisper z fork-u, host navyše filtruje `<pad>` garbage a preferuje
full-decode pred partial preview.

## Stav v0.1.0

- [x] Plugin skeleton + sidecar JSON protokol
- [x] Model download cez onnx-asr (INT8 Parakeet v3)
- [x] NPU/GPU/CPU prepinac v UI
- [ ] Plny native OpenVINO NPU encoder (bucket compile) — dalsia iteracia
- [ ] onnxruntime-openvino EP tuning pre NPU

Aktualne inferencia ide cez **onnx-asr**; device sa predava sidecaru pre OpenVINO cache a buduci NPU offload.

## Licencia

GPLv3 (kompatibilne s TypeWhisper). Fork: `typewhisper-win` + tento plugin.
