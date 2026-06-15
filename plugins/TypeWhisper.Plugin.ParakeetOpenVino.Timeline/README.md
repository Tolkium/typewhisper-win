# Parakeet OpenVINO Timeline (experimental)

Duplicate of the baseline OpenVINO Parakeet plugin with a separate plugin ID so both can be installed side by side.

| | Baseline | Timeline |
|---|---|---|
| Plugin ID | `com.whispernpu.parakeet-openvino` | `com.tolkium.parakeet-openvino-timeline` |
| Version | 0.3.3 | 0.4.0-alpha |
| Host merge | `RollingTranscriptAligner` | `TranscriptTimelineMerger` (token timeline WIP) |
| Python sidecar | Same scripts / model cache | Same scripts / model cache |

## Build

```powershell
# Both plugins (default on this branch)
.\run-typewhisper-fork.ps1

# Baseline only
.\run-typewhisper-baseline.ps1

# Timeline only
.\run-typewhisper-timeline.ps1
```

Pick the model in TypeWhisper settings to switch between baseline and timeline merge behavior.
