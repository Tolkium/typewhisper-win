# Plan: Parakeet on OpenVINO without Python

Goal: replace the Python sidecar (`parakeet_server.py` + `onnx-asr` + OpenVINO Python API) in
`TypeWhisper.Plugin.ParakeetOpenVino` with a C# engine that runs in-process, so users no longer
need Python and the `C:\WhisperNPU` paths disappear. Behaviour stays the same: GPU any length,
NPU first 20 s, CPU fallback.

Branch: `feature/native-openvino` (from `main`, tag `parakeet-openvino-v0.5.0`). The Python
plugin on `main` stays the working fallback until this branch is merged.

---

## 1. What the Python side does today

| Stage | Today (Python) | Size of C# work |
|---|---|---|
| Audio → features | `onnx_asr` `NemoPreprocessorNumpy` (NumPy, because the sidecar uses `CPUExecutionProvider`) | ~80 lines + 512-point FFT |
| Encoder | OpenVINO: GPU dynamic shape `[1,128,T]`; NPU fixed 2000 frames; CPU INT8 via ONNX Runtime | Interop + reshape |
| Decoder (TDT greedy) | `onnx_asr` `_AsrWithTransducerDecoding._decoding` + `NemoConformerTdt._decode` | ~60 lines |
| Tokens → text | `vocab.txt`, `▁` → space, `DECODE_SPACE_PATTERN` regex | ~20 lines |
| Model download | `huggingface_hub` from `istupakov/parakeet-tdt-0.6b-v3-onnx` | `HttpClient`, plain HTTPS URLs |
| Static decoder IR | `bootstrap_npu.py` reshapes decoder to `[1,1024,1]` and saves IR | Reshape at load time instead |

### Exact reference values (from `onnx_asr` 0.11.0, MIT licence)

**Preprocessor `nemo128`** (`preprocessors/numpy_preprocessor.py`, `NemoPreprocessorNumpy`):

1. Pre-emphasis: `x[n] - 0.97 * x[n-1]` (first sample unchanged), samples past the length set to 0.
2. Zero-pad 256 samples on both sides (`n_fft // 2`).
3. Frames: window 512, hop 160 → frame count `1 + (N + 512 - 512) // 160`.
4. Window: `np.hanning(400)` (symmetric Hann, **not** periodic), zero-padded by 56 on each side to 512.
5. Power spectrum: `|rfft(frame, 512)|²` → 257 bins, float32.
6. Mel: multiply by the `nemo128` matrix `[257 × 128]` from `preprocessors/data/fbanks.npz`
   (export once to a binary resource; do not recompute).
7. `log(mel + 2^-24)`.
8. `features_len = num_samples // 160`.
9. Per mel bin, over the first `features_len` frames: mean, then variance with `/(len − 1)`;
   `(x − mean) / (sqrt(var) + 1e-5)`; frames past `features_len` = 0.
10. Output transposed to `[1, 128, frames]`.

**Encoder** `encoder-model.onnx` (+ `.onnx.data`, 2.4 GB FP32):
in `audio_signal [B,128,T] f32`, `length [B] i64`; out `outputs [B,1024,T/8] f32`, `encoded_lengths [B] i64`.

**Decoder** `decoder_joint-model.onnx`:
in `encoder_outputs [B,1024,1] f32`, `targets [B,1] i32`, `target_length [B] i32`,
`input_states_1 [2,B,640] f32`, `input_states_2 [2,B,640] f32`;
out `outputs [B,1,1,8198]`, `output_states_1`, `output_states_2`.
8198 = 8193 vocab logits + 5 duration logits (durations 0–4).

**TDT greedy loop** (`asr.py`):

```
blank = 8192, max_tokens_per_step = 10
state = zeros; tokens = []; t = 0; emitted = 0
while t < encoded_len:
    out, new_state = decoder(enc[t], last token or blank, state)
    token = argmax(out[0:8193]); step = argmax(out[8193:8198])
    if token != blank: state = new_state; tokens.append(token); emitted += 1
    if step > 0: t += step; emitted = 0
    elif token == blank or emitted == 10: t += 1; emitted = 0
```

**Detokenizer**: `vocab.txt` lines `"<token> <id>"`, replace `U+2581` with space, join, then
regex `\A\s|\s\B|(\s)\b` → keep a single space only where group 1 matched.

**Files**: `config.json` = `{"model_type":"nemo-conformer-tdt","features_size":128,"subsampling_factor":8}`.

---

## 2. Research: how to call OpenVINO from C#

| Option | Status (Oct 2026) | NPU | Verdict |
|---|---|---|---|
| **A. Own P/Invoke to `openvino_c.dll`** (OpenVINO C API) | Official, stable C API; ~25 functions needed | Yes (any device string) | **Primary.** No third-party wrapper, full control of reshape and static shapes. |
| B. `OpenVINO.CSharp.API` (GuojinYan) | NuGet marked **deprecated/legacy**; successor `JYPPX.OpenVINO.CSharp.API` | Not documented | Fallback only, or as a reference for interop signatures. |
| C. `Intel.ML.OnnxRuntime.OpenVino` NuGet | Last 1.22 (2025); docs say C# needs a custom build and `setupvars.bat` | Yes | Not recommended. |
| D. Windows ML (`Microsoft.WindowsAppSDK.ML`, OpenVINO EP shipped by Windows Update, e.g. KB5096140) | Win 11 24H2+ only; unpackaged-app support unclear | Yes | Interesting later: no DLLs to ship. Spike only if A fails. |

Facts checked on this machine:

- **OpenVINO cannot load `nemo128.onnx`** (STFT op needs static input) → the preprocessor must be C#.
- Native runtime needed (from Python `openvino` 2026.2 `libs/`): `openvino_c.dll`, `openvino.dll`,
  `openvino_intel_{cpu,gpu,npu}_plugin.dll`, `openvino_intel_npu_compiler*.dll`,
  `openvino_onnx_frontend.dll`, `openvino_ir_frontend.dll`, `openvino_auto_plugin.dll`,
  `tbb12.dll`, `tbbmalloc.dll`, `tbbbind_2_5.dll` ≈ **190 MB** unpacked
  (the NPU compiler alone is 73 MB). `OpenVINO.runtime.win` 2026.4.1 on NuGet is 75 MB compressed.
- TypeWhisper loads plugins in a `PluginAssemblyLoadContext` that resolves unmanaged DLLs next to
  the plugin; the Sherpa-ONNX plugin already downloads a native runtime into its data folder and uses
  `NativeLibrary.SetDllImportResolver`. Same pattern works here.
- NPU still needs static shapes in 2026 releases → keep the fixed 2000-frame (20 s) encoder for NPU.

Sources: ONNX Runtime OpenVINO EP docs (onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html),
nuget.org pages for `OpenVINO.CSharp.API`, `OpenVINO.runtime.win`, `Intel.ML.OnnxRuntime.OpenVino`,
`Microsoft.WindowsAppSDK.ML`; Microsoft KB5096140; Intel "Accelerating AI PC experiences with OpenVINO and Windows ML".

---

## 3. Phases

Every phase ends with a **gate** (a check that must pass) and a **fallback** if it does not.
Work happens in order; nothing reaches the plugin before phase 5.

### Phase 0 — Golden reference data (Python, one time)

Write `tools/parakeet-reference/dump_reference.py` (uses the existing sidecar code) that saves,
for each test clip, raw little-endian `.bin` files + a `.json` with shapes:

- clips: JFK 11 s, 3 s, 19.9 s, 45 s, 0.5 s, 2 s silence, one Slovak recording
- per clip: waveform, features (NumPy preprocessor), encoder output (CPU FP32, GPU, NPU),
  token ids, durations, final text

Gate: all files exist; Python text for each clip recorded.
Fallback: none needed; this is plain Python that already works.

### Phase 1 — Interop spike (separate console project, not the plugin)

`tools/OpenVinoSpike/` (net10.0 console):

1. P/Invoke: `ov_core_create`, `ov_get_available_devices`, `ov_core_read_model`, `ov_model_reshape`,
   `ov_core_compile_model`, `ov_compiled_model_create_infer_request`, `ov_tensor_create_from_host_ptr`,
   `ov_infer_request_set_tensor`, `ov_infer_request_infer`, `ov_infer_request_get_output_tensor_by_index`,
   `ov_tensor_data`, `ov_tensor_get_shape`, the matching `*_free` functions, `ov_get_last_err_msg`.
2. Load the DLLs from a folder (use the Python `openvino/libs` copy first).
3. Print devices, then run the encoder on Phase 0 features on CPU, GPU, NPU.

Gate: devices list shows `CPU GPU NPU`; encoder output vs reference max abs diff
< 1e-3 (CPU), < 5e-2 (GPU/NPU FP16).

If it fails:
- DLL not found / `0x7E` → load `openvino_c.dll` by full path with `LoadLibraryEx(…, LOAD_WITH_ALTERED_SEARCH_PATH)`,
  or `AddDllDirectory` + `SetDefaultDllDirectories`.
- Crash / wrong values → check struct layouts (`ov_shape_t`, `ov_partial_shape_t`, `ov_rank_t`) against
  `openvino/c/*.h` of the same version; compare with `OpenVINO.CSharp.API` source.
- Still stuck after one day → switch to option B (wrapper) for the spike to separate
  "my interop is wrong" from "OpenVINO setup is wrong".
- Option B also fails → spike option D (Windows ML).

### Phase 2 — C# preprocessor

`NemoFeatureExtractor` (pure C#, no OpenVINO): steps 1–10 above, own radix-2 FFT, mel matrix as an
embedded resource exported from `fbanks.npz`.

Gate: vs reference features max abs diff < 1e-3 on every clip, identical frame count.
If it fails: dump each intermediate step (pre-emphasis, frames, windowed, power, mel, log, normalised)
from Python and C# and compare step by step. Usual suspects: Hann symmetric vs periodic, float vs double
accumulation in mean/variance, `len − 1` divisor, padding.

### Phase 3 — TDT decoder and detokenizer

`TdtGreedyDecoder` + `ParakeetVocabulary`, fed with the **reference encoder output** (so only the
decoder is tested).

Gate: token ids identical to reference on every clip; text byte-identical.
If it fails: log `(t, token, step)` per iteration in both languages and diff the first divergence.
Usual suspects: state updated on blank, `targets` int32 vs int64, duration slice offset.

### Phase 4 — End-to-end engine in the spike

`ParakeetOpenVinoEngine`: load once, `Transcribe(float[] samples) → string`.

- GPU: encoder dynamic shape; decoder on GPU **and** CPU, keep whichever is faster.
- NPU: encoder reshaped to `[1,128,2000]`, pad features, keep first 20 s; decoder on CPU.
- CPU: try INT8 ONNX in OpenVINO; if it fails to load, use FP32 on CPU.
- `CACHE_DIR` for compiled blobs.

Gate: text equals Python for every clip on every device; timings recorded vs Python
(11 s GPU ≈ 0.09 s, NPU ≈ 0.14 s, CPU ≈ 0.8 s). Accept up to 1.5× Python on first version.
If slow: profile per stage; decoder per-step calls are the usual cost → reuse infer request and
tensors, keep decoder on CPU.

### Phase 5 — Plugin integration

- Replace sidecar process code in `ParakeetOpenVinoPlugin` with the engine; keep public plugin API,
  settings (device GPU/NPU/CPU), NPU 20 s rule and `truncated` warning.
- Remove Python path, `ov_shims.py`, `parakeet_server.py`, `requirements.txt`, `C:\WhisperNPU` paths.
- Setup step (`DownloadModelAsync`): download model files with progress + download/unpack OpenVINO
  runtime into the plugin data folder (Sherpa CUDA pattern); verify sizes/hashes; resume on failure.
- Run inference on a background thread with the existing `SemaphoreSlim`.
- Update unit tests; add a test that runs `NemoFeatureExtractor` + decoder on a tiny fixture.

Gate: all tests pass; dictation works in the real TypeWhisper app on GPU, NPU, CPU; the plugin works
on a user account without Python on PATH.
Fallback: `main` keeps the Python plugin; nothing merged until the gate passes.

### Phase 6 — Packaging and store readiness

- Decide: runtime downloaded at setup (small plugin zip) vs shipped in the zip (~75 MB). Default: download.
- Unload: dispose OpenVINO objects in `UnloadModelAsync`/`Dispose`; check memory returns.
- English UI text, clean plugin ID, README, licences (OpenVINO Apache-2.0, onnx-asr MIT for ported code,
  model CC-BY-4.0).

Gate: fresh Windows user, install plugin → download → dictate, on GPU and NPU.

---

## 4. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Native DLL loading inside the plugin load context | Engine fails to start | Phase 1 tests the exact loading method; Sherpa plugin is a working precedent |
| C API struct layout mistakes | Crashes / wrong numbers | Golden data catches it; compare with wrapper source |
| Decoder per-token calls slow on GPU | Slower than Python | Measure; run decoder on CPU |
| INT8 ONNX not supported by OpenVINO CPU | No fast CPU path | Use FP32 on CPU or keep only GPU/NPU |
| 190 MB native runtime | Large download | Download on setup; drop unused frontends |
| NPU driver vs OpenVINO version mismatch | NPU fails to compile | Show clear error, fall back to GPU |
| First GPU compile ~15 s | Slow first load | `CACHE_DIR`; show progress |

## 5. Retry rules

- One change at a time; rerun the phase gate after each.
- Never debug two stages at once: phases 2 and 3 use reference inputs, not C# outputs of earlier stages.
- Record every failed attempt and its cause in section 6 so it is not tried again.
- If a phase takes more than twice its estimate, stop and reassess the option (A → B → D).

## 6. Log

| Date | Phase | Result |
|---|---|---|
| 2026-10-13 | Research | OpenVINO cannot import `nemo128.onnx` (STFT) → C# preprocessor required |
| 2026-10-13 | 0 | Pass. 9 clips × OVCPU/CPU/GPU/NPU in `C:\WhisperNPU\parakeet-reference`. FP32 OpenVINO text identical on OVCPU, GPU, NPU (NPU 45 s = first 20 s). Current INT8 CPU path differs: hallucinates "Thank you." on silence, more errors on Slovak → C# CPU path should use FP32. |
| 2026-10-13 | 1 | Pass. `openvino_c.dll` (2026.2, Python `openvino/libs`) loads via `NativeLibrary.Load(full path)`; devices CPU, GPU, NPU. Encoder output bit-identical to Phase 0 on CPU/GPU/NPU. Gotchas: dynamic dim must be `{min:-1,max:-1}` (else INVALID_C_PARAM -14); `ov_core_compile_model` varargs work as fixed overloads on win-x64. Cold NPU compile 56 s; first GPU infer per new length ~0.5–0.7 s (needs warm-up). |
| 2026-10-13 | 2 | Pass. `NemoFeatureExtractor` vs NumPy: max diff ≤ 4.3e-5 on 8 clips, frame counts equal. Known exception: digital silence (std = 0) → NumPy gives rounding constant −0.655, C# exact 0; both "no signal", transcript checked in phase 4. |
| 2026-10-13 | 3 | Pass. `TdtGreedyDecoder` + `ParakeetVocabulary` on reference encoder output: token ids and text identical for 9 clips × (CPU, NPU→CPU decoder, GPU) = 27/27, incl. Slovak. Decoder 3–120 ms. |
| 2026-10-13 | 4 | Pass. `ParakeetOpenVinoEngine` waveform → text equals Python on 9 clips × 5 configs (45/45); silence → empty. 11 s clip: NPU ~180 ms steady (Python ~132); GPU first ~650 / repeat ~110 ms (Python 549 / 89) → ~1.2× Python, within 1.5× gate. GPU first-call cost per new length exists in Python too (OpenVINO dynamic shapes). Padding to whole seconds made first calls worse (up to 2.1 s) → dropped. Decoder on CPU faster for GPU first calls → default. CPU FP32 ~1.3 s per 11 s. |
