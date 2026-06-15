using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows.Controls;
using TypeWhisper.PluginSDK;
using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Parakeet TDT 0.6B v3 on Intel OpenVINO (NPU/GPU/CPU) via Python sidecar.
/// SupportsModelDownload lets TypeWhisper manage setup/download state for the local model.
/// </summary>
public sealed class ParakeetOpenVinoPlugin : ITypeWhisperPlugin, ITranscriptionEnginePlugin, ITranscriptionAccelerationAdvisor
{
    public const string ModelId = "parakeet-tdt-0.6b-v3-openvino";

    public static readonly IReadOnlyList<string> DeviceOptions = ["NPU", "GPU", "CPU"];

    private static readonly string DefaultPythonPath =
        Path.Combine(@"C:\WhisperNPU", "venv", "Scripts", "python.exe");

    private static readonly string DefaultOvCache =
        Path.Combine(@"C:\WhisperNPU", "ovcache");

    private static readonly string LogFilePath =
        Path.Combine(@"C:\WhisperNPU", "logs", "parakeet-plugin.log");

    private readonly SemaphoreSlim _sidecarLock = new(1, 1);
    private IPluginHostServices? _host;
    private Process? _sidecar;
    private StreamWriter? _sidecarIn;
    private StreamReader? _sidecarOut;
    private CancellationTokenSource? _stderrCts;
    private Task? _stderrDrainTask;
    private string? _selectedModelId;
    private string? _loadedModelId;
    private string? _loadedDevice;
    private string? _activeBackend;
    private string _targetDevice = "NPU";
    private string _pythonPath = DefaultPythonPath;
    private string _ovCache = DefaultOvCache;
    private int _requestId;
    private TranscriptionAccelerationPreference _accelerationPreference =
        TranscriptionAccelerationPreference.Auto;
    private TranscriptionAccelerationStatus _accelerationStatus = new(
        TranscriptionAccelerationBackend.Cpu,
        "OpenVINO NPU encoder + CPU decoder");

    public string PluginId => "com.whispernpu.parakeet-openvino";
    public string PluginName => "Parakeet OpenVINO (NPU/GPU)";
    public string PluginVersion => "0.3.2";

    public string ProviderId => "parakeet-openvino";
    public string ProviderDisplayName => "Parakeet OpenVINO (lokalne)";
    public bool IsConfigured => File.Exists(PythonExecutablePath);
    public string? SelectedModelId => _selectedModelId;
    public bool SupportsTranslation => false;
    public bool SupportsModelDownload => true;

    public IReadOnlyList<TranscriptionAccelerationBackend> SupportedAccelerationBackends { get; } =
    [
        TranscriptionAccelerationBackend.Cpu,
        TranscriptionAccelerationBackend.NvidiaCuda,
    ];

    public TranscriptionAccelerationPreference AccelerationPreference => _accelerationPreference;

    public TranscriptionAccelerationStatus AccelerationStatus => _accelerationStatus;

    public IReadOnlyList<PluginModelInfo> TranscriptionModels { get; } =
    [
        new(ModelId, "Parakeet TDT 0.6B v3 (OpenVINO)")
        {
            SizeDescription = "~3.2 GB (INT8 + NPU FP32 encoder)",
            EstimatedSizeMB = 3200,
            IsRecommended = true,
            LanguageCount = 25,
        }
    ];

    public IReadOnlyList<string> SupportedLanguages => [];

    public string TargetDevice
    {
        get => _targetDevice;
        set
        {
            var normalized = NormalizeDevice(value);
            if (string.Equals(_targetDevice, normalized, StringComparison.Ordinal))
                return;

            _targetDevice = normalized;
            TrySetSetting("targetDevice", _targetDevice);
            _accelerationStatus = ParakeetOpenVinoAcceleration.ToStatus(
                new ParakeetOpenVinoAcceleration.Resolution(
                    normalized,
                    normalized == "CPU"
                        ? TranscriptionAccelerationBackend.Cpu
                        : TranscriptionAccelerationBackend.NvidiaCuda,
                    normalized switch
                    {
                        "GPU" => "OpenVINO GPU encoder + decoder",
                        "CPU" => "OpenVINO CPU INT8",
                        _ => "OpenVINO NPU encoder + CPU decoder",
                    },
                    "Manual device override in plugin settings. Engine acceleration may differ until reload.",
                    null),
                _loadedDevice,
                _activeBackend);
            Log(PluginLogLevel.Info, $"Target device changed to {_targetDevice}; reload model to apply");
        }
    }

    public string PythonExecutablePath
    {
        get => _pythonPath;
        set
        {
            _pythonPath = string.IsNullOrWhiteSpace(value) ? DefaultPythonPath : value.Trim();
            TrySetSetting("pythonPath", _pythonPath);
        }
    }

    public string OpenVinoCacheDirectory
    {
        get => _ovCache;
        set
        {
            _ovCache = string.IsNullOrWhiteSpace(value) ? DefaultOvCache : value.Trim();
            TrySetSetting("ovCache", _ovCache);
        }
    }

    public Task ActivateAsync(IPluginHostServices host)
    {
        _host = host;
        TryLoadPersistedSettings();
        ApplyAccelerationPreference(_accelerationPreference);
        // Always pin WhisperNPU venv unless user explicitly set a working path in settings.
        if (!File.Exists(_pythonPath) || !_pythonPath.StartsWith(@"C:\WhisperNPU\", StringComparison.OrdinalIgnoreCase))
        {
            _pythonPath = DefaultPythonPath;
            TrySetSetting("pythonPath", _pythonPath);
        }
        Log(
            PluginLogLevel.Info,
            $"Activated plugin v{PluginVersion}; python='{_pythonPath}', device='{_targetDevice}', cache='{_ovCache}', data='{GetDataDirectory()}'");
        return Task.CompletedTask;
    }

    public Task DeactivateAsync()
    {
        StopSidecar();
        return Task.CompletedTask;
    }

    public UserControl? CreateSettingsView() => new ParakeetOpenVinoSettingsView(this);

    public void SetAccelerationPreference(TranscriptionAccelerationPreference preference) =>
        ApplyAccelerationPreference(preference);

    public string? GetSuggestedHostAcceleration(TranscriptionAccelerationPreference preference) =>
        ParakeetOpenVinoAcceleration.GetSuggestedHostAcceleration(preference);

    private void ApplyAccelerationPreference(TranscriptionAccelerationPreference preference)
    {
        _accelerationPreference = preference;
        var resolution = ParakeetOpenVinoAcceleration.Resolve(preference);
        _targetDevice = resolution.TargetDevice;
        TrySetSetting("targetDevice", _targetDevice);
        _accelerationStatus = ParakeetOpenVinoAcceleration.ToStatus(
            resolution,
            _loadedDevice,
            _activeBackend);
        Log(PluginLogLevel.Info, $"Acceleration preference {preference} -> OpenVINO device {_targetDevice}");
    }

    public void SelectModel(string modelId)
    {
        if (modelId != ModelId)
            throw new ArgumentException($"Unknown model: {modelId}");
        _selectedModelId = modelId;
    }

    public bool IsModelDownloaded(string modelId) =>
        File.Exists(Path.Combine(GetDataDirectory(), ".setup-complete"));

    public async Task DownloadModelAsync(string modelId, IProgress<double>? progress, CancellationToken ct)
    {
        EnsurePythonExists();
        var dataDir = GetDataDirectory();
        Directory.CreateDirectory(dataDir);

        await EnsurePythonPackagesAsync(progress, ct);

        progress?.Report(0.25);
        Log(PluginLogLevel.Info, "Downloading Parakeet model via onnx-asr...");

        var scriptPath = GetScriptPath("parakeet_server.py");
        var psi = new ProcessStartInfo
        {
            FileName = PythonExecutablePath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        psi.ArgumentList.Add(scriptPath);
        psi.ArgumentList.Add("--setup");
        psi.ArgumentList.Add("--model-dir");
        psi.ArgumentList.Add(dataDir);
        ConfigurePythonProcess(psi);

        using var proc = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start parakeet setup");

        string? line;
        while ((line = await proc.StandardOutput.ReadLineAsync(ct)) is not null)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;
                if (root.TryGetProperty("progress", out var prog))
                    progress?.Report(0.25 + prog.GetDouble() * 0.75);
                if (root.TryGetProperty("error", out var err))
                    throw new InvalidOperationException(err.GetString());
            }
            catch (JsonException)
            {
                Debug.WriteLine($"[ParakeetOV] setup: {line}");
            }
        }

        await proc.WaitForExitAsync(ct);
        if (proc.ExitCode != 0)
        {
            var stderr = await proc.StandardError.ReadToEndAsync(ct);
            throw new InvalidOperationException(
                $"Parakeet setup failed (exit {proc.ExitCode}): {Tail(stderr, 800)}");
        }

        progress?.Report(1.0);
        Log(PluginLogLevel.Info, "Parakeet setup complete");
    }

    public async Task LoadModelAsync(string modelId, CancellationToken ct)
    {
        var loadWatch = Stopwatch.StartNew();
        Log(PluginLogLevel.Info, $"LoadModelAsync start: model='{modelId}', downloaded={IsModelDownloaded(modelId)}");
        if (!IsModelDownloaded(modelId))
            throw new FileNotFoundException("Model not set up. Download it first.");

        EnsurePythonExists();
        await VerifyPythonRuntimeAsync(ct);
        await _sidecarLock.WaitAsync(ct);
        try
        {
            StopSidecar();
            KillOrphanSidecars();
            StartSidecar();

            using var loadCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            loadCts.CancelAfter(TimeSpan.FromMinutes(6));

            var response = await SendCommandAsync(new
            {
                cmd = "load",
                device = TargetDevice,
            }, loadCts.Token);

            if (response.TryGetProperty("error", out var err))
                throw new InvalidOperationException($"Load failed: {err.GetString()}");

            var loadedDevice = response.TryGetProperty("device", out var dev)
                ? NormalizeDevice(dev.GetString())
                : TargetDevice;
            var backend = response.TryGetProperty("backend", out var be)
                ? be.GetString()
                : null;

            _loadedModelId = modelId;
            _selectedModelId = modelId;
            _loadedDevice = loadedDevice;
            _activeBackend = backend;
            _accelerationStatus = ParakeetOpenVinoAcceleration.ToStatus(
                ParakeetOpenVinoAcceleration.Resolve(_accelerationPreference),
                _loadedDevice,
                _activeBackend);
            Log(
                PluginLogLevel.Info,
                $"Parakeet loaded on {_loadedDevice} backend='{_activeBackend ?? "unknown"}' in {loadWatch.Elapsed.TotalSeconds:F1}s");
        }
        catch (Exception ex)
        {
            Log(PluginLogLevel.Error, $"LoadModelAsync failed after {loadWatch.Elapsed.TotalSeconds:F1}s: {ex}");
            throw;
        }
        finally
        {
            _sidecarLock.Release();
        }
    }

    public async Task<PluginTranscriptionResult> TranscribeAsync(
        byte[] wavAudio, string? language, bool translate, string? prompt, CancellationToken ct)
    {
        if (translate)
            throw new NotSupportedException(
                "Parakeet OpenVINO does not translate during transcription. " +
                "Set a target language in TypeWhisper to translate after transcription.");

        var transcribeWatch = Stopwatch.StartNew();
        await _sidecarLock.WaitAsync(ct);
        try
        {
            if (_sidecar is null || _sidecar.HasExited)
                throw new InvalidOperationException("No model loaded. Call LoadModelAsync first.");

            if (_loadedModelId is null)
                throw new InvalidOperationException("Model is still loading. Wait for load to complete.");

            var response = await SendCommandAsync(new
            {
                cmd = "transcribe",
                audio_base64 = Convert.ToBase64String(wavAudio),
                language,
            }, ct);

            if (response.TryGetProperty("error", out var err))
                throw new InvalidOperationException($"Transcription failed: {err.GetString()}");

            var text = SanitizeOutput(response.GetProperty("text").GetString() ?? "");
            var duration = response.GetProperty("duration").GetDouble();
            Log(
                PluginLogLevel.Info,
                $"Transcribe ok: audio={duration:F2}s, elapsed={transcribeWatch.Elapsed.TotalSeconds:F2}s, textLen={text.Length}, hadPad={response.GetProperty("text").GetString()?.Contains("<pad>", StringComparison.OrdinalIgnoreCase) == true}");
            return new PluginTranscriptionResult(text, language, duration, NoSpeechProbability: null);
        }
        catch (Exception ex)
        {
            Log(PluginLogLevel.Error, $"Transcribe failed after {transcribeWatch.Elapsed.TotalSeconds:F2}s: {ex}");
            throw;
        }
        finally
        {
            _sidecarLock.Release();
        }
    }

    internal static string SanitizeOutput(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return "";

        var stripped = PadTokenRegex.Replace(text, "").Trim();
        if (stripped.Length == 0)
            return "";

        if (PadTokenRegex.Matches(text).Count >= 3 && stripped.Length < 8)
            return "";

        return stripped;
    }

    private static readonly Regex PadTokenRegex =
        new(@"<pad>\s*", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    public async Task UnloadModelAsync()
    {
        await _sidecarLock.WaitAsync();
        try
        {
            StopSidecar();
            _loadedModelId = null;
            _loadedDevice = null;
            _activeBackend = null;
            ApplyAccelerationPreference(_accelerationPreference);
        }
        finally
        {
            _sidecarLock.Release();
        }
    }

    public void Dispose()
    {
        StopSidecar();
        _sidecarLock.Dispose();
    }

    private async Task EnsurePythonPackagesAsync(IProgress<double>? progress, CancellationToken ct)
    {
        progress?.Report(0.05);
        Log(PluginLogLevel.Info, "Installing Python packages (openvino, onnx-asr)...");

        var reqPath = GetScriptPath("requirements.txt");
        var args = $"-m pip install -q -r \"{reqPath}\"";
        await RunProcessAsync(PythonExecutablePath, args, ct, timeoutMs: 1_800_000);
        progress?.Report(0.2);
    }

    private void StartSidecar()
    {
        var scriptPath = GetScriptPath("parakeet_server.py");
        var dataDir = GetDataDirectory();
        var python = Path.GetFullPath(PythonExecutablePath);
        if (!python.StartsWith(@"C:\WhisperNPU\", StringComparison.OrdinalIgnoreCase))
            Log(PluginLogLevel.Warning, $"Non-WhisperNPU Python path: {python}");

        var psi = new ProcessStartInfo
        {
            FileName = python,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        psi.ArgumentList.Add(scriptPath);
        psi.ArgumentList.Add("--serve");
        psi.ArgumentList.Add("--model-dir");
        psi.ArgumentList.Add(dataDir);
        psi.ArgumentList.Add("--device");
        psi.ArgumentList.Add(TargetDevice);
        psi.ArgumentList.Add("--ov-cache");
        psi.ArgumentList.Add(OpenVinoCacheDirectory);
        ConfigurePythonProcess(psi);

        _sidecar = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start Parakeet sidecar");
        _sidecarIn = _sidecar.StandardInput;
        _sidecarOut = _sidecar.StandardOutput;
        Log(PluginLogLevel.Info, $"Started sidecar pid={_sidecar.Id}: {python} {string.Join(" ", psi.ArgumentList)}");
        StartStderrDrain(_sidecar);
    }

    private void StartStderrDrain(Process proc)
    {
        StopStderrDrain();
        _stderrCts = new CancellationTokenSource();
        var ct = _stderrCts.Token;
        var stderr = proc.StandardError;
        _stderrDrainTask = Task.Run(async () =>
        {
            try
            {
                while (!ct.IsCancellationRequested && !proc.HasExited)
                {
                    var line = await stderr.ReadLineAsync(ct).ConfigureAwait(false);
                    if (line is null)
                        break;
                    Log(PluginLogLevel.Info, line);
                }
            }
            catch (OperationCanceledException)
            {
                // expected on shutdown
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ParakeetOV] stderr drain failed: {ex.Message}");
            }
        }, ct);
    }

    private void StopStderrDrain()
    {
        try
        {
            _stderrCts?.Cancel();
        }
        catch
        {
            // ignore
        }

        _stderrCts?.Dispose();
        _stderrCts = null;
        _stderrDrainTask = null;
    }

    private void StopSidecar()
    {
        if (_sidecar is null)
            return;

        StopStderrDrain();

        try
        {
            if (!_sidecar.HasExited && _sidecarIn is not null)
            {
                _sidecarIn.WriteLine(JsonSerializer.Serialize(new { cmd = "quit" }));
                _sidecarIn.Flush();
                _sidecar.WaitForExit(3000);
            }
        }
        catch
        {
            // ignore
        }

        if (!_sidecar.HasExited)
        {
            try
            {
                _sidecar.Kill();
            }
            catch
            {
                // ignore
            }
        }

        _sidecar.Dispose();
        _sidecar = null;
        _sidecarIn = null;
        _sidecarOut = null;
        _loadedModelId = null;
        _loadedDevice = null;
        _activeBackend = null;
    }

    private async Task<JsonElement> SendCommandAsync(object command, CancellationToken ct)
    {
        if (_sidecarIn is null || _sidecarOut is null)
            throw new InvalidOperationException("Sidecar not running");

        var reqId = Interlocked.Increment(ref _requestId);
        var wrapper = new Dictionary<string, object?>();
        foreach (var prop in JsonSerializer.SerializeToElement(command).EnumerateObject())
            wrapper[prop.Name] = prop.Value;
        wrapper["req_id"] = reqId;
        var commandName = wrapper.TryGetValue("cmd", out var cmd) ? cmd?.ToString() : "unknown";
        Log(PluginLogLevel.Info, $"sidecar request #{reqId}: {commandName}");

        var requestJson = JsonSerializer.Serialize(wrapper);
        Log(PluginLogLevel.Info, $"sidecar request #{reqId} json: {requestJson}");
        await _sidecarIn.WriteLineAsync(requestJson.AsMemory(), ct);
        await _sidecarIn.FlushAsync(ct);

        while (true)
        {
            var response = await _sidecarOut.ReadLineAsync(ct)
                ?? throw new InvalidOperationException("Parakeet sidecar closed unexpectedly");

            using var doc = JsonDocument.Parse(response);
            var root = doc.RootElement.Clone();
            if (!root.TryGetProperty("req_id", out var id) || id.GetInt32() != reqId)
                continue;

            if (root.TryGetProperty("pending", out var pending) && pending.GetBoolean())
            {
                var phase = root.TryGetProperty("phase", out var ph) ? ph.GetString() : null;
                if (!string.IsNullOrWhiteSpace(phase))
                    Log(PluginLogLevel.Info, $"sidecar request #{reqId} pending: {phase}");
                continue;
            }

            Log(PluginLogLevel.Info, $"sidecar request #{reqId} completed");
            return root;
        }
    }

    private void EnsurePythonExists()
    {
        if (!File.Exists(PythonExecutablePath))
            throw new FileNotFoundException(
                $"Python not found: {PythonExecutablePath}. Set path in plugin settings to C:\\WhisperNPU\\venv\\Scripts\\python.exe",
                PythonExecutablePath);
    }

    private async Task VerifyPythonRuntimeAsync(CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            FileName = Path.GetFullPath(PythonExecutablePath),
            Arguments = "-c \"import openvino, onnx_asr; print('ok')\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        ConfigurePythonProcess(psi);

        using var proc = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to verify Python runtime");

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(TimeSpan.FromSeconds(45));
        try
        {
            await proc.WaitForExitAsync(cts.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            proc.Kill(true);
            throw new TimeoutException("Python runtime check timed out");
        }

        var stderr = await proc.StandardError.ReadToEndAsync(ct);
        if (proc.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"Python at {PythonExecutablePath} is missing openvino/onnx-asr. Use C:\\WhisperNPU\\venv\\Scripts\\python.exe. {Tail(stderr, 300)}");
        }
    }

    private static void KillOrphanSidecars()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = "-NoProfile -Command \"Get-CimInstance Win32_Process -Filter \\\"Name='python.exe'\\\" | Where-Object { $_.CommandLine -like '*parakeet_server.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }\"",
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi);
            proc?.WaitForExit(8000);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ParakeetOV] KillOrphanSidecars: {ex.Message}");
        }
    }

    private void Log(PluginLogLevel level, string message)
    {
        var line = $"[{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss.fff zzz}] [{level}] {message}";
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(LogFilePath)!);
            File.AppendAllText(LogFilePath, line + Environment.NewLine, Encoding.UTF8);
        }
        catch
        {
            // Logging must never break transcription.
        }

        try
        {
            _host?.Log(level, message);
        }
        catch
        {
            // Older host/plugin SDK mismatches should not take the plugin down.
        }

        Debug.WriteLine($"[ParakeetOV] {message}");
    }

    private string GetDataDirectory()
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TypeWhisper",
            "PluginData",
            PluginId);
    }

    private void TryLoadPersistedSettings()
    {
        if (_host is null)
            return;

        try
        {
            _targetDevice = NormalizeDevice(_host.GetSetting<string>("targetDevice") ?? "NPU");
            _pythonPath = _host.GetSetting<string>("pythonPath") ?? DefaultPythonPath;
            _ovCache = _host.GetSetting<string>("ovCache") ?? DefaultOvCache;
            return;
        }
        catch (MissingMethodException)
        {
            // Host SDK older than plugin build; fall back to settings.json.
        }

        try
        {
            var settingsPath = Path.Combine(GetDataDirectory(), "settings.json");
            if (!File.Exists(settingsPath))
                return;

            using var doc = JsonDocument.Parse(File.ReadAllText(settingsPath));
            var root = doc.RootElement;
            if (root.TryGetProperty("targetDevice", out var device))
                _targetDevice = NormalizeDevice(device.GetString());
            if (root.TryGetProperty("pythonPath", out var python))
                _pythonPath = python.GetString() ?? DefaultPythonPath;
            if (root.TryGetProperty("ovCache", out var cache))
                _ovCache = cache.GetString() ?? DefaultOvCache;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ParakeetOV] Failed to load settings.json: {ex.Message}");
        }
    }

    private void TrySetSetting(string key, string value)
    {
        if (_host is null)
            return;

        try
        {
            _host.SetSetting(key, value);
        }
        catch (MissingMethodException)
        {
            Debug.WriteLine($"[ParakeetOV] Host does not support SetSetting for '{key}'.");
        }
    }

    private static void ConfigurePythonProcess(ProcessStartInfo psi)
    {
        psi.StandardOutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        psi.StandardErrorEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        if (psi.RedirectStandardInput)
            psi.StandardInputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        psi.Environment["PYTHONUNBUFFERED"] = "1";
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        psi.Environment["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1";
        psi.Environment["PARAKEET_SIDECAR_LOG"] =
            Path.Combine(@"C:\WhisperNPU", "logs", "parakeet-sidecar.log");
    }

    private static string GetScriptPath(string fileName)
    {
        var pluginDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? ".";
        return Path.Combine(pluginDir, "Scripts", fileName);
    }

    private static string NormalizeDevice(string? device) =>
        device?.ToUpperInvariant() switch
        {
            "GPU" => "GPU",
            "CPU" => "CPU",
            _ => "NPU",
        };

    private static string Tail(string text, int max) =>
        text.Length <= max ? text : text[^max..];

    private static async Task RunProcessAsync(string exe, string args, CancellationToken ct, int timeoutMs)
    {
        var psi = new ProcessStartInfo(exe, args)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        psi.Environment["PYTHONIOENCODING"] = "utf-8";

        using var proc = Process.Start(psi)
            ?? throw new InvalidOperationException($"Failed to start {exe}");

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(timeoutMs);
        try
        {
            await proc.WaitForExitAsync(cts.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            proc.Kill();
            throw new TimeoutException($"Process timed out: {exe}");
        }

        if (proc.ExitCode != 0)
        {
            var stderr = await proc.StandardError.ReadToEndAsync(ct);
            throw new InvalidOperationException($"{exe} failed: {Tail(stderr, 500)}");
        }
    }
}

