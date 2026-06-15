using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows.Controls;
using TypeWhisper.PluginSDK;
using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Plugin.ParakeetIntelOnnx;

/// <summary>
/// Parakeet TDT 0.6B v3 on Intel iGPU via ONNX Runtime DirectML, with CPU fallback.
/// Integrates with the host "Local model acceleration" dropdown.
/// </summary>
public sealed class ParakeetIntelOnnxPlugin : ITypeWhisperPlugin, ITranscriptionEnginePlugin
{
    public const string ModelId = "parakeet-tdt-0.6b-v3-onnx";

    private static readonly Regex PadTokenRegex =
        new(@"<pad>\s*", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    private readonly SemaphoreSlim _sidecarLock = new(1, 1);
    private IPluginHostServices? _host;
    private Process? _sidecar;
    private StreamWriter? _sidecarIn;
    private StreamReader? _sidecarOut;
    private CancellationTokenSource? _stderrCts;
    private Task? _stderrDrainTask;
    private string? _selectedModelId;
    private string? _loadedModelId;
    private string? _loadedBackend;
    private string _pythonPath = "";
    private bool? _directMlAvailable;
    private int _requestId;
    private TranscriptionAccelerationPreference _accelerationPreference =
        TranscriptionAccelerationPreference.Auto;
    private TranscriptionAccelerationStatus _accelerationStatus = new(
        TranscriptionAccelerationBackend.Cpu,
        "CPU (ONNX Runtime)");

    public string PluginId => "com.tolkium.parakeet-intel-onnx";
    public string PluginName => "Parakeet Intel iGPU (ONNX DirectML)";
    public string PluginVersion => "1.0.0";

    public string ProviderId => "parakeet-intel-onnx";
    public string ProviderDisplayName => "Parakeet Intel iGPU (local)";
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
        new(ModelId, "Parakeet TDT 0.6B v3 (ONNX)")
        {
            SizeDescription = "~700 MB (INT8 ONNX)",
            EstimatedSizeMB = 700,
            IsRecommended = true,
            LanguageCount = 25,
        }
    ];

    public IReadOnlyList<string> SupportedLanguages => [];

    public string PythonExecutablePath
    {
        get => _pythonPath;
        set
        {
            _pythonPath = string.IsNullOrWhiteSpace(value) ? ResolveDefaultPythonPath() : value.Trim();
            TrySetSetting("pythonPath", _pythonPath);
        }
    }

    public Task ActivateAsync(IPluginHostServices host)
    {
        _host = host;
        EnsureLogDirectory();
        TryLoadPersistedSettings();
        _pythonPath = string.IsNullOrWhiteSpace(_pythonPath)
            ? ResolveDefaultPythonPath()
            : _pythonPath;

        _accelerationStatus = ParakeetIntelOnnxAcceleration.CreatePendingStatus(
            _accelerationPreference,
            QueryDirectMlAvailable());

        Log(
            PluginLogLevel.Info,
            $"Activated v{PluginVersion}; python='{_pythonPath}', data='{GetDataDirectory()}', directml={_directMlAvailable}");
        return Task.CompletedTask;
    }

    public Task DeactivateAsync()
    {
        StopSidecar();
        return Task.CompletedTask;
    }

    public UserControl? CreateSettingsView() => new ParakeetIntelOnnxSettingsView(this);

    public void SetAccelerationPreference(TranscriptionAccelerationPreference preference)
    {
        _accelerationPreference = preference;
        var directMlAvailable = QueryDirectMlAvailable();
        var desiredPath = ParakeetIntelOnnxAcceleration.ResolveExecutionPath(
            preference,
            directMlAvailable);
        var desiredBackend = desiredPath == ParakeetIntelOnnxAcceleration.ExecutionPath.DirectMl
            ? ParakeetIntelOnnxAcceleration.DirectMlBackendId
            : ParakeetIntelOnnxAcceleration.CpuBackendId;

        if (_loadedBackend is not null
            && !string.Equals(_loadedBackend, desiredBackend, StringComparison.OrdinalIgnoreCase))
        {
            _accelerationStatus = ParakeetIntelOnnxAcceleration.CreateReloadRequiredStatus(
                _loadedBackend,
                desiredBackend);
            return;
        }

        _accelerationStatus = ParakeetIntelOnnxAcceleration.CreatePendingStatus(
            preference,
            directMlAvailable);
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
        Log(PluginLogLevel.Info, "Downloading Parakeet ONNX model...");

        var scriptPath = GetScriptPath("intel_onnx_server.py");
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
            ?? throw new InvalidOperationException("Failed to start intel_onnx setup");

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
                Debug.WriteLine($"[ParakeetIntelOnnx] setup: {line}");
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
        Log(PluginLogLevel.Info, "Parakeet ONNX setup complete");
    }

    public async Task LoadModelAsync(string modelId, CancellationToken ct)
    {
        var loadWatch = Stopwatch.StartNew();
        if (!IsModelDownloaded(modelId))
            throw new FileNotFoundException("Model not set up. Download it first.");

        EnsurePythonExists();
        await VerifyPythonRuntimeAsync(ct);

        var directMlAvailable = QueryDirectMlAvailable();
        var executionPath = ParakeetIntelOnnxAcceleration.ResolveExecutionPath(
            _accelerationPreference,
            directMlAvailable);
        var sidecarDevice = ParakeetIntelOnnxAcceleration.ToSidecarDevice(executionPath);
        var desiredBackend = executionPath == ParakeetIntelOnnxAcceleration.ExecutionPath.DirectMl
            ? ParakeetIntelOnnxAcceleration.DirectMlBackendId
            : ParakeetIntelOnnxAcceleration.CpuBackendId;

        if (executionPath == ParakeetIntelOnnxAcceleration.ExecutionPath.DirectMl && !directMlAvailable)
        {
            _accelerationStatus = ParakeetIntelOnnxAcceleration.CreateDirectMlUnavailableStatus(
                "Install onnxruntime-directml in the plugin Python environment.");
            throw new InvalidOperationException(_accelerationStatus.Detail);
        }

        await _sidecarLock.WaitAsync(ct);
        try
        {
            StopSidecar();
            KillOrphanSidecars();
            StartSidecar(sidecarDevice);

            using var loadCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            loadCts.CancelAfter(TimeSpan.FromMinutes(5));

            var response = await SendCommandAsync(new
            {
                cmd = "load",
                device = sidecarDevice,
            }, loadCts.Token);

            if (response.TryGetProperty("error", out var err))
                throw new InvalidOperationException($"Load failed: {err.GetString()}");

            _loadedBackend = response.TryGetProperty("backend", out var be)
                ? be.GetString()
                : desiredBackend;
            _loadedModelId = modelId;
            _selectedModelId = modelId;

            _accelerationStatus = ParakeetIntelOnnxAcceleration.CreateLoadedStatus(
                _loadedBackend ?? desiredBackend,
                _accelerationPreference);

            Log(
                PluginLogLevel.Info,
                $"Loaded backend='{_loadedBackend}' device={sidecarDevice} in {loadWatch.Elapsed.TotalSeconds:F1}s");
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
        {
            throw new NotSupportedException(
                "This engine does not translate during transcription. " +
                "Set a target language in TypeWhisper settings to translate after transcription.");
        }

        var normalizedLanguage = NormalizeLanguage(language);
        var transcribeWatch = Stopwatch.StartNew();
        await _sidecarLock.WaitAsync(ct);
        try
        {
            if (_sidecar is null || _sidecar.HasExited)
                throw new InvalidOperationException("No model loaded. Call LoadModelAsync first.");

            if (_loadedModelId is null)
                throw new InvalidOperationException("Model is still loading.");

            var response = await SendCommandAsync(new
            {
                cmd = "transcribe",
                audio_base64 = Convert.ToBase64String(wavAudio),
                language = normalizedLanguage,
            }, ct);

            if (response.TryGetProperty("error", out var err))
                throw new InvalidOperationException($"Transcription failed: {err.GetString()}");

            var rawText = response.GetProperty("text").GetString() ?? "";
            var text = SanitizeOutput(rawText);
            var duration = response.GetProperty("duration").GetDouble();
            Log(
                PluginLogLevel.Info,
                $"Transcribe ok: audio={duration:F2}s, elapsed={transcribeWatch.Elapsed.TotalSeconds:F2}s, lang={normalizedLanguage ?? "auto"}, textLen={text.Length}");
            return new PluginTranscriptionResult(text, normalizedLanguage, duration, NoSpeechProbability: null);
        }
        finally
        {
            _sidecarLock.Release();
        }
    }

    public async Task UnloadModelAsync()
    {
        await _sidecarLock.WaitAsync();
        try
        {
            StopSidecar();
            _loadedModelId = null;
            _loadedBackend = null;
            SetAccelerationPreference(_accelerationPreference);
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

    internal static string? NormalizeLanguage(string? language)
    {
        if (string.IsNullOrWhiteSpace(language) || string.Equals(language, "auto", StringComparison.OrdinalIgnoreCase))
            return null;
        return language.Trim().ToLowerInvariant();
    }

    internal bool QueryDirectMlAvailableForTests() => QueryDirectMlAvailable();

    internal void MarkLoadedBackendForTests(string backend) => _loadedBackend = backend;

    private bool QueryDirectMlAvailable()
    {
        if (_directMlAvailable.HasValue)
            return _directMlAvailable.Value;

        if (!File.Exists(PythonExecutablePath))
        {
            _directMlAvailable = false;
            return false;
        }

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = Path.GetFullPath(PythonExecutablePath),
                Arguments =
                    "-c \"import onnxruntime as ort; print('1' if 'DmlExecutionProvider' in ort.get_available_providers() else '0')\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
            };
            ConfigurePythonProcess(psi);
            using var proc = Process.Start(psi);
            if (proc is null)
            {
                _directMlAvailable = false;
                return false;
            }

            proc.WaitForExit(15000);
            var output = proc.StandardOutput.ReadToEnd().Trim();
            _directMlAvailable = output == "1";
            return _directMlAvailable.Value;
        }
        catch
        {
            _directMlAvailable = false;
            return false;
        }
    }

    private async Task EnsurePythonPackagesAsync(IProgress<double>? progress, CancellationToken ct)
    {
        progress?.Report(0.05);
        Log(PluginLogLevel.Info, "Installing Python packages (onnxruntime-directml, onnx-asr)...");

        var reqPath = GetScriptPath("requirements.txt");
        var args = $"-m pip install -q -r \"{reqPath}\"";
        await RunProcessAsync(PythonExecutablePath, args, ct, timeoutMs: 1_800_000);
        _directMlAvailable = null;
        progress?.Report(0.2);
    }

    private void StartSidecar(string device)
    {
        var scriptPath = GetScriptPath("intel_onnx_server.py");
        var dataDir = GetDataDirectory();
        var python = Path.GetFullPath(PythonExecutablePath);

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
        psi.ArgumentList.Add(device);
        ConfigurePythonProcess(psi);

        _sidecar = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start intel_onnx sidecar");
        _sidecarIn = _sidecar.StandardInput;
        _sidecarOut = _sidecar.StandardOutput;
        Log(PluginLogLevel.Info, $"Started sidecar pid={_sidecar.Id} device={device}");
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
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ParakeetIntelOnnx] stderr drain: {ex.Message}");
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
        }

        if (!_sidecar.HasExited)
        {
            try
            {
                _sidecar.Kill();
            }
            catch
            {
            }
        }

        _sidecar.Dispose();
        _sidecar = null;
        _sidecarIn = null;
        _sidecarOut = null;
        _loadedModelId = null;
        _loadedBackend = null;
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

        await _sidecarIn.WriteLineAsync(JsonSerializer.Serialize(wrapper).AsMemory(), ct);
        await _sidecarIn.FlushAsync(ct);

        while (true)
        {
            var response = await _sidecarOut.ReadLineAsync(ct)
                ?? throw new InvalidOperationException("Sidecar closed unexpectedly");

            using var doc = JsonDocument.Parse(response);
            var root = doc.RootElement.Clone();
            if (!root.TryGetProperty("req_id", out var id) || id.GetInt32() != reqId)
                continue;

            if (root.TryGetProperty("pending", out var pending) && pending.GetBoolean())
                continue;

            return root;
        }
    }

    private void EnsurePythonExists()
    {
        if (!File.Exists(PythonExecutablePath))
        {
            throw new FileNotFoundException(
                $"Python not found: {PythonExecutablePath}. Configure it in plugin settings.",
                PythonExecutablePath);
        }
    }

    private async Task VerifyPythonRuntimeAsync(CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            FileName = Path.GetFullPath(PythonExecutablePath),
            Arguments = "-c \"import onnxruntime, onnx_asr; print('ok')\"",
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
        await proc.WaitForExitAsync(cts.Token);

        if (proc.ExitCode != 0)
        {
            var stderr = await proc.StandardError.ReadToEndAsync(ct);
            throw new InvalidOperationException(
                $"Python missing onnxruntime/onnx-asr. pip install -r Scripts/requirements.txt. {Tail(stderr, 300)}");
        }

        _directMlAvailable = null;
    }

    private static void KillOrphanSidecars()
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments =
                    "-NoProfile -Command \"Get-CimInstance Win32_Process -Filter \\\"Name='python.exe'\\\" | Where-Object { $_.CommandLine -like '*intel_onnx_server.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }\"",
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi);
            proc?.WaitForExit(8000);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ParakeetIntelOnnx] KillOrphanSidecars: {ex.Message}");
        }
    }

    private void Log(PluginLogLevel level, string message)
    {
        var line = $"[{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss.fff zzz}] [{level}] {message}";
        try
        {
            File.AppendAllText(GetLogFilePath(), line + Environment.NewLine, Encoding.UTF8);
        }
        catch
        {
        }

        try
        {
            _host?.Log(level, message);
        }
        catch
        {
        }

        Debug.WriteLine($"[ParakeetIntelOnnx] {message}");
    }

    private string GetDataDirectory()
    {
        if (_host is not null)
            return _host.PluginDataDirectory;

        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TypeWhisper",
            "PluginData",
            PluginId);
    }

    private string GetLogFilePath()
    {
        EnsureLogDirectory();
        return Path.Combine(GetDataDirectory(), "logs", "parakeet-intel-onnx-plugin.log");
    }

    private void EnsureLogDirectory() =>
        Directory.CreateDirectory(Path.Combine(GetDataDirectory(), "logs"));

    private void TryLoadPersistedSettings()
    {
        if (_host is null)
            return;

        try
        {
            _pythonPath = _host.GetSetting<string>("pythonPath") ?? ResolveDefaultPythonPath();
            return;
        }
        catch (MissingMethodException)
        {
        }

        try
        {
            var settingsPath = Path.Combine(GetDataDirectory(), "settings.json");
            if (!File.Exists(settingsPath))
                return;

            using var doc = JsonDocument.Parse(File.ReadAllText(settingsPath));
            if (doc.RootElement.TryGetProperty("pythonPath", out var python))
                _pythonPath = python.GetString() ?? ResolveDefaultPythonPath();
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"[ParakeetIntelOnnx] settings load failed: {ex.Message}");
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
        }
    }

    private static string ResolveDefaultPythonPath()
    {
        foreach (var root in new[] { @"C:\MyWhisper", @"C:\WhisperNPU" })
        {
            var candidate = Path.Combine(root, "venv", "Scripts", "python.exe");
            if (File.Exists(candidate))
                return candidate;
        }

        return Path.Combine(@"C:\MyWhisper", "venv", "Scripts", "python.exe");
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
        psi.Environment["PARAKEET_INTEL_ONNX_LOG"] =
            Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "TypeWhisper",
                "PluginData",
                "com.tolkium.parakeet-intel-onnx",
                "logs",
                "parakeet-intel-onnx-sidecar.log");
    }

    private static string GetScriptPath(string fileName)
    {
        var pluginDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? ".";
        return Path.Combine(pluginDir, "Scripts", fileName);
    }

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
