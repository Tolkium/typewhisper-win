using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows.Controls;
using TypeWhisper.Plugin.ParakeetOpenVino.Engine;
using TypeWhisper.PluginSDK;
using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Parakeet TDT 0.6B v3 on Intel OpenVINO (GPU/NPU/CPU), running in-process through the
/// OpenVINO C API. No Python: the model and the OpenVINO runtime are downloaded by the
/// host's model download step into the plugin data folder.
/// </summary>
public sealed class ParakeetOpenVinoPlugin : ITypeWhisperPlugin, ITranscriptionEnginePlugin
{
    public const string ModelId = "parakeet-tdt-0.6b-v3-openvino";

    public static readonly IReadOnlyList<string> DeviceOptions = ["GPU", "NPU", "CPU"];

    /// <summary>How long Dispose waits for a running transcription or load before leaking the engine.</summary>
    private static readonly TimeSpan DisposeLockTimeout = TimeSpan.FromSeconds(5);

    private readonly SemaphoreSlim _engineLock = new(1, 1);
    private IPluginHostServices? _host;
    private ParakeetOpenVinoEngine? _engine;
    private ParakeetSetup? _setup;
    private bool _setupReady;
    private bool _logDirectoryCreated;
    private NpuLimitWarning? _limitWarning;
    private readonly List<IDisposable> _eventSubscriptions = [];
    private string? _selectedModelId;
    private string? _loadedDevice;
    private string _targetDevice = "GPU";
    private string _preferredDevice = "GPU";
    private TranscriptionAccelerationPreference _accelerationPreference =
        TranscriptionAccelerationPreference.Auto;
    private TranscriptionAccelerationStatus? _accelerationStatus;
    private PluginText? _text;

    public string PluginId => "com.whispernpu.parakeet-openvino";
    public string PluginName => "Parakeet OpenVINO (Intel GPU/NPU)";
    public string PluginVersion => "0.6.0";

    public string ProviderId => "parakeet-openvino";
    public string ProviderDisplayName => "Parakeet OpenVINO (local)";
    public bool IsConfigured => true;
    public string? SelectedModelId => _selectedModelId;
    public bool SupportsTranslation => false;
    public bool SupportsModelDownload => true;

    public IReadOnlyList<TranscriptionAccelerationBackend> SupportedAccelerationBackends { get; } =
    [
        TranscriptionAccelerationBackend.Cpu,
        TranscriptionAccelerationBackend.NvidiaCuda,
    ];

    public TranscriptionAccelerationPreference AccelerationPreference => _accelerationPreference;

    public TranscriptionAccelerationStatus AccelerationStatus => _accelerationStatus ??= ComputeAccelerationStatus();

    /// <summary>User-facing text in the app language (Localization/*.json).</summary>
    internal PluginText Text => _text ??= new PluginText(_host?.Localization);

    public IReadOnlyList<PluginModelInfo> TranscriptionModels { get; } =
    [
        new(ModelId, "Parakeet TDT 0.6B v3 (OpenVINO)")
        {
            SizeDescription = "~2.6 GB model + ~0.2 GB OpenVINO runtime",
            EstimatedSizeMB = 2750,
            IsRecommended = true,
            LanguageCount = 25,
        }
    ];

    public IReadOnlyList<string> SupportedLanguages => [];

    /// <summary>Input window of the loaded engine (NPU: 20 s), or null when it accepts any length.</summary>
    internal TimeSpan? ActiveAudioLimit => _engine?.MaxAudioDuration;

    /// <summary>Device the next model load will use, resolved from host acceleration and <see cref="PreferredDevice"/>.</summary>
    public string TargetDevice => _targetDevice;

    /// <summary>Device chosen in plugin settings; used whenever the host acceleration is not CPU.</summary>
    public string PreferredDevice
    {
        get => _preferredDevice;
        set
        {
            var normalized = NormalizeDevice(value);
            if (string.Equals(_preferredDevice, normalized, StringComparison.Ordinal))
                return;

            _preferredDevice = normalized;
            TrySetSetting("targetDevice", _preferredDevice);
            ApplyAccelerationPreference(_accelerationPreference);
            Log(PluginLogLevel.Info, $"Preferred device changed to {_preferredDevice}; reload model to apply");
        }
    }

    /// <summary>Folder holding the model, the OpenVINO runtime, the compile cache and logs.</summary>
    public string DataDirectory => _host?.PluginDataDirectory ?? Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "TypeWhisper",
        "PluginData",
        PluginId);

    /// <summary>Download/readiness state for <see cref="DataDirectory"/>; reset when the host changes it.</summary>
    private ParakeetSetup Setup => _setup ??= new ParakeetSetup(DataDirectory);

    private string LogFilePath => Path.Combine(DataDirectory, "logs", "parakeet-openvino.log");

    public Task ActivateAsync(IPluginHostServices host)
    {
        _host = host;
        _setup = null; // the data directory and localization now come from the host
        _text = null;
        _setupReady = false;
        TryLoadPersistedSettings();
        ApplyAccelerationPreference(_accelerationPreference);
        SubscribeToRecordingEvents(host);
        Log(PluginLogLevel.Info, $"Activated plugin v{PluginVersion}; device='{_targetDevice}', data='{DataDirectory}'");
        return Task.CompletedTask;
    }

    public async Task DeactivateAsync()
    {
        UnsubscribeFromRecordingEvents();
        // Wait for any running load/transcription: disposing native handles under it would crash.
        await _engineLock.WaitAsync();
        try
        {
            DisposeEngine();
        }
        finally
        {
            _engineLock.Release();
        }
    }

    public UserControl? CreateSettingsView() => new ParakeetOpenVinoSettingsView(this);

    public void SetAccelerationPreference(TranscriptionAccelerationPreference preference) =>
        ApplyAccelerationPreference(preference);

    private void ApplyAccelerationPreference(TranscriptionAccelerationPreference preference)
    {
        _accelerationPreference = preference;
        _targetDevice = ParakeetOpenVinoAcceleration.Resolve(preference, _preferredDevice, Text).TargetDevice;
        _accelerationStatus = ComputeAccelerationStatus();
        Log(PluginLogLevel.Info, $"Acceleration preference {preference} -> OpenVINO device {_targetDevice}");
    }

    public void SelectModel(string modelId)
    {
        if (modelId != ModelId)
            throw new ArgumentException($"Unknown model: {modelId}");
        _selectedModelId = modelId;
    }

    /// <summary>Called often by the host UI; a positive result is cached until the next download or failed load.</summary>
    public bool IsModelDownloaded(string modelId) => _setupReady || (_setupReady = Setup.IsReady);

    public async Task DownloadModelAsync(string modelId, IProgress<double>? progress, CancellationToken ct)
    {
        var watch = Stopwatch.StartNew();
        try
        {
            _setupReady = false;
            await Setup.EnsureAsync(progress, message => Log(PluginLogLevel.Info, message), ct);
            Log(PluginLogLevel.Info, $"Setup complete in {watch.Elapsed.TotalSeconds:F0}s");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            Log(PluginLogLevel.Error, $"Setup failed after {watch.Elapsed.TotalSeconds:F0}s: {ex}");
            throw;
        }
    }

    public async Task LoadModelAsync(string modelId, CancellationToken ct)
    {
        var watch = Stopwatch.StartNew();
        var setup = Setup;
        _setupReady = setup.IsReady;
        if (!_setupReady)
            throw new FileNotFoundException(Text.Get("Error.NotSetUp"));

        await _engineLock.WaitAsync(ct);
        try
        {
            DisposeEngine();
            var device = TargetDevice;
            Log(PluginLogLevel.Info, $"Loading Parakeet on {device} (first NPU load compiles for ~1 min)");

            // Native compile cannot be cancelled once started; run it off the UI thread.
            _engine = await Task.Run(() =>
            {
                OpenVinoLoader.Load(setup.RuntimeDirectory);
                var engine = new ParakeetOpenVinoEngine(setup.WeightsDirectory, device, setup.CacheDirectory);
                try
                {
                    engine.WarmUp();
                }
                catch
                {
                    engine.Dispose();
                    throw;
                }
                return engine;
            }, ct);

            _selectedModelId = modelId;
            _loadedDevice = device;
            ApplyAccelerationPreference(_accelerationPreference);
            Log(PluginLogLevel.Info, $"Parakeet loaded on {device} with OpenVINO {OpenVinoLoader.Version()} in {watch.Elapsed.TotalSeconds:F1}s");
        }
        catch (Exception ex)
        {
            Log(PluginLogLevel.Error, $"LoadModelAsync failed after {watch.Elapsed.TotalSeconds:F1}s: {ex}");
            DisposeEngine();
            ApplyAccelerationPreference(_accelerationPreference); // status must not show the old engine
            throw;
        }
        finally
        {
            _engineLock.Release();
        }
    }

    public async Task<PluginTranscriptionResult> TranscribeAsync(
        byte[] wavAudio, string? language, bool translate, string? prompt, CancellationToken ct)
    {
        if (translate)
            throw new NotSupportedException(Text.Get("Error.TranslationNotSupported"));

        var watch = Stopwatch.StartNew();
        await _engineLock.WaitAsync(ct);
        try
        {
            var engine = _engine ?? throw new InvalidOperationException(Text.Get("Error.NoModelLoaded"));
            var result = await Task.Run(() => engine.Transcribe(WavReader.ReadMono16k(wavAudio)), ct);

            if (result.Truncated)
            {
                Log(
                    PluginLogLevel.Warning,
                    $"NPU limit: only the first {ParakeetOpenVinoAcceleration.NpuMaxAudioSeconds} s of {result.AudioSeconds:F1} s were transcribed");
            }

            var text = SanitizeOutput(result.Text);
            Log(PluginLogLevel.Info, $"Transcribe ok: audio={result.AudioSeconds:F2}s, elapsed={watch.Elapsed.TotalSeconds:F2}s, textLen={text.Length}");
            return new PluginTranscriptionResult(text, language, result.AudioSeconds, NoSpeechProbability: null);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            Log(PluginLogLevel.Error, $"Transcribe failed after {watch.Elapsed.TotalSeconds:F2}s: {ex}");
            throw;
        }
        finally
        {
            _engineLock.Release();
        }
    }

    /// <summary>Drops &lt;pad&gt; tokens; output that is mostly padding (short live-poll buffers) becomes empty.</summary>
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
        await _engineLock.WaitAsync();
        try
        {
            DisposeEngine();
            ApplyAccelerationPreference(_accelerationPreference);
        }
        finally
        {
            _engineLock.Release();
        }
    }

    public void Dispose()
    {
        UnsubscribeFromRecordingEvents();

        // Same race as DeactivateAsync, but synchronous. If a native compile is still running, leak the
        // engine rather than free handles it is using. The semaphore is not disposed (it has no wait
        // handle), so a caller still waiting on it gets "No model loaded" instead of ObjectDisposedException.
        if (!_engineLock.Wait(DisposeLockTimeout))
            return;

        try
        {
            DisposeEngine();
        }
        finally
        {
            _engineLock.Release();
        }
    }

    /// <summary>
    /// Times each dictation from the host's recording events so the user is warned when it passes the NPU
    /// window. Stock TypeWhisper offers plugins no way to stop a recording, so this is a warning only.
    /// </summary>
    private void SubscribeToRecordingEvents(IPluginHostServices host)
    {
        UnsubscribeFromRecordingEvents();
        _limitWarning = new NpuLimitWarning(
            () => ActiveAudioLimit,
            () => NpuLimitPopup.ShowMessage(Text.Get("Limit.Warning", ParakeetOpenVinoAcceleration.NpuMaxAudioSeconds)));
        try
        {
            _eventSubscriptions.Add(host.EventBus.Subscribe<RecordingStartedEvent>(_ =>
            {
                _limitWarning?.OnRecordingStarted();
                return Task.CompletedTask;
            }));
            _eventSubscriptions.Add(host.EventBus.Subscribe<RecordingStoppedEvent>(_ =>
            {
                _limitWarning?.OnRecordingStopped();
                return Task.CompletedTask;
            }));
        }
        catch (Exception ex) when (ex is NotSupportedException or MissingMethodException)
        {
            Log(PluginLogLevel.Warning, $"Recording events unavailable; no NPU limit warning: {ex.Message}");
        }
    }

    private void UnsubscribeFromRecordingEvents()
    {
        foreach (var subscription in _eventSubscriptions)
            subscription.Dispose();
        _eventSubscriptions.Clear();
        _limitWarning?.Dispose();
        _limitWarning = null;
    }

    private string? ActiveBackendText =>
        _loadedDevice is null ? null : Text.Get("Status.Runtime", ParakeetSetup.OpenVinoVersion);

    private TranscriptionAccelerationStatus ComputeAccelerationStatus() =>
        ParakeetOpenVinoAcceleration.ToStatus(
            ParakeetOpenVinoAcceleration.Resolve(_accelerationPreference, _preferredDevice, Text),
            _loadedDevice,
            ActiveBackendText,
            Text);

    private void DisposeEngine()
    {
        _engine?.Dispose();
        _engine = null;
        _loadedDevice = null;
    }

    private void Log(PluginLogLevel level, string message)
    {
        try
        {
            var path = LogFilePath;
            if (!_logDirectoryCreated)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                _logDirectoryCreated = true;
            }
            File.AppendAllText(path, $"[{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss.fff zzz}] [{level}] {message}{Environment.NewLine}", Encoding.UTF8);
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

    private void TryLoadPersistedSettings()
    {
        if (_host is null)
            return;

        try
        {
            _preferredDevice = NormalizeDevice(_host.GetSetting<string>("targetDevice") ?? "GPU");
            return;
        }
        catch (MissingMethodException)
        {
            // Host SDK older than plugin build; fall back to settings.json.
        }

        try
        {
            var settingsPath = Path.Combine(DataDirectory, "settings.json");
            if (!File.Exists(settingsPath))
                return;

            using var doc = JsonDocument.Parse(File.ReadAllText(settingsPath));
            if (doc.RootElement.TryGetProperty("targetDevice", out var device))
                _preferredDevice = NormalizeDevice(device.GetString());
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

    private static string NormalizeDevice(string? device) =>
        device?.ToUpperInvariant() switch
        {
            "CPU" => "CPU",
            "NPU" => "NPU",
            _ => "GPU",
        };
}
