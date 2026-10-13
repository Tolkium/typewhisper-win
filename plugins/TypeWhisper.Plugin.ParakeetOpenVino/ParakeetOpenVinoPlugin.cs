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

    private readonly SemaphoreSlim _engineLock = new(1, 1);
    private IPluginHostServices? _host;
    private ParakeetOpenVinoEngine? _engine;
    private string? _selectedModelId;
    private string? _loadedModelId;
    private string? _loadedDevice;
    private string _targetDevice = "GPU";
    private string _preferredDevice = "GPU";
    private TranscriptionAccelerationPreference _accelerationPreference =
        TranscriptionAccelerationPreference.Auto;
    private TranscriptionAccelerationStatus _accelerationStatus = new(
        TranscriptionAccelerationBackend.NvidiaCuda,
        ParakeetOpenVinoAcceleration.GpuDisplayText);

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

    public TranscriptionAccelerationStatus AccelerationStatus => _accelerationStatus;

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

    /// <summary>
    /// On NPU the static encoder window holds 20 s of audio. The host stops dictation early enough that
    /// the audio it sends, including its own tail padding, fits in the window.
    /// Hosts without this SDK member ignore it; the engine then still cuts audio to 20 s.
    /// </summary>
    public TimeSpan? MaxRecordingDuration =>
        (_loadedDevice ?? _targetDevice) == "NPU"
            ? TimeSpan.FromSeconds(ParakeetOpenVinoAcceleration.NpuMaxAudioSeconds)
            : null;

    public string? MaxRecordingDurationReason =>
        $"NPU limit: {ParakeetOpenVinoAcceleration.NpuMaxAudioSeconds} s per recording. Switch to GPU for longer dictation.";

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

    private ParakeetSetup Setup => new(DataDirectory);

    private string LogFilePath => Path.Combine(DataDirectory, "logs", "parakeet-openvino.log");

    public Task ActivateAsync(IPluginHostServices host)
    {
        _host = host;
        TryLoadPersistedSettings();
        ApplyAccelerationPreference(_accelerationPreference);
        Log(PluginLogLevel.Info, $"Activated plugin v{PluginVersion}; device='{_targetDevice}', data='{DataDirectory}'");
        return Task.CompletedTask;
    }

    public Task DeactivateAsync()
    {
        DisposeEngine();
        return Task.CompletedTask;
    }

    public UserControl? CreateSettingsView() => new ParakeetOpenVinoSettingsView(this);

    public void SetAccelerationPreference(TranscriptionAccelerationPreference preference) =>
        ApplyAccelerationPreference(preference);

    private void ApplyAccelerationPreference(TranscriptionAccelerationPreference preference)
    {
        _accelerationPreference = preference;
        var resolution = ParakeetOpenVinoAcceleration.Resolve(preference, _preferredDevice);
        _targetDevice = resolution.TargetDevice;
        _accelerationStatus = ParakeetOpenVinoAcceleration.ToStatus(resolution, _loadedDevice, null);
        Log(PluginLogLevel.Info, $"Acceleration preference {preference} -> OpenVINO device {_targetDevice}");
    }

    public void SelectModel(string modelId)
    {
        if (modelId != ModelId)
            throw new ArgumentException($"Unknown model: {modelId}");
        _selectedModelId = modelId;
    }

    public bool IsModelDownloaded(string modelId) => Setup.IsReady;

    public async Task DownloadModelAsync(string modelId, IProgress<double>? progress, CancellationToken ct)
    {
        var watch = Stopwatch.StartNew();
        try
        {
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
        if (!setup.IsReady)
            throw new FileNotFoundException("Model or OpenVINO runtime not set up. Download the model first.");

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

            _loadedModelId = modelId;
            _selectedModelId = modelId;
            _loadedDevice = device;
            _accelerationStatus = ParakeetOpenVinoAcceleration.ToStatus(
                ParakeetOpenVinoAcceleration.Resolve(_accelerationPreference, _preferredDevice),
                _loadedDevice,
                $"OpenVINO {ParakeetSetup.OpenVinoVersion}");
            Log(PluginLogLevel.Info, $"Parakeet loaded on {device} (decoder {_engine.DecoderDevice}) in {watch.Elapsed.TotalSeconds:F1}s");
        }
        catch (Exception ex)
        {
            Log(PluginLogLevel.Error, $"LoadModelAsync failed after {watch.Elapsed.TotalSeconds:F1}s: {ex}");
            DisposeEngine();
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
            throw new NotSupportedException(
                "Parakeet OpenVINO does not translate during transcription. " +
                "Set a target language in TypeWhisper to translate after transcription.");

        var watch = Stopwatch.StartNew();
        await _engineLock.WaitAsync(ct);
        try
        {
            var engine = _engine ?? throw new InvalidOperationException("No model loaded. Call LoadModelAsync first.");
            var samples = WavReader.ReadMono16k(wavAudio);
            var result = await Task.Run(() => engine.Transcribe(samples), ct);

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
        DisposeEngine();
        _engineLock.Dispose();
    }

    private void DisposeEngine()
    {
        _engine?.Dispose();
        _engine = null;
        _loadedModelId = null;
        _loadedDevice = null;
    }

    private void Log(PluginLogLevel level, string message)
    {
        try
        {
            var path = LogFilePath;
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
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
