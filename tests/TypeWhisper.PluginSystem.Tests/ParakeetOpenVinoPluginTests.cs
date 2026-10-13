using System.IO;
using System.Text.Json;
using TypeWhisper.Plugin.ParakeetOpenVino;
using TypeWhisper.PluginSDK;
using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.PluginSystem.Tests;

public class ParakeetOpenVinoPluginTests
{
    [Fact]
    public void PluginVersion_MatchesManifestVersion()
    {
        var repoRoot = Path.GetFullPath(Path.Join(
            AppContext.BaseDirectory,
            "..", "..", "..", "..", ".."));
        var manifestPath = Path.Join(
            repoRoot,
            "plugins",
            "TypeWhisper.Plugin.ParakeetOpenVino",
            "manifest.json");
        var manifest = JsonSerializer.Deserialize<PluginManifest>(
            File.ReadAllText(manifestPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        var sut = new ParakeetOpenVinoPlugin();

        Assert.NotNull(manifest);
        Assert.Equal(sut.PluginVersion, manifest.Version);
    }

    [Theory]
    [InlineData(TranscriptionAccelerationPreference.Auto, "GPU", "GPU")]
    [InlineData(TranscriptionAccelerationPreference.Auto, "NPU", "NPU")]
    [InlineData(TranscriptionAccelerationPreference.Cpu, "NPU", "CPU")]
    [InlineData(TranscriptionAccelerationPreference.NvidiaCuda, "GPU", "GPU")]
    [InlineData(TranscriptionAccelerationPreference.AmdVulkan, "NPU", "NPU")]
    [InlineData(TranscriptionAccelerationPreference.AmdRocm, "GPU", "GPU")]
    public void Resolve_UsesPreferredDeviceUnlessHostAsksForCpu(
        TranscriptionAccelerationPreference preference,
        string preferredDevice,
        string expectedDevice)
    {
        var resolution = ParakeetOpenVinoAcceleration.Resolve(preference, preferredDevice, new PluginText(null));
        Assert.Equal(expectedDevice, resolution.TargetDevice);
    }

    [Fact]
    public void SetAccelerationPreference_NvidiaCuda_ExplainsRemapToIntelGpu()
    {
        var sut = new ParakeetOpenVinoPlugin();
        sut.SetAccelerationPreference(TranscriptionAccelerationPreference.NvidiaCuda);

        Assert.Contains("NVIDIA CUDA is not supported", sut.AccelerationStatus.Detail ?? "");
        Assert.Contains("Intel GPU", sut.AccelerationStatus.Detail ?? "");
        Assert.Equal("GPU", sut.TargetDevice);
    }

    [Fact]
    public void PreferredNpu_SurvivesHostAutoPreference()
    {
        var sut = new ParakeetOpenVinoPlugin { PreferredDevice = "NPU" };
        sut.SetAccelerationPreference(TranscriptionAccelerationPreference.Auto);

        Assert.Equal("NPU", sut.TargetDevice);
        Assert.Equal("NPU", sut.PreferredDevice);
    }

    [Fact]
    public void NpuStatus_StatesTwentySecondLimit()
    {
        var sut = new ParakeetOpenVinoPlugin { PreferredDevice = "NPU" };

        Assert.Contains("first 20 s", sut.AccelerationStatus.Detail ?? "");
    }

    [Fact]
    public async Task LimitWarning_ShowsAfterLimitWhenRecordingContinues()
    {
        var delay = new ManualDelay();
        var shown = 0;
        using var warning = new NpuLimitWarning(() => TimeSpan.FromSeconds(20), () => shown++, delay.WaitAsync);

        warning.OnRecordingStarted();
        Assert.Equal(TimeSpan.FromSeconds(20), delay.Requested);
        await delay.ElapseAsync();

        Assert.Equal(1, shown);
    }

    [Fact]
    public async Task LimitWarning_NotShownWhenRecordingStopsInTime()
    {
        var delay = new ManualDelay();
        var shown = 0;
        using var warning = new NpuLimitWarning(() => TimeSpan.FromSeconds(20), () => shown++, delay.WaitAsync);

        warning.OnRecordingStarted();
        warning.OnRecordingStopped();
        await delay.ElapseAsync();

        Assert.Equal(0, shown);
    }

    [Fact]
    public void LimitWarning_NotTimedWithoutLimit()
    {
        var delay = new ManualDelay();
        using var warning = new NpuLimitWarning(() => null, () => { }, delay.WaitAsync);

        warning.OnRecordingStarted();

        Assert.Null(delay.Requested);
    }

    [Fact]
    public void DeviceOptions_OfferGpuNpuAndCpu()
    {
        Assert.Equal(["GPU", "NPU", "CPU"], ParakeetOpenVinoPlugin.DeviceOptions);
    }

    [Theory]
    [InlineData("npu", "NPU")]
    [InlineData("gpu", "GPU")]
    [InlineData("cpu", "CPU")]
    [InlineData("unknown", "GPU")]
    public void PreferredDevice_NormalizesInput(string input, string expected)
    {
        var sut = new ParakeetOpenVinoPlugin { PreferredDevice = input };
        Assert.Equal(expected, sut.PreferredDevice);
    }

    [Fact]
    public async Task TranscribeAsync_ThrowsWhenTranslateRequested()
    {
        var sut = new ParakeetOpenVinoPlugin();
        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => sut.TranscribeAsync([], "en", translate: true, prompt: null, CancellationToken.None));

        Assert.Contains("target language", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Localization_GermanHasEveryEnglishKey()
    {
        var english = ReadLocalization("en");
        var german = ReadLocalization("de");

        Assert.Empty(english.Keys.Except(german.Keys));
        Assert.Empty(german.Keys.Except(english.Keys));
    }

    [Fact]
    public async Task LimitWarning_UsesHostLanguage()
    {
        var host = new FakeHost(new DictionaryLocalization(ReadLocalization("de")));
        var sut = new ParakeetOpenVinoPlugin();
        await sut.ActivateAsync(host);

        Assert.StartsWith("NPU-Limit erreicht: 20 s pro Aufnahme.", sut.Text.Get("Limit.Warning", 20));
    }

    private static Dictionary<string, string> ReadLocalization(string language)
    {
        var path = Path.Join(AppContext.BaseDirectory, "Localization", $"{language}.json");
        if (!File.Exists(path))
        {
            path = Path.GetFullPath(Path.Join(
                AppContext.BaseDirectory, "..", "..", "..", "..", "..",
                "plugins", "TypeWhisper.Plugin.ParakeetOpenVino", "Localization", $"{language}.json"));
        }

        return JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(path))!;
    }

    private sealed class DictionaryLocalization(Dictionary<string, string> strings) : IPluginLocalization
    {
        public string CurrentLanguage => "de";
        public IReadOnlyList<string> AvailableLanguages => ["de"];
        public string GetString(string key) => strings.GetValueOrDefault(key, key);
        public string GetString(string key, params object[] args) => string.Format(GetString(key), args);
    }

    private sealed class FakeHost(IPluginLocalization localization) : IPluginHostServices
    {
        public IPluginLocalization Localization => localization;
        public string PluginDataDirectory => Path.GetTempPath();
        public string? ActiveAppProcessName => null;
        public string? ActiveAppName => null;
        public IPluginEventBus EventBus => throw new NotSupportedException();
        public IReadOnlyList<string> AvailableProfileNames => [];
        public Task StoreSecretAsync(string key, string value) => Task.CompletedTask;
        public Task<string?> LoadSecretAsync(string key) => Task.FromResult<string?>(null);
        public Task DeleteSecretAsync(string key) => Task.CompletedTask;
        public T? GetSetting<T>(string key) => default;
        public void SetSetting<T>(string key, T value) { }
        public void Log(PluginLogLevel level, string message) { }
        public void NotifyCapabilitiesChanged() { }
    }

    /// <summary>Delay that completes only when the test says so, so timing tests do not sleep.</summary>
    private sealed class ManualDelay
    {
        private readonly TaskCompletionSource _elapsed = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private Task? _pending;

        public TimeSpan? Requested { get; private set; }

        public Task WaitAsync(TimeSpan delay, CancellationToken token)
        {
            Requested = delay;
            _pending = _elapsed.Task.WaitAsync(token);
            return _pending;
        }

        public async Task ElapseAsync()
        {
            _elapsed.TrySetResult();
            if (_pending is not null)
                await Task.WhenAny(_pending, Task.Delay(1000));
            await Task.Delay(50); // let the continuation run
        }
    }
}
