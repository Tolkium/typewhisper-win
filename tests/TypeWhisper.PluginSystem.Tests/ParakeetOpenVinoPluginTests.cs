using System.IO;
using System.Text.Json;
using TypeWhisper.Plugin.ParakeetOpenVino;
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
        var resolution = ParakeetOpenVinoAcceleration.Resolve(preference, preferredDevice);
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
    public void MaxRecordingDuration_OnlyLimitsNpu()
    {
        var sut = new ParakeetOpenVinoPlugin { PreferredDevice = "NPU" };
        Assert.Equal(TimeSpan.FromSeconds(20), sut.MaxRecordingDuration);
        Assert.Contains("20 s", sut.MaxRecordingDurationReason);

        sut.PreferredDevice = "GPU";
        Assert.Null(sut.MaxRecordingDuration);
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
}
