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
    [InlineData(TranscriptionAccelerationPreference.Auto, "GPU")]
    [InlineData(TranscriptionAccelerationPreference.Cpu, "CPU")]
    [InlineData(TranscriptionAccelerationPreference.NvidiaCuda, "GPU")]
    [InlineData(TranscriptionAccelerationPreference.AmdVulkan, "GPU")]
    [InlineData(TranscriptionAccelerationPreference.AmdRocm, "GPU")]
    public void Resolve_MapsHostAccelerationToOpenVinoDevice(
        TranscriptionAccelerationPreference preference,
        string expectedDevice)
    {
        var resolution = ParakeetOpenVinoAcceleration.Resolve(preference);
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
    public void DeviceOptions_DoNotOfferNpu()
    {
        Assert.Equal(["GPU", "CPU"], ParakeetOpenVinoPlugin.DeviceOptions);
    }

    [Theory]
    [InlineData("NPU", "GPU")]
    [InlineData("gpu", "GPU")]
    [InlineData("cpu", "CPU")]
    public void TargetDevice_NormalizesLegacyAndCaseVariants(string input, string expected)
    {
        var sut = new ParakeetOpenVinoPlugin { TargetDevice = input };
        Assert.Equal(expected, sut.TargetDevice);
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
