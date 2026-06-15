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
    [InlineData(TranscriptionAccelerationPreference.Auto, "NPU")]
    [InlineData(TranscriptionAccelerationPreference.Cpu, "CPU")]
    [InlineData(TranscriptionAccelerationPreference.NvidiaCuda, "NPU")]
    [InlineData(TranscriptionAccelerationPreference.AmdVulkan, "NPU")]
    public void Resolve_MapsHostAccelerationToOpenVinoDevice(
        TranscriptionAccelerationPreference preference,
        string expectedDevice)
    {
        var resolution = ParakeetOpenVinoAcceleration.Resolve(preference);
        Assert.Equal(expectedDevice, resolution.TargetDevice);
    }

    [Theory]
    [InlineData(TranscriptionAccelerationPreference.NvidiaCuda)]
    [InlineData(TranscriptionAccelerationPreference.AmdVulkan)]
    [InlineData(TranscriptionAccelerationPreference.AmdRocm)]
    public void GetSuggestedHostAcceleration_UnsupportedGpuSelectionsSuggestAuto(
        TranscriptionAccelerationPreference preference)
    {
        var suggested = ParakeetOpenVinoAcceleration.GetSuggestedHostAcceleration(preference);
        Assert.Equal(ParakeetOpenVinoAcceleration.HostAccelerationAuto, suggested);
    }

    [Fact]
    public void SetAccelerationPreference_NvidiaCuda_ExplainsUnsupportedBackend()
    {
        var sut = new ParakeetOpenVinoPlugin();
        sut.SetAccelerationPreference(TranscriptionAccelerationPreference.NvidiaCuda);

        Assert.Contains("NVIDIA CUDA is not supported", sut.AccelerationStatus.Detail ?? "");
        Assert.Contains("Switched to Auto", sut.AccelerationStatus.Detail ?? "");
        Assert.Equal("NPU", sut.TargetDevice);
    }

    [Fact]
    public void GetSuggestedHostAcceleration_AutoAndCpu_ReturnNull()
    {
        var sut = new ParakeetOpenVinoPlugin();
        Assert.Null(sut.GetSuggestedHostAcceleration(TranscriptionAccelerationPreference.Auto));
        Assert.Null(sut.GetSuggestedHostAcceleration(TranscriptionAccelerationPreference.Cpu));
        Assert.Equal(
            ParakeetOpenVinoAcceleration.HostAccelerationAuto,
            sut.GetSuggestedHostAcceleration(TranscriptionAccelerationPreference.NvidiaCuda));
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
