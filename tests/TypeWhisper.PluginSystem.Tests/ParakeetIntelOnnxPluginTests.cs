using System.IO;
using System.Text.Json;
using TypeWhisper.Plugin.ParakeetIntelOnnx;
using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.PluginSystem.Tests;

public class ParakeetIntelOnnxPluginTests
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
            "TypeWhisper.Plugin.ParakeetIntelOnnx",
            "manifest.json");
        var manifest = JsonSerializer.Deserialize<PluginManifest>(
            File.ReadAllText(manifestPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        var sut = new ParakeetIntelOnnxPlugin();

        Assert.NotNull(manifest);
        Assert.Equal("1.0.0", manifest.Version);
        Assert.Equal(manifest.Version, sut.PluginVersion);
        Assert.Equal("com.tolkium.parakeet-intel-onnx", manifest.Id);
        Assert.Equal(manifest.Id, sut.PluginId);
    }

    [Theory]
    [InlineData(TranscriptionAccelerationPreference.Auto, true, ParakeetIntelOnnxAcceleration.ExecutionPath.DirectMl)]
    [InlineData(TranscriptionAccelerationPreference.Auto, false, ParakeetIntelOnnxAcceleration.ExecutionPath.Cpu)]
    [InlineData(TranscriptionAccelerationPreference.Cpu, true, ParakeetIntelOnnxAcceleration.ExecutionPath.Cpu)]
    [InlineData(TranscriptionAccelerationPreference.Cpu, false, ParakeetIntelOnnxAcceleration.ExecutionPath.Cpu)]
    [InlineData(TranscriptionAccelerationPreference.NvidiaCuda, true, ParakeetIntelOnnxAcceleration.ExecutionPath.DirectMl)]
    [InlineData(TranscriptionAccelerationPreference.NvidiaCuda, false, ParakeetIntelOnnxAcceleration.ExecutionPath.Cpu)]
    [InlineData(TranscriptionAccelerationPreference.AmdVulkan, true, ParakeetIntelOnnxAcceleration.ExecutionPath.DirectMl)]
    [InlineData(TranscriptionAccelerationPreference.AmdRocm, false, ParakeetIntelOnnxAcceleration.ExecutionPath.Cpu)]
    public void ResolveExecutionPath_MapsHostAcceleration(
        TranscriptionAccelerationPreference preference,
        bool directMlAvailable,
        ParakeetIntelOnnxAcceleration.ExecutionPath expected)
    {
        var path = ParakeetIntelOnnxAcceleration.ResolveExecutionPath(preference, directMlAvailable);
        Assert.Equal(expected, path);
    }

    [Theory]
    [InlineData(ParakeetIntelOnnxAcceleration.ExecutionPath.DirectMl, "DML")]
    [InlineData(ParakeetIntelOnnxAcceleration.ExecutionPath.Cpu, "CPU")]
    public void ToSidecarDevice_MapsExecutionPath(
        ParakeetIntelOnnxAcceleration.ExecutionPath path,
        string expected)
    {
        Assert.Equal(expected, ParakeetIntelOnnxAcceleration.ToSidecarDevice(path));
    }

    [Fact]
    public void SetAccelerationPreference_NvidiaCuda_ExplainsDirectMlRemap_WhenDirectMlAvailable()
    {
        var status = ParakeetIntelOnnxAcceleration.CreatePendingStatus(
            TranscriptionAccelerationPreference.NvidiaCuda,
            directMlAvailable: true);

        Assert.Contains("NVIDIA CUDA", status.Detail ?? "");
        Assert.Contains("DirectML", status.Detail ?? "");
        Assert.Equal("Intel iGPU (DirectML)", status.DisplayText);
    }

    [Fact]
    public void SetAccelerationPreference_NvidiaCuda_FallsBackToCpu_WhenDirectMlUnavailable()
    {
        var sut = new ParakeetIntelOnnxPlugin();
        sut.SetAccelerationPreference(TranscriptionAccelerationPreference.NvidiaCuda);

        Assert.Equal(TranscriptionAccelerationBackend.Cpu, sut.AccelerationStatus.ActiveBackend);
        Assert.Contains("DirectML is unavailable", sut.AccelerationStatus.Detail ?? "");
    }

    [Fact]
    public void SetAccelerationPreference_Cpu_UsesCpuStatus()
    {
        var sut = new ParakeetIntelOnnxPlugin();
        sut.SetAccelerationPreference(TranscriptionAccelerationPreference.Cpu);

        Assert.Equal(TranscriptionAccelerationBackend.Cpu, sut.AccelerationStatus.ActiveBackend);
        Assert.Equal("CPU (ONNX Runtime)", sut.AccelerationStatus.DisplayText);
    }

    [Fact]
    public void SetAccelerationPreference_WhenLoadedBackendDiffers_RequiresReload()
    {
        var sut = new ParakeetIntelOnnxPlugin();
        sut.MarkLoadedBackendForTests(ParakeetIntelOnnxAcceleration.DirectMlBackendId);
        sut.SetAccelerationPreference(TranscriptionAccelerationPreference.Cpu);

        Assert.Contains("Reload the model", sut.AccelerationStatus.Detail ?? "");
    }

    [Theory]
    [InlineData("auto", null)]
    [InlineData("AUTO", null)]
    [InlineData("", null)]
    [InlineData(null, null)]
    [InlineData("sk", "sk")]
    [InlineData("en-US", "en-us")]
    public void NormalizeLanguage_MapsAutoToNull(string? input, string? expected)
    {
        Assert.Equal(expected, ParakeetIntelOnnxPlugin.NormalizeLanguage(input));
    }

    [Theory]
    [InlineData("hello world", "hello world")]
    [InlineData("<pad><pad> test", "test")]
    [InlineData("<pad><pad><pad> x", "")]
    public void SanitizeOutput_RemovesPadTokens(string input, string expected)
    {
        Assert.Equal(expected, ParakeetIntelOnnxPlugin.SanitizeOutput(input));
    }

    [Fact]
    public async Task TranscribeAsync_ThrowsWhenTranslateRequested()
    {
        var sut = new ParakeetIntelOnnxPlugin();
        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => sut.TranscribeAsync([], "en", translate: true, prompt: null, CancellationToken.None));

        Assert.Contains("target language", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SupportedAccelerationBackends_IncludesCpuAndGpuSlotForHostUi()
    {
        var sut = new ParakeetIntelOnnxPlugin();

        Assert.Contains(TranscriptionAccelerationBackend.Cpu, sut.SupportedAccelerationBackends);
        Assert.Contains(TranscriptionAccelerationBackend.NvidiaCuda, sut.SupportedAccelerationBackends);
    }
}
