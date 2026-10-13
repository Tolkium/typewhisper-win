using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Maps TypeWhisper host acceleration preferences to OpenVINO GPU/CPU devices.
/// The host has no Intel GPU option, so GPU vendor selections are remapped to the Intel iGPU.
/// </summary>
public static class ParakeetOpenVinoAcceleration
{
    public const string GpuDisplayText = "OpenVINO Intel GPU encoder + decoder";
    public const string CpuDisplayText = "OpenVINO CPU INT8";

    public sealed record Resolution(
        string TargetDevice,
        TranscriptionAccelerationBackend ActiveBackend,
        string DisplayText,
        string? Detail);

    public static Resolution Resolve(TranscriptionAccelerationPreference preference) =>
        preference switch
        {
            TranscriptionAccelerationPreference.Cpu => CreateCpu(null),
            TranscriptionAccelerationPreference.NvidiaCuda => CreateGpu(
                "NVIDIA CUDA is not supported by this engine. Using the Intel GPU via OpenVINO."),
            TranscriptionAccelerationPreference.AmdVulkan => CreateGpu(
                "AMD Vulkan is not supported by this engine. Using the Intel GPU via OpenVINO."),
            TranscriptionAccelerationPreference.AmdRocm => CreateGpu(
                "AMD ROCm is not supported by this engine. Using the Intel GPU via OpenVINO."),
            _ => CreateGpu(null),
        };

    public static Resolution ForDevice(string device, string? detail) =>
        device == "CPU" ? CreateCpu(detail) : CreateGpu(detail);

    public static TranscriptionAccelerationStatus ToStatus(
        Resolution resolution,
        string? loadedDevice,
        string? activeBackend)
    {
        var needsReload = loadedDevice is not null
            && !string.Equals(loadedDevice, resolution.TargetDevice, StringComparison.OrdinalIgnoreCase);

        if (needsReload)
        {
            return CreateReloadRequiredStatus(
                loadedDevice!,
                resolution.TargetDevice,
                resolution.DisplayText);
        }

        var detail = resolution.Detail;
        if (!string.IsNullOrWhiteSpace(activeBackend) && string.IsNullOrWhiteSpace(detail))
            detail = activeBackend;

        return new TranscriptionAccelerationStatus(
            resolution.ActiveBackend,
            resolution.DisplayText,
            detail,
            needsReload);
    }

    public static TranscriptionAccelerationStatus CreateReloadRequiredStatus(
        string loadedDevice,
        string targetDevice,
        string displayText) =>
        new(
            TranscriptionAccelerationBackend.Cpu,
            displayText,
            $"Loaded on {loadedDevice}; selection maps to {targetDevice}. Reload the model to apply.",
            RequiresRestart: false);

    // The SDK has no Intel GPU backend value; NvidiaCuda is the host's generic "GPU active" marker.
    private static Resolution CreateGpu(string? detail) =>
        new("GPU", TranscriptionAccelerationBackend.NvidiaCuda, GpuDisplayText, detail);

    private static Resolution CreateCpu(string? detail) =>
        new("CPU", TranscriptionAccelerationBackend.Cpu, CpuDisplayText, detail);
}
