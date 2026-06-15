using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Maps TypeWhisper host acceleration preferences to OpenVINO NPU/GPU/CPU devices.
/// </summary>
public static class ParakeetOpenVinoAcceleration
{
    public const string HostAccelerationAuto = "auto";
    public const string HostAccelerationCpu = "cpu";

    public sealed record Resolution(
        string TargetDevice,
        TranscriptionAccelerationBackend ActiveBackend,
        string DisplayText,
        string? Detail,
        string? SuggestedHostAcceleration);

    public static Resolution Resolve(TranscriptionAccelerationPreference preference) =>
        preference switch
        {
            TranscriptionAccelerationPreference.Cpu => new(
                "CPU",
                TranscriptionAccelerationBackend.Cpu,
                "OpenVINO CPU INT8",
                null,
                null),
            TranscriptionAccelerationPreference.NvidiaCuda => new(
                "NPU",
                TranscriptionAccelerationBackend.Cpu,
                "OpenVINO NPU encoder + CPU decoder",
                "NVIDIA CUDA is not supported by this engine. Switched to Auto (OpenVINO NPU).",
                HostAccelerationAuto),
            TranscriptionAccelerationPreference.AmdVulkan => new(
                "NPU",
                TranscriptionAccelerationBackend.Cpu,
                "OpenVINO NPU encoder + CPU decoder",
                "AMD Vulkan is not supported by this engine. Switched to Auto (OpenVINO NPU).",
                HostAccelerationAuto),
            TranscriptionAccelerationPreference.AmdRocm => new(
                "NPU",
                TranscriptionAccelerationBackend.Cpu,
                "OpenVINO NPU encoder + CPU decoder",
                "AMD ROCm is not supported by this engine. Switched to Auto (OpenVINO NPU).",
                HostAccelerationAuto),
            _ => new(
                "NPU",
                TranscriptionAccelerationBackend.Cpu,
                "OpenVINO NPU encoder + CPU decoder",
                null,
                null),
        };

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
            $"Loaded on {loadedDevice}; host selection maps to {targetDevice}. Reload the model to apply.",
            RequiresRestart: false);

    public static string? GetSuggestedHostAcceleration(TranscriptionAccelerationPreference preference) =>
        Resolve(preference).SuggestedHostAcceleration;
}
