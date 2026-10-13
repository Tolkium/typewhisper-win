using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Maps TypeWhisper host acceleration preferences to OpenVINO NPU/GPU/CPU devices.
/// The host has no Intel option, so Auto and GPU vendor selections use the device
/// chosen in plugin settings (default: Intel GPU).
/// </summary>
public static class ParakeetOpenVinoAcceleration
{
    public const string GpuDisplayText = "OpenVINO Intel GPU encoder + CPU decoder";
    public const string NpuDisplayText = "OpenVINO Intel NPU encoder + CPU decoder";
    public const string CpuDisplayText = "OpenVINO CPU";

    /// <summary>Known limit: NPU transcribes only the first 20 s of a recording (the fixed NPU encoder window).</summary>
    public const int NpuMaxAudioSeconds = Engine.ParakeetOpenVinoEngine.NpuMaxSeconds;

    public const string NpuLimitDetail =
        "NPU transcribes only the first 20 s of each recording. Use GPU for longer dictation.";

    public sealed record Resolution(
        string TargetDevice,
        TranscriptionAccelerationBackend ActiveBackend,
        string DisplayText,
        string? Detail);

    public static Resolution Resolve(TranscriptionAccelerationPreference preference, string preferredDevice) =>
        preference switch
        {
            TranscriptionAccelerationPreference.Cpu => ForDevice("CPU", null),
            TranscriptionAccelerationPreference.NvidiaCuda => ForDevice(
                preferredDevice,
                $"NVIDIA CUDA is not supported by this engine. Using Intel {preferredDevice} via OpenVINO."),
            TranscriptionAccelerationPreference.AmdVulkan => ForDevice(
                preferredDevice,
                $"AMD Vulkan is not supported by this engine. Using Intel {preferredDevice} via OpenVINO."),
            TranscriptionAccelerationPreference.AmdRocm => ForDevice(
                preferredDevice,
                $"AMD ROCm is not supported by this engine. Using Intel {preferredDevice} via OpenVINO."),
            _ => ForDevice(preferredDevice, null),
        };

    public static Resolution ForDevice(string device, string? detail) =>
        device switch
        {
            "CPU" => new("CPU", TranscriptionAccelerationBackend.Cpu, CpuDisplayText, detail),
            "NPU" => new(
                "NPU",
                TranscriptionAccelerationBackend.Cpu,
                NpuDisplayText,
                detail is null ? NpuLimitDetail : $"{detail} {NpuLimitDetail}"),
            // The SDK has no Intel GPU backend value; NvidiaCuda is the host's generic "GPU active" marker.
            _ => new("GPU", TranscriptionAccelerationBackend.NvidiaCuda, GpuDisplayText, detail),
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
            $"Loaded on {loadedDevice}; selection maps to {targetDevice}. Reload the model to apply.",
            RequiresRestart: false);
}
