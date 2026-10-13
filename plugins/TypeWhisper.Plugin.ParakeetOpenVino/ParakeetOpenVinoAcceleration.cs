using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Maps TypeWhisper host acceleration preferences to OpenVINO NPU/GPU/CPU devices.
/// The host has no Intel option, so Auto and GPU vendor selections use the device
/// chosen in plugin settings (default: Intel GPU). Texts come from Localization/*.json.
/// </summary>
internal static class ParakeetOpenVinoAcceleration
{
    /// <summary>Known limit: NPU transcribes only the first 20 s of a recording (the fixed NPU encoder window).</summary>
    public const int NpuMaxAudioSeconds = Engine.ParakeetOpenVinoEngine.NpuMaxSeconds;

    public sealed record Resolution(
        string TargetDevice,
        TranscriptionAccelerationBackend ActiveBackend,
        string DisplayText,
        string? Detail);

    public static Resolution Resolve(
        TranscriptionAccelerationPreference preference, string preferredDevice, PluginText text) =>
        preference switch
        {
            TranscriptionAccelerationPreference.Cpu => ForDevice("CPU", null, text),
            TranscriptionAccelerationPreference.NvidiaCuda =>
                ForDevice(preferredDevice, text.Get("Status.UnsupportedNvidiaCuda", preferredDevice), text),
            TranscriptionAccelerationPreference.AmdVulkan =>
                ForDevice(preferredDevice, text.Get("Status.UnsupportedAmdVulkan", preferredDevice), text),
            TranscriptionAccelerationPreference.AmdRocm =>
                ForDevice(preferredDevice, text.Get("Status.UnsupportedAmdRocm", preferredDevice), text),
            _ => ForDevice(preferredDevice, null, text),
        };

    private static Resolution ForDevice(string device, string? detail, PluginText text)
    {
        switch (device)
        {
            case "CPU":
                return new("CPU", TranscriptionAccelerationBackend.Cpu, text.Get("Status.Cpu"), detail);
            case "NPU":
                var limit = text.Get("Status.NpuLimit", NpuMaxAudioSeconds);
                return new(
                    "NPU",
                    TranscriptionAccelerationBackend.Cpu,
                    text.Get("Status.Npu"),
                    detail is null ? limit : $"{detail} {limit}");
            default:
                // The SDK has no Intel GPU backend value; NvidiaCuda is the host's generic "GPU active" marker.
                return new("GPU", TranscriptionAccelerationBackend.NvidiaCuda, text.Get("Status.Gpu"), detail);
        }
    }

    public static TranscriptionAccelerationStatus ToStatus(
        Resolution resolution,
        string? loadedDevice,
        string? activeBackend,
        PluginText text)
    {
        if (loadedDevice is not null
            && !string.Equals(loadedDevice, resolution.TargetDevice, StringComparison.OrdinalIgnoreCase))
        {
            return new TranscriptionAccelerationStatus(
                TranscriptionAccelerationBackend.Cpu,
                resolution.DisplayText,
                text.Get("Status.ReloadRequired", loadedDevice, resolution.TargetDevice));
        }

        var detail = string.IsNullOrWhiteSpace(resolution.Detail) ? activeBackend : resolution.Detail;
        return new TranscriptionAccelerationStatus(resolution.ActiveBackend, resolution.DisplayText, detail);
    }
}
