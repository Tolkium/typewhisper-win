using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Plugin.ParakeetIntelOnnx;

/// <summary>
/// Maps TypeWhisper host acceleration preferences to ONNX Runtime execution paths.
/// This engine uses DirectML (Intel/AMD iGPU on Windows), not NVIDIA CUDA or AMD ROCm/Vulkan.
/// </summary>
public static class ParakeetIntelOnnxAcceleration
{
    internal const string DirectMlBackendId = "directml";
    internal const string CpuBackendId = "cpu";

    public enum ExecutionPath
    {
        DirectMl,
        Cpu,
    }

    internal static ExecutionPath ResolveExecutionPath(
        TranscriptionAccelerationPreference preference,
        bool directMlAvailable)
    {
        return preference switch
        {
            TranscriptionAccelerationPreference.Cpu => ExecutionPath.Cpu,
            TranscriptionAccelerationPreference.Auto =>
                directMlAvailable ? ExecutionPath.DirectMl : ExecutionPath.Cpu,
            TranscriptionAccelerationPreference.NvidiaCuda
                or TranscriptionAccelerationPreference.AmdVulkan
                or TranscriptionAccelerationPreference.AmdRocm =>
                directMlAvailable ? ExecutionPath.DirectMl : ExecutionPath.Cpu,
            _ => directMlAvailable ? ExecutionPath.DirectMl : ExecutionPath.Cpu,
        };
    }

    internal static string ToSidecarDevice(ExecutionPath path) =>
        path == ExecutionPath.DirectMl ? "DML" : "CPU";

    internal static TranscriptionAccelerationStatus CreatePendingStatus(
        TranscriptionAccelerationPreference preference,
        bool directMlAvailable)
    {
        var path = ResolveExecutionPath(preference, directMlAvailable);
        if (path == ExecutionPath.DirectMl)
        {
            var remapped = preference is TranscriptionAccelerationPreference.NvidiaCuda
                or TranscriptionAccelerationPreference.AmdVulkan
                or TranscriptionAccelerationPreference.AmdRocm;

            var detail = remapped
                ? DescribeHostPreferenceRemap(preference)
                : null;

            return new TranscriptionAccelerationStatus(
                TranscriptionAccelerationBackend.NvidiaCuda,
                "Intel iGPU (DirectML)",
                detail);
        }

        return new TranscriptionAccelerationStatus(
            TranscriptionAccelerationBackend.Cpu,
            "CPU (ONNX Runtime)",
            preference == TranscriptionAccelerationPreference.Auto && !directMlAvailable
                ? "DirectML is not available; using CPU."
                : preference is TranscriptionAccelerationPreference.NvidiaCuda
                    or TranscriptionAccelerationPreference.AmdVulkan
                    or TranscriptionAccelerationPreference.AmdRocm
                    ? $"{DescribeHostPreferenceRemap(preference)} DirectML is unavailable; using CPU."
                    : null);
    }

    internal static TranscriptionAccelerationStatus CreateLoadedStatus(
        string backendId,
        TranscriptionAccelerationPreference preference)
    {
        if (string.Equals(backendId, DirectMlBackendId, StringComparison.OrdinalIgnoreCase))
        {
            var detail = preference is TranscriptionAccelerationPreference.NvidiaCuda
                or TranscriptionAccelerationPreference.AmdVulkan
                or TranscriptionAccelerationPreference.AmdRocm
                ? DescribeHostPreferenceRemap(preference)
                : null;

            return new TranscriptionAccelerationStatus(
                TranscriptionAccelerationBackend.NvidiaCuda,
                "Intel iGPU (DirectML)",
                detail);
        }

        return new TranscriptionAccelerationStatus(
            TranscriptionAccelerationBackend.Cpu,
            "CPU (ONNX Runtime INT8)");
    }

    internal static TranscriptionAccelerationStatus CreateDirectMlUnavailableStatus(string detail) =>
        new(
            TranscriptionAccelerationBackend.Cpu,
            "DirectML unavailable",
            detail);

    internal static TranscriptionAccelerationStatus CreateReloadRequiredStatus(
        string loadedBackend,
        string desiredBackend)
    {
        var active = string.Equals(loadedBackend, DirectMlBackendId, StringComparison.OrdinalIgnoreCase)
            ? TranscriptionAccelerationBackend.NvidiaCuda
            : TranscriptionAccelerationBackend.Cpu;

        return new TranscriptionAccelerationStatus(
            active,
            string.Equals(loadedBackend, DirectMlBackendId, StringComparison.OrdinalIgnoreCase)
                ? "Intel iGPU (DirectML)"
                : "CPU (ONNX Runtime)",
            $"Loaded with {loadedBackend}; selected {desiredBackend}. Reload the model to apply.",
            RequiresRestart: false);
    }

    private static string DescribeHostPreferenceRemap(TranscriptionAccelerationPreference preference) =>
        preference switch
        {
            TranscriptionAccelerationPreference.NvidiaCuda =>
                "Host acceleration is set to NVIDIA CUDA; this engine uses Intel iGPU via DirectML instead.",
            TranscriptionAccelerationPreference.AmdVulkan =>
                "Host acceleration is set to AMD Vulkan; this engine uses Intel iGPU via DirectML instead.",
            TranscriptionAccelerationPreference.AmdRocm =>
                "Host acceleration is set to AMD ROCm; this engine uses Intel iGPU via DirectML instead.",
            _ => "Host GPU acceleration maps to Intel iGPU (DirectML) for this engine.",
        };
}
