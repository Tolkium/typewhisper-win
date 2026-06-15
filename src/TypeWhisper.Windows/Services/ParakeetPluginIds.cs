namespace TypeWhisper.Windows.Services;

/// <summary>
/// Canonical plugin and model IDs for Parakeet OpenVINO baseline vs timeline variants.
/// </summary>
internal static class ParakeetPluginIds
{
    internal const string BaselinePluginId = "com.whispernpu.parakeet-openvino";
    internal const string TimelinePluginId = "com.tolkium.parakeet-openvino-timeline";

    internal const string BaselineModelId =
        "plugin:com.whispernpu.parakeet-openvino:parakeet-tdt-0.6b-v3-openvino";
    internal const string TimelineModelId =
        "plugin:com.tolkium.parakeet-openvino-timeline:parakeet-tdt-0.6b-v3-openvino-timeline";

    internal static bool UsesTimelineMerge(string? modelOrPluginId) =>
        !string.IsNullOrEmpty(modelOrPluginId)
        && (string.Equals(modelOrPluginId, TimelineModelId, StringComparison.Ordinal)
            || string.Equals(modelOrPluginId, TimelinePluginId, StringComparison.Ordinal)
            || modelOrPluginId.StartsWith($"plugin:{TimelinePluginId}:", StringComparison.Ordinal));

    internal static bool IsOpenVinoParakeet(string? modelId) =>
        !string.IsNullOrEmpty(modelId)
        && (string.Equals(modelId, BaselineModelId, StringComparison.Ordinal)
            || string.Equals(modelId, TimelineModelId, StringComparison.Ordinal)
            || modelId.StartsWith($"plugin:{BaselinePluginId}:", StringComparison.Ordinal)
            || modelId.StartsWith($"plugin:{TimelinePluginId}:", StringComparison.Ordinal));
}
