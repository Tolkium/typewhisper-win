using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.PluginSDK;

/// <summary>
/// Optional plugin hook for correcting unsupported host "Local model acceleration" selections.
/// When implemented, the host may replace the stored preference with the suggested value.
/// </summary>
public interface ITranscriptionAccelerationAdvisor
{
    /// <summary>
    /// Returns a normalized host acceleration token (for example "auto" or "cpu")
    /// when the requested preference is not supported and should be replaced, or null to keep the selection.
    /// </summary>
    string? GetSuggestedHostAcceleration(TranscriptionAccelerationPreference preference);
}
