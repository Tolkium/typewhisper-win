using TypeWhisper.Core.Models;
using TypeWhisper.PluginSDK;

namespace TypeWhisper.Windows.Services;

internal static class TranscriptionAccelerationAdvisorSupport
{
    internal static string ApplyPluginAdvisor(
        ITranscriptionEnginePlugin plugin,
        string normalizedHostAcceleration)
    {
        if (plugin is not ITranscriptionAccelerationAdvisor advisor)
            return normalizedHostAcceleration;

        var preference = ModelManagerService.GetAccelerationPreference(normalizedHostAcceleration);
        var suggested = advisor.GetSuggestedHostAcceleration(preference);
        if (string.IsNullOrWhiteSpace(suggested))
            return normalizedHostAcceleration;

        return AppSettings.NormalizeLocalModelAcceleration(suggested);
    }
}
