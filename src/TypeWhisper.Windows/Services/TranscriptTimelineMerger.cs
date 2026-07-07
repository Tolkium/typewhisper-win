namespace TypeWhisper.Windows.Services;

/// <summary>
/// Merge pipeline for the timeline plugin variant using committed/provisional tokens.
/// </summary>
internal static class TranscriptTimelineMerger
{
    public static string Merge(
        string confirmed,
        string windowText,
        bool rollingTailWindow = false,
        int frozenPrefixWordCount = 0) =>
        new TranscriptTokenTimeline().Merge(
            confirmed,
            windowText,
            rollingTailWindow,
            frozenPrefixWordCount);
}
