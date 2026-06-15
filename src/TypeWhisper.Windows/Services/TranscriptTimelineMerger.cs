namespace TypeWhisper.Windows.Services;

/// <summary>
/// Merge pipeline for the timeline plugin variant. Currently delegates to the fuzzy aligner
/// while the committed/provisional token timeline is implemented on this branch.
/// </summary>
internal static class TranscriptTimelineMerger
{
    public static string Merge(
        string confirmed,
        string windowText,
        bool rollingTailWindow = false,
        int frozenPrefixWordCount = 0) =>
        RollingTranscriptAligner.Merge(
            confirmed,
            windowText,
            rollingTailWindow,
            frozenPrefixWordCount);
}
