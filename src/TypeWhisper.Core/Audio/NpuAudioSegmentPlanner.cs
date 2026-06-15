namespace TypeWhisper.Core.Audio;

/// <summary>
/// Describes one PCM segment to transcribe for NPU-limited engines.
/// </summary>
/// <param name="StartSample">Inclusive start index in the full recording.</param>
/// <param name="Length">Number of samples in the segment.</param>
public readonly record struct AudioSegmentRange(int StartSample, int Length);

/// <summary>
/// Plans NPU-safe audio segments using live sentence-end hints with overlap fallback.
/// </summary>
public static class NpuAudioSegmentPlanner
{
    private const int SampleRate = 16000;
    /// <summary>Maximum segment duration before NPU encoder truncation.</summary>
    public const double MaxSegmentSeconds = 18.0;
    /// <summary>Overlap after a forced mid-speech cut, recovering words sliced apart.</summary>
    public const double OverlapSeconds = 4.0;
    /// <summary>Smaller overlap after a clean sentence-end cut, only a stitching anchor.</summary>
    public const double SentenceOverlapSeconds = 1.0;
    /// <summary>Minimum segment duration to avoid tiny decode windows.</summary>
    public const double MinSegmentSeconds = 2.0;

    internal static readonly int MaxSegmentSamples = (int)(SampleRate * MaxSegmentSeconds);
    internal static readonly int OverlapSamples = (int)(SampleRate * OverlapSeconds);
    internal static readonly int SentenceOverlapSamples = (int)(SampleRate * SentenceOverlapSeconds);
    internal static readonly int MinSegmentSamples = (int)(SampleRate * MinSegmentSeconds);

    /// <summary>
    /// Builds segment ranges for the supplied audio length.
    /// </summary>
    public static IReadOnlyList<AudioSegmentRange> PlanSegments(
        int totalSamples,
        IReadOnlyList<int>? sentenceEndSampleHints)
    {
        if (totalSamples <= 0)
            return [];

        if (totalSamples <= MaxSegmentSamples)
            return [new AudioSegmentRange(0, totalSamples)];

        var hints = NormalizeHints(sentenceEndSampleHints, totalSamples);
        var segments = new List<AudioSegmentRange>();
        var start = 0;

        while (start < totalSamples)
        {
            var remaining = totalSamples - start;
            if (remaining <= MaxSegmentSamples)
            {
                segments.Add(new AudioSegmentRange(start, remaining));
                break;
            }

            var targetEnd = start + MaxSegmentSamples;
            var cutEnd = SelectCutEnd(start, targetEnd, hints, out var cutAtSentenceEnd);
            var length = cutEnd - start;

            if (length < MinSegmentSamples)
            {
                cutEnd = Math.Min(targetEnd, totalSamples);
                length = cutEnd - start;
                cutAtSentenceEnd = false;
            }

            if (length <= 0)
                break;

            segments.Add(new AudioSegmentRange(start, length));

            if (cutEnd >= totalSamples)
                break;

            // A clean sentence boundary needs no recovery overlap, only a small stitching anchor.
            var overlap = cutAtSentenceEnd ? SentenceOverlapSamples : OverlapSamples;
            var nextStart = cutEnd - overlap;
            if (nextStart <= start)
                nextStart = start + Math.Max(MinSegmentSamples, MaxSegmentSamples - overlap);

            start = nextStart;
        }

        return segments;
    }

    private static int SelectCutEnd(
        int start,
        int targetEnd,
        IReadOnlyList<int> hints,
        out bool cutAtSentenceEnd)
    {
        cutAtSentenceEnd = false;
        if (hints.Count == 0)
            return targetEnd;

        var minCut = start + MinSegmentSamples;
        var selected = -1;
        foreach (var hint in hints)
        {
            if (hint > minCut && hint <= targetEnd)
                selected = hint;
        }

        if (selected <= 0)
            return targetEnd;

        cutAtSentenceEnd = true;
        return selected;
    }

    private static List<int> NormalizeHints(IReadOnlyList<int>? hints, int totalSamples)
    {
        if (hints is null || hints.Count == 0)
            return [];

        return hints
            .Where(h => h > 0 && h < totalSamples)
            .Distinct()
            .OrderBy(h => h)
            .ToList();
    }
}
