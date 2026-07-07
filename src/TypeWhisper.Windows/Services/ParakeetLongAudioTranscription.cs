using TypeWhisper.Core.Audio;
using TypeWhisper.Core.Models;
using TypeWhisper.PluginSDK;
using TypeWhisper.PluginSDK.Models;

namespace TypeWhisper.Windows.Services;

/// <summary>
/// Splits long PCM audio into NPU-safe windows and merges chunk transcripts.
/// </summary>
internal static class ParakeetLongAudioTranscription
{
    internal static bool ShouldChunk(int sampleCount) =>
        sampleCount > StreamingTranscriptState.MaxPollingSamples;

    internal static async Task<TranscriptionResult> TranscribeAsync(
        ITranscriptionEnginePlugin plugin,
        float[] audioSamples,
        string? language,
        bool translate,
        IReadOnlyList<int>? sentenceEndSampleHints,
        bool useTimelineMerge,
        CancellationToken cancellationToken)
    {
        if (!ShouldChunk(audioSamples.Length))
        {
            var wavBytes = WavEncoder.Encode(audioSamples);
            var single = await plugin.TranscribeAsync(wavBytes, language, translate, null, cancellationToken);
            return ToResult(single);
        }

        var segmentPlans = NpuAudioSegmentPlanner.PlanSegments(
            audioSamples.Length,
            sentenceEndSampleHints);

        if (TranscriptTraceLog.IsEnabled)
        {
            TranscriptTraceLog.Write("final_plan", new
            {
                totalSamples = audioSamples.Length,
                sentenceEndSampleHints = sentenceEndSampleHints,
                segments = segmentPlans.Select(p => new { p.StartSample, p.Length }),
            });
        }

        var mergedText = "";
        double totalDuration = 0;
        string? detectedLanguage = null;
        var segmentIndex = 0;

        foreach (var plan in segmentPlans)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (plan.Length <= 0)
                continue;

            var chunk = new float[plan.Length];
            Array.Copy(audioSamples, plan.StartSample, chunk, 0, plan.Length);

            var wavBytes = WavEncoder.Encode(chunk);
            var result = await plugin.TranscribeAsync(wavBytes, language, translate, null, cancellationToken);
            totalDuration += result.DurationSeconds;
            detectedLanguage ??= result.DetectedLanguage;

            var text = result.Text?.Trim() ?? "";
            if (string.IsNullOrEmpty(text))
                continue;

            var beforeMerge = mergedText;
            mergedText = string.IsNullOrEmpty(mergedText)
                ? text
                : (useTimelineMerge
                    ? TranscriptTimelineMerger.Merge(mergedText, text, rollingTailWindow: true)
                    : StreamingHandler.StabilizeRollingText(mergedText, text, rollingTailWindow: true));

            if (TranscriptTraceLog.IsEnabled)
            {
                TranscriptTraceLog.Write("final_segment", new
                {
                    index = segmentIndex,
                    plan.StartSample,
                    plan.Length,
                    rawDecode = text,
                    mergedBefore = beforeMerge,
                    mergedAfter = mergedText,
                });
            }

            segmentIndex++;
        }

        if (TranscriptTraceLog.IsEnabled)
            TranscriptTraceLog.Write("final_result", new { text = mergedText });

        return new TranscriptionResult
        {
            Text = mergedText,
            DetectedLanguage = detectedLanguage,
            Duration = totalDuration,
        };
    }

    private static TranscriptionResult ToResult(PluginTranscriptionResult result) =>
        new()
        {
            Text = result.Text,
            DetectedLanguage = result.DetectedLanguage,
            Duration = result.DurationSeconds,
            NoSpeechProbability = result.NoSpeechProbability,
            Segments = result.Segments
                .Select(seg => new TranscriptionSegment(seg.Text, seg.Start, seg.End))
                .ToList(),
        };
}
