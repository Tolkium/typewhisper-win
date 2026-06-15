using TypeWhisper.Core.Audio;

namespace TypeWhisper.PluginSystem.Tests;

public class LiveSentenceBoundaryTrackerTests
{
    [Fact]
    public void TryRecord_AddsHint_AfterTwoStableSentenceEndingPolls()
    {
        var tracker = new LiveSentenceBoundaryTracker();

        tracker.TryRecord("Ahoj.", 80_000);
        Assert.Empty(tracker.GetHints());

        tracker.TryRecord("Ahoj.", 83_000);

        Assert.Equal([80_000], tracker.GetHints());
    }

    [Fact]
    public void TryRecord_DoesNotDuplicateHint_WhenStableTextUnchanged()
    {
        var tracker = new LiveSentenceBoundaryTracker();

        tracker.TryRecord("Ahoj.", 80_000);
        tracker.TryRecord("Ahoj.", 83_000);
        tracker.TryRecord("Ahoj.", 86_000);

        Assert.Single(tracker.GetHints());
        Assert.Equal(80_000, tracker.GetHints()[0]);
    }

    [Fact]
    public void TryRecord_DoesNotAddHint_WhenTextGrowsWithoutSentenceEnding()
    {
        var tracker = new LiveSentenceBoundaryTracker();

        tracker.TryRecord("Ahoj, ", 80_000);
        tracker.TryRecord("Ahoj, svet", 83_000);

        Assert.Empty(tracker.GetHints());
    }

    [Fact]
    public void TryRecord_CancelsPendingHint_WhenSentenceEndingIsRevisedAway()
    {
        var tracker = new LiveSentenceBoundaryTracker();

        tracker.TryRecord("Povieme.", 80_000);
        tracker.TryRecord("Poviete ale", 83_000);

        Assert.Empty(tracker.GetHints());
    }

    [Fact]
    public void TryRecord_AddsSecondHint_WhenAnotherSentenceCompletes()
    {
        var tracker = new LiveSentenceBoundaryTracker();

        tracker.TryRecord("Prva veta.", 80_000);
        tracker.TryRecord("Prva veta.", 83_000);
        tracker.TryRecord("Prva veta. Druha veta.", 160_000);
        tracker.TryRecord("Prva veta. Druha veta.", 163_000);

        Assert.Equal([80_000, 160_000], tracker.GetHints());
    }

    [Fact]
    public void Reset_ClearsHints()
    {
        var tracker = new LiveSentenceBoundaryTracker();
        tracker.TryRecord("Hotovo.", 10_000);
        tracker.TryRecord("Hotovo.", 13_000);

        tracker.Reset();

        Assert.Empty(tracker.GetHints());
    }
}

public class NpuAudioSegmentPlannerTests
{
    private const int SampleRate = 16000;
    private static readonly int MaxSegmentSamples =
        (int)(NpuAudioSegmentPlanner.MaxSegmentSeconds * SampleRate);

    [Fact]
    public void PlanSegments_ReturnsSingleSegment_WhenAudioWithinLimit()
    {
        var total = MaxSegmentSamples - 1;

        var plans = NpuAudioSegmentPlanner.PlanSegments(total, null);

        Assert.Equal([new AudioSegmentRange(0, total)], plans);
    }

    [Fact]
    public void PlanSegments_CutsAtLastHintWithinWindow()
    {
        var hintAtFiveSeconds = 5 * SampleRate;
        var total = 60 * SampleRate;

        var plans = NpuAudioSegmentPlanner.PlanSegments(total, [hintAtFiveSeconds, 12 * SampleRate]);

        Assert.True(plans[0].StartSample == 0);
        Assert.Equal(12 * SampleRate, plans[0].Length);
    }

    [Fact]
    public void PlanSegments_DoesNotCutAtHintOutsideCurrentWindow()
    {
        var hintAtTwentyFiveSeconds = 25 * SampleRate;
        var total = 60 * SampleRate;

        var plans = NpuAudioSegmentPlanner.PlanSegments(total, [hintAtTwentyFiveSeconds]);

        Assert.True(plans[0].Length <= MaxSegmentSamples);
        Assert.True(plans[0].Length < hintAtTwentyFiveSeconds);
    }

    [Fact]
    public void PlanSegments_FallsBackToForcedWindow_WhenNoHints()
    {
        var total = 60 * SampleRate;

        var plans = NpuAudioSegmentPlanner.PlanSegments(total, null);

        Assert.True(plans.Count > 1);
        Assert.Equal(MaxSegmentSamples, plans[0].Length);
    }

    [Fact]
    public void PlanSegments_UsesSmallOverlap_AfterCleanSentenceHintCut()
    {
        var hint = 12 * SampleRate;
        var total = 60 * SampleRate;

        var plans = NpuAudioSegmentPlanner.PlanSegments(total, [hint]);

        var sentenceOverlap = (int)(NpuAudioSegmentPlanner.SentenceOverlapSeconds * SampleRate);
        Assert.Equal(hint, plans[0].Length);
        Assert.Equal(hint - sentenceOverlap, plans[1].StartSample);
    }

    [Fact]
    public void PlanSegments_UsesFullOverlap_AfterForcedCut()
    {
        var total = 60 * SampleRate;

        var plans = NpuAudioSegmentPlanner.PlanSegments(total, null);

        var overlap = (int)(NpuAudioSegmentPlanner.OverlapSeconds * SampleRate);
        Assert.Equal(MaxSegmentSamples, plans[0].Length);
        Assert.Equal(MaxSegmentSamples - overlap, plans[1].StartSample);
    }

    [Fact]
    public void PlanSegments_CoversEntireAudio()
    {
        var total = 95 * SampleRate;
        var hints = new[] { 8 * SampleRate, 19 * SampleRate, 33 * SampleRate, 51 * SampleRate, 72 * SampleRate };

        var plans = NpuAudioSegmentPlanner.PlanSegments(total, hints);

        Assert.Equal(0, plans[0].StartSample);
        Assert.Equal(total, plans[^1].StartSample + plans[^1].Length);
    }

    [Fact]
    public void PlanSegments_IgnoresInvalidHints()
    {
        var total = 40 * SampleRate;

        var plans = NpuAudioSegmentPlanner.PlanSegments(total, [0, total, -1, 10 * SampleRate]);

        Assert.True(plans[0].Length <= MaxSegmentSamples);
    }
}
