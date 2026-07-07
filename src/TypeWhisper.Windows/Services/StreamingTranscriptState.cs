using System.Threading;
using TypeWhisper.PluginSDK;

namespace TypeWhisper.Windows.Services;

internal sealed class StreamingTranscriptState
{
    private const int SampleRate = 16000;
    /// <summary>Stay below NPU encoder bucket (20s) with margin for live polling.</summary>
    internal const double MaxPollingAudioSeconds = 18.0;
    internal static readonly int MaxPollingSamples =
        (int)(SampleRate * MaxPollingAudioSeconds);

    private int _sessionVersion;
    private string _confirmedText = "";
    private string _lastDisplayedText = "";
    private string _lastFinalSegment = "";

    private int _frozenPrefixWordCount;
    private string _immutablePrefix = "";
    private const double CommittedWordsPerSecond = 2.0;
    private const int MinRevisableTailWords = 12;

    private TranscriptMergeMode _mergeMode = TranscriptMergeMode.BaselineAligner;
    private TranscriptTokenTimeline? _tokenTimeline;

    /// <summary>
    /// Gets the most recent text shown in the live preview.
    /// </summary>
    public string LastDisplayedText => _lastDisplayedText;

    /// <summary>
    /// Starts a new streaming transcript session and returns its version token.
    /// </summary>
    public int StartSession(TranscriptMergeMode mergeMode = TranscriptMergeMode.BaselineAligner)
    {
        _mergeMode = mergeMode;
        _tokenTimeline = mergeMode == TranscriptMergeMode.Timeline
            ? new TranscriptTokenTimeline()
            : null;
        _confirmedText = "";
        _lastDisplayedText = "";
        _lastFinalSegment = "";
        _frozenPrefixWordCount = 0;
        _immutablePrefix = "";
        return Interlocked.Increment(ref _sessionVersion);
    }

    /// <summary>
    /// Stops the current session and returns the most recent display text.
    /// </summary>
    public string StopSession()
    {
        var finalText = !string.IsNullOrWhiteSpace(_lastDisplayedText)
            ? _lastDisplayedText
            : _confirmedText;
        InvalidateSession();
        _tokenTimeline = null;
        _confirmedText = "";
        _lastDisplayedText = "";
        _lastFinalSegment = "";
        _frozenPrefixWordCount = 0;
        _immutablePrefix = "";
        return finalText;
    }

    /// <summary>
    /// Returns whether the supplied version still belongs to the active session.
    /// </summary>
    public bool IsCurrentSession(int sessionVersion) =>
        sessionVersion == Volatile.Read(ref _sessionVersion);

    /// <summary>
    /// Invalidates in-flight transcript updates from older sessions.
    /// </summary>
    public void InvalidateSession()
    {
        Interlocked.Increment(ref _sessionVersion);
        _tokenTimeline?.Reset();
    }

    /// <summary>
    /// Applies a real-time transcript update and returns false for stale or empty updates.
    /// </summary>
    public bool TryApplyRealtime(
        int sessionVersion,
        StreamingTranscriptEvent evt,
        Func<string, string> corrector,
        out string displayText)
    {
        displayText = "";
        if (!IsCurrentSession(sessionVersion))
            return false;

        var text = evt.Text?.Trim() ?? "";
        if (string.IsNullOrEmpty(text))
            return false;

        text = corrector(text);
        if (string.IsNullOrEmpty(text))
            return false;

        if (evt.IsFinal)
        {
            // Some providers send final events as cumulative text while others send only the
            // newest segment, so final updates are merged with duplicate protection.
            _confirmedText = MergeFinalSegment(_confirmedText, _lastFinalSegment, text);
            _lastFinalSegment = text;
            _lastDisplayedText = _confirmedText;
            displayText = _confirmedText;
            return true;
        }

        displayText = string.IsNullOrEmpty(_confirmedText)
            ? text
            : _confirmedText + " " + text;
        _lastDisplayedText = displayText;
        return true;
    }

    /// <summary>
    /// Applies a polled transcript snapshot and stabilizes it against confirmed text.
    /// </summary>
    public bool TryApplyPolling(
        int sessionVersion,
        string rawText,
        Func<string, string> corrector,
        out string displayText,
        bool rollingWindow = false,
        int totalSamplesRecorded = 0)
    {
        displayText = "";
        if (!IsCurrentSession(sessionVersion))
            return false;

        var text = rawText.Trim();
        if (string.IsNullOrEmpty(text))
            return false;

        text = corrector(text);
        if (string.IsNullOrEmpty(text))
            return false;

        if (rollingWindow && totalSamplesRecorded > MaxPollingSamples)
            TryGrowImmutablePrefix(totalSamplesRecorded, CountWords(text));

        string stable;
        if (rollingWindow)
        {
            var revisable = GetRevisableText(_confirmedText);
            var mergedTail = MergeRolling(revisable, text, rollingTailWindow: true);
            stable = string.IsNullOrEmpty(_immutablePrefix)
                ? mergedTail
                : StreamingHandler.JoinPreviewParts(_immutablePrefix, mergedTail);
        }
        else
        {
            stable = MergeNonRolling(_confirmedText, text);
        }

        if (rollingWindow && !IsAcceptableRollingMerge(stable))
            return false;

        if (string.Equals(stable, _confirmedText, StringComparison.Ordinal))
            return false;

        _confirmedText = stable;
        _lastDisplayedText = stable;
        displayText = stable;
        return true;
    }

    private string MergeRolling(string confirmed, string windowText, bool rollingTailWindow)
    {
        if (_mergeMode != TranscriptMergeMode.Timeline)
            return StreamingHandler.StabilizeRollingText(confirmed, windowText, rollingTailWindow);

        _tokenTimeline ??= new TranscriptTokenTimeline();
        if (_tokenTimeline.TokenCount == 0 && !string.IsNullOrWhiteSpace(confirmed))
            _tokenTimeline.ImportConfirmed(confirmed, committedPrefixCount: 0, commitThreshold: 2);

        return _tokenTimeline.MergeWindow(windowText, rollingTailWindow);
    }

    private string MergeNonRolling(string confirmed, string windowText)
    {
        if (_mergeMode != TranscriptMergeMode.Timeline)
            return StreamingHandler.StabilizeText(confirmed, windowText);

        _tokenTimeline ??= new TranscriptTokenTimeline();
        if (_tokenTimeline.TokenCount == 0 && !string.IsNullOrWhiteSpace(confirmed))
            _tokenTimeline.ImportConfirmed(confirmed, committedPrefixCount: 0, commitThreshold: 1);

        return _tokenTimeline.MergeWindow(windowText, rollingTailWindow: false);
    }

    private void TryGrowImmutablePrefix(int totalSamplesRecorded, int windowWordCount)
    {
        if (string.IsNullOrWhiteSpace(_confirmedText))
            return;

        var words = _confirmedText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0)
            return;

        var secondsBeforeTail = (totalSamplesRecorded - MaxPollingSamples) / (double)SampleRate;
        var audioBasedLock = (int)Math.Floor(secondsBeforeTail * CommittedWordsPerSecond);
        var preserveWords = Math.Max(0, words.Length - windowWordCount - 1);
        var minRevisable = Math.Min(MinRevisableTailWords, Math.Max(4, words.Length / 4));
        var targetWords = Math.Max(audioBasedLock, preserveWords);
        targetWords = Math.Min(targetWords, Math.Max(0, words.Length - minRevisable));

        if (targetWords <= _frozenPrefixWordCount)
            return;

        var candidate = string.Join(" ", words[..targetWords]);
        if (!string.IsNullOrEmpty(_immutablePrefix)
            && !candidate.StartsWith(_immutablePrefix, StringComparison.Ordinal))
        {
            return;
        }

        _immutablePrefix = candidate;
        _frozenPrefixWordCount = targetWords;
    }

    private static int CountWords(string text) =>
        text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

    private string GetRevisableText(string confirmed)
    {
        if (string.IsNullOrEmpty(_immutablePrefix))
            return confirmed;

        return confirmed.StartsWith(_immutablePrefix, StringComparison.Ordinal)
            ? confirmed[_immutablePrefix.Length..].TrimStart()
            : confirmed;
    }

    private bool IsAcceptableRollingMerge(string stable)
    {
        if (!string.IsNullOrEmpty(_immutablePrefix)
            && !stable.StartsWith(_immutablePrefix, StringComparison.Ordinal))
        {
            return false;
        }

        var previousWords = CountWords(_confirmedText);
        var nextWords = CountWords(stable);
        return previousWords <= 30 || nextWords >= previousWords - 4;
    }

    /// <summary>
    /// Takes the most recent audio window for batch polling when the buffer exceeds NPU limits.
    /// </summary>
    internal static float[] PreparePollingBuffer(float[] buffer, out bool usedTailWindow)
    {
        if (buffer.Length <= MaxPollingSamples)
        {
            usedTailWindow = false;
            return buffer;
        }

        usedTailWindow = true;
        var tail = new float[MaxPollingSamples];
        Array.Copy(buffer, buffer.Length - MaxPollingSamples, tail, 0, MaxPollingSamples);
        return tail;
    }

    private static string MergeFinalSegment(string confirmedText, string lastFinalSegment, string newText)
    {
        if (string.IsNullOrEmpty(confirmedText))
            return newText;

        if (string.Equals(newText, confirmedText, StringComparison.Ordinal)
            || string.Equals(newText, lastFinalSegment, StringComparison.Ordinal))
        {
            return confirmedText;
        }

        if (newText.StartsWith(confirmedText, StringComparison.Ordinal))
            return newText;

        return confirmedText + " " + newText;
    }
}
