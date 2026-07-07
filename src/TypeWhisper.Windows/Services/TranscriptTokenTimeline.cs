using System.Globalization;
using System.Text;

namespace TypeWhisper.Windows.Services;

/// <summary>
/// Token-level transcript timeline with committed and provisional regions.
/// Committed tokens survive rolling-window revisions; provisional tail is replaced each poll.
/// </summary>
internal sealed class TranscriptTokenTimeline
{
    private const int MaxSearchTailTokens = 180;
    private const int RollingCommitPolls = 2;
    private const int ChunkCommitPolls = 1;
    private const double MostlyContainedCoverage = 0.78;

    private readonly List<TimelineToken> _tokens = [];

    public int TokenCount => _tokens.Count;

    public void Reset() => _tokens.Clear();

    /// <summary>
    /// One-shot merge for chunk stitching or live/full selection without session state.
    /// </summary>
    public string Merge(
        string confirmed,
        string windowText,
        bool rollingTailWindow = false,
        int committedPrefixCount = 0)
    {
        var timeline = new TranscriptTokenTimeline();
        timeline.ImportConfirmed(
            confirmed,
            committedPrefixCount,
            rollingTailWindow ? RollingCommitPolls : ChunkCommitPolls);
        timeline.ApplyWindow(windowText, rollingTailWindow);
        return timeline.Render();
    }

    public void ImportConfirmed(string confirmed, int committedPrefixCount, int commitThreshold)
    {
        _tokens.Clear();
        confirmed = confirmed.Trim();
        if (string.IsNullOrEmpty(confirmed))
            return;

        var words = Tokenize(confirmed);
        for (var i = 0; i < words.Length; i++)
        {
            var committed = i < committedPrefixCount;
            _tokens.Add(new TimelineToken(
                words[i].Original,
                words[i].Normalized,
                committed ? commitThreshold : 0,
                committed));
        }
    }

    public string MergeWindow(string windowText, bool rollingTailWindow)
    {
        var before = Render();
        ApplyWindow(windowText, rollingTailWindow);
        var after = Render();

        if (rollingTailWindow && IsDestructiveShrink(before, after))
            return before;

        return after;
    }

    private void ApplyWindow(string windowText, bool rollingTailWindow)
    {
        windowText = windowText.Trim();
        if (string.IsNullOrEmpty(windowText))
            return;

        var windowTokens = Tokenize(windowText);
        if (windowTokens.Length == 0)
            return;

        if (_tokens.Count == 0)
        {
            foreach (var token in windowTokens)
                _tokens.Add(NewProvisional(token));
            return;
        }

        if (IsMostlyContained(_tokens, windowTokens))
            return;

        var alignment = FindBestAlignment(_tokens, windowTokens, rollingTailWindow);
        if (alignment is null)
        {
            if (rollingTailWindow)
                return;

            foreach (var token in windowTokens)
                _tokens.Add(NewProvisional(token));
            return;
        }

        var commitThreshold = rollingTailWindow ? RollingCommitPolls : ChunkCommitPolls;
        var timelineStart = alignment.Value.TimelineStart;
        var windowStart = alignment.Value.WindowStart;
        var matchCount = alignment.Value.Matches;
        var rebuilt = new List<TimelineToken>(timelineStart + matchCount + windowTokens.Length);

        rebuilt.AddRange(_tokens.Take(timelineStart));

        for (var i = 0; i < matchCount; i++)
        {
            var existing = _tokens[timelineStart + i];
            var incoming = windowTokens[windowStart + i];

            if (existing.IsCommitted)
            {
                rebuilt.Add(existing with
                {
                    Stability = Math.Max(existing.Stability, commitThreshold),
                });
                continue;
            }

            var stability = existing.Stability + 1;
            var committed = stability >= commitThreshold;
            rebuilt.Add(new TimelineToken(
                incoming.Original,
                incoming.Normalized,
                stability,
                committed));
        }

        for (var j = windowStart + matchCount; j < windowTokens.Length; j++)
            rebuilt.Add(NewProvisional(windowTokens[j]));

        _tokens.Clear();
        _tokens.AddRange(rebuilt);
    }

    private string Render() =>
        CollapseAdjacentDuplicatePhrase(string.Join(" ", _tokens.Select(token => token.Text)));

    private static TimelineToken NewProvisional(WordToken token) =>
        new(token.Original, token.Normalized, 1, false);

    private static bool IsDestructiveShrink(string before, string after)
    {
        var beforeWords = Tokenize(before);
        if (beforeWords.Length < 24)
            return false;

        var afterWords = Tokenize(after);
        return afterWords.Length < (int)(beforeWords.Length * 0.55);
    }

    private static Alignment? FindBestAlignment(
        IReadOnlyList<TimelineToken> timelineTokens,
        WordToken[] windowTokens,
        bool rollingTailWindow)
    {
        var timelineWords = timelineTokens
            .Select(token => new WordToken(token.Text, token.Normalized))
            .ToArray();

        var timelineSearchStart = Math.Max(0, timelineWords.Length - MaxSearchTailTokens);
        var best = default(Alignment);
        var hasBest = false;

        for (var timelineStart = timelineSearchStart; timelineStart < timelineWords.Length; timelineStart++)
        {
            for (var windowStart = 0; windowStart < windowTokens.Length; windowStart++)
            {
                var scored = ScoreConsecutiveMatch(timelineWords, timelineStart, windowTokens, windowStart);
                if (scored.Matches == 0)
                    continue;

                var score = scored.Score
                    + Math.Min(timelineStart, timelineWords.Length - 1) * 0.01
                    - windowStart * 0.02;

                if (!hasBest || score > best.Score)
                {
                    best = new Alignment(timelineStart, windowStart, scored.Matches, score);
                    hasBest = true;
                }
            }
        }

        if (!hasBest)
            return null;

        var minOverlap = MinimumOverlapWords(windowTokens.Length, rollingTailWindow);
        return best.Matches >= minOverlap ? best : null;
    }

    private static ScoredMatch ScoreConsecutiveMatch(
        WordToken[] timelineWords,
        int timelineStart,
        WordToken[] windowWords,
        int windowStart)
    {
        var matches = 0;
        double score = 0;
        var timelineIndex = timelineStart;
        var windowIndex = windowStart;

        while (timelineIndex < timelineWords.Length && windowIndex < windowWords.Length)
        {
            var wordScore = WordScore(timelineWords[timelineIndex].Normalized, windowWords[windowIndex].Normalized);
            if (wordScore <= 0)
                break;

            matches++;
            score += wordScore;
            timelineIndex++;
            windowIndex++;
        }

        return new ScoredMatch(matches, score);
    }

    private static bool IsMostlyContained(IReadOnlyList<TimelineToken> timelineTokens, WordToken[] windowTokens)
    {
        if (windowTokens.Length <= 2 || timelineTokens.Count < windowTokens.Length)
            return false;

        var timelineWords = timelineTokens
            .Select(token => new WordToken(token.Text, token.Normalized))
            .ToArray();

        var best = 0;
        for (var start = 0; start < timelineWords.Length; start++)
        {
            for (var windowStart = 0; windowStart < Math.Min(3, windowTokens.Length); windowStart++)
            {
                var scored = ScoreConsecutiveMatch(timelineWords, start, windowTokens, windowStart);
                best = Math.Max(best, scored.Matches);
            }
        }

        var required = Math.Max(3, (int)Math.Ceiling(windowTokens.Length * MostlyContainedCoverage));
        return best >= required;
    }

    private static int MinimumOverlapWords(int windowWordCount, bool rollingTailWindow)
    {
        if (!rollingTailWindow && windowWordCount <= 4)
            return 1;

        var divisor = rollingTailWindow ? 9 : 7;
        return Math.Min(7, Math.Max(2, (int)Math.Ceiling(windowWordCount / (double)divisor)));
    }

    private static double WordScore(string left, string right)
    {
        if (left.Length == 0 || right.Length == 0)
            return 0;

        if (string.Equals(left, right, StringComparison.Ordinal))
            return 4;

        if (left.Length >= 4 && right.Length >= 4)
        {
            var prefixLength = Math.Min(5, Math.Min(left.Length, right.Length));
            if (left.AsSpan(0, prefixLength).SequenceEqual(right.AsSpan(0, prefixLength)))
                return 2.5;
        }

        if (Math.Min(left.Length, right.Length) >= 5 && LevenshteinWithin(left, right, 1))
            return 2;

        return 0;
    }

    private static bool LevenshteinWithin(string left, string right, int maxDistance)
    {
        if (Math.Abs(left.Length - right.Length) > maxDistance)
            return false;

        var previous = new int[right.Length + 1];
        var current = new int[right.Length + 1];

        for (var j = 0; j <= right.Length; j++)
            previous[j] = j;

        for (var i = 1; i <= left.Length; i++)
        {
            current[0] = i;
            var rowMinimum = current[0];

            for (var j = 1; j <= right.Length; j++)
            {
                var cost = left[i - 1] == right[j - 1] ? 0 : 1;
                current[j] = Math.Min(
                    Math.Min(current[j - 1] + 1, previous[j] + 1),
                    previous[j - 1] + cost);
                rowMinimum = Math.Min(rowMinimum, current[j]);
            }

            if (rowMinimum > maxDistance)
                return false;

            (previous, current) = (current, previous);
        }

        return previous[right.Length] <= maxDistance;
    }

    private static WordToken[] Tokenize(string text) =>
        text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Select(word => new WordToken(word, NormalizeWord(word)))
            .Where(word => word.Normalized.Length > 0)
            .ToArray();

    private static string NormalizeWord(string word)
    {
        var trimmed = word.Trim().Trim(
            '.', ',', ';', ':', '!', '?', '"', '\'', '(', ')', '[', ']', '{', '}');
        if (trimmed.Length == 0)
            return "";

        var normalized = trimmed.ToLowerInvariant().Normalize(NormalizationForm.FormD);
        var builder = new StringBuilder(normalized.Length);
        foreach (var ch in normalized)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(ch) == UnicodeCategory.NonSpacingMark)
                continue;
            if (char.IsLetterOrDigit(ch))
                builder.Append(ch);
        }

        return builder.ToString().Normalize(NormalizationForm.FormC);
    }

    private static string CollapseAdjacentDuplicatePhrase(string text)
    {
        var current = text;
        for (var pass = 0; pass < 8; pass++)
        {
            var collapsed = CollapseFirstAdjacentDuplicatePhrase(current);
            if (ReferenceEquals(collapsed, current))
                return current;

            current = collapsed;
        }

        return current;
    }

    private static string CollapseFirstAdjacentDuplicatePhrase(string text)
    {
        var words = Tokenize(text);
        if (words.Length < 8)
            return text;

        for (var phraseLength = Math.Min(14, words.Length / 2); phraseLength >= 4; phraseLength--)
        {
            for (var start = 0; start + phraseLength * 2 <= words.Length; start++)
            {
                if (!PhraseMatches(words, start, start + phraseLength, phraseLength))
                    continue;

                var kept = words[..(start + phraseLength)]
                    .Concat(words[(start + phraseLength * 2)..])
                    .Select(word => word.Original);
                return string.Join(" ", kept);
            }
        }

        return text;
    }

    private static bool PhraseMatches(WordToken[] words, int leftStart, int rightStart, int length)
    {
        var matches = 0;
        for (var i = 0; i < length; i++)
        {
            if (string.Equals(
                    words[leftStart + i].Normalized,
                    words[rightStart + i].Normalized,
                    StringComparison.Ordinal))
            {
                matches++;
            }
        }

        return matches >= Math.Max(4, (int)Math.Ceiling(length * 0.8));
    }

    private readonly record struct TimelineToken(
        string Text,
        string Normalized,
        int Stability,
        bool IsCommitted);

    private readonly record struct WordToken(string Original, string Normalized);
    private readonly record struct Alignment(int TimelineStart, int WindowStart, int Matches, double Score);
    private readonly record struct ScoredMatch(int Matches, double Score);
}
