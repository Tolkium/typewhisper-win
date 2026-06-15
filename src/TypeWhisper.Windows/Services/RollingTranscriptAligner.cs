using System.Globalization;
using System.Text;

namespace TypeWhisper.Windows.Services;

/// <summary>
/// Word-level aligner for merging unstable ASR rolling windows into a stable transcript.
/// </summary>
internal static class RollingTranscriptAligner
{
    private const int MaxSearchTailWords = 180;
    private const double MostlyContainedCoverage = 0.78;

    public static string Merge(
        string confirmed,
        string windowText,
        bool rollingTailWindow = false,
        int frozenPrefixWordCount = 0)
    {
        confirmed = confirmed.Trim();
        windowText = windowText.Trim();

        if (string.IsNullOrEmpty(confirmed))
            return CollapseAdjacentDuplicatePhrase(windowText);
        if (string.IsNullOrEmpty(windowText))
            return confirmed;

        var confirmedWords = Tokenize(confirmed);
        var windowWords = Tokenize(windowText);
        if (confirmedWords.Length == 0 || windowWords.Length == 0)
            return JoinParts(confirmed, windowText);

        if (rollingTailWindow && frozenPrefixWordCount > 0 && frozenPrefixWordCount < confirmedWords.Length)
        {
            var frozen = Rebuild(confirmedWords[..frozenPrefixWordCount]);
            var revisable = Rebuild(confirmedWords[frozenPrefixWordCount..]);
            return JoinParts(
                frozen,
                Merge(revisable, windowText, rollingTailWindow: true));
        }

        if (IsMostlyContained(confirmedWords, windowWords))
            return confirmed;

        var alignment = FindBestAlignment(confirmedWords, windowWords, rollingTailWindow);
        if (alignment is not null)
        {
            if (!rollingTailWindow
                && alignment.Value.ConfirmedStart == 0
                && alignment.Value.WindowStart == 0
                && alignment.Value.Matches < confirmedWords.Length)
            {
                var append = Rebuild(windowWords[alignment.Value.Matches..]);
                return CollapseAdjacentDuplicatePhrase(JoinParts(confirmed, append));
            }

            var prefixText = alignment.Value.ConfirmedStart > 0
                ? Rebuild(confirmedWords[..alignment.Value.ConfirmedStart])
                : "";
            // Fresh tail starts at the alignment anchor inside the window, not at word 0.
            // Using the full windowText here re-inserts words before the anchor and causes
            // the "No tak dobrú noc… No tak dobrú noc…" duplicates seen in live preview.
            var freshTail = Rebuild(windowWords[alignment.Value.WindowStart..]);
            var merged = JoinParts(prefixText, freshTail);
            return IsDestructiveMerge(confirmedWords, Tokenize(merged))
                ? confirmed
                : CollapseAdjacentDuplicatePhrase(merged);
        }

        // For live rolling windows, no reliable overlap usually means the model restarted
        // or hallucinated. Keep the stable text instead of duplicating a large old block.
        if (rollingTailWindow)
            return confirmed;

        return CollapseAdjacentDuplicatePhrase(JoinParts(confirmed, windowText));
    }

    private static Alignment? FindBestAlignment(
        WordToken[] confirmedWords,
        WordToken[] windowWords,
        bool rollingTailWindow)
    {
        var confirmedSearchStart = Math.Max(0, confirmedWords.Length - MaxSearchTailWords);
        var best = default(Alignment);
        var hasBest = false;

        for (var confirmedStart = confirmedSearchStart; confirmedStart < confirmedWords.Length; confirmedStart++)
        {
            for (var windowStart = 0; windowStart < windowWords.Length; windowStart++)
            {
                var scored = ScoreConsecutiveMatch(confirmedWords, confirmedStart, windowWords, windowStart);
                if (scored.Matches == 0)
                    continue;

                var score = scored.Score
                    + Math.Min(confirmedStart, confirmedWords.Length - 1) * 0.01
                    - windowStart * 0.02;

                if (!hasBest || score > best.Score)
                {
                    best = new Alignment(confirmedStart, windowStart, scored.Matches, score);
                    hasBest = true;
                }
            }
        }

        if (!hasBest)
            return null;

        var minOverlap = MinimumOverlapWords(windowWords.Length, rollingTailWindow);
        return best.Matches >= minOverlap ? best : null;
    }

    private static ScoredMatch ScoreConsecutiveMatch(
        WordToken[] confirmedWords,
        int confirmedStart,
        WordToken[] windowWords,
        int windowStart)
    {
        var matches = 0;
        double score = 0;
        var confirmedIndex = confirmedStart;
        var windowIndex = windowStart;

        while (confirmedIndex < confirmedWords.Length && windowIndex < windowWords.Length)
        {
            var wordScore = WordScore(confirmedWords[confirmedIndex].Normalized, windowWords[windowIndex].Normalized);
            if (wordScore <= 0)
                break;

            matches++;
            score += wordScore;
            confirmedIndex++;
            windowIndex++;
        }

        return new ScoredMatch(matches, score);
    }

    private static bool IsMostlyContained(WordToken[] confirmedWords, WordToken[] windowWords)
    {
        if (windowWords.Length <= 2 || confirmedWords.Length < windowWords.Length)
            return false;

        var best = 0;
        for (var start = 0; start < confirmedWords.Length; start++)
        {
            for (var windowStart = 0; windowStart < Math.Min(3, windowWords.Length); windowStart++)
            {
                var scored = ScoreConsecutiveMatch(confirmedWords, start, windowWords, windowStart);
                best = Math.Max(best, scored.Matches);
            }
        }

        var required = Math.Max(3, (int)Math.Ceiling(windowWords.Length * MostlyContainedCoverage));
        return best >= required;
    }

    private static bool IsDestructiveMerge(WordToken[] confirmedWords, WordToken[] mergedWords)
    {
        if (confirmedWords.Length < 24)
            return false;

        return mergedWords.Length < (int)(confirmedWords.Length * 0.55);
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
        // Real dictation restarts can repeat more than once, so collapse iteratively
        // with a hard cap to stay bounded on pathological input.
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
        // Duplicate deletion must be conservative: only exact normalized repeats count,
        // so prefix-sharing words (slovo1/slovo15, inflections) are never collapsed away.
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

    private static string Rebuild(IEnumerable<WordToken> words) =>
        string.Join(" ", words.Select(word => word.Original));

    internal static string JoinParts(string prefix, string append)
    {
        if (string.IsNullOrEmpty(append))
            return prefix;
        if (string.IsNullOrEmpty(prefix))
            return append;

        var needsSpace = !prefix.EndsWith(' ') && !append.StartsWith(' ');
        return needsSpace ? prefix + " " + append : prefix + append;
    }

    private readonly record struct WordToken(string Original, string Normalized);
    private readonly record struct Alignment(int ConfirmedStart, int WindowStart, int Matches, double Score);
    private readonly record struct ScoredMatch(int Matches, double Score);
}
