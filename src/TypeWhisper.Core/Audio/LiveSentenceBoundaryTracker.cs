using System.Text.RegularExpressions;

namespace TypeWhisper.Core.Audio;

/// <summary>
/// Records absolute sample offsets when live text ends a sentence on consecutive stable polls.
/// </summary>
public sealed class LiveSentenceBoundaryTracker
{
    private static readonly Regex SentenceEndingRegex = new(@"[.!?]\s*$", RegexOptions.CultureInvariant);

    private readonly List<int> _hints = [];
    private string _lastStable = "";
    private string? _pendingSentenceText;
    private int _pendingSentenceSample;
    private int _pendingSentenceConfirmations;

    /// <summary>
    /// Number of consecutive polls required before a sentence ending is committed.
    /// </summary>
    public int RequiredConfirmations { get; set; } = 2;

    /// <summary>
    /// Clears recorded hints and internal state.
    /// </summary>
    public void Reset()
    {
        _hints.Clear();
        _lastStable = "";
        _pendingSentenceText = null;
        _pendingSentenceSample = 0;
        _pendingSentenceConfirmations = 0;
    }

    /// <summary>
    /// Records a hint when sentence punctuation is stable across consecutive polls.
    /// </summary>
    public void TryRecord(string newStable, int totalSamplesRecorded)
    {
        if (totalSamplesRecorded <= 0)
            return;

        newStable = newStable.Trim();
        if (string.IsNullOrEmpty(newStable))
            return;

        var grew = newStable.Length > _lastStable.Length;

        if (SentenceEndingRegex.IsMatch(newStable))
        {
            if (grew)
            {
                if (string.Equals(_pendingSentenceText, newStable, StringComparison.Ordinal))
                    _pendingSentenceConfirmations++;
                else
                {
                    _pendingSentenceText = newStable;
                    _pendingSentenceSample = totalSamplesRecorded;
                    _pendingSentenceConfirmations = 1;
                }
            }
            else if (string.Equals(_pendingSentenceText, newStable, StringComparison.Ordinal))
            {
                _pendingSentenceConfirmations++;
            }

            if (_pendingSentenceConfirmations >= RequiredConfirmations)
                CommitHint(_pendingSentenceSample);
        }
        else
        {
            _pendingSentenceText = null;
            _pendingSentenceConfirmations = 0;
        }

        _lastStable = newStable;
    }

    private void CommitHint(int sampleOffset)
    {
        if (_hints.Count > 0 && _hints[^1] == sampleOffset)
            return;

        _hints.Add(sampleOffset);
        _pendingSentenceText = null;
        _pendingSentenceConfirmations = 0;
    }

    /// <summary>
    /// Returns a snapshot of sentence-end sample hints collected during the session.
    /// A copy is returned so callers are unaffected by a later <see cref="Reset"/>.
    /// </summary>
    public IReadOnlyList<int> GetHints() => _hints.ToArray();
}
