namespace TypeWhisper.Windows.Services;

/// <summary>
/// Result of stopping a live transcription session.
/// </summary>
/// <param name="LiveText">Stabilized live preview text at stop time.</param>
/// <param name="SentenceEndSampleHints">Absolute sample offsets where live text ended a sentence.</param>
public readonly record struct StreamingSessionStop(
    string LiveText,
    IReadOnlyList<int> SentenceEndSampleHints);
