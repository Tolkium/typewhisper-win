using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace TypeWhisper.Plugin.ParakeetOpenVino.Engine;

/// <summary>
/// SentencePiece vocabulary from <c>vocab.txt</c> ("token id" per line) and the onnx-asr
/// detokenisation rule (<c>_AsrWithDecoding._decode_tokens</c>).
/// </summary>
internal sealed partial class ParakeetVocabulary
{
    private readonly string[] _tokens;

    public ParakeetVocabulary(string vocabPath)
    {
        var entries = new Dictionary<int, string>();
        foreach (var line in File.ReadLines(vocabPath, Encoding.UTF8))
        {
            if (line.Length == 0)
                continue;
            var separator = line.LastIndexOf(' ');
            if (separator <= 0)
                throw new InvalidDataException($"Bad vocab line: '{line}'");
            var id = int.Parse(line.AsSpan(separator + 1));
            entries[id] = line[..separator].Replace('▁', ' ');
        }

        _tokens = new string[entries.Count];
        foreach (var (id, token) in entries)
            _tokens[id] = token;

        BlankId = Array.IndexOf(_tokens, "<blk>");
        if (BlankId < 0)
            throw new InvalidDataException("Vocabulary has no <blk> token.");
    }

    /// <summary>Number of token logits in the joint output (blank included).</summary>
    public int Size => _tokens.Length;

    public int BlankId { get; }

    public string Decode(IEnumerable<int> tokenIds)
    {
        var builder = new StringBuilder();
        foreach (var id in tokenIds)
            builder.Append(_tokens[id]);

        // Leading space dropped, spaces before a non-word boundary dropped, word-boundary spaces kept.
        return SpacePattern().Replace(builder.ToString(), m => m.Groups[1].Success ? " " : "");
    }

    [GeneratedRegex(@"\A\s|\s\B|(\s)\b")]
    private static partial Regex SpacePattern();
}
