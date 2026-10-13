using System.Diagnostics;
using System.Text.Json;
using TypeWhisper.Plugin.ParakeetOpenVino.Engine;

internal static partial class Checks
{
    /// <summary>
    /// Phase 3: C# TDT decoder fed with the Python reference encoder output; token ids and text
    /// must be identical. Decoder device matches what produced the reference (NPU used a CPU decoder).
    /// </summary>
    public static int Decoder(string weights, string reference, string libs)
    {
        OpenVinoLoader.Load(libs);
        using var core = new OvCore();
        var vocabulary = new ParakeetVocabulary(Path.Combine(weights, "vocab.txt"));
        Console.WriteLine($"vocab {vocabulary.Size}, blank {vocabulary.BlankId}");

        var failed = false;
        foreach (var (key, device) in new[] { ("ovcpu", "CPU"), ("npu", "CPU"), ("gpu", "GPU") })
        {
            using var decoder = new TdtGreedyDecoder(
                core, Path.Combine(weights, "decoder_joint-model.onnx"), device, null, vocabulary);

            foreach (var clip in AllClips)
            {
                var dir = Path.Combine(reference, clip);
                using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
                var root = meta.RootElement;
                if (!root.TryGetProperty($"encoder_{key}", out var encoderMeta))
                    continue;

                var rows = encoderMeta.GetProperty("shape")[0].GetInt32();
                var timeMajor = ReadFloats(Path.Combine(dir, $"encoder_{key}.bin")); // [rows, 1024]
                var channelMajor = new float[TdtGreedyDecoder.EncoderChannels * rows];
                for (var t = 0; t < rows; t++)
                    for (var c = 0; c < TdtGreedyDecoder.EncoderChannels; c++)
                        channelMajor[c * rows + t] = timeMajor[t * TdtGreedyDecoder.EncoderChannels + c];

                var encodedLength = root.GetProperty($"encoder_len_{key}").GetInt32();
                var expectedTokens = root.GetProperty($"tokens_{key}").EnumerateArray().Select(e => e.GetInt32()).ToList();
                var expectedText = root.GetProperty($"text_{key}").GetString() ?? "";

                var watch = Stopwatch.StartNew();
                var tokens = decoder.Decode(channelMajor, rows, encodedLength);
                var ms = watch.Elapsed.TotalMilliseconds;
                var text = vocabulary.Decode(tokens);

                var firstDiff = Enumerable.Range(0, Math.Min(tokens.Count, expectedTokens.Count))
                    .FirstOrDefault(i => tokens[i] != expectedTokens[i], -1);
                var ok = tokens.SequenceEqual(expectedTokens) && text == expectedText;
                failed |= !ok;
                Console.WriteLine(
                    $"[decoder {key}/{device}] {clip}: tokens {tokens.Count}/{expectedTokens.Count}, " +
                    $"text {(text == expectedText ? "same" : "DIFFERENT")}, {ms:F0} ms -> {(ok ? "PASS" : "FAIL")}" +
                    (ok ? "" : $"\n    first token diff at {firstDiff}\n    got: {text}\n    exp: {expectedText}"));
            }
        }

        return failed ? 1 : 0;
    }
}
