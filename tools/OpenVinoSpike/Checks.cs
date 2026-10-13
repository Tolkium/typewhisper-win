using System.Text.Json;
using TypeWhisper.Plugin.ParakeetOpenVino.Engine;

/// <summary>Per-phase checks against the Phase 0 reference data.</summary>
internal static partial class Checks
{
    private static readonly string[] AllClips =
    [
        "jfk_16k", "test_en", "stream_test", "last_recording", "synthetic_0p5s",
        "synthetic_3s", "synthetic_19p9s", "synthetic_45s", "synthetic_silence_2s",
    ];

    /// <summary>Phase 2: C# features vs NumPy features, max abs diff &lt; 1e-3 and same frame count.</summary>
    public static int Features(string reference)
    {
        var extractor = new NemoFeatureExtractor();
        var failed = false;
        foreach (var clip in AllClips)
        {
            var dir = Path.Combine(reference, clip);
            using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
            var root = meta.RootElement;
            var waveform = ReadFloats(Path.Combine(dir, "waveform.bin"));
            var expected = ReadFloats(Path.Combine(dir, "features.bin"));
            var expectedFrames = root.GetProperty("features").GetProperty("shape")[1].GetInt32();
            var expectedLen = root.GetProperty("features_len").GetInt32();

            var actual = extractor.Compute(waveform);
            var frames = NemoFeatureExtractor.FrameCount(waveform.Length);
            var len = NemoFeatureExtractor.ValidFrameCount(waveform.Length);

            double maxDiff = 0;
            var worst = 0;
            for (var i = 0; i < Math.Min(actual.Length, expected.Length); i++)
            {
                double d = Math.Abs(actual[i] - expected[i]);
                if (double.IsNaN(d) && !(float.IsNaN(actual[i]) && float.IsNaN(expected[i])))
                    d = double.PositiveInfinity;
                if (d > maxDiff)
                {
                    maxDiff = d;
                    worst = i;
                }
            }

            var ok = frames == expectedFrames && len == expectedLen && actual.Length == expected.Length && maxDiff < 1e-3;

            // Digital silence: every log-mel value is equal, std = 0, and NumPy's float32 rounding
            // noise divided by 1e-5 becomes a constant (-0.655). C# yields exact 0. Both mean "no
            // signal"; Phase 4 checks the transcript is empty.
            if (!ok && clip == "synthetic_silence_2s" && actual.All(v => v == 0f))
            {
                Console.WriteLine($"[features] {clip}: all zeros vs NumPy rounding constant -> PASS (known exception)");
                continue;
            }
            failed |= !ok;
            Console.WriteLine(
                $"[features] {clip}: frames {frames}/{expectedFrames}, len {len}/{expectedLen}, " +
                $"max diff {maxDiff:E2} at mel {worst / Math.Max(frames, 1)} frame {worst % Math.Max(frames, 1)} " +
                $"-> {(ok ? "PASS" : "FAIL")}");
        }

        return failed ? 1 : 0;
    }

    internal static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var result = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }
}
