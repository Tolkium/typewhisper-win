using System.Diagnostics;
using System.Text.Json;
using TypeWhisper.Plugin.ParakeetOpenVino.Engine;

internal static partial class Checks
{
    /// <summary>
    /// Phase 4: full C# pipeline (waveform → text) per configuration; text must equal the Python
    /// reference of the same device. Reports load, warm-up, first-call and repeat-call times.
    /// </summary>
    public static int EndToEnd(string weights, string reference, string libs, string? cache, string configs)
    {
        OpenVinoLoader.Load(libs);
        var failed = false;
        foreach (var config in configs.Split(';', StringSplitOptions.RemoveEmptyEntries))
        {
            // config: DEVICE[/DECODER][@PAD]  e.g. GPU, GPU/CPU, GPU@100, NPU, CPU
            var pad = config.Contains('@') ? int.Parse(config[(config.IndexOf('@') + 1)..]) : 0;
            var spec = config.Split('@')[0].Split('/');
            var device = spec[0];
            var decoderDevice = spec.Length > 1 ? spec[1] : null;
            var key = device == "CPU" ? "ovcpu" : device.ToLowerInvariant();

            var watch = Stopwatch.StartNew();
            using var engine = new ParakeetOpenVinoEngine(weights, device, cache, decoderDevice, pad);
            var loadSeconds = watch.Elapsed.TotalSeconds;
            watch.Restart();
            engine.WarmUp();
            Console.WriteLine(
                $"== {config}: encoder {engine.Device}, decoder {engine.DecoderDevice}, pad {pad}, " +
                $"load {loadSeconds:F1}s, warm-up {watch.Elapsed.TotalSeconds:F2}s");

            foreach (var clip in AllClips)
            {
                var dir = Path.Combine(reference, clip);
                using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
                var root = meta.RootElement;
                var waveform = ReadFloats(Path.Combine(dir, "waveform.bin"));
                var expected = waveform.Length < 8000 ? "" : root.GetProperty($"text_{key}").GetString() ?? "";

                watch.Restart();
                var result = engine.Transcribe(waveform);
                var first = watch.Elapsed.TotalSeconds;
                watch.Restart();
                engine.Transcribe(waveform);
                var again = watch.Elapsed.TotalSeconds;

                var ok = result.Text == expected;
                failed |= !ok;
                Console.WriteLine(
                    $"  {clip,-21} {result.AudioSeconds,5:F1}s audio  first {first * 1000,5:F0} ms  again {again * 1000,5:F0} ms" +
                    $"{(result.Truncated ? "  [cut to 20 s]" : "")}  -> {(ok ? "PASS" : "FAIL")}" +
                    (ok ? "" : $"\n    got: {result.Text}\n    exp: {expected}"));
            }
        }

        return failed ? 1 : 0;
    }
}
