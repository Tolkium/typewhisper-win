using System.Diagnostics;
using TypeWhisper.Plugin.ParakeetOpenVino.Engine;

internal static partial class Checks
{
    /// <summary>
    /// Phase 5: plugin setup path. Runs ParakeetSetup in <paramref name="dataDirectory"/> (downloads
    /// whatever is missing), then loads the engine from the downloaded runtime and transcribes a WAV.
    /// </summary>
    public static int Setup(string dataDirectory, string device, string wavPath)
    {
        var setup = new ParakeetSetup(dataDirectory);
        Console.WriteLine($"before: model {setup.IsModelReady}, runtime {setup.IsRuntimeReady}");

        var watch = Stopwatch.StartNew();
        var lastPercent = -1;
        var progress = new Progress<double>(p =>
        {
            var percent = (int)(p * 100);
            if (percent / 10 != lastPercent / 10)
                Console.WriteLine($"  progress {percent}%");
            lastPercent = percent;
        });
        setup.EnsureAsync(progress, message => Console.WriteLine($"  {message}"), CancellationToken.None)
            .GetAwaiter().GetResult();
        Console.WriteLine($"after: ready {setup.IsReady} in {watch.Elapsed.TotalSeconds:F0}s");

        watch.Restart();
        OpenVinoLoader.Load(setup.RuntimeDirectory);
        using var engine = new ParakeetOpenVinoEngine(setup.WeightsDirectory, device, setup.CacheDirectory);
        engine.WarmUp();
        Console.WriteLine($"OpenVINO {OpenVinoLoader.Version()} from {setup.RuntimeDirectory}; {device} loaded in {watch.Elapsed.TotalSeconds:F1}s");

        var samples = WavReader.ReadMono16k(File.ReadAllBytes(wavPath));
        watch.Restart();
        var result = engine.Transcribe(samples);
        Console.WriteLine($"{result.AudioSeconds:F1}s in {watch.Elapsed.TotalMilliseconds:F0} ms: {result.Text}");
        return setup.IsReady && result.Text.Length > 0 ? 0 : 1;
    }
}
