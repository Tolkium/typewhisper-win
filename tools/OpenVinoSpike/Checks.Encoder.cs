using System.Diagnostics;
using TypeWhisper.Plugin.ParakeetOpenVino.Engine;
using static TypeWhisper.Plugin.ParakeetOpenVino.Engine.OpenVinoNative;

internal static partial class Checks
{
    private const int NpuFrames = ParakeetOpenVinoEngine.NpuMaxSeconds * NemoFeatureExtractor.FramesPerSecond;

    /// <summary>
    /// Phase 1: the encoder through the OpenVINO C API, fed with the reference features; output must match
    /// the Python reference of the same device (CPU within 1e-3, GPU/NPU FP16 within 5e-2).
    /// </summary>
    public static unsafe int Encoder(string libs, string weights, string reference, string devices, string? cache)
    {
        OpenVinoLoader.Load(libs);
        Console.WriteLine($"OpenVINO {OpenVinoLoader.Version()}");

        using var core = new OvCore();
        Console.WriteLine($"Devices: {string.Join(", ", core.AvailableDevices())}");

        string[] clips = ["jfk_16k", "synthetic_3s", "synthetic_19p9s", "synthetic_45s", "stream_test"];
        var failed = false;
        foreach (var device in devices.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            var key = ReferenceKey(device);
            var watch = Stopwatch.StartNew();
            using var model = core.ReadModel(Path.Combine(weights, "encoder-model.onnx"));
            model.Reshape("audio_signal", 1, NemoFeatureExtractor.MelBins, device == "NPU" ? NpuFrames : -1);
            model.Reshape("length", 1);
            using var compiled = core.Compile(model, device, cache);
            using var request = compiled.CreateInferRequest();
            Console.WriteLine($"[{device}] encoder ready in {watch.Elapsed.TotalSeconds:F1}s");

            foreach (var clip in clips)
            {
                using var meta = ReadMeta(reference, clip);
                var root = meta.RootElement;
                if (!root.TryGetProperty($"encoder_{key}", out _))
                    continue;

                var features = ReadFloats(Path.Combine(reference, clip, "features.bin")); // [128, frames]
                var frames = root.GetProperty("features").GetProperty("shape")[1].GetInt32();
                var featuresLen = root.GetProperty("features_len").GetInt64();
                if (device == "NPU" && frames > NpuFrames + 1)
                {
                    Console.WriteLine($"[{device}] {clip}: skipped (cut clip, compared end-to-end in phase 4)");
                    continue;
                }

                var inputFrames = device == "NPU" ? NpuFrames : frames;
                var input = new float[NemoFeatureExtractor.MelBins * inputFrames];
                for (var m = 0; m < NemoFeatureExtractor.MelBins; m++)
                    Array.Copy(features, m * frames, input, m * inputFrames, Math.Min(frames, inputFrames));
                long[] length = [Math.Min(featuresLen, inputFrames)];

                watch.Restart();
                fixed (float* f = input)
                fixed (long* l = length)
                {
                    request.SetInput("audio_signal", ElementType.F32, f, 1, NemoFeatureExtractor.MelBins, inputFrames);
                    request.SetInput("length", ElementType.I64, l, 1);
                    request.Infer();
                }
                var inferMs = watch.Elapsed.TotalMilliseconds;

                using var output = request.GetTensor("outputs");   // [1, 1024, T]
                using var lengths = request.GetTensor("encoded_lengths");
                var stride = (int)output.Shape()[2];
                var actual = output.Data<float>();
                var encodedLength = lengths.Data<long>()[0];

                var expected = ReadFloats(Path.Combine(reference, clip, $"encoder_{key}.bin")); // [T', 1024]
                var expectedLength = root.GetProperty($"encoder_len_{key}").GetInt64();
                double maxDiff = 0;
                for (var t = 0; t < expectedLength; t++)
                    for (var c = 0; c < TdtGreedyDecoder.EncoderChannels; c++)
                        maxDiff = Math.Max(maxDiff, Math.Abs(actual[c * stride + t] - expected[t * TdtGreedyDecoder.EncoderChannels + c]));

                var tolerance = device == "CPU" ? 1e-3 : 5e-2;
                var ok = encodedLength == expectedLength && maxDiff < tolerance;
                failed |= !ok;
                Console.WriteLine(
                    $"[{device}] {clip}: len {encodedLength}/{expectedLength}, max diff {maxDiff:E2} (tol {tolerance:E0}), " +
                    $"{inferMs:F0} ms -> {(ok ? "PASS" : "FAIL")}");
            }
        }

        return failed ? 1 : 0;
    }
}
