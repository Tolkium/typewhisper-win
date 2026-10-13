using System.Diagnostics;
using System.Text.Json;
using TypeWhisper.Plugin.ParakeetOpenVino.Engine;
using static TypeWhisper.Plugin.ParakeetOpenVino.Engine.OpenVinoNative;

// Phase 1 of docs/plans/native-openvino-engine.md: run the Parakeet encoder through the
// OpenVINO C API and compare with the Phase 0 reference encoder outputs.
//
// Usage: OpenVinoSpike <openvino libs dir> <model weights dir> <reference dir> [devices] [cache dir]

// Modes for later phases: "features <reference>" (phase 2), "decoder <weights> <reference> <ov-libs>" (phase 3).
if (args.Length > 0 && args[0] == "features")
    return Checks.Features(args[1]);
if (args.Length > 0 && args[0] == "decoder")
    return Checks.Decoder(args[1], args[2], args[3]);

if (args.Length < 3)
{
    Console.Error.WriteLine("Usage: OpenVinoSpike <ov-libs> <weights> <reference> [CPU,GPU,NPU] [cache]");
    return 2;
}

var libs = args[0];
var weights = args[1];
var reference = args[2];
var devices = (args.Length > 3 ? args[3] : "CPU,GPU,NPU").Split(',', StringSplitOptions.RemoveEmptyEntries);
var cache = args.Length > 4 ? args[4] : null;
const int NpuFrames = 2000;

OpenVinoLoader.Load(libs);
Console.WriteLine($"OpenVINO {OpenVinoLoader.Version()}");

using var core = new OvCore();
Console.WriteLine($"Devices: {string.Join(", ", core.AvailableDevices())}");

var clips = new[] { "jfk_16k", "synthetic_3s", "synthetic_19p9s", "synthetic_45s", "stream_test" };
var failed = false;

foreach (var device in devices)
{
    var referenceKey = device == "CPU" ? "ovcpu" : device.ToLowerInvariant();
    var watch = Stopwatch.StartNew();
    using var model = core.ReadModel(Path.Combine(weights, "encoder-model.onnx"));
    if (device == "NPU")
    {
        model.Reshape("audio_signal", 1, 128, NpuFrames);
        model.Reshape("length", 1);
    }
    else
    {
        model.Reshape("audio_signal", 1, 128, -1);
        model.Reshape("length", 1);
    }
    using var compiled = core.Compile(model, device, cache);
    using var request = compiled.CreateInferRequest();
    Console.WriteLine($"[{device}] encoder ready in {watch.Elapsed.TotalSeconds:F1}s");

    foreach (var clip in clips)
    {
        var clipDir = Path.Combine(reference, clip);
        using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(clipDir, "meta.json")));
        var root = meta.RootElement;
        if (!root.TryGetProperty($"encoder_{referenceKey}", out var encoderMeta))
            continue;

        // Reference features for this device's input (NPU input is cut to the first 20 s).
        var features = Checks.ReadFloats(Path.Combine(clipDir, "features.bin"));
        var featureShape = root.GetProperty("features").GetProperty("shape");
        var frames = featureShape[1].GetInt32();
        var featuresLen = root.GetProperty("features_len").GetInt64();
        if (device == "NPU")
        {
            var npuSeconds = root.GetProperty("input_seconds_npu").GetDouble();
            var npuSamples = (long)Math.Round(npuSeconds * 16000);
            featuresLen = npuSamples / 160;
            if (frames > featuresLen + 1)
            {
                // Features of the 20 s cut differ only in normalisation; skip exact compare for cut clips.
                Console.WriteLine($"[{device}] {clip}: skipped (cut clip, compare in phase 4)");
                continue;
            }
        }

        var inputFrames = device == "NPU" ? NpuFrames : frames;
        var input = new float[128 * inputFrames];
        for (var m = 0; m < 128; m++)
            Array.Copy(features, m * frames, input, m * inputFrames, Math.Min(frames, inputFrames));
        var length = new[] { Math.Min(featuresLen, inputFrames) };

        watch.Restart();
        float[] output;
        long[] outShape;
        long encodedLen;
        unsafe
        {
            fixed (float* f = input)
            fixed (long* l = length)
            {
                request.SetInput("audio_signal", ElementType.F32, f, 1, 128, inputFrames);
                request.SetInput("length", ElementType.I64, l, 1);
                request.Infer();
            }
            (output, outShape) = request.GetOutput<float>("outputs");
            encodedLen = request.GetOutput<long>("encoded_lengths").Data[0];
        }
        var inferSeconds = watch.Elapsed.TotalSeconds;

        // Reference is [T', 1024]; ours is [1, 1024, T].
        var expected = Checks.ReadFloats(Path.Combine(clipDir, $"encoder_{referenceKey}.bin"));
        var refLen = root.GetProperty($"encoder_len_{referenceKey}").GetInt64();
        var tOut = (int)outShape[2];
        double maxDiff = 0;
        for (var t = 0; t < refLen; t++)
            for (var c = 0; c < 1024; c++)
                maxDiff = Math.Max(maxDiff, Math.Abs(output[c * tOut + t] - expected[t * 1024 + c]));

        var tolerance = device == "CPU" ? 1e-3 : 5e-2;
        var ok = encodedLen == refLen && maxDiff < tolerance;
        failed |= !ok;
        Console.WriteLine(
            $"[{device}] {clip}: len {encodedLen}/{refLen}, max diff {maxDiff:E2} (tol {tolerance:E0}), " +
            $"{inferSeconds * 1000:F0} ms -> {(ok ? "PASS" : "FAIL")}");
    }
}

return failed ? 1 : 0;

