using System.IO;
using static TypeWhisper.Plugin.ParakeetOpenVino.Engine.OpenVinoNative;

namespace TypeWhisper.Plugin.ParakeetOpenVino.Engine;

/// <summary>Result of one transcription.</summary>
internal sealed record ParakeetTranscription(string Text, double AudioSeconds, bool Truncated);

/// <summary>
/// Parakeet TDT 0.6B v3 on OpenVINO, in-process: C# features → encoder → TDT greedy decoder.
/// GPU and CPU use a dynamic-length encoder; NPU needs a fixed shape, so it gets a 20 s window
/// and longer audio is cut to its first 20 s.
/// </summary>
internal sealed unsafe class ParakeetOpenVinoEngine : IDisposable
{
    public const int NpuMaxSeconds = 20;
    private const int NpuFrames = NpuMaxSeconds * 100;
    private const float MinimumSeconds = 1.0f;

    private readonly OvCore _core;
    private readonly OvCompiledModel _encoder;
    private readonly OvInferRequest _encoderRequest;
    private readonly TdtGreedyDecoder _decoder;
    private readonly ParakeetVocabulary _vocabulary;
    private readonly NemoFeatureExtractor _features = new();
    private readonly int _gpuPadFrames;

    /// <param name="weightsDirectory">Folder with encoder-model.onnx(.data), decoder_joint-model.onnx, vocab.txt.</param>
    /// <param name="device">"GPU", "NPU" or "CPU".</param>
    /// <param name="cacheDirectory">OpenVINO compiled-model cache folder, or null for none.</param>
    /// <param name="decoderDevice">Device for the per-token decoder; null = CPU (measured faster on GPU first calls).</param>
    /// <param name="gpuPadFrames">Round GPU/CPU encoder input up to a multiple of this many frames
    /// (0 = no padding) so repeated recording lengths reuse already-prepared GPU kernels.</param>
    public ParakeetOpenVinoEngine(
        string weightsDirectory,
        string device,
        string? cacheDirectory,
        string? decoderDevice = null,
        int gpuPadFrames = 0)
    {
        Device = device;
        _gpuPadFrames = device == "NPU" ? 0 : gpuPadFrames;
        _core = new OvCore();
        _vocabulary = new ParakeetVocabulary(Path.Combine(weightsDirectory, "vocab.txt"));

        using (var model = _core.ReadModel(Path.Combine(weightsDirectory, "encoder-model.onnx")))
        {
            model.Reshape("audio_signal", 1, NemoFeatureExtractor.MelBins, device == "NPU" ? NpuFrames : -1);
            model.Reshape("length", 1);
            _encoder = _core.Compile(model, device, cacheDirectory);
        }
        _encoderRequest = _encoder.CreateInferRequest();

        DecoderDevice = decoderDevice ?? "CPU";
        _decoder = new TdtGreedyDecoder(
            _core, Path.Combine(weightsDirectory, "decoder_joint-model.onnx"), DecoderDevice, cacheDirectory, _vocabulary);
    }

    public string Device { get; }

    public string DecoderDevice { get; }

    /// <summary>Runs a short silent clip so GPU kernels are built before the first dictation.</summary>
    public void WarmUp() => Transcribe(new float[5 * NemoFeatureExtractor.SampleRate]);

    public ParakeetTranscription Transcribe(ReadOnlySpan<float> samples)
    {
        var seconds = samples.Length / (double)NemoFeatureExtractor.SampleRate;
        if (seconds < MinimumSeconds)
            return new ParakeetTranscription("", seconds, false); // short live-poll buffers decode to garbage

        var truncated = false;
        if (Device == "NPU" && samples.Length > NpuMaxSeconds * NemoFeatureExtractor.SampleRate)
        {
            samples = samples[..(NpuMaxSeconds * NemoFeatureExtractor.SampleRate)];
            truncated = true;
        }

        var features = _features.Compute(samples); // [128, frames]
        var frames = NemoFeatureExtractor.FrameCount(samples.Length);
        var validFrames = NemoFeatureExtractor.ValidFrameCount(samples.Length);

        var inputFrames = Device == "NPU" ? NpuFrames : PadFrames(frames);
        if (inputFrames != frames)
            features = PadTime(features, frames, inputFrames);

        var length = new long[] { validFrames };
        fixed (float* f = features)
        fixed (long* l = length)
        {
            _encoderRequest.SetInput("audio_signal", ElementType.F32, f, 1, NemoFeatureExtractor.MelBins, inputFrames);
            _encoderRequest.SetInput("length", ElementType.I64, l, 1);
            _encoderRequest.Infer();
        }

        using var output = _encoderRequest.GetTensor("outputs");
        using var lengths = _encoderRequest.GetTensor("encoded_lengths");
        var shape = output.Shape(); // [1, 1024, T]
        var stride = (int)shape[2];
        var encodedLength = (int)Math.Min(lengths.Data<long>()[0], stride);
        var encoder = new ReadOnlySpan<float>(output.Data<float>(), TdtGreedyDecoder.EncoderChannels * stride);

        var tokens = _decoder.Decode(encoder, stride, encodedLength);
        return new ParakeetTranscription(_vocabulary.Decode(tokens), seconds, truncated);
    }

    private int PadFrames(int frames) =>
        _gpuPadFrames <= 0 ? frames : (frames + _gpuPadFrames - 1) / _gpuPadFrames * _gpuPadFrames;

    private static float[] PadTime(float[] features, int frames, int targetFrames)
    {
        var padded = new float[NemoFeatureExtractor.MelBins * targetFrames];
        for (var m = 0; m < NemoFeatureExtractor.MelBins; m++)
            Array.Copy(features, m * frames, padded, m * targetFrames, Math.Min(frames, targetFrames));
        return padded;
    }

    public void Dispose()
    {
        _decoder.Dispose();
        _encoderRequest.Dispose();
        _encoder.Dispose();
        _core.Dispose();
    }
}
