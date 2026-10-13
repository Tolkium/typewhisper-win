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
    private const int NpuFrames = NpuMaxSeconds * NemoFeatureExtractor.FramesPerSecond;
    private const int NpuMaxSamples = NpuMaxSeconds * NemoFeatureExtractor.SampleRate;
    private const int WarmUpSeconds = 5;

    /// <summary>The per-token decoder runs on CPU: measured faster than GPU for this loop, and NPU cannot run it.</summary>
    private const string DecoderDevice = "CPU";
    /// <summary>
    /// Shorter buffers (early live-preview polls) decode to garbage and return empty text. Kept below the
    /// host's 0.75 s padding of short final clips so a quick "Yes." is still transcribed.
    /// </summary>
    private const float MinimumSeconds = 0.5f;

    private readonly OvCore _core;
    private readonly OvCompiledModel _encoder;
    private readonly OvInferRequest _encoderRequest;
    private readonly TdtGreedyDecoder _decoder;
    private readonly ParakeetVocabulary _vocabulary;
    private readonly NemoFeatureExtractor _features = new();

    /// <param name="weightsDirectory">Folder with encoder-model.onnx(.data), decoder_joint-model.onnx, vocab.txt.</param>
    /// <param name="device">"GPU", "NPU" or "CPU".</param>
    /// <param name="cacheDirectory">OpenVINO compiled-model cache folder, or null for none.</param>
    public ParakeetOpenVinoEngine(string weightsDirectory, string device, string? cacheDirectory)
    {
        Device = device;
        IsFixedShape = device == "NPU";
        _core = new OvCore();
        try
        {
            _vocabulary = new ParakeetVocabulary(Path.Combine(weightsDirectory, "vocab.txt"));

            using (var model = _core.ReadModel(Path.Combine(weightsDirectory, "encoder-model.onnx")))
            {
                model.Reshape("audio_signal", 1, NemoFeatureExtractor.MelBins, IsFixedShape ? NpuFrames : -1);
                model.Reshape("length", 1);
                _encoder = _core.Compile(model, device, cacheDirectory);
            }
            _encoderRequest = _encoder.CreateInferRequest();

            _decoder = new TdtGreedyDecoder(
                _core, Path.Combine(weightsDirectory, "decoder_joint-model.onnx"), DecoderDevice, cacheDirectory, _vocabulary);
        }
        catch
        {
            Dispose(); // free the native handles created so far (the compiled encoder holds GPU memory)
            throw;
        }
    }

    public string Device { get; }

    /// <summary>True when the encoder has a fixed input window (NPU); longer audio is cut to it.</summary>
    private bool IsFixedShape { get; }

    /// <summary>Longest audio transcribed in one call, or null when the encoder accepts any length.</summary>
    public TimeSpan? MaxAudioDuration => IsFixedShape ? TimeSpan.FromSeconds(NpuMaxSeconds) : null;

    /// <summary>Runs a short silent clip so GPU kernels are built before the first dictation.</summary>
    public void WarmUp() => Transcribe(new float[WarmUpSeconds * NemoFeatureExtractor.SampleRate]);

    public ParakeetTranscription Transcribe(ReadOnlySpan<float> samples)
    {
        var seconds = samples.Length / (double)NemoFeatureExtractor.SampleRate;
        if (seconds < MinimumSeconds)
            return new ParakeetTranscription("", seconds, false);

        var truncated = false;
        if (IsFixedShape && samples.Length > NpuMaxSamples)
        {
            samples = samples[..NpuMaxSamples];
            truncated = true;
        }

        var inputFrames = IsFixedShape ? NpuFrames : NemoFeatureExtractor.FrameCount(samples.Length);
        var features = _features.Compute(samples, inputFrames); // [128, inputFrames], zero beyond the audio
        var validFrames = NemoFeatureExtractor.ValidFrameCount(samples.Length);

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

    public void Dispose()
    {
        // Null-tolerant: also called from the constructor after a partial failure.
        _decoder?.Dispose();
        _encoderRequest?.Dispose();
        _encoder?.Dispose();
        _core?.Dispose();
    }
}
