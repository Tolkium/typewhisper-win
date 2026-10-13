using static TypeWhisper.Plugin.ParakeetOpenVino.Engine.OpenVinoNative;

namespace TypeWhisper.Plugin.ParakeetOpenVino.Engine;

/// <summary>
/// Greedy TDT decoding with the Parakeet decoder/joint network. Port of onnx-asr 0.11.0
/// <c>_AsrWithTransducerDecoding._decoding</c> + <c>NemoConformerTdt._decode</c> (MIT licence).
/// </summary>
internal sealed unsafe class TdtGreedyDecoder : IDisposable
{
    public const int EncoderChannels = 1024;
    private const int StateSize = 2 * 1 * 640;
    private const int MaxTokensPerStep = 10;

    private readonly OvCompiledModel _compiled;
    private readonly OvInferRequest _request;
    private readonly OvTensor _encoderInput;
    private readonly OvTensor _targetsInput;
    private readonly OvTensor _targetLengthInput;
    private readonly OvTensor _state1Input;
    private readonly OvTensor _state2Input;
    private readonly int _vocabSize;
    private readonly int _blankId;
    private readonly int _outputSize;

    public TdtGreedyDecoder(OvCore core, string decoderModelPath, string device, string? cacheDir, ParakeetVocabulary vocabulary)
    {
        _vocabSize = vocabulary.Size;
        _blankId = vocabulary.BlankId;

        using (var model = core.ReadModel(decoderModelPath))
        {
            model.Reshape("encoder_outputs", 1, EncoderChannels, 1);
            model.Reshape("targets", 1, 1);
            model.Reshape("target_length", 1);
            model.Reshape("input_states_1", 2, 1, 640);
            model.Reshape("input_states_2", 2, 1, 640);
            _compiled = core.Compile(model, device, cacheDir);
        }

        _request = _compiled.CreateInferRequest();
        _encoderInput = _request.GetTensor("encoder_outputs");
        _targetsInput = _request.GetTensor("targets");
        _targetLengthInput = _request.GetTensor("target_length");
        _state1Input = _request.GetTensor("input_states_1");
        _state2Input = _request.GetTensor("input_states_2");
        _targetLengthInput.Data<int>()[0] = 1;

        using var outputs = _request.GetTensor("outputs");
        _outputSize = (int)outputs.Shape()[^1];
        if (_outputSize <= _vocabSize)
            throw new InvalidOperationException($"Joint output {_outputSize} has no duration logits (vocab {_vocabSize}).");
    }

    /// <summary>
    /// Decodes encoder output laid out as [1024, <paramref name="frameStride"/>] (the encoder's native
    /// [1, 1024, T] layout); only the first <paramref name="encodedLength"/> frames are used.
    /// </summary>
    public List<int> Decode(ReadOnlySpan<float> encoder, int frameStride, int encodedLength)
    {
        var tokens = new List<int>();
        new Span<float>(_state1Input.Data<float>(), StateSize).Clear();
        new Span<float>(_state2Input.Data<float>(), StateSize).Clear();

        var encoderFrame = _encoderInput.Data<float>();
        var targets = _targetsInput.Data<int>();
        var t = 0;
        var emitted = 0;
        while (t < encodedLength)
        {
            for (var c = 0; c < EncoderChannels; c++)
                encoderFrame[c] = encoder[c * frameStride + t];
            targets[0] = tokens.Count > 0 ? tokens[^1] : _blankId;

            _request.Infer();

            int token, step;
            using (var output = _request.GetTensor("outputs"))
            {
                var logits = output.Data<float>();
                token = ArgMax(logits, 0, _vocabSize);
                step = ArgMax(logits, _vocabSize, _outputSize);
            }

            if (token != _blankId)
            {
                // The state only advances when a token is emitted.
                CopyOutputState("output_states_1", _state1Input);
                CopyOutputState("output_states_2", _state2Input);
                tokens.Add(token);
                emitted++;
            }

            if (step > 0)
            {
                t += step;
                emitted = 0;
            }
            else if (token == _blankId || emitted == MaxTokensPerStep)
            {
                t++;
                emitted = 0;
            }
        }

        return tokens;
    }

    private void CopyOutputState(string name, OvTensor destination)
    {
        using var source = _request.GetTensor(name);
        new ReadOnlySpan<float>(source.Data<float>(), StateSize)
            .CopyTo(new Span<float>(destination.Data<float>(), StateSize));
    }

    /// <summary>Index of the first maximum in [start, end), relative to <paramref name="start"/> (NumPy argmax).</summary>
    private static int ArgMax(float* values, int start, int end)
    {
        var best = start;
        for (var i = start + 1; i < end; i++)
        {
            if (values[i] > values[best])
                best = i;
        }
        return best - start;
    }

    public void Dispose()
    {
        _encoderInput.Dispose();
        _targetsInput.Dispose();
        _targetLengthInput.Dispose();
        _state1Input.Dispose();
        _state2Input.Dispose();
        _request.Dispose();
        _compiled.Dispose();
    }
}
