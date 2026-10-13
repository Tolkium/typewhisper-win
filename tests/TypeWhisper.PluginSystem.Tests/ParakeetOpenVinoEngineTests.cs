using System.IO;
using TypeWhisper.Core.Audio;
using TypeWhisper.Plugin.ParakeetOpenVino.Engine;

namespace TypeWhisper.PluginSystem.Tests;

/// <summary>
/// Pure C# parts of the native OpenVINO engine. Model-level equality with the Python
/// reference is checked by tools/OpenVinoSpike against tools/parakeet-reference data.
/// </summary>
public class ParakeetOpenVinoEngineTests
{
    [Fact]
    public void WavReader_ReadsHostWavEncoderOutput()
    {
        float[] samples = [0f, 0.5f, -0.5f, 0.25f];
        var wav = WavEncoder.Encode(samples);

        var decoded = WavReader.ReadMono16k(wav);

        Assert.Equal(samples.Length, decoded.Length);
        for (var i = 0; i < samples.Length; i++)
            Assert.Equal(samples[i], decoded[i], 3);
    }

    [Fact]
    public void WavReader_RejectsWrongSampleRate()
    {
        var wav = WavEncoder.Encode(new float[100], sampleRate: 8000);

        Assert.Throws<InvalidDataException>(() => WavReader.ReadMono16k(wav));
    }

    [Theory]
    [InlineData(16000, 101, 100)]
    [InlineData(176000, 1101, 1100)]
    [InlineData(159, 1, 0)]
    public void FeatureExtractor_FrameCountsMatchNumPy(int samples, int frames, int validFrames)
    {
        Assert.Equal(frames, NemoFeatureExtractor.FrameCount(samples));
        Assert.Equal(validFrames, NemoFeatureExtractor.ValidFrameCount(samples));
    }

    [Fact]
    public void FeatureExtractor_ToneGivesNormalisedFiniteFeatures()
    {
        var samples = new float[16000];
        for (var i = 0; i < samples.Length; i++)
            samples[i] = 0.3f * MathF.Sin(2 * MathF.PI * 440 * i / 16000f);

        var features = new NemoFeatureExtractor().Compute(samples);
        var frames = NemoFeatureExtractor.FrameCount(samples.Length);

        Assert.Equal(NemoFeatureExtractor.MelBins * frames, features.Length);
        Assert.All(features, v => Assert.True(float.IsFinite(v)));
        // Padding frame (past the valid length) is zero for every mel bin.
        for (var m = 0; m < NemoFeatureExtractor.MelBins; m++)
            Assert.Equal(0f, features[m * frames + frames - 1]);
    }

    [Fact]
    public void Vocabulary_DecodesSentencePieceSpacing()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(path, ["▁Hello 0", ", 1", "▁wor 2", "ld 3", ". 4", "<blk> 5"]);
            var vocabulary = new ParakeetVocabulary(path);

            Assert.Equal(5, vocabulary.BlankId);
            Assert.Equal(6, vocabulary.Size);
            Assert.Equal("Hello, world.", vocabulary.Decode([0, 1, 2, 3, 4]));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
