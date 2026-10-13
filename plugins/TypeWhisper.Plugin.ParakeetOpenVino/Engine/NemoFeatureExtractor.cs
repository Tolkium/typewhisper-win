using System.IO;
using System.Reflection;

namespace TypeWhisper.Plugin.ParakeetOpenVino.Engine;

/// <summary>
/// NeMo 128-bin log-mel features for Parakeet. Port of onnx-asr 0.11.0
/// <c>NemoPreprocessorNumpy</c> (MIT licence).
/// </summary>
internal sealed class NemoFeatureExtractor
{
    public const int SampleRate = 16000;
    public const int MelBins = 128;
    public const int HopLength = 160;
    public const int FramesPerSecond = SampleRate / HopLength;

    private const int FftSize = 512;
    private const int WindowLength = 400;
    private const int SpectrumBins = FftSize / 2 + 1;
    private const float PreEmphasis = 0.97f;
    private const float LogZeroGuard = 1f / (1 << 24);
    private const double NormalizationEpsilon = 1e-5;

    private readonly float[] _melFilters; // [MelBins, SpectrumBins]: one contiguous row per mel bin
    private readonly int[] _melStart;     // per mel bin: first spectrum bin with a nonzero weight
    private readonly int[] _melEnd;       // per mel bin: one past the last nonzero weight
    private readonly double[] _window;    // [FftSize]
    private readonly Fft _fft = new(FftSize);

    public NemoFeatureExtractor()
    {
        (_melFilters, _melStart, _melEnd) = LoadMelFilters();
        _window = CreateWindow();
    }

    /// <summary>Number of feature frames produced for <paramref name="sampleCount"/> samples.</summary>
    public static int FrameCount(int sampleCount) => 1 + sampleCount / HopLength;

    /// <summary>Number of valid (non-padding) frames, as passed to the encoder <c>length</c> input.</summary>
    public static int ValidFrameCount(int sampleCount) => sampleCount / HopLength;

    /// <summary>
    /// Computes features laid out as [MelBins, <paramref name="outputFrames"/>] (row-major), ready for the
    /// encoder. Columns past the valid frames are zero, so a fixed-shape encoder input is filled directly;
    /// by default the width is <see cref="FrameCount"/>. The width must hold all valid frames.
    /// </summary>
    public float[] Compute(ReadOnlySpan<float> samples, int? outputFrames = null)
    {
        var frames = FrameCount(samples.Length);
        var columns = outputFrames ?? frames;
        ArgumentOutOfRangeException.ThrowIfLessThan(columns, ValidFrameCount(samples.Length), nameof(outputFrames));
        var validFrames = ValidFrameCount(samples.Length);
        var padded = PreEmphasizeAndPad(samples);
        var logMel = new float[frames * MelBins]; // [frames, MelBins]

        var real = new double[FftSize];
        var imag = new double[FftSize];
        var power = new float[SpectrumBins];
        for (var f = 0; f < frames; f++)
        {
            var start = f * HopLength;
            for (var i = 0; i < FftSize; i++)
            {
                real[i] = padded[start + i] * _window[i];
                imag[i] = 0;
            }

            _fft.Transform(real, imag);
            for (var k = 0; k < SpectrumBins; k++)
            {
                // NumPy: abs(rfft) in float64, cast to float32, then squared in float32.
                var magnitude = (float)Math.Sqrt(real[k] * real[k] + imag[k] * imag[k]);
                power[k] = magnitude * magnitude;
            }

            var row = logMel.AsSpan(f * MelBins, MelBins);
            for (var m = 0; m < MelBins; m++)
            {
                // Triangular filters: summing only the nonzero span skips exact zeros, same result.
                var filter = _melFilters.AsSpan(m * SpectrumBins, SpectrumBins);
                float sum = 0;
                for (var k = _melStart[m]; k < _melEnd[m]; k++)
                    sum += power[k] * filter[k];
                row[m] = MathF.Log(sum + LogZeroGuard);
            }
        }

        return Normalize(logMel, columns, validFrames);
    }

    private static float[] PreEmphasizeAndPad(ReadOnlySpan<float> samples)
    {
        const int pad = FftSize / 2;
        var padded = new float[samples.Length + 2 * pad];
        if (samples.Length == 0)
            return padded;

        padded[pad] = samples[0];
        for (var i = 1; i < samples.Length; i++)
            padded[pad + i] = samples[i] - PreEmphasis * samples[i - 1];
        return padded;
    }

    /// <summary>Per mel bin: (x - mean) / (std + 1e-5) over valid frames; padding frames become 0.</summary>
    private static float[] Normalize(float[] logMel, int columns, int validFrames)
    {
        var features = new float[MelBins * columns]; // [MelBins, columns]
        if (validFrames == 0)
            return features;

        for (var m = 0; m < MelBins; m++)
        {
            double sum = 0;
            for (var f = 0; f < validFrames; f++)
                sum += logMel[f * MelBins + m];
            var mean = (float)(sum / validFrames);

            double squares = 0;
            for (var f = 0; f < validFrames; f++)
            {
                var d = logMel[f * MelBins + m] - mean;
                squares += d * d;
            }
            var variance = (float)(squares / (validFrames - 1)); // NaN for one frame, as in NumPy
            var scale = (float)(Math.Sqrt(variance) + NormalizationEpsilon);

            var row = features.AsSpan(m * columns, columns);
            for (var f = 0; f < validFrames; f++)
                row[f] = (logMel[f * MelBins + m] - mean) / scale;
        }

        return features;
    }

    /// <summary>Symmetric Hann window of 400 (<c>np.hanning</c>) centred in 512 with zeros.</summary>
    private static double[] CreateWindow()
    {
        var window = new double[FftSize];
        var offset = (FftSize - WindowLength) / 2;
        for (var n = 0; n < WindowLength; n++)
            window[offset + n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / (WindowLength - 1));
        return window;
    }

    /// <summary>Loads the [SpectrumBins, MelBins] filter bank, transposed, with each bin's nonzero span.</summary>
    private static (float[] Filters, int[] Start, int[] End) LoadMelFilters()
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("nemo128_fbanks.bin")
            ?? throw new InvalidOperationException("Embedded resource nemo128_fbanks.bin is missing.");
        var bytes = new byte[SpectrumBins * MelBins * sizeof(float)];
        stream.ReadExactly(bytes);
        var stored = new float[SpectrumBins * MelBins];
        Buffer.BlockCopy(bytes, 0, stored, 0, bytes.Length);

        var filters = new float[MelBins * SpectrumBins];
        var start = new int[MelBins];
        var end = new int[MelBins];
        for (var m = 0; m < MelBins; m++)
        {
            start[m] = SpectrumBins;
            for (var k = 0; k < SpectrumBins; k++)
            {
                var weight = stored[k * MelBins + m];
                filters[m * SpectrumBins + k] = weight;
                if (weight != 0)
                {
                    start[m] = Math.Min(start[m], k);
                    end[m] = k + 1;
                }
            }
        }
        return (filters, start, end);
    }

    /// <summary>In-place iterative radix-2 complex FFT.</summary>
    private sealed class Fft
    {
        private readonly int _size;
        private readonly int[] _bitReverse;
        private readonly double[] _cos;
        private readonly double[] _sin;

        public Fft(int size)
        {
            _size = size;
            var bits = (int)Math.Log2(size);
            _bitReverse = new int[size];
            for (var i = 0; i < size; i++)
            {
                var r = 0;
                for (var b = 0; b < bits; b++)
                    r |= ((i >> b) & 1) << (bits - 1 - b);
                _bitReverse[i] = r;
            }

            _cos = new double[size / 2];
            _sin = new double[size / 2];
            for (var i = 0; i < size / 2; i++)
            {
                _cos[i] = Math.Cos(-2 * Math.PI * i / size);
                _sin[i] = Math.Sin(-2 * Math.PI * i / size);
            }
        }

        public void Transform(double[] real, double[] imag)
        {
            for (var i = 0; i < _size; i++)
            {
                var j = _bitReverse[i];
                if (j > i)
                {
                    (real[i], real[j]) = (real[j], real[i]);
                    (imag[i], imag[j]) = (imag[j], imag[i]);
                }
            }

            for (var length = 2; length <= _size; length <<= 1)
            {
                var half = length / 2;
                var step = _size / length;
                for (var start = 0; start < _size; start += length)
                {
                    for (var k = 0; k < half; k++)
                    {
                        var wr = _cos[k * step];
                        var wi = _sin[k * step];
                        var a = start + k;
                        var b = a + half;
                        var tr = real[b] * wr - imag[b] * wi;
                        var ti = real[b] * wi + imag[b] * wr;
                        real[b] = real[a] - tr;
                        imag[b] = imag[a] - ti;
                        real[a] += tr;
                        imag[a] += ti;
                    }
                }
            }
        }
    }
}
