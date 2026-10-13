using System.Buffers.Binary;
using System.IO;

namespace TypeWhisper.Plugin.ParakeetOpenVino.Engine;

/// <summary>Reads the 16 kHz mono WAV the host sends into float samples (int16 / 32768, as onnx-asr).</summary>
internal static class WavReader
{
    public static float[] ReadMono16k(ReadOnlySpan<byte> wav)
    {
        if (wav.Length < 12 || !wav[..4].SequenceEqual("RIFF"u8) || !wav.Slice(8, 4).SequenceEqual("WAVE"u8))
            throw new InvalidDataException("Audio is not a RIFF/WAVE file.");

        int format = 0, channels = 0, sampleRate = 0, bits = 0;
        var offset = 12;
        while (offset + 8 <= wav.Length)
        {
            var id = wav.Slice(offset, 4);
            var size = BinaryPrimitives.ReadUInt32LittleEndian(wav.Slice(offset + 4, 4));
            var body = offset + 8;
            var available = wav.Length - body;
            if (id.SequenceEqual("fmt "u8))
            {
                if (size < 16 || available < 16)
                    throw new InvalidDataException("WAV fmt chunk is truncated.");
                format = BinaryPrimitives.ReadUInt16LittleEndian(wav.Slice(body, 2));
                channels = BinaryPrimitives.ReadUInt16LittleEndian(wav.Slice(body + 2, 2));
                sampleRate = BinaryPrimitives.ReadInt32LittleEndian(wav.Slice(body + 4, 4));
                bits = BinaryPrimitives.ReadUInt16LittleEndian(wav.Slice(body + 14, 2));
            }
            else if (id.SequenceEqual("data"u8))
            {
                if (channels != 1 || sampleRate != NemoFeatureExtractor.SampleRate)
                    throw new InvalidDataException($"Expected 16 kHz mono WAV, got {sampleRate} Hz, {channels} channels.");

                // Streaming writers may leave the size as 0xFFFFFFFF; read what is there.
                var data = wav.Slice(body, (int)Math.Min(size, (uint)available));
                return (format, bits) switch
                {
                    (1, 16) => FromInt16(data),
                    (3, 32) => FromFloat32(data),
                    _ => throw new InvalidDataException($"Unsupported WAV format {format}, {bits} bits."),
                };
            }

            var next = (long)body + size + (size & 1);
            if (next > wav.Length)
                break;
            offset = (int)next;
        }

        throw new InvalidDataException("WAV has no data chunk.");
    }

    private static float[] FromInt16(ReadOnlySpan<byte> data)
    {
        var samples = new float[data.Length / 2];
        for (var i = 0; i < samples.Length; i++)
            samples[i] = BinaryPrimitives.ReadInt16LittleEndian(data.Slice(i * 2, 2)) / 32768f;
        return samples;
    }

    private static float[] FromFloat32(ReadOnlySpan<byte> data)
    {
        var samples = new float[data.Length / 4];
        for (var i = 0; i < samples.Length; i++)
            samples[i] = BinaryPrimitives.ReadSingleLittleEndian(data.Slice(i * 4, 4));
        return samples;
    }
}
