using System.IO;
using System.Text;
using System.Text.Json;

namespace TypeWhisper.Windows.Services;

/// <summary>
/// Opt-in diagnostic trace that records raw ASR decodes and merge results to a JSONL file.
/// Enabled only when the TYPEWHISPER_TRACE environment variable is set to 1/true, so it is
/// zero-cost in normal use. Lets a long dictation be replayed offline to locate where
/// duplication or text loss is introduced (raw decode vs live merge vs final stitch).
/// </summary>
internal static class TranscriptTraceLog
{
    private static readonly Lazy<string?> FilePath = new(ResolveFilePath);
    private static readonly Lock WriteLock = new();

    /// <summary>
    /// Gets whether tracing is enabled for this process.
    /// </summary>
    public static bool IsEnabled => FilePath.Value is not null;

    /// <summary>
    /// Appends a single trace record. No-op unless tracing is enabled.
    /// </summary>
    public static void Write(string stage, object payload)
    {
        var path = FilePath.Value;
        if (path is null)
            return;

        try
        {
            var record = new
            {
                ts = DateTime.UtcNow.ToString("O"),
                stage,
                payload,
            };
            var line = JsonSerializer.Serialize(record) + Environment.NewLine;

            lock (WriteLock)
                File.AppendAllText(path, line, Encoding.UTF8);
        }
        catch
        {
            // Diagnostics must never disrupt transcription.
        }
    }

    private static string? ResolveFilePath()
    {
        var flag = Environment.GetEnvironmentVariable("TYPEWHISPER_TRACE");
        if (string.IsNullOrWhiteSpace(flag))
            return null;

        if (!flag.Equals("1", StringComparison.Ordinal)
            && !flag.Equals("true", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        var custom = Environment.GetEnvironmentVariable("TYPEWHISPER_TRACE_PATH");
        var path = !string.IsNullOrWhiteSpace(custom)
            ? custom
            : Path.Combine(Path.GetTempPath(), "typewhisper-transcript-trace.jsonl");

        try
        {
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);
        }
        catch
        {
            return null;
        }

        return path;
    }
}
