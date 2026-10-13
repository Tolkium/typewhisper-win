using System.IO;
using System.IO.Compression;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace TypeWhisper.Plugin.ParakeetOpenVino.Engine;

/// <summary>
/// Downloads and verifies everything the engine needs: the Parakeet ONNX model from Hugging Face
/// and the OpenVINO runtime DLLs (taken from Intel's official PyPI wheel, which is a zip).
/// Files are downloaded to "*.part" first, resumed after interruptions, and checked by size and SHA-256.
/// </summary>
internal sealed class ParakeetSetup(string dataDirectory)
{
    public const string OpenVinoVersion = "2026.2.0";

    private const string ModelRevision = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce";
    private const string ModelBaseUrl =
        $"https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/{ModelRevision}/";

    private const string RuntimeWheelUrl =
        "https://files.pythonhosted.org/packages/63/4f/80400a0b506acf913ceeb1dccd7b7fd4424828feeea7d958f2d1577b88da/" +
        "openvino-2026.2.0-21903-cp312-cp312-win_amd64.whl";
    private const long RuntimeWheelSize = 76_100_000; // approximate, only used for progress weighting
    private const string RuntimeWheelSha256 = "fd9ba2eae04f76ea7266a461aa35b4e8bfd5408d732de96e88060ba9700792e0";

    private sealed record Asset(string Name, long Size, string? Sha256);

    private static readonly Asset[] ModelFiles =
    [
        new("config.json", 97, null),
        new("vocab.txt", 93_939, null),
        new("decoder_joint-model.onnx", 72_520_893, "e978ddf6688527182c10fde2eb4b83068421648985ef23f7a86be732be8706c1"),
        new("encoder-model.onnx", 41_770_866, "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1"),
        new("encoder-model.onnx.data", 2_435_420_160, "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36"),
    ];

    /// <summary>Runtime files needed for CPU/GPU/NPU and ONNX import; other frontends are skipped.</summary>
    private static readonly string[] RuntimeFiles =
    [
        "cache.json", "openvino.dll", "openvino_c.dll", "openvino_auto_plugin.dll",
        "openvino_intel_cpu_plugin.dll", "openvino_intel_gpu_plugin.dll",
        "openvino_intel_npu_plugin.dll", "openvino_intel_npu_compiler.dll", "openvino_intel_npu_compiler_loader.dll",
        "openvino_ir_frontend.dll", "openvino_onnx_frontend.dll",
        "tbb12.dll", "tbbbind_2_5.dll", "tbbmalloc.dll",
    ];

    private static readonly HttpClient Http = CreateHttpClient();

    public string WeightsDirectory => Path.Combine(dataDirectory, "weights");

    public string RuntimeDirectory => Path.Combine(dataDirectory, $"openvino-{OpenVinoVersion}");

    public string CacheDirectory => Path.Combine(dataDirectory, "ovcache");

    private string RuntimeMarker => Path.Combine(RuntimeDirectory, ".ready");

    public bool IsModelReady =>
        ModelFiles.All(f => FileHasSize(Path.Combine(WeightsDirectory, f.Name), f.Size));

    public bool IsRuntimeReady =>
        File.Exists(RuntimeMarker) && RuntimeFiles.All(f => File.Exists(Path.Combine(RuntimeDirectory, f)));

    public bool IsReady => IsModelReady && IsRuntimeReady;

    /// <summary>Downloads whatever is missing. Progress is reported from 0 to 1 by bytes.</summary>
    public async Task EnsureAsync(IProgress<double>? progress, Action<string> log, CancellationToken ct)
    {
        Directory.CreateDirectory(WeightsDirectory);
        var missingModel = ModelFiles.Where(f => !FileHasSize(Path.Combine(WeightsDirectory, f.Name), f.Size)).ToList();
        var needRuntime = !IsRuntimeReady;
        long total = missingModel.Sum(f => f.Size) + (needRuntime ? RuntimeWheelSize : 0);
        long done = 0;
        void Report(long bytes) => progress?.Report(total == 0 ? 1 : Math.Min(1, (done + bytes) / (double)total));

        if (needRuntime)
        {
            log($"Downloading OpenVINO {OpenVinoVersion} runtime");
            var wheel = Path.Combine(dataDirectory, "openvino-runtime.whl");
            await DownloadAsync(RuntimeWheelUrl, wheel, null, RuntimeWheelSha256, Report, ct);
            done += RuntimeWheelSize;
            ExtractRuntime(wheel);
            File.Delete(wheel);
            log($"OpenVINO runtime ready in {RuntimeDirectory}");
        }

        foreach (var file in missingModel)
        {
            log($"Downloading {file.Name} ({file.Size / 1048576.0:F0} MB)");
            await DownloadAsync(ModelBaseUrl + file.Name, Path.Combine(WeightsDirectory, file.Name), file.Size, file.Sha256, Report, ct);
            done += file.Size;
        }

        progress?.Report(1);
    }

    private void ExtractRuntime(string wheelPath)
    {
        Directory.CreateDirectory(RuntimeDirectory);
        File.Delete(RuntimeMarker);
        using var archive = ZipFile.OpenRead(wheelPath);
        foreach (var name in RuntimeFiles)
        {
            var entry = archive.GetEntry($"openvino/libs/{name}")
                ?? throw new InvalidDataException($"OpenVINO wheel has no openvino/libs/{name}");
            entry.ExtractToFile(Path.Combine(RuntimeDirectory, name), overwrite: true);
        }
        File.WriteAllText(RuntimeMarker, OpenVinoVersion);
    }

    private static async Task DownloadAsync(
        string url, string destination, long? expectedSize, string? sha256, Action<long> report, CancellationToken ct)
    {
        var part = destination + ".part";
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                await DownloadToPartAsync(url, part, report, ct);
                break;
            }
            catch (Exception ex) when (attempt < 4 && ex is HttpRequestException or IOException && !ct.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromSeconds(2 * attempt), ct); // resume from the .part file
            }
        }

        var size = new FileInfo(part).Length;
        if (expectedSize is { } expected && size != expected)
        {
            File.Delete(part);
            throw new InvalidDataException($"{Path.GetFileName(destination)}: got {size} bytes, expected {expected}.");
        }

        if (sha256 is not null)
        {
            var actual = await HashAsync(part, ct);
            if (!string.Equals(actual, sha256, StringComparison.OrdinalIgnoreCase))
            {
                File.Delete(part);
                throw new InvalidDataException($"{Path.GetFileName(destination)}: SHA-256 mismatch, download deleted.");
            }
        }

        File.Move(part, destination, overwrite: true);
    }

    private static async Task DownloadToPartAsync(string url, string part, Action<long> report, CancellationToken ct)
    {
        var existing = File.Exists(part) ? new FileInfo(part).Length : 0;
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (existing > 0)
            request.Headers.Range = new RangeHeaderValue(existing, null);

        using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
            return; // already complete
        response.EnsureSuccessStatusCode();

        var append = existing > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        if (!append)
            existing = 0;

        await using var source = await response.Content.ReadAsStreamAsync(ct);
        await using var target = new FileStream(part, append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true);
        var buffer = new byte[1 << 20];
        var written = existing;
        int read;
        while ((read = await source.ReadAsync(buffer, ct)) > 0)
        {
            await target.WriteAsync(buffer.AsMemory(0, read), ct);
            written += read;
            report(written);
        }
    }

    private static async Task<string> HashAsync(string path, CancellationToken ct)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
        var hash = await SHA256.HashDataAsync(stream, ct);
        return Convert.ToHexStringLower(hash);
    }

    private static bool FileHasSize(string path, long size) => File.Exists(path) && new FileInfo(path).Length == size;

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("TypeWhisper-ParakeetOpenVino");
        return client;
    }
}
