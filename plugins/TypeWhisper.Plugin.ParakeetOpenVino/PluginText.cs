using System.IO;
using System.Reflection;
using System.Text.Json;
using TypeWhisper.PluginSDK;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// User-facing text from <c>Localization/*.json</c>. Uses the host's localization (app language, English
/// fallback); without a host (unit tests) it reads the bundled <c>en.json</c>, so English lives in one place.
/// </summary>
internal sealed class PluginText(IPluginLocalization? localization)
{
    private static readonly Lazy<IReadOnlyDictionary<string, string>> BundledEnglish = new(LoadBundledEnglish);

    public string Get(string key, params object[] args)
    {
        var text = localization?.GetString(key) is { } hostText && hostText != key
            ? hostText
            : BundledEnglish.Value.GetValueOrDefault(key, key);
        return args.Length == 0 ? text : string.Format(text, args);
    }

    private static IReadOnlyDictionary<string, string> LoadBundledEnglish()
    {
        try
        {
            var directory = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? ".";
            var path = Path.Combine(directory, "Localization", "en.json");
            return File.Exists(path)
                ? JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(path)) ?? []
                : new Dictionary<string, string>();
        }
        catch (Exception ex) when (ex is IOException or JsonException)
        {
            return new Dictionary<string, string>();
        }
    }
}
