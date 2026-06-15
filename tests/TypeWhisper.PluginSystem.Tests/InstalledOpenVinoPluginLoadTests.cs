using System.IO;
using System.Windows.Controls;
using TypeWhisper.Windows.Services.Plugins;

namespace TypeWhisper.PluginSystem.Tests;

public class InstalledOpenVinoPluginLoadTests
{
    [Fact]
    public void InstalledOpenVinoPlugin_LoadsAndCreatesSettingsView()
    {
        var pluginsRoot = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TypeWhisper",
            "Plugins");

        var loader = new PluginLoader();
        var loaded = loader.DiscoverAndLoad([pluginsRoot]);
        var plugin = loaded.FirstOrDefault(p =>
            string.Equals(p.Manifest.Id, "com.tolkium.parakeet-openvino", StringComparison.OrdinalIgnoreCase));

        Assert.NotNull(plugin);
        Assert.Equal("1.0.2", plugin.Manifest.Version);

        UserControl? view = null;
        Exception? error = null;
        var thread = new Thread(() =>
        {
            try
            {
                view = plugin.Instance.CreateSettingsView();
            }
            catch (Exception ex)
            {
                error = ex;
            }
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join(TimeSpan.FromSeconds(10));

        Assert.Null(error);
        Assert.NotNull(view);
    }
}
