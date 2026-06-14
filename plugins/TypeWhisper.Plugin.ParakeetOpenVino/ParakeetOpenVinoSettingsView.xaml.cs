using System.Windows;
using System.Windows.Controls;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

public sealed class ParakeetOpenVinoSettingsView : UserControl
{
    private readonly ParakeetOpenVinoPlugin _plugin;
    private readonly ComboBox _deviceCombo;
    private readonly TextBox _pythonPathBox;
    private readonly TextBox _ovCacheBox;

    public ParakeetOpenVinoSettingsView(ParakeetOpenVinoPlugin plugin)
    {
        _plugin = plugin;
        MinWidth = 360;

        var title = new TextBlock
        {
            Text = "Parakeet OpenVINO",
            FontWeight = FontWeights.SemiBold,
            FontSize = 14,
            Margin = new Thickness(0, 0, 0, 8),
        };

        var deviceHint = new TextBlock
        {
            Text = "Zariadenie pre inferenciu (OpenVINO / ORT).",
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 0, 8),
        };

        _deviceCombo = new ComboBox
        {
            MinWidth = 200,
            Margin = new Thickness(0, 0, 0, 12),
            ItemsSource = ParakeetOpenVinoPlugin.DeviceOptions,
            SelectedItem = plugin.TargetDevice,
        };
        _deviceCombo.SelectionChanged += DeviceCombo_OnSelectionChanged;

        var pythonLabel = new TextBlock
        {
            Text = "Python (WhisperNPU venv):",
            Margin = new Thickness(0, 0, 0, 4),
        };

        _pythonPathBox = new TextBox
        {
            Text = plugin.PythonExecutablePath,
            Margin = new Thickness(0, 0, 0, 8),
        };
        _pythonPathBox.LostFocus += PythonPathBox_OnLostFocus;

        var cacheLabel = new TextBlock
        {
            Text = "OV cache (volitelne):",
            Margin = new Thickness(0, 0, 0, 4),
        };

        _ovCacheBox = new TextBox
        {
            Text = plugin.OpenVinoCacheDirectory,
            Margin = new Thickness(0, 0, 0, 8),
        };
        _ovCacheBox.LostFocus += OvCacheBox_OnLostFocus;

        var hintText = new TextBlock
        {
            Text =
                "NPU: encoder na NPU, decoder na CPU (stabilnejsie). Prvy load ~1 min (kompilacia bucketov). " +
                "Po zmene zariadenia znovu nacitaj model. Live preview: polling kazde 3 s pocas nahravania.",
            TextWrapping = TextWrapping.Wrap,
            Opacity = 0.75,
            FontSize = 12,
        };

        Content = new StackPanel
        {
            Margin = new Thickness(12),
            Children =
            {
                title,
                deviceHint,
                _deviceCombo,
                pythonLabel,
                _pythonPathBox,
                cacheLabel,
                _ovCacheBox,
                hintText,
            },
        };
    }

    private void DeviceCombo_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_deviceCombo.SelectedItem is string device)
            _plugin.TargetDevice = device;
    }

    private void PythonPathBox_OnLostFocus(object sender, RoutedEventArgs e) =>
        _plugin.PythonExecutablePath = _pythonPathBox.Text.Trim();

    private void OvCacheBox_OnLostFocus(object sender, RoutedEventArgs e) =>
        _plugin.OpenVinoCacheDirectory = _ovCacheBox.Text.Trim();
}
