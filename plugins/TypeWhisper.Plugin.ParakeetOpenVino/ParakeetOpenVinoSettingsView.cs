using System.Windows;
using System.Windows.Controls;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

public sealed class ParakeetOpenVinoSettingsView : UserControl
{
    private readonly ParakeetOpenVinoPlugin _plugin;
    private readonly TextBlock _deviceStatus;
    private readonly ComboBox _deviceCombo;
    private readonly TextBox _pythonPathBox;
    private readonly TextBox _ovCacheBox;
    private bool _isRefreshing;

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

        var accelerationHint = new TextBlock
        {
            Text =
                "Primary device selection: Engine → Local model acceleration in TypeWhisper.\n" +
                "• Auto → device chosen below (default: Intel GPU)\n" +
                "• CPU → OpenVINO CPU\n" +
                "• NVIDIA CUDA / AMD Vulkan / ROCm → not supported; the device chosen below is used instead.\n" +
                "NPU transcribes only the first 20 s of each recording; use GPU for longer dictation.",
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 0, 8),
        };

        _deviceStatus = new TextBlock
        {
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 0, 8),
            FontWeight = FontWeights.SemiBold,
        };

        var advancedLabel = new TextBlock
        {
            Text = "Intel device:",
            Margin = new Thickness(0, 0, 0, 4),
        };

        _deviceCombo = new ComboBox
        {
            MinWidth = 200,
            Margin = new Thickness(0, 0, 0, 12),
            ItemsSource = ParakeetOpenVinoPlugin.DeviceOptions,
            SelectedItem = plugin.PreferredDevice,
        };
        _deviceCombo.SelectionChanged += DeviceCombo_OnSelectionChanged;

        var pythonLabel = new TextBlock
        {
            Text = "Python executable:",
            Margin = new Thickness(0, 0, 0, 4),
        };

        _pythonPathBox = new TextBox
        {
            Text = plugin.PythonExecutablePath,
            Margin = new Thickness(0, 0, 0, 8),
        };
        _pythonPathBox.LostFocus += (_, _) =>
            _plugin.PythonExecutablePath = _pythonPathBox.Text.Trim();

        var cacheLabel = new TextBlock
        {
            Text = "OpenVINO cache (optional):",
            Margin = new Thickness(0, 0, 0, 4),
        };

        _ovCacheBox = new TextBox
        {
            Text = plugin.OpenVinoCacheDirectory,
            Margin = new Thickness(0, 0, 0, 8),
        };
        _ovCacheBox.LostFocus += (_, _) =>
            _plugin.OpenVinoCacheDirectory = _ovCacheBox.Text.Trim();

        Loaded += (_, _) => RefreshFromPlugin();
        IsVisibleChanged += (_, _) =>
        {
            if (IsVisible)
                RefreshFromPlugin();
        };

        Content = new StackPanel
        {
            Margin = new Thickness(12),
            Children =
            {
                title,
                accelerationHint,
                _deviceStatus,
                advancedLabel,
                _deviceCombo,
                pythonLabel,
                _pythonPathBox,
                cacheLabel,
                _ovCacheBox,
            },
        };
    }

    private void DeviceCombo_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_isRefreshing)
            return;

        if (_deviceCombo.SelectedItem is string device)
            _plugin.PreferredDevice = device;
    }

    private void RefreshFromPlugin()
    {
        _isRefreshing = true;
        try
        {
            var status = _plugin.AccelerationStatus;
            _deviceStatus.Text = string.IsNullOrWhiteSpace(status.Detail)
                ? status.DisplayText
                : $"{status.DisplayText}: {status.Detail}";
            _deviceCombo.SelectedItem = _plugin.PreferredDevice;
            _pythonPathBox.Text = _plugin.PythonExecutablePath;
            _ovCacheBox.Text = _plugin.OpenVinoCacheDirectory;
        }
        finally
        {
            _isRefreshing = false;
        }
    }
}
