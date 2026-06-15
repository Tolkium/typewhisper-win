using System.Windows;
using System.Windows.Controls;

namespace TypeWhisper.Plugin.ParakeetIntelOnnx;

public sealed class ParakeetIntelOnnxSettingsView : UserControl
{
    private readonly ParakeetIntelOnnxPlugin _plugin;
    private readonly TextBox _pythonPathBox;

    public ParakeetIntelOnnxSettingsView(ParakeetIntelOnnxPlugin plugin)
    {
        _plugin = plugin;
        MinWidth = 360;

        var title = new TextBlock
        {
            Text = "Parakeet Intel iGPU (ONNX DirectML)",
            FontWeight = FontWeights.SemiBold,
            FontSize = 14,
            Margin = new Thickness(0, 0, 0, 8),
        };

        var accelerationHint = new TextBlock
        {
            Text =
                "Use Engine → Local model acceleration in TypeWhisper settings:\n" +
                "• Auto — Intel iGPU (DirectML) when available, else CPU\n" +
                "• CPU — ONNX Runtime on CPU only\n" +
                "• NVIDIA CUDA / AMD Vulkan / ROCm — mapped to Intel iGPU (DirectML) on this engine\n\n" +
                "Spoken language and target-language translation are handled by TypeWhisper host settings.",
            TextWrapping = TextWrapping.Wrap,
            Margin = new Thickness(0, 0, 0, 12),
            Opacity = 0.9,
        };

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
            plugin.PythonExecutablePath = _pythonPathBox.Text.Trim();

        Loaded += (_, _) => _pythonPathBox.Text = plugin.PythonExecutablePath;
        IsVisibleChanged += (_, _) =>
        {
            if (IsVisible)
                _pythonPathBox.Text = plugin.PythonExecutablePath;
        };

        Content = new StackPanel
        {
            Margin = new Thickness(12),
            Children =
            {
                title,
                accelerationHint,
                pythonLabel,
                _pythonPathBox,
            },
        };
    }
}
