using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Threading;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Small red banner at the top of the screen. It never takes focus or clicks, because dictated text is
/// typed into the focused window; it closes itself after a few seconds.
/// </summary>
internal sealed class NpuLimitPopup : Window
{
    private static readonly TimeSpan DisplayDuration = TimeSpan.FromSeconds(6);
    private const double TopMargin = 24;

    private const int GwlExStyle = -20;
    private const int WsExTransparent = 0x00000020;
    private const int WsExToolWindow = 0x00000080;
    private const int WsExNoActivate = 0x08000000;

    private static NpuLimitPopup? _current;

    private NpuLimitPopup(string message)
    {
        WindowStyle = WindowStyle.None;
        AllowsTransparency = true;
        Background = Brushes.Transparent;
        ResizeMode = ResizeMode.NoResize;
        SizeToContent = SizeToContent.WidthAndHeight;
        ShowInTaskbar = false;
        ShowActivated = false;
        Topmost = true;
        Focusable = false;
        IsHitTestVisible = false;

        Content = new Border
        {
            Background = new SolidColorBrush(Color.FromRgb(0xC4, 0x2B, 0x1C)),
            CornerRadius = new CornerRadius(8),
            Padding = new Thickness(16, 10, 16, 10),
            MaxWidth = 520,
            Child = new TextBlock
            {
                Text = message,
                Foreground = Brushes.White,
                FontSize = 14,
                FontWeight = FontWeights.SemiBold,
                TextWrapping = TextWrapping.Wrap,
            },
        };

        SourceInitialized += (_, _) => MakeNonActivating();
        Loaded += (_, _) => PlaceAtTopCenter();
    }

    /// <summary>Shows the banner on the UI thread, replacing one that is still visible.</summary>
    public static void ShowMessage(string message)
    {
        var dispatcher = Application.Current?.Dispatcher;
        dispatcher?.InvokeAsync(() =>
        {
            _current?.Close();
            var popup = new NpuLimitPopup(message);
            _current = popup;
            popup.Closed += (_, _) =>
            {
                if (ReferenceEquals(_current, popup))
                    _current = null;
            };
            popup.Show();

            var timer = new DispatcherTimer { Interval = DisplayDuration };
            timer.Tick += (_, _) =>
            {
                timer.Stop();
                popup.Close();
            };
            timer.Start();
        });
    }

    private void PlaceAtTopCenter()
    {
        var area = SystemParameters.WorkArea;
        Left = area.Left + (area.Width - ActualWidth) / 2;
        Top = area.Top + TopMargin;
    }

    private void MakeNonActivating()
    {
        var handle = new WindowInteropHelper(this).Handle;
        var style = GetWindowLongPtr(handle, GwlExStyle);
        SetWindowLongPtr(handle, GwlExStyle, style | WsExNoActivate | WsExToolWindow | WsExTransparent);
    }

    [DllImport("user32.dll", EntryPoint = "GetWindowLongPtrW")]
    private static extern nint GetWindowLongPtr(nint hWnd, int index);

    [DllImport("user32.dll", EntryPoint = "SetWindowLongPtrW")]
    private static extern nint SetWindowLongPtr(nint hWnd, int index, nint value);
}
