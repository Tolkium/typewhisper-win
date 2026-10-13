namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Warns when a dictation runs past the engine's input window (NPU: 20 s). Stock TypeWhisper gives plugins
/// no way to stop a recording, so the plugin times the recording from the host's start/stop events and
/// shows a warning; the engine itself cuts the audio to the window.
/// </summary>
internal sealed class NpuLimitWarning(
    Func<TimeSpan?> activeLimit,
    Action showWarning,
    Func<TimeSpan, CancellationToken, Task>? delay = null) : IDisposable
{
    private readonly Func<TimeSpan, CancellationToken, Task> _delay = delay ?? Task.Delay;
    private readonly object _sync = new();
    private CancellationTokenSource? _pending;

    /// <summary>Starts timing a recording when the active engine has a limit.</summary>
    public void OnRecordingStarted()
    {
        var limit = activeLimit();
        CancellationToken token;
        lock (_sync)
        {
            CancelPending();
            if (limit is null)
                return;
            _pending = new CancellationTokenSource();
            token = _pending.Token;
        }

        _ = WarnAfterAsync(limit.Value, token);
    }

    public void OnRecordingStopped()
    {
        lock (_sync)
            CancelPending();
    }

    public void Dispose() => OnRecordingStopped();

    private async Task WarnAfterAsync(TimeSpan limit, CancellationToken token)
    {
        try
        {
            await _delay(limit, token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            return; // recording stopped in time
        }

        if (!token.IsCancellationRequested)
            showWarning();
    }

    private void CancelPending()
    {
        _pending?.Cancel();
        _pending?.Dispose();
        _pending = null;
    }
}
