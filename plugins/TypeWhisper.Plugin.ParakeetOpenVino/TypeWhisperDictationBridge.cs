using System.Reflection;
using System.Windows;
using System.Windows.Input;

namespace TypeWhisper.Plugin.ParakeetOpenVino;

/// <summary>
/// Stops the host's dictation and shows a message in the host's own feedback area.
/// <para>
/// The plugin SDK has no API for this, so the bridge locates TypeWhisper's dictation view model at runtime
/// (the object exposing <c>StopRecordingCommand</c>, found through the app's window data contexts) and uses
/// its members by name. Verified against TypeWhisper 0.8.3 and 1.0.1. Every step is optional: when a member
/// is missing (a future host version), <see cref="TryStopWithMessage"/> returns false and the caller falls
/// back to the plugin's own banner.
/// </para>
/// </summary>
internal static class TypeWhisperDictationBridge
{
    private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    private const int FeedbackMilliseconds = 6000;

    private static WeakReference<object>? _dictation;

    /// <summary>
    /// On the UI thread: shows <paramref name="message"/> as a red host notification, then stops dictation
    /// like the stop shortcut. Returns false when the host's dictation controller cannot be found.
    /// </summary>
    public static bool TryStopWithMessage(string message)
    {
        try
        {
            var dictation = FindDictation();
            if (dictation is null || GetProperty<ICommand>(dictation, "StopRecordingCommand") is not { } stop)
                return false;

            // Message first, so a "No speech"/"Too short" message from the stop itself still wins.
            ShowFeedback(dictation, message);

            // Hosts before 1.0 do not reset the shortcut state when stopped this way; resetting it twice is harmless.
            var hotkey = GetField(dictation, "_hotkey");
            hotkey?.GetType().GetMethod("ForceStop", Type.EmptyTypes)?.Invoke(hotkey, null);

            if (!stop.CanExecute(null))
                return false;
            stop.Execute(null);
            return true;
        }
        catch (Exception ex) when (ex is TargetInvocationException or MemberAccessException or InvalidCastException)
        {
            System.Diagnostics.Debug.WriteLine($"[ParakeetOV] Host dictation bridge failed: {ex}");
            return false;
        }
    }

    private static void ShowFeedback(object dictation, string message)
    {
        SetProperty(dictation, "FeedbackText", message);
        SetProperty(dictation, "FeedbackIsError", true);
        SetField(dictation, "_feedbackAutoHideMilliseconds", FeedbackMilliseconds);
        SetProperty(dictation, "ShowFeedback", false); // re-trigger the host's show/auto-hide
        SetProperty(dictation, "ShowFeedback", true);
    }

    /// <summary>The object with <c>StopRecordingCommand</c>: a window's data context or one of its fields.</summary>
    private static object? FindDictation()
    {
        if (_dictation?.TryGetTarget(out var cached) == true)
            return cached;

        foreach (Window window in Application.Current?.Windows ?? new WindowCollection())
        {
            if (window.DataContext is not { } context)
                continue;

            var found = HasStopCommand(context)
                ? context
                : context.GetType().GetFields(Instance)
                    .Select(field => field.GetValue(context))
                    .FirstOrDefault(value => value is not null && HasStopCommand(value));
            if (found is not null)
            {
                _dictation = new WeakReference<object>(found);
                return found;
            }
        }

        return null;
    }

    private static bool HasStopCommand(object target) =>
        target.GetType().GetProperty("StopRecordingCommand", Instance)?.PropertyType is { } type
        && typeof(ICommand).IsAssignableFrom(type);

    private static T? GetProperty<T>(object target, string name) where T : class =>
        target.GetType().GetProperty(name, Instance)?.GetValue(target) as T;

    private static void SetProperty(object target, string name, object value)
    {
        var property = target.GetType().GetProperty(name, Instance);
        if (property?.CanWrite == true && property.PropertyType.IsInstanceOfType(value))
            property.SetValue(target, value);
    }

    private static object? GetField(object target, string name) =>
        target.GetType().GetField(name, Instance)?.GetValue(target);

    private static void SetField(object target, string name, object value)
    {
        var field = target.GetType().GetField(name, Instance);
        if (field is not null && field.FieldType.IsInstanceOfType(value))
            field.SetValue(target, value);
    }
}
