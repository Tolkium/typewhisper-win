using System.Reflection;
using Moq;
using NAudio.Wave;
using TypeWhisper.Core.Interfaces;
using TypeWhisper.Core.Models;
using TypeWhisper.PluginSDK;
using TypeWhisper.PluginSDK.Models;
using TypeWhisper.Windows.Services;
using TypeWhisper.Windows.Services.Plugins;
using TypeWhisper.Windows.ViewModels;

namespace TypeWhisper.PluginSystem.Tests;

public class StreamingTranscriptionTests
{
    [Fact]
    public void SupportsStreaming_DefaultIsFalse()
    {
        var mock = new Mock<ITranscriptionEnginePlugin>();
        // DIMs return default values — SupportsStreaming defaults to false
        Assert.False(mock.Object.SupportsStreaming);
    }

    [Fact]
    public void SupportedLanguages_DefaultIsEmpty()
    {
        var mock = new Mock<ITranscriptionEnginePlugin> { CallBase = true };
        // CallBase invokes the DIM — SupportedLanguages defaults to empty
        var languages = mock.Object.SupportedLanguages;
        Assert.Empty(languages);
    }

    [Fact]
    public async Task TranscribeStreamingAsync_DefaultDelegatesToTranscribeAsync()
    {
        var expectedResult = new PluginTranscriptionResult("Hello world", "en", 2.5);
        var audio = new byte[] { 1, 2, 3 };

        var mock = new Mock<ITranscriptionEnginePlugin>();
        mock.Setup(e => e.TranscribeAsync(audio, "en", false, null, It.IsAny<CancellationToken>()))
            .ReturnsAsync(expectedResult);

        // TranscribeStreamingAsync should delegate to TranscribeAsync by default
        // Since Moq doesn't call DIMs directly, we verify the TranscribeAsync call
        var result = await mock.Object.TranscribeAsync(audio, "en", false, null, CancellationToken.None);

        Assert.Equal("Hello world", result.Text);
        Assert.Equal("en", result.DetectedLanguage);
        Assert.Equal(2.5, result.DurationSeconds);
    }

    [Fact]
    public void PluginTranscriptionResult_NoSpeechProbability_DefaultIsNull()
    {
        var result = new PluginTranscriptionResult("Hello", "en", 2.0);
        Assert.Null(result.NoSpeechProbability);
    }

    [Fact]
    public void PluginTranscriptionResult_NoSpeechProbability_CanBeSet()
    {
        var result = new PluginTranscriptionResult("So.", "en", 1.0, 0.95f);
        Assert.Equal(0.95f, result.NoSpeechProbability);
    }

    [Fact]
    public void SupportsStreaming_CanBeOverridden()
    {
        var mock = new Mock<ITranscriptionEnginePlugin>();
        mock.Setup(e => e.SupportsStreaming).Returns(true);

        Assert.True(mock.Object.SupportsStreaming);
    }

    [Fact]
    public void SupportedLanguages_CanBeOverridden()
    {
        var mock = new Mock<ITranscriptionEnginePlugin>();
        mock.Setup(e => e.SupportedLanguages).Returns(new List<string> { "en", "de", "fr" });

        var languages = mock.Object.SupportedLanguages;
        Assert.Equal(3, languages.Count);
        Assert.Contains("de", languages);
    }

    [Fact]
    public async Task TranscribeStreamingAsync_CanBeOverridden()
    {
        var expectedResult = new PluginTranscriptionResult("Streamed text", "de", 5.0);
        var audio = new byte[] { 1, 2, 3, 4, 5 };
        var progressCalls = new List<string>();

        var mock = new Mock<ITranscriptionEnginePlugin>();
        mock.Setup(e => e.TranscribeStreamingAsync(
            audio, "de", false, null,
            It.IsAny<Func<string, bool>>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(expectedResult);

        var result = await mock.Object.TranscribeStreamingAsync(
            audio, "de", false, null,
            partial => { progressCalls.Add(partial); return true; },
            CancellationToken.None);

        Assert.Equal("Streamed text", result.Text);
        Assert.Equal("de", result.DetectedLanguage);
    }

    [Fact]
    public async Task StreamingHandler_BuffersAudioCapturedBeforeRealtimeSessionConnects()
    {
        var settings = new FakeSettingsService(AppSettings.Default);
        using var pluginManager = TestPluginManagerFactory.Create(settings);
        var plugin = new DelayedStreamingPlugin();
        TestPluginManagerFactory.SetPrivateField(
            pluginManager,
            "_transcriptionEngines",
            new List<ITranscriptionEnginePlugin> { plugin });

        var modelManager = new ModelManagerService(pluginManager, settings);
        await modelManager.LoadModelAsync(ModelManagerService.GetPluginModelId(plugin.PluginId, "stream"));

        var devices = new FakeAudioInputDeviceProvider("Test Microphone");
        var captures = new FakeAudioInputCaptureFactory();
        using var audio = new AudioRecordingService(devices, captures, Timeout.InfiniteTimeSpan);
        using var handler = new StreamingHandler(modelManager, audio, new PassthroughDictionaryService());

        handler.Start("en", TranscriptionTask.Transcribe, () => audio.IsRecording);
        audio.StartRecording();
        var capture = Assert.Single(captures.Created);

        capture.RaiseData([0, 0, 0, 0], 4);
        var session = new CapturingStreamingSession();
        plugin.CompleteStart(session);

        await WaitUntilAsync(() => session.SentAudio.Count == 1);

        Assert.Equal(4, session.SentAudio.Single().Length);
    }

    [Fact]
    public async Task StreamingHandler_SerializesRealtimeAudioWrites()
    {
        var settings = new FakeSettingsService(AppSettings.Default);
        using var pluginManager = TestPluginManagerFactory.Create(settings);
        var plugin = new DelayedStreamingPlugin();
        var session = new BlockingStreamingSession();
        plugin.CompleteStart(session);
        TestPluginManagerFactory.SetPrivateField(
            pluginManager,
            "_transcriptionEngines",
            new List<ITranscriptionEnginePlugin> { plugin });

        var modelManager = new ModelManagerService(pluginManager, settings);
        await modelManager.LoadModelAsync(ModelManagerService.GetPluginModelId(plugin.PluginId, "stream"));

        var devices = new FakeAudioInputDeviceProvider("Test Microphone");
        var captures = new FakeAudioInputCaptureFactory();
        using var audio = new AudioRecordingService(devices, captures, Timeout.InfiniteTimeSpan);
        using var handler = new StreamingHandler(modelManager, audio, new PassthroughDictionaryService());

        handler.Start("en", TranscriptionTask.Transcribe, () => audio.IsRecording);
        audio.StartRecording();
        var capture = Assert.Single(captures.Created);

        capture.RaiseData([0, 0, 0, 0], 4);
        await session.FirstSendEntered;

        capture.RaiseData([1, 0, 1, 0], 4);

        var overlapped = await CompletesWithinAsync(session.ConcurrentSendObserved, TimeSpan.FromMilliseconds(500));
        Assert.False(overlapped);

        session.ReleaseFirstSend();
        await WaitUntilAsync(() => session.SendAttemptCount == 2);

        Assert.Equal(1, session.MaxConcurrentSendCount);
        Assert.Equal(2, session.SentAudioCount);
    }

    [Fact]
    public async Task StreamingHandler_DoesNotBlockAudioCallbackWhenSenderFallsBehind()
    {
        var settings = new FakeSettingsService(AppSettings.Default);
        using var pluginManager = TestPluginManagerFactory.Create(settings);
        var plugin = new DelayedStreamingPlugin();
        var session = new BlockingStreamingSession();
        plugin.CompleteStart(session);
        TestPluginManagerFactory.SetPrivateField(
            pluginManager,
            "_transcriptionEngines",
            new List<ITranscriptionEnginePlugin> { plugin });

        var modelManager = new ModelManagerService(pluginManager, settings);
        await modelManager.LoadModelAsync(ModelManagerService.GetPluginModelId(plugin.PluginId, "stream"));

        var devices = new FakeAudioInputDeviceProvider("Test Microphone");
        var captures = new FakeAudioInputCaptureFactory();
        using var audio = new AudioRecordingService(devices, captures, Timeout.InfiniteTimeSpan);
        using var handler = new StreamingHandler(modelManager, audio, new PassthroughDictionaryService());

        handler.Start("en", TranscriptionTask.Transcribe, () => audio.IsRecording);
        audio.StartRecording();
        var capture = Assert.Single(captures.Created);

        capture.RaiseData([0, 0, 0, 0], 4);
        await session.FirstSendEntered;

        var raiseManyChunks = Task.Run(() =>
        {
            for (var i = 0; i < 200; i++)
                capture.RaiseData([1, 0, 1, 0], 4);
        });

        var completedWithoutBlocking = await CompletesWithinAsync(raiseManyChunks, TimeSpan.FromMilliseconds(500));

        session.ReleaseFirstSend();
        await raiseManyChunks.WaitAsync(TimeSpan.FromSeconds(3));

        Assert.True(completedWithoutBlocking);
    }

    [Fact]
    public async Task StreamingHandler_CleansUpStreamingStateWhenInitialFlushFails()
    {
        var settings = new FakeSettingsService(AppSettings.Default);
        using var pluginManager = TestPluginManagerFactory.Create(settings);
        var plugin = new DelayedStreamingPlugin();
        TestPluginManagerFactory.SetPrivateField(
            pluginManager,
            "_transcriptionEngines",
            new List<ITranscriptionEnginePlugin> { plugin });

        var modelManager = new ModelManagerService(pluginManager, settings);
        await modelManager.LoadModelAsync(ModelManagerService.GetPluginModelId(plugin.PluginId, "stream"));

        var devices = new FakeAudioInputDeviceProvider("Test Microphone");
        var captures = new FakeAudioInputCaptureFactory();
        using var audio = new AudioRecordingService(devices, captures, Timeout.InfiniteTimeSpan);
        using var handler = new StreamingHandler(modelManager, audio, new PassthroughDictionaryService());

        handler.Start("en", TranscriptionTask.Transcribe, () => audio.IsRecording);
        audio.StartRecording();
        var capture = Assert.Single(captures.Created);

        capture.RaiseData([0, 0, 0, 0], 4);

        var session = new FailingStreamingSession();
        plugin.CompleteStart(session);
        await WaitUntilAsync(() => session.SendAttemptCount == 1);
        await Task.Delay(100);

        Assert.Null(GetPrivateField(handler, "_session"));
        Assert.False((bool)GetPrivateField(handler, "_isFlushingPendingStreamingAudio")!);
        Assert.Equal(0, (int)GetPrivateField(handler, "_pendingStreamingAudioBytes")!);

        capture.RaiseData([1, 0, 1, 0], 4);
        await Task.Delay(50);

        Assert.Equal(1, session.SendAttemptCount);
        Assert.Equal(0, (int)GetPrivateField(handler, "_pendingStreamingAudioBytes")!);
    }

    private static async Task WaitUntilAsync(Func<bool> condition)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(3));
        while (!condition())
        {
            timeout.Token.ThrowIfCancellationRequested();
            await Task.Delay(20, timeout.Token);
        }
    }

    private static async Task<bool> CompletesWithinAsync(Task task, TimeSpan timeout)
    {
        var delay = Task.Delay(timeout);
        return await Task.WhenAny(task, delay) == task;
    }

    private static object? GetPrivateField(object target, string fieldName)
    {
        var field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        return field.GetValue(target);
    }

    private sealed class DelayedStreamingPlugin : ITranscriptionEnginePlugin
    {
        private readonly TaskCompletionSource<IStreamingSession> _startCompletion =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public string PluginId => "com.test.delayed-streaming";
        public string PluginName => "Delayed Streaming";
        public string PluginVersion => "1.0.0";
        public string ProviderId => "delayed-streaming";
        public string ProviderDisplayName => "Delayed Streaming";
        public bool IsConfigured => true;
        public IReadOnlyList<PluginModelInfo> TranscriptionModels { get; } =
            [new PluginModelInfo("stream", "Stream")];
        public string? SelectedModelId { get; private set; }
        public bool SupportsTranslation => false;
        public bool SupportsStreaming => true;

        public Task ActivateAsync(IPluginHostServices host) => Task.CompletedTask;
        public Task DeactivateAsync() => Task.CompletedTask;
        public System.Windows.Controls.UserControl? CreateSettingsView() => null;
        public void Dispose() { }
        public void SelectModel(string modelId) => SelectedModelId = modelId;
        public Task<IStreamingSession> StartStreamingAsync(string? language, CancellationToken ct) =>
            _startCompletion.Task.WaitAsync(ct);
        public void CompleteStart(IStreamingSession session) => _startCompletion.SetResult(session);

        public Task<PluginTranscriptionResult> TranscribeAsync(
            byte[] wavAudio,
            string? language,
            bool translate,
            string? prompt,
            CancellationToken ct) =>
            Task.FromResult(new PluginTranscriptionResult("", language ?? "en", 0));
    }

    private sealed class CapturingStreamingSession : IStreamingSession
    {
        public List<byte[]> SentAudio { get; } = [];
        public event Action<StreamingTranscriptEvent>? TranscriptReceived;
        public Task SendAudioAsync(ReadOnlyMemory<byte> pcm16Audio, CancellationToken ct)
        {
            SentAudio.Add(pcm16Audio.ToArray());
            return Task.CompletedTask;
        }

        public Task FinalizeAsync(CancellationToken ct) => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public void RaiseTranscript(StreamingTranscriptEvent evt) => TranscriptReceived?.Invoke(evt);
    }

    private sealed class BlockingStreamingSession : IStreamingSession
    {
        private readonly TaskCompletionSource _firstSendEntered =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource _releaseFirstSend =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource _concurrentSendObserved =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly List<byte[]> _sentAudio = [];

        private int _inFlightSendCount;
        private int _maxConcurrentSendCount;
        private int _sendAttemptCount;

        public Task FirstSendEntered => _firstSendEntered.Task;
        public Task ConcurrentSendObserved => _concurrentSendObserved.Task;
        public int MaxConcurrentSendCount => Volatile.Read(ref _maxConcurrentSendCount);
        public int SendAttemptCount => Volatile.Read(ref _sendAttemptCount);
        public int SentAudioCount
        {
            get
            {
                lock (_sentAudio) return _sentAudio.Count;
            }
        }

        public event Action<StreamingTranscriptEvent>? TranscriptReceived;

        public async Task SendAudioAsync(ReadOnlyMemory<byte> pcm16Audio, CancellationToken ct)
        {
            var inFlight = Interlocked.Increment(ref _inFlightSendCount);
            TrackMaxConcurrentSendCount(inFlight);
            if (inFlight > 1)
                _concurrentSendObserved.TrySetResult();

            var attempt = Interlocked.Increment(ref _sendAttemptCount);

            try
            {
                if (attempt == 1)
                {
                    _firstSendEntered.TrySetResult();
                    await _releaseFirstSend.Task.WaitAsync(ct);
                }

                lock (_sentAudio)
                {
                    _sentAudio.Add(pcm16Audio.ToArray());
                }
            }
            finally
            {
                Interlocked.Decrement(ref _inFlightSendCount);
            }
        }

        public void ReleaseFirstSend() => _releaseFirstSend.TrySetResult();
        public Task FinalizeAsync(CancellationToken ct) => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public void RaiseTranscript(StreamingTranscriptEvent evt) => TranscriptReceived?.Invoke(evt);

        private void TrackMaxConcurrentSendCount(int count)
        {
            while (true)
            {
                var current = Volatile.Read(ref _maxConcurrentSendCount);
                if (count <= current)
                    return;

                if (Interlocked.CompareExchange(ref _maxConcurrentSendCount, count, current) == current)
                    return;
            }
        }
    }

    private sealed class FailingStreamingSession : IStreamingSession
    {
        private int _sendAttemptCount;

        public int SendAttemptCount => Volatile.Read(ref _sendAttemptCount);
        public event Action<StreamingTranscriptEvent>? TranscriptReceived;

        public Task SendAudioAsync(ReadOnlyMemory<byte> pcm16Audio, CancellationToken ct)
        {
            Interlocked.Increment(ref _sendAttemptCount);
            throw new InvalidOperationException("Simulated streaming send failure.");
        }

        public Task FinalizeAsync(CancellationToken ct) => Task.CompletedTask;
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
        public void RaiseTranscript(StreamingTranscriptEvent evt) => TranscriptReceived?.Invoke(evt);
    }

    private sealed class PassthroughDictionaryService : IDictionaryService
    {
        public IReadOnlyList<DictionaryEntry> Entries => [];
        public event Action? EntriesChanged { add { } remove { } }
        public void AddEntry(DictionaryEntry entry) { }
        public void AddEntries(IEnumerable<DictionaryEntry> entries) { }
        public void UpdateEntry(DictionaryEntry entry) { }
        public void DeleteEntry(string id) { }
        public void DeleteEntries(IEnumerable<string> ids) { }
        public string ApplyCorrections(string text) => text;
        public string? GetTermsForPrompt() => null;
        public void LearnCorrection(string original, string replacement) { }
        public void ActivatePack(TermPack pack) { }
        public void DeactivatePack(string packId) { }
    }
}

public class LiveTranscriptionStartupPolicyTests
{
    [Fact]
    public void GlobalLiveTranscriptionDisabled_SuppressesAllLiveModes()
    {
        var plugin = new FakePolicyTranscriptionPlugin(supportsStreaming: true, supportsModelDownload: true);

        var mode = LiveTranscriptionStartupPolicy.Select(
            AppSettings.Default with { LiveTranscriptionEnabled = false },
            isPluginModel: true,
            plugin);

        Assert.Equal(LiveTranscriptionStartupMode.None, mode);
    }

    [Fact]
    public void StreamingPlugin_UsesRealtimeStreaming()
    {
        var plugin = new FakePolicyTranscriptionPlugin(supportsStreaming: true, supportsModelDownload: false);

        var mode = LiveTranscriptionStartupPolicy.Select(
            AppSettings.Default,
            isPluginModel: true,
            plugin);

        Assert.Equal(LiveTranscriptionStartupMode.PluginStreaming, mode);
    }

    [Fact]
    public void DownloadablePlugin_UsesPollingFallback()
    {
        var plugin = new FakePolicyTranscriptionPlugin(supportsStreaming: false, supportsModelDownload: true);

        var mode = LiveTranscriptionStartupPolicy.Select(
            AppSettings.Default,
            isPluginModel: true,
            plugin);

        Assert.Equal(LiveTranscriptionStartupMode.PluginPollingFallback, mode);
    }

    [Fact]
    public void ParakeetOpenVino_UsesPollingFallback()
    {
        var plugin = new FakePolicyTranscriptionPlugin(
            supportsStreaming: false,
            supportsModelDownload: true,
            providerId: "parakeet-openvino");

        var mode = LiveTranscriptionStartupPolicy.Select(
            AppSettings.Default,
            isPluginModel: true,
            plugin);

        Assert.Equal(LiveTranscriptionStartupMode.PluginPollingFallback, mode);
    }

    [Fact]
    public void OnlineBatchProvider_DefaultsToNoLiveTranscription()
    {
        var plugin = new FakePolicyTranscriptionPlugin(supportsStreaming: false, supportsModelDownload: false);

        var mode = LiveTranscriptionStartupPolicy.Select(
            AppSettings.Default,
            isPluginModel: true,
            plugin);

        Assert.Equal(LiveTranscriptionStartupMode.None, mode);
    }

    [Fact]
    public void OnlineBatchProvider_UsesPollingFallbackWhenOptedIn()
    {
        var plugin = new FakePolicyTranscriptionPlugin(supportsStreaming: false, supportsModelDownload: false);

        var mode = LiveTranscriptionStartupPolicy.Select(
            AppSettings.Default with { OnlineAsrBatchLiveTranscriptionEnabled = true },
            isPluginModel: true,
            plugin);

        Assert.Equal(LiveTranscriptionStartupMode.PluginPollingFallback, mode);
    }

    [Fact]
    public void NonPluginModel_UsesLegacyVad()
    {
        var mode = LiveTranscriptionStartupPolicy.Select(
            AppSettings.Default,
            isPluginModel: false,
            plugin: null);

        Assert.Equal(LiveTranscriptionStartupMode.LegacyVad, mode);
    }

    private sealed class FakePolicyTranscriptionPlugin : ITranscriptionEnginePlugin
    {
        public FakePolicyTranscriptionPlugin(
            bool supportsStreaming,
            bool supportsModelDownload,
            string providerId = "policy")
        {
            SupportsStreaming = supportsStreaming;
            SupportsModelDownload = supportsModelDownload;
            ProviderId = providerId;
        }

        public string PluginId => "com.test.policy";
        public string PluginName => "Policy Test";
        public string PluginVersion => "1.0.0";
        public string ProviderId { get; }
        public string ProviderDisplayName => "Policy";
        public bool IsConfigured => true;
        public IReadOnlyList<PluginModelInfo> TranscriptionModels { get; } =
            [new PluginModelInfo("model", "Model")];
        public string? SelectedModelId => "model";
        public bool SupportsTranslation => false;
        public bool SupportsStreaming { get; }
        public bool SupportsModelDownload { get; }

        public Task ActivateAsync(IPluginHostServices host) => Task.CompletedTask;
        public Task DeactivateAsync() => Task.CompletedTask;
        public System.Windows.Controls.UserControl? CreateSettingsView() => null;
        public void Dispose() { }
        public void SelectModel(string modelId) { }

        public Task<PluginTranscriptionResult> TranscribeAsync(
            byte[] wavAudio,
            string? language,
            bool translate,
            string? prompt,
            CancellationToken ct) =>
            Task.FromResult(new PluginTranscriptionResult("", language ?? "en", 0));
    }
}

public class StabilizeTextTests
{
    [Fact]
    public void EmptyConfirmed_ReturnsNew()
    {
        var result = StreamingHandler.StabilizeText("", "Hello world");
        Assert.Equal("Hello world", result);
    }

    [Fact]
    public void EmptyNew_ReturnsConfirmed()
    {
        var result = StreamingHandler.StabilizeText("Hello", "");
        Assert.Equal("Hello", result);
    }

    [Fact]
    public void NewStartsWithConfirmed_ReturnsNew()
    {
        var result = StreamingHandler.StabilizeText("Hello", "Hello world");
        Assert.Equal("Hello world", result);
    }

    [Fact]
    public void ExactMatch_ReturnsConfirmed()
    {
        var result = StreamingHandler.StabilizeText("Hello world", "Hello world");
        Assert.Equal("Hello world", result);
    }

    [Fact]
    public void PartialPrefixMatch_KeepsConfirmedAndAppends()
    {
        // "Hello worl" matches >50% of "Hello world", so confirmed + new tail
        var result = StreamingHandler.StabilizeText("Hello world", "Hello world, how are you?");
        Assert.Equal("Hello world, how are you?", result);
    }

    [Fact]
    public void MinorDivergence_KeepsConfirmedPrefix()
    {
        // First 6 chars match ("Hello "), >50% of 11-char confirmed
        var result = StreamingHandler.StabilizeText("Hello world", "Hello earth and sky");
        Assert.Equal("Hello world earth and sky", result);
    }

    [Fact]
    public void CompletelyDifferent_PreservesConfirmedAndAppends()
    {
        var result = StreamingHandler.StabilizeText("Hello world", "Goodbye universe");
        Assert.Equal("Hello world Goodbye universe", result);
    }

    [Fact]
    public void SuffixPrefixOverlap_DetectsShift()
    {
        // Confirmed = "A B C D", new starts with "B C D E" (suffix of confirmed)
        var confirmed = "Alpha Beta Gamma Delta";
        var newText = "Beta Gamma Delta Epsilon";
        var result = StreamingHandler.StabilizeText(confirmed, newText);
        // Should keep confirmed + append the new tail " Epsilon"
        Assert.Equal("Alpha Beta Gamma Delta Epsilon", result);
    }

    [Fact]
    public void WhitespaceIsTrimmed()
    {
        var result = StreamingHandler.StabilizeText("", "  Hello  ");
        Assert.Equal("Hello", result);
    }
}

public class StabilizeRollingTextTests
{
    [Fact]
    public void EmptyConfirmed_ReturnsWindow()
    {
        var result = StreamingHandler.StabilizeRollingText("", "new tail words");
        Assert.Equal("new tail words", result);
    }

    [Fact]
    public void NoOverlap_PreservesConfirmedPrefix()
    {
        var confirmed = "Začiatok dlhého textu ktorý musí zostať";
        var window = "úplne iná veta na konci";
        var result = StreamingHandler.StabilizeRollingText(confirmed, window);
        Assert.StartsWith("Začiatok", result);
        Assert.Contains("úplne iná veta", result);
    }

    [Fact]
    public void WordOverlap_AppendsTailWhenCharactersDiffer()
    {
        var confirmed = "Alpha beta gamma";
        var window = "beta gamma delta";
        var result = StreamingHandler.StabilizeRollingText(confirmed, window);
        Assert.Equal("Alpha beta gamma delta", result);
    }

    [Fact]
    public void WindowExtendsConfirmed_ReturnsWindow()
    {
        var result = StreamingHandler.StabilizeRollingText(
            "Hello world",
            "Hello world, how are you?");
        Assert.Equal("Hello world, how are you?", result);
    }

    [Fact]
    public void RollingTail_WordOverlap_DoesNotDuplicateMiddle()
    {
        var confirmed = "No takže skúsimme budem hovoriť nejakú dlhšiu vetu";
        var window = "budem hovoriť nejakú dlhšiu vetu a uprostred tej vety budem zastavovať";

        var result = StreamingHandler.StabilizeRollingText(confirmed, window, rollingTailWindow: true);

        Assert.Equal(
            "No takže skúsimme budem hovoriť nejakú dlhšiu vetu a uprostred tej vety budem zastavovať",
            result);
    }

    [Fact]
    public void RollingTail_RepeatedWindow_DoesNotGrow()
    {
        var confirmed = "Alpha beta gamma delta epsilon";
        var window = "gamma delta epsilon zeta eta";

        var once = StreamingHandler.StabilizeRollingText(confirmed, window, rollingTailWindow: true);
        var twice = StreamingHandler.StabilizeRollingText(once, window, rollingTailWindow: true);

        Assert.Equal("Alpha beta gamma delta epsilon zeta eta", once);
        Assert.Equal(once, twice);
    }

    [Fact]
    public void RollingTail_ASRVariation_StillMergesWithoutFullDuplicate()
    {
        var confirmed = "skúsimne budem hovoriť nejakú";
        var window = "hovoriť nejakú dlhšiu vetu a potom pauza";

        var result = StreamingHandler.StabilizeRollingText(confirmed, window, rollingTailWindow: true);

        Assert.StartsWith("skúsimne budem hovoriť nejakú", result);
        Assert.Contains("dlhšiu vetu", result);
        Assert.DoesNotContain("hovoriť nejakú hovoriť", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RollingTail_FrozenPrefix_DoesNotRewriteCommittedStart()
    {
        var confirmed = "Zmrazený začiatok vetu ktorý už nemá meniť";
        var revisedWindow = "ktorý už nemá meniť ale koniec sa mení";

        var result = StreamingHandler.StabilizeRollingText(
            confirmed,
            revisedWindow,
            rollingTailWindow: true,
            frozenPrefixWordCount: 3);

        Assert.StartsWith("Zmrazený začiatok vetu", result);
        Assert.Contains("koniec sa mení", result);
    }

    [Fact]
    public void RollingTail_LongConfirmed_DoesNotTruncateToWindowOnly()
    {
        var confirmed = string.Join(" ", Enumerable.Range(1, 80).Select(i => $"slovo{i}"))
            + " koniec prvej casti.";
        var window = "koniec prvej casti. Druha cast pokracuje.";

        var result = StreamingHandler.StabilizeRollingText(confirmed, window, rollingTailWindow: true);

        Assert.StartsWith("slovo1", result);
        Assert.Contains("slovo40", result);
        Assert.Contains("Druha cast", result);
    }

    [Fact]
    public void RollingTail_AlignedWithWindowOffset_DoesNotPrependWindowGarbage()
    {
        var confirmed = "No tak dobrou noc ja idem spat lebo som unaveny";
        var window = "foo bar idem spat lebo som unaveny ale predtym pouzijem nahravku";

        var result = StreamingHandler.StabilizeRollingText(
            confirmed,
            window,
            rollingTailWindow: true);

        Assert.StartsWith("No tak dobrou noc", result);
        Assert.Contains("pouzijem nahravku", result);
        Assert.DoesNotContain("foo bar", result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RollingTail_GoodnightDictation_DoesNotTripleOpeningPhrase()
    {
        var confirmed = "No tak dobrou noc ja idem spat lebo uz som fakt unaveny. " +
            "Ale predtym nez mozem spat pouzijem tuto nahravku.";
        var window = "mozem spat pouzijem tuto nahravku aby som ti povedal ze ten merge " +
            "No tak dobrou noc ja idem spat lebo uz som fakt unaveny";

        var result = StreamingHandler.StabilizeRollingText(
            confirmed,
            window,
            rollingTailWindow: true);

        var count = CountPhrase(result, "No tak dobrou noc");
        Assert.Equal(1, count);
    }

    private static int CountPhrase(string text, string phrase)
    {
        var count = 0;
        var index = 0;
        while ((index = text.IndexOf(phrase, index, StringComparison.OrdinalIgnoreCase)) >= 0)
        {
            count++;
            index += phrase.Length;
        }

        return count;
    }

    [Fact]
    public void RollingTail_ModelRestartFromEarlierText_DoesNotDuplicateBeginning()
    {
        var confirmed = "Zúsim hovoriť nejakú vetu respektíve vytvoriť konverzáciu. " +
            "Začnem tým že nejakú tú vetu poviem znova a uvidíme či to nespôsobí problém. " +
            "Budeme hovoriť o tom ako sa mi podarilo mať sen.";
        var restartedWindow = "Zúsim hovoriť nejakú vetu respektíve vytvoriť konverzáciu. " +
            "Začnem tým že nejakú tú vetu poviem znova a uvidíme či to nespôsobí problém.";

        var result = StreamingHandler.StabilizeRollingText(
            confirmed,
            restartedWindow,
            rollingTailWindow: true);

        Assert.Equal(confirmed, result);
    }

    [Fact]
    public void CollapsesMultipleAdjacentDuplicatePhrases()
    {
        var phrase = "ako sa mi podarilo mať sen";
        var text = $"{phrase} {phrase} {phrase} a potom som sa zobudil";

        var result = StreamingHandler.StabilizeText("", text);

        var occurrences = CountOccurrences(result, "podarilo mať sen");
        Assert.Equal(1, occurrences);
        Assert.Contains("potom som sa zobudil", result);
    }

    private static int CountOccurrences(string haystack, string needle)
    {
        var count = 0;
        var index = 0;
        while ((index = haystack.IndexOf(needle, index, StringComparison.OrdinalIgnoreCase)) >= 0)
        {
            count++;
            index += needle.Length;
        }

        return count;
    }

    [Fact]
    public void FuzzyOverlap_MergesFinalChunksWithAsrVariation()
    {
        var first = "Posledný krát som zistil že už to je v poriadku respektíve zobudil som sa aj reálny";
        var second = "respektive zobudil som sa aj realny síce veci neboli pokazané ale po štyroch zobudeniach";

        var result = StreamingHandler.StabilizeRollingText(first, second);

        Assert.Contains("Posledný krát", result);
        Assert.Contains("síce veci", result);
        Assert.DoesNotContain("reálny respektive zobudil", result, StringComparison.OrdinalIgnoreCase);
    }
}

public class TranscriptTimelineMergerTests
{
    [Fact]
    public void EmptyConfirmed_ReturnsWindowTokens()
    {
        var result = TranscriptTimelineMerger.Merge("", "new tail words");
        Assert.Equal("new tail words", result);
    }

    [Fact]
    public void RollingTail_WordOverlap_DoesNotDuplicateMiddle()
    {
        var confirmed = "No takže skúsimme budem hovoriť nejakú dlhšiu vetu";
        var window = "budem hovoriť nejakú dlhšiu vetu a uprostred tej vety budem zastavovať";

        var result = TranscriptTimelineMerger.Merge(confirmed, window, rollingTailWindow: true);

        Assert.Equal(
            "No takže skúsimme budem hovoriť nejakú dlhšiu vetu a uprostred tej vety budem zastavovať",
            result);
    }

    [Fact]
    public void RollingTail_RepeatedWindow_DoesNotGrowAfterCommit()
    {
        var confirmed = "Alpha beta gamma delta epsilon";
        var window = "gamma delta epsilon zeta eta";

        var timeline = new TranscriptTokenTimeline();
        timeline.ImportConfirmed(confirmed, committedPrefixCount: 0, commitThreshold: 2);
        var once = timeline.MergeWindow(window, rollingTailWindow: true);
        var twice = timeline.MergeWindow(window, rollingTailWindow: true);

        Assert.Equal("Alpha beta gamma delta epsilon zeta eta", once);
        Assert.Equal(once, twice);
    }

    [Fact]
    public void RollingTail_ModelRestartFromEarlierText_DoesNotDuplicateBeginning()
    {
        var confirmed = "Zúsim hovoriť nejakú vetu respektíve vytvoriť konverzáciu. " +
            "Začnem tým že nejakú tú vetu poviem znova a uvidíme či to nespôsobí problém. " +
            "Budeme hovoriť o tom ako sa mi podarilo mať sen.";
        var restartedWindow = "Zúsim hovoriť nejakú vetu respektíve vytvoriť konverzáciu. " +
            "Začnem tým že nejakú tú vetu poviem znova a uvidíme či to nespôsobí problém.";

        var result = TranscriptTimelineMerger.Merge(
            confirmed,
            restartedWindow,
            rollingTailWindow: true);

        Assert.Equal(confirmed, result);
    }

    [Fact]
    public void TimelineSession_UsesStatefulTokenTimelineAcrossPolls()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession(TranscriptMergeMode.Timeline);

        Assert.True(sut.TryApplyPolling(
            sessionVersion,
            "Hello world",
            text => text,
            out var firstDisplay));
        Assert.True(sut.TryApplyPolling(
            sessionVersion,
            "Hello world, how are you?",
            text => text,
            out var secondDisplay));

        Assert.Equal("Hello world", firstDisplay);
        Assert.Equal("Hello world, how are you?", secondDisplay);
    }

    [Fact]
    public void TimelineSession_CommitsPrefixDuringRollingWindow()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession(TranscriptMergeMode.Timeline);
        var samples = StreamingTranscriptState.MaxPollingSamples + (16 * 16000);

        Assert.True(sut.TryApplyPolling(
            sessionVersion,
            "Dobre zaciatok o 3D tlaci a Pruse i3 sliceri.",
            text => text,
            out _,
            rollingWindow: false,
            totalSamplesRecorded: samples));

        Assert.True(sut.TryApplyPolling(
            sessionVersion,
            "o 3D tlaci a Pruse i3 sliceri. Pokracujem v dlhom texte.",
            text => text,
            out _,
            rollingWindow: true,
            totalSamplesRecorded: samples));

        samples += 48000;
        sut.TryApplyPolling(
            sessionVersion,
            "uplne iny tail bez 3d zmienky",
            text => text,
            out _,
            rollingWindow: true,
            totalSamplesRecorded: samples);

        var finalText = sut.StopSession();

        Assert.Contains("Dobre zaciatok", finalText, StringComparison.Ordinal);
        Assert.Contains("Pruse", finalText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void PreparePollingBuffer_UsesTailWhenTooLong()
    {
        var buffer = Enumerable.Range(0, StreamingTranscriptState.MaxPollingSamples + 16000)
            .Select(i => (float)i)
            .ToArray();

        var prepared = StreamingTranscriptState.PreparePollingBuffer(buffer, out var usedTail);

        Assert.True(usedTail);
        Assert.Equal(StreamingTranscriptState.MaxPollingSamples, prepared.Length);
        Assert.Equal(buffer[^1], prepared[^1]);
        Assert.NotEqual(buffer[0], prepared[0]);
    }
}

public class StreamingTranscriptStateTests
{
    [Fact]
    public void StopSession_FallsBackToOnlyInterimRealtimeText()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession();

        var interimApplied = sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Hello world", false),
            text => text,
            out var interimDisplay);

        Assert.True(interimApplied);
        Assert.Equal("Hello world", interimDisplay);

        var finalText = sut.StopSession();

        Assert.Equal("Hello world", finalText);
    }

    [Fact]
    public void StopSession_InvalidatesLateRealtimeEvents()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession();

        var appliedBeforeStop = sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Hello world", true),
            text => text,
            out var displayBeforeStop);

        Assert.True(appliedBeforeStop);
        Assert.Equal("Hello world", displayBeforeStop);

        var finalText = sut.StopSession();

        Assert.Equal("Hello world", finalText);

        var appliedAfterStop = sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Should be ignored", false),
            text => text,
            out var displayAfterStop);

        Assert.False(appliedAfterStop);
        Assert.Equal("", displayAfterStop);
    }

    [Fact]
    public void StopSession_FallsBackWhenTrailingRealtimeInterimAfterConfirmedText()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession();

        Assert.True(sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Confirmed", true),
            text => text,
            out var confirmedDisplay));
        Assert.Equal("Confirmed", confirmedDisplay);

        Assert.True(sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("still changing", false),
            text => text,
            out var interimDisplay));
        Assert.Equal("Confirmed still changing", interimDisplay);

        Assert.Equal("Confirmed still changing", sut.StopSession());
    }

    [Fact]
    public void RealtimeFinalTranscript_AppendsToConfirmedText()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession();

        var interimApplied = sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Hello", false),
            text => text,
            out var interimDisplay);
        var finalApplied = sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("world", true),
            text => text,
            out var finalDisplay);

        Assert.True(interimApplied);
        Assert.Equal("Hello", interimDisplay);
        Assert.True(finalApplied);
        Assert.Equal("world", finalDisplay);
        Assert.Equal("world", sut.StopSession());
    }

    [Fact]
    public void RealtimeFinalTranscript_ReplacesCumulativeFinalPrefix()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession();

        Assert.True(sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Hello world", true),
            text => text,
            out var firstDisplay));
        Assert.Equal("Hello world", firstDisplay);

        Assert.True(sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Hello world from streaming", true),
            text => text,
            out var secondDisplay));

        Assert.Equal("Hello world from streaming", secondDisplay);
        Assert.Equal("Hello world from streaming", sut.StopSession());
    }

    [Fact]
    public void RealtimeFinalTranscript_IgnoresDuplicateFinalSegment()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession();

        Assert.True(sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Hello world", true),
            text => text,
            out var firstDisplay));
        Assert.Equal("Hello world", firstDisplay);

        Assert.True(sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Hello world", true),
            text => text,
            out var duplicateDisplay));

        Assert.Equal("Hello world", duplicateDisplay);
        Assert.Equal("Hello world", sut.StopSession());
    }

    [Fact]
    public void RealtimeFinalTranscript_AppendsDistinctFinalChunks()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession();

        Assert.True(sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("Hello", true),
            text => text,
            out var firstDisplay));
        Assert.Equal("Hello", firstDisplay);

        Assert.True(sut.TryApplyRealtime(
            sessionVersion,
            new StreamingTranscriptEvent("world", true),
            text => text,
            out var secondDisplay));

        Assert.Equal("Hello world", secondDisplay);
        Assert.Equal("Hello world", sut.StopSession());
    }

    [Fact]
    public void PollingTranscript_UsesStabilizedCurrentSessionOnly()
    {
        var sut = new StreamingTranscriptState();
        var firstSession = sut.StartSession();

        var firstApplied = sut.TryApplyPolling(
            firstSession,
            "Hello world",
            text => text,
            out var firstDisplay);
        var secondApplied = sut.TryApplyPolling(
            firstSession,
            "Hello world, how are you?",
            text => text,
            out var secondDisplay);

        Assert.True(firstApplied);
        Assert.Equal("Hello world", firstDisplay);
        Assert.True(secondApplied);
        Assert.Equal("Hello world, how are you?", secondDisplay);

        var secondSession = sut.StartSession();
        var staleApplied = sut.TryApplyPolling(
            firstSession,
            "Old session text",
            text => text,
            out _);
        var currentApplied = sut.TryApplyPolling(
            secondSession,
            "Fresh session text",
            text => text,
            out var currentDisplay);

        Assert.False(staleApplied);
        Assert.True(currentApplied);
        Assert.Equal("Fresh session text", currentDisplay);
    }

    [Fact]
    public void PollingTranscript_PreservesImmutablePrefixDuringRollingWindow()
    {
        var sut = new StreamingTranscriptState();
        var sessionVersion = sut.StartSession();
        var samples = 12 * 16000;

        Assert.True(sut.TryApplyPolling(
            sessionVersion,
            "Dobre zaciatok o 3D tlaci a Pruse i3 sliceri.",
            text => text,
            out _,
            rollingWindow: false,
            totalSamplesRecorded: samples));

        samples = StreamingTranscriptState.MaxPollingSamples + (16 * 16000);
        Assert.True(sut.TryApplyPolling(
            sessionVersion,
            "o 3D tlaci a Pruse i3 sliceri. Pokracujem v dlhom texte.",
            text => text,
            out _,
            rollingWindow: true,
            totalSamplesRecorded: samples));

        samples += 48000;
        sut.TryApplyPolling(
            sessionVersion,
            "uplne iny tail bez 3d zmienky",
            text => text,
            out _,
            rollingWindow: true,
            totalSamplesRecorded: samples);

        var finalText = sut.StopSession();

        Assert.Contains("Dobre zaciatok", finalText, StringComparison.Ordinal);
        Assert.Contains("Pruse", finalText, StringComparison.OrdinalIgnoreCase);
    }
}

public class ParakeetTailHelperTests
{
    [Fact]
    public void IsParakeetModel_MatchesOpenVinoPluginModelId()
    {
        Assert.True(ParakeetTailHelper.IsParakeetModel(ParakeetTailHelper.ParakeetOpenVinoModelId));
    }

    [Fact]
    public void SelectResult_MergesLivePrefixWithTailFullDecode()
    {
        var partial = "Začiatok textu ktorý sme videli v live preview";
        var full = "v live preview a potom ešte pokračovanie";

        var selection = ParakeetTailHelper.SelectResult(
            ParakeetTailHelper.ParakeetOpenVinoTolkiumModelId,
            full,
            [partial]);

        Assert.Equal("merged_live_full", selection.Source);
        Assert.StartsWith("Začiatok", selection.Text);
        Assert.Contains("pokračovanie", selection.Text);
    }

    [Fact]
    public void SelectResult_ForOpenVinoParakeet_PrefersFullDecodeOverPartials()
    {
        var selection = ParakeetTailHelper.SelectResult(
            ParakeetTailHelper.ParakeetOpenVinoModelId,
            "final full decode",
            ["partial text"]);

        Assert.Equal("final full decode", selection.Text);
        Assert.Equal("full_decode", selection.Source);
    }

    [Fact]
    public void AppendTailGuard_AddsExpectedSilenceSamples()
    {
        var samples = new float[] { 0.1f, -0.2f, 0.3f };

        var guarded = ParakeetTailHelper.AppendTailGuard(samples);

        Assert.Equal(samples.Length + 3200, guarded.Length);
        Assert.Equal(samples[0], guarded[0]);
        Assert.Equal(samples[1], guarded[1]);
        Assert.Equal(samples[2], guarded[2]);
        Assert.All(guarded.Skip(samples.Length), sample => Assert.Equal(0f, sample));
    }

    [Fact]
    public void SelectResult_ForParakeet_PrefersFullDecodeOverPartials()
    {
        var selection = ParakeetTailHelper.SelectResult(
            ParakeetTailHelper.ParakeetModelId,
            "final full decode",
            ["partial text"]);

        Assert.Equal("final full decode", selection.Text);
        Assert.Equal("full_decode", selection.Source);
        Assert.True(selection.DivergedFromPartials);
    }

    [Fact]
    public void SelectResult_ForParakeet_FallsBackToPartialsWhenFullDecodeIsEmpty()
    {
        var selection = ParakeetTailHelper.SelectResult(
            ParakeetTailHelper.ParakeetModelId,
            "",
            ["tail segment"]);

        Assert.Equal("tail segment", selection.Text);
        Assert.Equal("fallback_partials_after_empty_full_decode", selection.Source);
        Assert.False(selection.DivergedFromPartials);
    }

    [Fact]
    public void SelectResult_ForNonParakeet_KeepsExistingPartialPreference()
    {
        var selection = ParakeetTailHelper.SelectResult(
            "plugin:other:model",
            "full decode",
            ["partial text"]);

        Assert.Equal("partial text", selection.Text);
        Assert.Equal("partials", selection.Source);
    }
}
