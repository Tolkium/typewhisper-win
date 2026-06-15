using TypeWhisper.Plugin.ParakeetOpenVino.Timeline;
using TypeWhisper.Windows.Services;

namespace TypeWhisper.PluginSystem.Tests;

public class ParakeetOpenVinoTimelinePluginTests
{
    [Fact]
    public void Plugin_HasDistinctIdAndVersion()
    {
        var sut = new ParakeetOpenVinoTimelinePlugin();

        Assert.Equal("com.tolkium.parakeet-openvino-timeline", sut.PluginId);
        Assert.Equal("0.4.0-alpha", sut.PluginVersion);
        Assert.Equal("parakeet-tdt-0.6b-v3-openvino-timeline", ParakeetOpenVinoTimelinePlugin.ModelId);
    }

    [Fact]
    public void UsesTimelineMerge_OnlyForTimelineModel()
    {
        Assert.True(ParakeetPluginIds.UsesTimelineMerge(ParakeetPluginIds.TimelineModelId));
        Assert.False(ParakeetPluginIds.UsesTimelineMerge(ParakeetPluginIds.BaselineModelId));
    }
}
