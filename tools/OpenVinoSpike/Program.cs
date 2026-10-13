// Checks for the native OpenVINO engine against the Python reference data (docs/plans/native-openvino-engine.md).
//
//   encoder  <ov-libs> <weights> <reference> [CPU,GPU,NPU] [cache]   phase 1: encoder output vs reference
//   features <reference>                                              phase 2: C# features vs NumPy
//   decoder  <weights> <reference> <ov-libs>                          phase 3: TDT decoder vs reference tokens
//   e2e      <weights> <reference> <ov-libs> <GPU;NPU;CPU> [cache]    phase 4: waveform -> text vs reference
//   setup    <data dir> <device> <wav>                                phase 5: plugin download + transcription

return args switch
{
    ["encoder", var libs, var weights, var reference, ..] =>
        Checks.Encoder(libs, weights, reference, args.ElementAtOrDefault(4) ?? "CPU,GPU,NPU", args.ElementAtOrDefault(5)),
    ["features", var reference] => Checks.Features(reference),
    ["decoder", var weights, var reference, var libs] => Checks.Decoder(weights, reference, libs),
    ["e2e", var weights, var reference, var libs, var devices, ..] =>
        Checks.EndToEnd(weights, reference, libs, args.ElementAtOrDefault(5), devices),
    ["setup", var dataDirectory, var device, var wav] => Checks.Setup(dataDirectory, device, wav),
    _ => Usage(),
};

static int Usage()
{
    Console.Error.WriteLine("Usage: OpenVinoSpike encoder|features|decoder|e2e|setup ... (see Program.cs)");
    return 2;
}
