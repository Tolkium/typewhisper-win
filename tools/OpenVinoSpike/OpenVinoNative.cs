using System.Runtime.InteropServices;

namespace OpenVinoSpike;

/// <summary>
/// P/Invoke declarations for the OpenVINO C API (openvino_c.dll), checked against the
/// 2026.2.0 headers in src/bindings/c/include/openvino/c.
/// </summary>
internal static unsafe partial class OpenVinoNative
{
    public const string Library = "openvino_c";

    public enum Status
    {
        Ok = 0,
    }

    public enum ElementType
    {
        F32 = 4,
        I32 = 9,
        I64 = 10,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Shape
    {
        public long Rank;
        public long* Dims;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Dimension
    {
        public long Min;
        public long Max;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PartialShape
    {
        public Dimension Rank;
        public Dimension* Dims;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct AvailableDevices
    {
        public byte** Devices;
        public nuint Size;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct OvVersion
    {
        public byte* BuildNumber;
        public byte* Description;
    }

    [DllImport(Library, EntryPoint = "ov_get_last_err_msg")]
    public static extern byte* GetLastErrorMessage();

    [DllImport(Library, EntryPoint = "ov_get_openvino_version")]
    public static extern Status GetVersion(OvVersion* version);

    [DllImport(Library, EntryPoint = "ov_version_free")]
    public static extern void FreeVersion(OvVersion* version);

    [DllImport(Library, EntryPoint = "ov_core_create")]
    public static extern Status CoreCreate(nint* core);

    [DllImport(Library, EntryPoint = "ov_core_free")]
    public static extern void CoreFree(nint core);

    [DllImport(Library, EntryPoint = "ov_core_get_available_devices")]
    public static extern Status CoreGetAvailableDevices(nint core, AvailableDevices* devices);

    [DllImport(Library, EntryPoint = "ov_available_devices_free")]
    public static extern void AvailableDevicesFree(AvailableDevices* devices);

    [DllImport(Library, EntryPoint = "ov_core_read_model")]
    public static extern Status CoreReadModel(
        nint core,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? binPath,
        nint* model);

    // ov_core_compile_model is variadic (key/value string pairs). On win-x64 pointer varargs use
    // the normal calling convention, so fixed-arity overloads are ABI-compatible.
    [DllImport(Library, EntryPoint = "ov_core_compile_model")]
    public static extern Status CoreCompileModel(
        nint core, nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string device, nuint argCount, nint* compiled);

    [DllImport(Library, EntryPoint = "ov_core_compile_model")]
    public static extern Status CoreCompileModel(
        nint core, nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string device, nuint argCount, nint* compiled,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key1, [MarshalAs(UnmanagedType.LPUTF8Str)] string value1);

    [DllImport(Library, EntryPoint = "ov_core_compile_model")]
    public static extern Status CoreCompileModel(
        nint core, nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string device, nuint argCount, nint* compiled,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key1, [MarshalAs(UnmanagedType.LPUTF8Str)] string value1,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string key2, [MarshalAs(UnmanagedType.LPUTF8Str)] string value2);

    [DllImport(Library, EntryPoint = "ov_model_free")]
    public static extern void ModelFree(nint model);

    [DllImport(Library, EntryPoint = "ov_model_reshape_input_by_name")]
    public static extern Status ModelReshapeInputByName(
        nint model, [MarshalAs(UnmanagedType.LPUTF8Str)] string tensorName, PartialShape shape);

    [DllImport(Library, EntryPoint = "ov_partial_shape_create")]
    public static extern Status PartialShapeCreate(long rank, Dimension* dims, PartialShape* shape);

    [DllImport(Library, EntryPoint = "ov_partial_shape_free")]
    public static extern void PartialShapeFree(PartialShape* shape);

    [DllImport(Library, EntryPoint = "ov_compiled_model_create_infer_request")]
    public static extern Status CompiledModelCreateInferRequest(nint compiled, nint* request);

    [DllImport(Library, EntryPoint = "ov_compiled_model_free")]
    public static extern void CompiledModelFree(nint compiled);

    [DllImport(Library, EntryPoint = "ov_infer_request_set_tensor")]
    public static extern Status InferRequestSetTensor(
        nint request, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, nint tensor);

    [DllImport(Library, EntryPoint = "ov_infer_request_get_tensor")]
    public static extern Status InferRequestGetTensor(
        nint request, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, nint* tensor);

    [DllImport(Library, EntryPoint = "ov_infer_request_infer")]
    public static extern Status InferRequestInfer(nint request);

    [DllImport(Library, EntryPoint = "ov_infer_request_free")]
    public static extern void InferRequestFree(nint request);

    [DllImport(Library, EntryPoint = "ov_tensor_create_from_host_ptr")]
    public static extern Status TensorCreateFromHostPtr(ElementType type, Shape shape, void* hostPtr, nint* tensor);

    [DllImport(Library, EntryPoint = "ov_tensor_get_shape")]
    public static extern Status TensorGetShape(nint tensor, Shape* shape);

    [DllImport(Library, EntryPoint = "ov_tensor_data")]
    public static extern Status TensorData(nint tensor, void** data);

    [DllImport(Library, EntryPoint = "ov_tensor_free")]
    public static extern void TensorFree(nint tensor);

    [DllImport(Library, EntryPoint = "ov_shape_create")]
    public static extern Status ShapeCreate(long rank, long* dims, Shape* shape);

    [DllImport(Library, EntryPoint = "ov_shape_free")]
    public static extern Status ShapeFree(Shape* shape);

    public static void Check(Status status, string what)
    {
        if (status == Status.Ok)
            return;

        var message = Marshal.PtrToStringUTF8((nint)GetLastErrorMessage()) ?? "";
        throw new InvalidOperationException($"{what} failed ({(int)status}): {message.Trim()}");
    }
}
