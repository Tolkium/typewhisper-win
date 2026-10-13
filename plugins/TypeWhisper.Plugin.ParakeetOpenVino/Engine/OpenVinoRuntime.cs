using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using static TypeWhisper.Plugin.ParakeetOpenVino.Engine.OpenVinoNative;

namespace TypeWhisper.Plugin.ParakeetOpenVino.Engine;

/// <summary>Thin owning wrappers over the OpenVINO C handles used by the spike.</summary>
internal static unsafe class OpenVinoLoader
{
    private static nint _handle;

    /// <summary>Loads openvino_c.dll from <paramref name="libraryDirectory"/> and routes DllImports to it.</summary>
    public static void Load(string libraryDirectory)
    {
        var path = Path.Combine(libraryDirectory, "openvino_c.dll");
        if (!File.Exists(path))
            throw new FileNotFoundException("openvino_c.dll not found", path);

        // Absolute path: Windows resolves openvino.dll / tbb12.dll from the same folder.
        _handle = NativeLibrary.Load(path);
        NativeLibrary.SetDllImportResolver(
            Assembly.GetExecutingAssembly(),
            (name, _, _) => name == Library ? _handle : nint.Zero);
    }

    public static string Version()
    {
        OvVersion version;
        Check(GetVersion(&version), "ov_get_openvino_version");
        var text = Marshal.PtrToStringUTF8((nint)version.BuildNumber) ?? "?";
        FreeVersion(&version);
        return text;
    }
}

internal sealed unsafe class OvCore : IDisposable
{
    public nint Handle { get; }

    public OvCore()
    {
        nint core;
        Check(CoreCreate(&core), "ov_core_create");
        Handle = core;
    }

    public IReadOnlyList<string> AvailableDevices()
    {
        AvailableDevices devices;
        Check(CoreGetAvailableDevices(Handle, &devices), "ov_core_get_available_devices");
        var result = new List<string>();
        for (nuint i = 0; i < devices.Size; i++)
            result.Add(Marshal.PtrToStringUTF8((nint)devices.Devices[i]) ?? "");
        AvailableDevicesFree(&devices);
        return result;
    }

    public OvModel ReadModel(string path)
    {
        nint model;
        Check(CoreReadModel(Handle, path, null, &model), $"ov_core_read_model({Path.GetFileName(path)})");
        return new OvModel(model);
    }

    public OvCompiledModel Compile(OvModel model, string device, string? cacheDir)
    {
        nint compiled;
        var status = cacheDir is null
            ? CoreCompileModel(Handle, model.Handle, device, 2, &compiled, "PERFORMANCE_HINT", "LATENCY")
            : CoreCompileModel(Handle, model.Handle, device, 4, &compiled,
                "PERFORMANCE_HINT", "LATENCY", "CACHE_DIR", cacheDir);
        Check(status, $"ov_core_compile_model({device})");
        return new OvCompiledModel(compiled);
    }

    public void Dispose() => CoreFree(Handle);
}

internal sealed unsafe class OvModel(nint handle) : IDisposable
{
    public nint Handle { get; } = handle;

    /// <summary>Reshapes one input; -1 marks a dynamic dimension.</summary>
    public void Reshape(string input, params long[] dims)
    {
        var dimensions = stackalloc Dimension[dims.Length];
        for (var i = 0; i < dims.Length; i++)
            dimensions[i] = dims[i] < 0 ? new Dimension { Min = -1, Max = -1 } : new Dimension { Min = dims[i], Max = dims[i] };

        PartialShape shape;
        Check(PartialShapeCreate(dims.Length, dimensions, &shape), "ov_partial_shape_create");
        try
        {
            Check(ModelReshapeInputByName(Handle, input, shape), $"ov_model_reshape_input_by_name({input})");
        }
        finally
        {
            PartialShapeFree(&shape);
        }
    }

    public void Dispose() => ModelFree(Handle);
}

internal sealed unsafe class OvCompiledModel(nint handle) : IDisposable
{
    public nint Handle { get; } = handle;

    public OvInferRequest CreateInferRequest()
    {
        nint request;
        Check(CompiledModelCreateInferRequest(Handle, &request), "ov_compiled_model_create_infer_request");
        return new OvInferRequest(request);
    }

    public void Dispose() => CompiledModelFree(Handle);
}

internal sealed unsafe class OvInferRequest(nint handle) : IDisposable
{
    public nint Handle { get; } = handle;

    /// <summary>Binds a managed array as input. The array must stay pinned until <see cref="Infer"/> returns.</summary>
    public void SetInput<T>(string name, ElementType type, T* data, params long[] dims) where T : unmanaged
    {
        Shape shape;
        fixed (long* d = dims)
            Check(ShapeCreate(dims.Length, d, &shape), "ov_shape_create");
        nint tensor;
        try
        {
            Check(TensorCreateFromHostPtr(type, shape, data, &tensor), $"ov_tensor_create_from_host_ptr({name})");
        }
        finally
        {
            ShapeFree(&shape);
        }

        try
        {
            Check(InferRequestSetTensor(Handle, name, tensor), $"ov_infer_request_set_tensor({name})");
        }
        finally
        {
            TensorFree(tensor);
        }
    }

    public void Infer() => Check(InferRequestInfer(Handle), "ov_infer_request_infer");

    /// <summary>Copies an output tensor to a new array and returns its shape.</summary>
    public (T[] Data, long[] Shape) GetOutput<T>(string name) where T : unmanaged
    {
        nint tensor;
        Check(InferRequestGetTensor(Handle, name, &tensor), $"ov_infer_request_get_tensor({name})");
        try
        {
            Shape shape;
            Check(TensorGetShape(tensor, &shape), "ov_tensor_get_shape");
            var dims = new long[shape.Rank];
            long count = 1;
            for (var i = 0; i < shape.Rank; i++)
            {
                dims[i] = shape.Dims[i];
                count *= dims[i];
            }
            ShapeFree(&shape);

            void* data;
            Check(TensorData(tensor, &data), "ov_tensor_data");
            var result = new T[count];
            new ReadOnlySpan<T>(data, checked((int)count)).CopyTo(result);
            return (result, dims);
        }
        finally
        {
            TensorFree(tensor);
        }
    }

    public void Dispose() => InferRequestFree(Handle);
}
