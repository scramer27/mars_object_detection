import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

def patch_model(input_path, output_path):
    print(f"--- 1. Reading {input_path} ---")
    with open(input_path, 'rb') as f:
        buf = bytearray(f.read())

    model_raw = schema_fb.Model.GetRootAsModel(buf, 0)
    model = schema_fb.ModelT.InitFromObj(model_raw)

    graph = model.subgraphs[0]

    # Inspect Node 0 (QUANTIZE) output tensor
    node0 = graph.operators[0]
    out_idx = node0.outputs[0]
    tensor = graph.tensors[out_idx]

    tensor_name = tensor.name if isinstance(tensor.name, str) else tensor.name.decode('utf-8')
    print(f"Node 0 Output Tensor {out_idx} ('{tensor_name}') Type: {tensor.type}")

    # Force strict INT8 type
    tensor.type = schema_fb.TensorType.INT8

    # Ensure QuantizationParameters object exists
    if tensor.quantization is None:
        tensor.quantization = schema_fb.QuantizationParametersT()

    q = tensor.quantization

    # Force scale & zeroPoint values if missing/empty
    if not q.scale or len(q.scale) == 0:
        print("  -> Injecting missing scale [0.0039215686]")
        q.scale = [1.0 / 255.0]

    if not q.zeroPoint or len(q.zeroPoint) == 0:
        print("  -> Injecting missing zeroPoint [0]")
        q.zeroPoint = [0]

    q.quantizedDimension = 0

    # Serialize back to FlatBuffer binary
    builder = flatbuffers.Builder(len(buf) * 2)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b'TFL3')

    # ALWAYS write output file outside conditional blocks
    with open(output_path, 'wb') as f:
        f.write(builder.Output())

    print(f"--- 2. Patched model saved to {output_path} ---")

if __name__ == "__main__":
    patch_model(
        "fpga_payload/mars_yolov8_int8_fixed.tflite", 
        "fpga_payload/mars_yolov8_int8_patched.tflite"
    )