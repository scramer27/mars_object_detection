#!/usr/bin/env python3
"""
fix_litert.py
Repairs YOLOv8 / LiteRT INT8 FlatBuffer models for Microchip VectorBlox SDK compatibility.
"""

import sys
import os
import flatbuffers

try:
    from tensorflow.lite.python import schema_py_generated as schema_fb
except ImportError:
    try:
        import tflite as schema_fb
    except ImportError:
        raise ImportError("Please install tensorflow or tflite package.")

input_path = "fpga_payload/mars_yolov8_int8.tflite"
output_path = "fpga_payload/mars_yolov8_int8_fixed.tflite"

if len(sys.argv) >= 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]

print(f"--- 1. Reading model from {input_path} ---")
with open(input_path, "rb") as f:
    file_bytes = f.read()

model = schema_fb.Model.GetRootAsModel(file_bytes, 0)
model_t = schema_fb.ModelT.UnPackToNew(model)

if model_t.buffers is None:
    model_t.buffers = []

# Byte width per TensorType, used to correctly size dummy constant buffers.
# NOTE: previously this only distinguished FLOAT32 (4 bytes) from "everything
# else" (1 byte), which silently under-sized INT16 (2 bytes) and INT32
# (4 bytes) dummy buffers -> buffer/tensor size mismatch or OOB read risk
# inside vnnx_compile.
TYPE_BYTE_SIZES = {
    schema_fb.TensorType.FLOAT32: 4,
    schema_fb.TensorType.INT32:   4,
    schema_fb.TensorType.INT16:   2,
    schema_fb.TensorType.INT8:    1,
    schema_fb.TensorType.UINT8:   1,
}

# Map binary opcodes (ADD=0, MUL=18, SUB=41, DIV=4, etc.)
binary_codes = {
    schema_fb.BuiltinOperator.ADD,
    schema_fb.BuiltinOperator.SUB,
    schema_fb.BuiltinOperator.MUL,
    schema_fb.BuiltinOperator.DIV,
    schema_fb.BuiltinOperator.MINIMUM,
    schema_fb.BuiltinOperator.MAXIMUM,
    schema_fb.BuiltinOperator.POW,
    schema_fb.BuiltinOperator.SQUARED_DIFFERENCE,
}

quant_op_indices = set()
dequant_op_indices = set()
binary_op_indices = set()

for idx, op_code in enumerate(model_t.operatorCodes):
    # CRITICAL: use max(), not `builtinCode if builtinCode != 0 else deprecated`.
    # BuiltinOperator.ADD == 0, so the old ternary treated ADD as "unset" and
    # fell through to deprecatedBuiltinCode, silently dropping ADD ops from
    # binary_op_indices.
    code = max(op_code.builtinCode, op_code.deprecatedBuiltinCode)
    if code == schema_fb.BuiltinOperator.QUANTIZE:
        quant_op_indices.add(idx)
    elif code == schema_fb.BuiltinOperator.DEQUANTIZE:
        dequant_op_indices.add(idx)
    elif code in binary_codes:
        binary_op_indices.add(idx)

print(f"Opcode Mapping -> QUANTIZE: {quant_op_indices}, DEQUANTIZE: {dequant_op_indices}, BINARY: {binary_op_indices}")

for sg_idx, subgraph in enumerate(model_t.subgraphs):
    # 1. Enforce node output types
    for op in subgraph.operators:
        if op.opcodeIndex in quant_op_indices:
            for out_idx in op.outputs:
                subgraph.tensors[out_idx].type = schema_fb.TensorType.INT8
        elif op.opcodeIndex in dequant_op_indices:
            for out_idx in op.outputs:
                subgraph.tensors[out_idx].type = schema_fb.TensorType.FLOAT32
                subgraph.tensors[out_idx].quantization = None

    # 2. Pad ANY binary operator (ADD, SUB, MUL, DIV, MIN, MAX, POW, SQDIFF)
    #    that has fewer than 2 inputs, so downstream lut_pattern() in
    #    transform_tflite.py never does op['inputs'][1] on a 1-length list.
    for op_idx, op in enumerate(subgraph.operators):
        if op.opcodeIndex in binary_op_indices:
            while len(op.inputs) < 2:
                print(f"Padding binary op #{op_idx} (Opcode index {op.opcodeIndex}) with dummy operand...")
                if len(op.inputs) == 1:
                    ref_tensor = subgraph.tensors[op.inputs[0]]
                else:
                    ref_tensor = subgraph.tensors[subgraph.inputs[0]]

                nbytes = TYPE_BYTE_SIZES.get(ref_tensor.type, 1)

                # Create zero constant buffer, correctly sized for the type.
                buf = schema_fb.BufferT()
                buf.data = [0] * nbytes
                model_t.buffers.append(buf)
                buf_idx = len(model_t.buffers) - 1

                # Append dummy tensor
                dummy_tensor = schema_fb.TensorT()
                dummy_tensor.name = f"repaired_dummy_operand_{len(subgraph.tensors)}"
                dummy_tensor.type = ref_tensor.type
                dummy_tensor.shape = [1]
                dummy_tensor.buffer = buf_idx

                if ref_tensor.type in [
                    schema_fb.TensorType.INT8,
                    schema_fb.TensorType.UINT8,
                    schema_fb.TensorType.INT16,
                    schema_fb.TensorType.INT32,
                ]:
                    dummy_tensor.quantization = schema_fb.QuantizationParametersT()
                    dummy_tensor.quantization.scale = [1.0]
                    dummy_tensor.quantization.zeroPoint = [0]
                    dummy_tensor.quantization.zero_point = [0]
                    dummy_tensor.quantization.quantizedDimension = 0
                    dummy_tensor.quantization.quantized_dimension = 0

                subgraph.tensors.append(dummy_tensor)
                op.inputs.append(len(subgraph.tensors) - 1)

    # 3. Clean up quantization parameters by tensor type
    for tensor in subgraph.tensors:
        if tensor.type in [
            schema_fb.TensorType.INT8,
            schema_fb.TensorType.UINT8,
            schema_fb.TensorType.INT16,
            schema_fb.TensorType.INT32,
        ]:
            if tensor.quantization is None:
                tensor.quantization = schema_fb.QuantizationParametersT()

            q = tensor.quantization

            scale_obj = getattr(q, 'scale', None)
            scale_vals = None
            if scale_obj is not None:
                try:
                    if len(scale_obj) > 0:
                        scale_vals = [float(x) for x in scale_obj]
                except Exception:
                    pass

            if not scale_vals:
                scale_vals = [1.0 / 255.0]

            zp_obj = getattr(q, 'zeroPoint', None)
            if zp_obj is None:
                zp_obj = getattr(q, 'zero_point', None)

            zp_vals = None
            if zp_obj is not None:
                try:
                    if len(zp_obj) == len(scale_vals):
                        zp_vals = [int(x) for x in zp_obj]
                except Exception:
                    pass

            if not zp_vals:
                zp_vals = [0] * len(scale_vals)

            q.scale = scale_vals
            q.zeroPoint = zp_vals
            q.zero_point = zp_vals
            q.quantizedDimension = 0
            q.quantized_dimension = 0

        elif tensor.type in [schema_fb.TensorType.FLOAT32, schema_fb.TensorType.FLOAT16]:
            tensor.quantization = None

# Pack repaired FlatBuffer
builder = flatbuffers.Builder(len(file_bytes) * 3)
packed = model_t.Pack(builder)
builder.Finish(packed, b"TFL3")
repacked_bytes = bytes(builder.Output())

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "wb") as f:
    f.write(repacked_bytes)

print(f"--- 2. Saved repaired model to {output_path} ---")

# Verify Native C++ Allocation
print("--- 3. Verifying Native C++ AllocateTensors() ---")
try:
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=output_path)
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    print(f"SUCCESS! Repaired model allocated clean in C++ engine ({len(ops)} ops total).")
except Exception as e:
    print(f"C++ Engine Error: {e}")