cd /mnt/c/Users/scramer/Documents/26X/mars_object_detection

# 1. Run Python to strictly re-quantize (Step 1) and patch LOGISTIC layers (Step 2)
python3 -c '
import tensorflow as tf
import numpy as np
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

print("==> [STEP 1] Re-quantizing model to strict INT8...")
converter = tf.lite.TFLiteConverter.from_saved_model("fpga_payload/tf_out")

def representative_data_gen():
    for _ in range(100):
        yield [np.random.rand(1, 256, 256, 3).astype(np.float32)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quant_model = converter.convert()

in_path = "fpga_payload/tf_out/best_full_integer_quant.tflite"
with open(in_path, "wb") as f:
    f.write(quant_model)

print("==> [STEP 2] Patching LOGISTIC layers...")
model = schema_fb.Model.GetRootAsModel(quant_model, 0)
model_t = schema_fb.ModelT.InitFromObj(model)

op_codes = [op.builtinCode if op.builtinCode != 0 else op.deprecatedBuiltinCode for op in model_t.operatorCodes]
exact_scale = float(np.float32(1.0 / 256.0))
count = 0

for subgraph in model_t.subgraphs:
    for op in subgraph.operators:
        if op_codes[op.opcodeIndex] == schema_fb.BuiltinOperator.LOGISTIC:
            for out_idx in op.outputs:
                tensor = subgraph.tensors[out_idx]
                if tensor.quantization:
                    tensor.quantization.scale = [exact_scale]
                    tensor.quantization.zeroPoint = [-128]
                    count += 1

builder = flatbuffers.Builder(1024 * 1024)
builder.Finish(model_t.Pack(builder), file_identifier=b"TFL3")

out_path = "fpga_payload/mars_yolov8_fpga_full_integer_quant.tflite"
with open(out_path, "wb") as f:
    f.write(builder.Output())

print(f"Patched {count} LOGISTIC layer(s) successfully!")
'

# 2. Preprocess and compile for VectorBlox FPGA (Step 3)
echo "==> [STEP 3] Preprocessing and Compiling for VectorBlox V1000..."
tflite_preprocess fpga_payload/mars_yolov8_fpga_full_integer_quant.tflite --scale 255

vnnx_compile \
    -s V1000 \
    -c ncomp \
    -t fpga_payload/mars_yolov8_fpga_full_integer_quant.pre.tflite \
    -o fpga_payload/mars_yolov8.vnnx