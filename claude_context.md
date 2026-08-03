First, let's clear up where everything lives so you don't have to guess:

* **Mac:** Where you trained your model and where your dataset (`ai4mars.yaml` / images) lives.
* **Windows Laptop (WSL):** WSL (*Windows Subsystem for Linux*) is a Linux environment running inside Windows. This is where your **VectorBlox SDK** (`tflite_preprocess` and `vnnx_compile`) lives.

You do **not** need to remake the model from scratch on Windows. The real issue is **YOLOv8's anchor box post-processing head**.

---

## 🔍 Why `onnx2tf` Keeps Failing (`fully_quantize: 0`)

When you export YOLOv8 to standard ONNX, it attaches a complex post-processing "Detect" head (DFL layers) for drawing bounding boxes.

When `onnx2tf` converts that ONNX file to TFLite:

1. It **cannot** convert those anchor math operations to 8-bit integers.
2. It leaves **151 operations (37% of the model) in Float32**.
3. Node 6 (`LOGISTIC`) stays tied to Float32 activations, causing `tflite_preprocess` to crash with `output->params.scale == 1. / 256 was not true`.

---

## 🛠️ The Clean 2-Step Fix

Instead of converting ONNX $\rightarrow$ TensorFlow $\rightarrow$ TFLite via `onnx2tf`, let **Ultralytics directly handle the INT8 TFLite export on your Mac** where your PyTorch weights and dataset live. Ultralytics properly strips and quantizes the YOLOv8 head.

### Step 1: On Your MAC (Export 100% INT8 TFLite)

1. On your Mac, create a file named `export_fpga.py` in your project directory:

```python
# export_fpga.py
from ultralytics import YOLO

# 1. Load your trained PyTorch weights
model = YOLO("runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt")

# 2. Export directly to TFLite INT8 using your dataset calibration
model.export(
    format="tflite",
    int8=True,
    data="ai4mars.yaml",  # Path to your dataset yaml on Mac
    imgsz=256,
)
print("Export complete!")

```

2. Run the script in your Mac terminal:

```bash
python3 export_fpga.py

```

This creates a folder named `best_saved_model/` containing `best_full_integer_quant.tflite`. This file is **100% INT8 quantized**.

3. Push `best_full_integer_quant.tflite` to Git (or copy it over to your Windows PC):

```bash
mkdir -p fpga_payload
cp runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_saved_model/best_full_integer_quant.tflite fpga_payload/mars_yolov8_int8.tflite

git add -f fpga_payload/mars_yolov8_int8.tflite
git commit -m "Add direct INT8 TFLite model from Mac"
git push

```

---

### Step 2: On Your Windows Laptop (Inside WSL)

Now switch to your WSL terminal on Windows:

```bash
cd /mnt/c/Users/scramer/Documents/26X/mars_object_detection
git pull

# 1. Activate VectorBlox environment
source ~/VectorBlox-SDK/setup_vars.sh

# 2. Patch the LOGISTIC layer scales for VectorBlox compatibility
python3 -c '
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

in_path = "fpga_payload/mars_yolov8_int8.tflite"
out_path = "fpga_payload/mars_yolov8_patched.tflite"

with open(in_path, "rb") as f:
    buf = f.read()

model = schema_fb.Model.GetRootAsModel(buf, 0)
model_t = schema_fb.ModelT.InitFromObj(model)

op_codes = [op.builtinCode if op.builtinCode != 0 else op.deprecatedBuiltinCode for op in model_t.operatorCodes]

count = 0
for subgraph in model_t.subgraphs:
    for op in subgraph.operators:
        if op_codes[op.opcodeIndex] == schema_fb.BuiltinOperator.LOGISTIC:
            for out_idx in op.outputs:
                tensor = subgraph.tensors[out_idx]
                if tensor.quantization:
                    tensor.quantization.scale = [1.0 / 256.0]
                    tensor.quantization.zeroPoint = [-128]
                    count += 1

builder = flatbuffers.Builder(1024 * 1024)
builder.Finish(model_t.Pack(builder), file_identifier=b"TFL3")

with open(out_path, "wb") as f:
    f.write(builder.Output())

print(f"Successfully patched {count} LOGISTIC layer(s)!")
'

# 3. Preprocess for VectorBlox
tflite_preprocess fpga_payload/mars_yolov8_patched.tflite --scale 255

# 4. Compile to VNNX FPGA binary
vnnx_compile \
    -s V1000 \
    -c ncomp \
    -t fpga_payload/mars_yolov8_patched.pre.tflite \
    -o fpga_payload/mars_yolov8.vnnx

```