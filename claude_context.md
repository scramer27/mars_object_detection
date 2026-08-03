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


st")
FileNotFoundError: 'mars.yaml' does not exist
(mars_object_detection) scramer@MT-400226 mars_object_detection % python export_fpga.py
WARNING ⚠️ 'int8' is deprecated and will be removed in the future. Use 'quantize' instead.
WARNING ⚠️ format='tflite' is deprecated as of 8.4.83 and has been replaced by the unified Google LiteRT format. Exporting format='litert' instead. See https://docs.ultralytics.com/integrations/litert/
Ultralytics 8.4.112 🚀 Python-3.11.15 torch-2.12.1 CPU (Apple M1 Max)
WARNING ⚠️ LiteRT INT8 export does not support end2end models, disabling end2end branch.
Model summary (fused): 73 layers, 3,006,428 parameters, 0 gradients, 1.3 GFLOPs

PyTorch: starting from 'runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt' with input shape (1, 3, 256, 256) BCHW and output shape(s) (1, 8, 1344) (5.9 MB)
LiteRT: collecting INT8 calibration images from 'data=ai4mars.yaml'
ERROR ❌ LiteRT: export failure 0.0s: 'ai4mars.yaml' does not exist
Export to tflite in the cloud with Ultralytics Platform: https://platform.ultralytics.com
Traceback (most recent call last):
  File "/Users/scramer/Documents/mars_object_detection/export_fpga.py", line 7, in <module>
    model.export(
  File "/Users/scramer/Documents/mars_object_detection/.venv/lib/python3.11/site-packages/ultralytics/engine/model.py", line 769, in export
    return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/scramer/Documents/mars_object_detection/.venv/lib/python3.11/site-packages/ultralytics/engine/exporter.py", line 945, in __call__
    f = getattr(self, f"export_{fmt}")()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/scramer/Documents/mars_object_detection/.venv/lib/python3.11/site-packages/ultralytics/engine/exporter.py", line 484, in outer_func
    f = inner_func(*args, **kwargs)  # exported file/dir or tuple of (file/dir, *)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/scramer/Documents/mars_object_detection/.venv/lib/python3.11/site-packages/ultralytics/engine/exporter.py", line 1223, in export_litert
    calibration_dataset=self.get_int8_calibration_dataloader(prefix)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/scramer/Documents/mars_object_detection/.venv/lib/python3.11/site-packages/ultralytics/engine/exporter.py", line 989, in get_int8_calibration_dataloader
    data = check_det_dataset(self.args.data, split=self.args.split)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/scramer/Documents/mars_object_detection/.venv/lib/python3.11/site-packages/ultralytics/data/utils.py", line 497, in check_det_dataset
    file = Path(check_file(dataset))
                ^^^^^^^^^^^^^^^^^^^
  File "/Users/scramer/Documents/mars_object_detection/.venv/lib/python3.11/site-packages/ultralytics/utils/checks.py", line 720, in check_file
    raise FileNotFoundError(f"'{file}' does not exist")
FileNotFoundError: 'ai4mars.yaml' does not exist
(mars_object_detection) scramer@MT-400226 mars_object_detection % 


The script crashed because `ai4mars.yaml` isn't directly in your root folder—it's likely buried inside a subfolder (like `data/yolo_mars/mars.yaml`).

Update your `export_fpga.py` script on your **Mac** with this version. It will automatically search your project for your dataset YAML file so you don't have to hunt down the exact path manually.

---

### 🍏 Step 1: Update `export_fpga.py` on your Mac

Replace the contents of `/Users/scramer/Documents/mars_object_detection/export_fpga.py` with:

```python
import glob
import os
from ultralytics import YOLO

# 1. Automatically find your dataset YAML file
yaml_files = glob.glob("**/*.yaml", recursive=True)
# Filter out venv or hidden folders
yaml_files = [f for f in yaml_files if ".venv" not in f and "venv" not in f]

if not yaml_files:
    raise FileNotFoundError("Could not find any dataset .yaml file in your project!")

# Prioritize files with 'mars' or 'data' in the name
dataset_yaml = yaml_files[0]
for y in yaml_files:
    if "mars" in y.lower() or "ai4mars" in y.lower():
        dataset_yaml = y
        break

print(f"--> Found calibration dataset config: {dataset_yaml}")

# 2. Load PyTorch model weights
weights = "runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt"
model = YOLO(weights)

# 3. Export to INT8 TFLite
model.export(
    format="tflite",
    int8=True,
    data=dataset_yaml,
    imgsz=256,
)

print("--> Export complete!")

```

---

### 🍏 Step 2: Run the Export on Mac & Push to WSL

In your **Mac terminal**:

```bash
python export_fpga.py

```

Once it finishes successfully, copy the new file into your `fpga_payload` folder and push it to Git:

```bash
mkdir -p fpga_payload

# Locate the newly exported tflite file
cp runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_saved_model/best_full_integer_quant.tflite fpga_payload/mars_yolov8_int8.tflite

# Push to Git for WSL to download
git add -f fpga_payload/mars_yolov8_int8.tflite
git commit -m "Add valid INT8 TFLite model from Mac export"
git push

```

---

### 🪟 Step 3: Patch & Compile in WSL (Windows)

Switch to your **WSL terminal**:

```bash
cd /mnt/c/Users/scramer/Documents/26X/mars_object_detection
git pull

source ~/VectorBlox-SDK/setup_vars.sh

# Patch LOGISTIC scales
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

# Preprocess and compile to .vnnx
tflite_preprocess fpga_payload/mars_yolov8_patched.tflite --scale 255

vnnx_compile \
    -s V1000 \
    -c ncomp \
    -t fpga_payload/mars_yolov8_patched.pre.tflite \
    -o fpga_payload/mars_yolov8.vnnx

```