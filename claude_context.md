Fair point. The issue isn't Git anymore—it's a compatibility wall between Ultralytics and VectorBlox.

---

## 🔍 The Root Cause

Look at the warning Ultralytics output on your Mac during export:

> `WARNING ⚠️ format='tflite' is deprecated as of 8.4.83 and has been replaced by the unified Google LiteRT format. Exporting format='litert' instead.`

In Ultralytics $\ge$ 8.4.83, exporting to `tflite` secretly routes through Google's new **`litert_torch`** backend. LiteRT generates FlatBuffers with newer 64-bit buffer offsets.

When Microchip VectorBlox's `tflite_preprocess` script tries to parse this new format in Python using `tf.lite.Interpreter(..., experimental_preserve_all_tensors=True)`, standard C++ TensorFlow inside VectorBlox chokes and throws:
`ValueError: Constant buffer 84 specified an out of range offset`

---

## 🛠️ The Fix: Export ONNX → Convert via VectorBlox (`onnx2tf`)

Microchip's VectorBlox SDK expects standard TensorFlow Lite flatbuffers. The official Microchip workflow converts YOLO models via **ONNX** using `onnx2tf` (which is pre-installed inside VectorBlox's `vbx_env`).

### Step 1: Export ONNX on your Mac & Push

Run this in your **Mac terminal**:

```bash
cd /Users/scramer/Documents/mars_object_detection

# 1. Export standard ONNX model (opset 12 is optimal for VectorBlox)
yolo export model=runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt format=onnx opset=12 simplify=True

# 2. Copy to fpga_payload and push
mkdir -p fpga_payload
cp runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.onnx fpga_payload/best.onnx

git add -f fpga_payload/best.onnx
git commit -m "Add ONNX model for VectorBlox onnx2tf conversion"
git push

```

---

### Step 2: Convert to Classic TFLite & Compile in WSL

Switch to your **WSL terminal** and run this sequence:

```bash
cd /mnt/c/Users/scramer/Documents/26X/mars_object_detection
git pull

# 1. Activate VectorBlox environment
source ~/VectorBlox-SDK/setup_vars.sh

# 2. Convert ONNX -> Classic INT8 TFLite using VectorBlox's built-in tool
onnx2tf -i fpga_payload/best.onnx -o fpga_payload/tf_out -oiqt

# 3. Copy the generated INT8 model to the expected filename
cp fpga_payload/tf_out/*_full_integer_quant.tflite fpga_payload/mars_yolov8_fpga_full_integer_quant.tflite

# 4. Preprocess and compile for FPGA
tflite_preprocess fpga_payload/mars_yolov8_fpga_full_integer_quant.tflite --scale 255

vnnx_compile \
    -s V1000 \
    -c ncomp \
    -t fpga_payload/mars_yolov8_fpga_full_integer_quant.pre.tflite \
    -o fpga_payload/mars_yolov8.vnnx

```

`onnx2tf` outputs standard TFLite buffers, allowing `tflite_preprocess` to parse the file without offset errors.