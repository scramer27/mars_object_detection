Using Git is much cleaner and eliminates any manual file copying! Since the 3.2 MB file is well under GitHub's 100 MB limit, standard Git commands will work seamlessly.

Here is the 2-part sequence:

---

### 🍏 Step 1: Commit & Push from your Mac

Run this in your **Mac terminal**:

```bash
cd /Users/scramer/Documents/mars_object_detection

# 1. Copy the new file into the payload folder
mkdir -p fpga_payload
cp runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best_int8.tflite fpga_payload/mars_yolov8_fpga_full_integer_quant.tflite

# 2. Stage, commit, and push (-f overrides any .gitignore rules for .tflite files)
git add -f fpga_payload/mars_yolov8_fpga_full_integer_quant.tflite
git commit -m "Add fresh INT8 TFLite export from Mac"
git push

```

---

### 🪟 Step 2: Pull & Compile in WSL

Switch to your **WSL terminal** and run this:

```bash
# 1. Go to repo and pull the fresh file
cd /mnt/c/Users/scramer/Documents/26X/mars_object_detection
git pull

# 2. Activate VectorBlox environment
source ~/VectorBlox-SDK/setup_vars.sh

# 3. Preprocess for VectorBlox
tflite_preprocess \
    fpga_payload/mars_yolov8_fpga_full_integer_quant.tflite \
    --scale 255

# 4. Compile into the FPGA binary
vnnx_compile \
    -s V1000 \
    -c ncomp \
    -t fpga_payload/mars_yolov8_fpga_full_integer_quant.pre.tflite \
    -o fpga_payload/mars_yolov8.vnnx

```

---

### Check Your Result

Once `vnnx_compile` finishes, verify your output:

```bash
ls -lh fpga_payload/mars_yolov8.vnnx

```

You should see your compiled `.vnnx` FPGA binary ready to deploy!