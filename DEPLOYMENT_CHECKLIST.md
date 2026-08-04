# FPGA Deployment Checklist

## Training on Mac → Deployment on Windows → FPGA

### Phase 1: Mac Training ✓

**Location:** `~/Documents/mars_object_detection/`

```bash
# 1. Train model (45-75 min on M1 Max)
python train_do_it_2.py

# 2. Verify training completed
ls output_new/mars_yolov8_rock_focus/weights/best.pt

# 3. Check results
open output_new/mars_yolov8_rock_focus/results.png
```

**Success criteria:**
- [x] Dataset created (2100 train, 450 val, 450 test)
- [ ] Training completes 120 epochs
- [ ] mAP@50 > 40%
- [ ] `best.pt` model saved

---

### Phase 2: Transfer to Windows

**Files to transfer:**

1. **Trained model** (required):
   ```
   output_new/mars_yolov8_rock_focus/weights/best.pt → fpga_payload/best.pt
   ```

2. **Training results** (optional, for reference):
   ```
   output_new/mars_yolov8_rock_focus/results.png
   output_new/mars_yolov8_rock_focus/confusion_matrix.png
   ```

**Transfer methods:**
- USB drive
- GitHub LFS (large file storage)
- Cloud storage (OneDrive, Google Drive)
- Network share

**On Windows, verify:**
```cmd
dir fpga_payload\best.pt
# Should be ~6 MB
```

---

### Phase 3: FPGA Test Set Preparation (Windows)

**Test images manifest:** `fpga_test_manifest.txt` (50 images reserved for FPGA validation)

**These images were NOT used for training** - they're from the test split.

**Already prepared:**
- ✓ 50 test images in `fpga_payload/raw_inputs/` (binary format for FPGA)
- ✓ Ground truth labels in `fpga_payload/labels/` (YOLO format)
- ✓ Manifest file tracks which images to use

---

### Phase 4: Windows Inference Verification

**Before FPGA deployment, verify on Windows CPU:**

```cmd
# Run inference on test set
python -c "
from ultralytics import YOLO
model = YOLO('fpga_payload/best.pt')
results = model('archive/test_data/test_images_reconstructed/', conf=0.30, imgsz=256)
print('Inference successful on', len(results), 'images')
"
```

**Success criteria:**
- [ ] Model loads without errors
- [ ] Inference completes on all 50 test images
- [ ] Detection quality looks reasonable

---

### Phase 5: WSL VNNX Compilation

**Location:** WSL Ubuntu environment

```bash
cd ~/build_temp

# Copy model from Windows
cp /mnt/c/Users/scramer/Documents/26X/mars_object_detection/fpga_payload/best.pt .

# Export to TFLite (with nms=False for VectorBlox)
python << 'EOF'
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='tflite', int8=True, imgsz=416, nms=False)
EOF

# Compile to VNNX
source /home/scramer/VectorBlox-SDK/setup_vars.sh
vnnx_compile -t best_full_integer_quant.tflite -s V1000 -c comp -o mars_yolov8.vnnx
```

**Success criteria:**
- [ ] TFLite export succeeds (no NMS errors)
- [ ] VNNX compilation completes without single-input ADD errors
- [ ] Output file: `mars_yolov8.vnnx` (~7-8 MB)

---

### Phase 6: FPGA Deployment Files

**Final package to transfer to PolarFire:**

```
fpga_deployment/
├── mars_yolov8.vnnx              # FPGA binary
├── fpga_test_manifest.txt        # List of test images
├── fpga_payload/
│   ├── raw_inputs/               # 50 test images (binary format)
│   └── labels/                   # Ground truth labels
```

---

## Quick Reference Commands

### Mac (Training)
```bash
python train_do_it_2.py
```

### Windows (Verification)
```cmd
python -c "from ultralytics import YOLO; YOLO('fpga_payload/best.pt')('archive/test_data/test_images_reconstructed/', conf=0.30, imgsz=256)"
```

### WSL (VNNX Compilation)
```bash
source /home/scramer/VectorBlox-SDK/setup_vars.sh
vnnx_compile -t best_full_integer_quant.tflite -s V1000 -c comp -o mars_yolov8.vnnx
```

### PolarFire (Inference)
```c
vbx_model_t* model = vbx_model_load("mars_yolov8.vnnx");
vbx_model_run(model, input_buffer, output_buffer);
```

---

## Critical Files (Do Not Lose)

1. **`fpga_payload/best.pt`** - Trained model (6 MB)
2. **`fpga_test_manifest.txt`** - Test image list (50 images)
3. **`fpga_payload/labels/*.txt`** - Ground truth for validation
4. **`mars_yolov8.vnnx`** - Final FPGA binary (7-8 MB)

---

Last updated: 2026-08-04
