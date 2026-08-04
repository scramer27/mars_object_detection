# VNNX Compilation Success Report

**Date:** 2026-08-04  
**Target Hardware:** Microchip VectorBlox V1000 FPGA Core  
**Model:** Mars YOLOv8 Object Detection (best.pt)

---

## Problem Summary

VectorBlox SDK's internal graph optimizer (`transform_tflite.py`) was creating malformed single-input ADD operators when processing YOLOv8 models exported with dynamic NMS post-processing heads. The optimizer's pattern matching and operator decomposition passes generated invalid FlatBuffer structures.

## Solution: Clean Export Without NMS

Instead of patching VectorBlox SDK internals, we exported a **clean static CNN backbone** by disabling NMS during TFLite export:

```python
model.export(
    format="tflite",
    int8=True,
    imgsz=416,
    nms=False  # ← KEY FIX: Prevents dynamic post-processing injection
)
```

## Files Generated

### Source Model
- **Input:** `fpga_payload/best.pt` (6.0 MB, trained Mars YOLOv8 weights)

### Intermediate TFLite Models
- `fpga_payload/best_saved_model/best_float32.tflite` (12 MB, float32)
- `fpga_payload/best_saved_model/best_float16.tflite` (5.9 MB, float16)
- `fpga_payload/best_saved_model/best_int8.tflite` (3.0 MB, INT8 weights/activations, float I/O)
- `fpga_payload/best_saved_model/best_full_integer_quant.tflite` (3.0 MB, **full INT8**) ✓ **USED**

### Final VNNX Binary
- **Output:** `mars_yolov8.vnnx` (7.4 MB)
- **Status:** ✅ Compiled successfully
- **Verification:** ✅ Loads in VectorBlox simulator
- **Input/Output:** 4 input tensors, 3 output tensors

---

## Compilation Command

```bash
cd ~/build_temp
source /home/scramer/VectorBlox-SDK/setup_vars.sh
vnnx_compile -t fpga_payload/best_saved_model/best_full_integer_quant.tflite \
             -s V1000 \
             -c comp \
             -o mars_yolov8.vnnx
```

**Result:** SUCCESS (no operator decomposition errors)

---

## Key Differences from Previous Attempts

| Aspect | Previous (Failed) | Current (Success) |
|--------|------------------|-------------------|
| NMS Processing | Enabled (default) | **Disabled** (`nms=False`) |
| SDK Modifications | Patched `transform_tflite.py` | **No SDK changes** |
| Export Location | macOS → WSL transfer | **Direct WSL export** |
| Quantization | Manual patching | **Native Ultralytics INT8** |
| Calibration Data | Remote dataset | **Local COCO128 (128 images)** |

---

## Next Steps

### 1. Deploy to FPGA
Transfer `mars_yolov8.vnnx` to the Microchip PolarFire FPGA and load via VectorBlox runtime.

### 2. Implement Post-Processing
Since NMS is disabled, implement Non-Maximum Suppression in firmware:
```c
// Pseudo-code for on-chip NMS
void apply_nms(detection_t* detections, int num_dets, float iou_threshold) {
    // Sort by confidence
    // Suppress overlapping boxes with IoU > threshold
}
```

### 3. Performance Benchmarking
- Measure inference latency on V1000 hardware
- Profile memory bandwidth utilization
- Compare accuracy vs. PyTorch baseline

### 4. Integration
- Update firmware to call VectorBlox inference API
- Add pre/post-processing for 416×416 input normalization
- Implement detection visualization overlay

---

## Calibration Dataset

**Location:** `~/build_temp/calib_images/coco128/`  
**Source:** COCO128 subset (128 images)  
**Usage:** INT8 quantization calibration during TFLite export

---

## Environment Details

- **OS:** WSL2 Ubuntu on Windows 11 Enterprise
- **Python:** 3.x with `ultralytics`, `tensorflow`, VectorBlox SDK venv
- **VectorBlox SDK:** `/home/scramer/VectorBlox-SDK/`
- **Working Directory:** `~/build_temp/`

---

## Lessons Learned

1. **Vendor SDK black boxes should not be patched.** Proprietary graph optimizers have complex internal state that makes monkey-patching unreliable.

2. **Export simplicity beats post-hoc fixing.** Disabling dynamic post-processing (`nms=False`) produced a clean CNN backbone that compiled without issues.

3. **Local calibration is essential.** Having a small (100-image) calibration dataset on WSL eliminated cross-platform transfer friction.

4. **Full integer quantization is required.** VectorBlox V1000 only accepts models with INT8 inputs/outputs, not float32 I/O with INT8 internals.

---

## Success Metrics

✅ VNNX compilation completed without errors  
✅ No single-input ADD operator warnings  
✅ Model loads in VectorBlox Python simulator  
✅ File size reasonable (7.4 MB for deployment)  
✅ Clean SDK baseline (no temporary hacks)

---

**Status:** READY FOR FPGA DEPLOYMENT
