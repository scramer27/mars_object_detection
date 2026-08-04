#!/usr/bin/env python3
"""
Clean YOLOv8 INT8 Export for VectorBlox VNNX Compilation
Exports without NMS post-processing to avoid VectorBlox optimizer bugs
"""
import os
import sys
from ultralytics import YOLO

# 1. Locate PyTorch weights
pt_path = "fpga_payload/best.pt"
if not os.path.exists(pt_path):
    pt_path = "best.pt"
if not os.path.exists(pt_path):
    print("WARNING: Mars weights not found. Downloading default yolov8n.pt for testing...")
    pt_path = "yolov8n.pt"

print(f"Loading model from: {pt_path}")
model = YOLO(pt_path)

# 2. Export clean static TFLite INT8 model
# CRITICAL: nms=False prevents injection of custom dynamic post-processing layers that crash VectorBlox
print("\n=== Exporting static INT8 TFLite model without NMS ===")
print("Parameters:")
print("  - format: tflite")
print("  - int8: True")
print("  - imgsz: [416, 416]")
print("  - batch: 1")
print("  - nms: False (CRITICAL - avoids VectorBlox optimizer bugs)")
print("")

try:
    result = model.export(
        format="tflite",
        int8=True,
        imgsz=416,
        batch=1,
        data="calib_images/coco128/coco128.yaml" if os.path.exists("calib_images/coco128/coco128.yaml") else None,
        nms=False,  # CRITICAL: Prevents dynamic NMS layers that VectorBlox can't compile
        simplify=True
    )
    print(f"\n=== Export complete ===")
    print(f"Output: {result}")
except Exception as e:
    print(f"\nERROR during export: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
