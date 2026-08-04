#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------------------
# 1️⃣ SETTINGS – Adjusted for High-RAM Apple Silicon Hardware
# --------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_YAML  = BASE_DIR / "data" / "yolo_mars" / "mars.yaml"
PROJECT    = BASE_DIR / "runs" / "detect" / "output_yolov8_fpga"
RUN_NAME   = "mars_yolov8n_fpga"

IMG_SIZE   = 256          # Standard VectorBlox resolution
EPOCHS     = 120
PATIENCE   = 30
BATCH      = 32           # High memory bandwidth allows batch size 32 or 64 smoothly

# --------------------------------------------------------------
# 2️⃣ DEVICE & ACCELERATION SETUP
# --------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"🚀 Launching Pipeline on Hardware Device: {DEVICE.upper()}")

# --------------------------------------------------------------
# 3️⃣ LOAD BASE CHECKPOINT
# --------------------------------------------------------------
model = YOLO("yolov8n.pt")

# --------------------------------------------------------------
# 4️⃣ TRAIN – Optimized for Mac Pro (RAM Caching & MPS Performance)
# --------------------------------------------------------------
model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    project=str(PROJECT),
    name=RUN_NAME,
    exist_ok=True,

    # ---- Hardware Performance Tweaks for 64GB Mac Pro ----
    cache="ram",    # Loads the entire dataset into RAM for instant epoch iteration
    workers=8,      # Utilizes M-series CPU performance cores for background loading
    half=True,      # Enables FP16 mixed precision on Metal/MPS for faster ops

    # ---- Custom Loss Weighting (Rock & Terrain Focus) ----
    box=9.5,
    cls=1.5,
    dfl=2.0,

    # ---- Data Augmentations ----
    mosaic=1.0,
    scale=0.7,
    mixup=0.15,
    degrees=10.0,
    fliplr=0.5,

    # ---- Validation & Logging ----
    val=True,
    plots=True,
    verbose=True,
)

# --------------------------------------------------------------
# 5️⃣ EVALUATE & EXPORT BEST WEIGHTS TO VECTORBLOX ONNX
# --------------------------------------------------------------
best_pt = PROJECT / RUN_NAME / "weights" / "best.pt"
print(f"\n✅ Training finished. Loading best checkpoint: {best_pt}")

# Load explicit best weights checkpoint
best_model = YOLO(str(best_pt))

# Run quick evaluation on the test split
print("\n--- Running Test Set Evaluation ---")
test_metrics = best_model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=IMG_SIZE,
    batch=1,
    device=DEVICE,
    conf=0.15,
    iou=0.6,
)

print(f"\n📊 Test Results:")
print(f"   - mAP@50:    {test_metrics.box.map50 * 100:.2f}%")
print(f"   - mAP@50-95: {test_metrics.box.map * 100:.2f}%")

# Export to Static VectorBlox ONNX (Opset 12)
print("\n--- Exporting to VectorBlox ONNX (Opset 12) ---")
try:
    onnx_path = best_model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=12,       # VectorBlox requirement
        dynamic=False,  # Static memory layout for FPGA allocation
        simplify=True,  # Graph optimization pass
    )
    print(f"🎉 Success! FPGA-ready ONNX model compiled to:\n{onnx_path}")
except Exception as e:
    print(f"⚠️ Export encountered an issue: {e}")