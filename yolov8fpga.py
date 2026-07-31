#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------------------
# 1️⃣ SETTINGS – adjust only paths if you move the repo
# --------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_YAML  = BASE_DIR / "data" / "yolo_mars" / "mars.yaml"
PROJECT    = BASE_DIR / "runs" / "detect" / "output_yolov8_fpga"
RUN_NAME   = "mars_yolov8n_fpga"

IMG_SIZE   = 256          # the size you used for YOLO‑11
EPOCHS     = 120
PATIENCE   = 30
BATCH      = 32

# --------------------------------------------------------------
# 2️⃣ DEVICE – use MPS > CUDA > CPU
# --------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"🚀 Training on {DEVICE.upper()}")

# --------------------------------------------------------------
# 3️⃣ LOAD the official YOLO‑8 Nano checkpoint
# --------------------------------------------------------------
model = YOLO("yolov8n.pt")   # <-- this is the only line that differs from YOLO‑11

# --------------------------------------------------------------
# 4️⃣ TRAIN – keep the custom loss weights & augmentations you used for YOLO‑11
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
    exist_ok=False,

    # ---- custom loss weighting (same values you used for YOLO‑11) ----
    box=9.5,
    cls=1.5,
    dfl=2.0,

    # ---- data‑augmentation (same as YOLO‑11) ----
    mosaic=1.0,
    scale=0.7,
    mixup=0.15,
    degrees=10.0,
    fliplr=0.5,

    # ---- validation & plots (helpful for debugging) ----
    val=True,
    plots=True,
)

# --------------------------------------------------------------
# 5️⃣ EXPORT to ONNX (static shapes, opset 12 – exactly what the SDK expects)
# --------------------------------------------------------------
best_pt = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
print(f"✅ Best checkpoint saved to {best_pt}")

# Export in one command – the `model.export` call returns the file path
onnx_path = model.export(
    format="onnx",
    imgsz=IMG_SIZE,
    opset=12,       # required by VectorBlox
    dynamic=False,  # static shapes → no reshapes that the compiler can't handle
    simplify=True,
)
print(f"✅ ONNX model written to {onnx_path}")