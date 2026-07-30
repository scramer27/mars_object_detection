import os
import torch
from ultralytics import YOLO

# Configuration
DATA_YAML = "data/yolo_mars/mars.yaml"
PROJECT_DIR = "output_clean"
RUN_NAME = "mars_yolo_fpga"
EPOCHS = 120
IMG_SIZE = 256

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Initializing Pipeline on Device: {DEVICE.upper()}")

# 1. Initialize YOLO11 Nano model
model = YOLO("yolo11n.pt")

# 2. Train the model (Optimized for rock/soil classes)
print("\n--- Phase 1: Training Model ---")
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=32,
    device=DEVICE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    box=9.5,  # Weighted box loss for small rock focus
    cls=1.5,  # Weighted class loss
    exist_ok=True,
    verbose=True,
)

# Retrieve the absolute path of the best weights safely
best_weights_path = str(model.trainer.best)
print(f"\n✅ Training Complete. Best weights saved at: {best_weights_path}")

# 3. Load Best Weights for Final Held-Out Test Evaluation
print("\n--- Phase 2: Evaluating on Held-Out Test Set ---")
best_model = YOLO(best_weights_path)

test_metrics = best_model.val(
    data=DATA_YAML,
    split="test",  # This is your untouched test set for the FPGA pipeline
    imgsz=IMG_SIZE,
    batch=1,
    device=DEVICE,
    conf=0.25,
    iou=0.6,
)

print(f"\n📊 Final Test Set Results:")
print(f"   - mAP@50:    {test_metrics.box.map50 * 100:.2f}%")
print(f"   - mAP@50-95: {test_metrics.box.map * 100:.2f}%")

# 4. Export to VectorBlox-Compatible ONNX (Opset 12, Static Shapes)
print("\n--- Phase 3: Exporting to VectorBlox ONNX (Opset 12) ---")
try:
    onnx_path = best_model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=12,          # Mandatory requirement for VectorBlox
        dynamic=False,     # Static dimensions required for FPGA fixed-memory mapping
        simplify=True,     # Graph optimization using onnxslim
    )
    print(f"\n🎉 Success! FPGA-ready ONNX model compiled to:\n{onnx_path}")
except Exception as e:
    print(f"⚠️ Export failed: {e}")