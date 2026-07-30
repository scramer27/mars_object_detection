import torch
from ultralytics import YOLO

# 1. Point directly to your completed 120-epoch best weights
BEST_WEIGHTS = "runs/detect/output_new/mars_yolo_rock_focus/weights/best.pt"
DATA_YAML = "data/yolo_mars/mars.yaml"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"📂 Loading pre-trained checkpoint: {BEST_WEIGHTS}")
model = YOLO(BEST_WEIGHTS)

# 2. Evaluate on Unseen Test Set
print("\n--- Evaluating Best Weights on Unseen Test Set ---")
metrics = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=256,
    batch=1,  # Single-frame inference simulation
    device=DEVICE,
    conf=0.15,
    iou=0.6,
)

print(f"\n✅ Unseen Test mAP@50:    {metrics.box.map50 * 100:.2f}%")
print(f"✅ Unseen Test mAP@50-95: {metrics.box.map * 100:.2f}%")

# 3. Export to Microchip VectorBlox-Compatible ONNX (Opset 12)
print("\n--- Exporting Model to VectorBlox ONNX (Opset 12) ---")
try:
    onnx_path = model.export(
        format="onnx",
        imgsz=256,
        opset=12,  # Required for VectorBlox
        dynamic=False,  # Static shapes required for FPGA memory
        simplify=True,  # Applies onnxslim graph optimization
    )
    print(f"\n🎉 Success! VectorBlox ONNX model saved to:\n{onnx_path}")
except Exception as e:
    print(f"⚠️ Export error: {e}")