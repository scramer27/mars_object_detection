import os
import torch
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION & DEVICE SETUP
# ==========================================
DATA_YAML = "data/yolo_mars/mars.yaml"  # Path to your dataset YAML
BASE_MODEL = "yolov8n.pt"
PROJECT_DIR = "output_new"  # Outputs saved to output_new directory
RUN_NAME = "mars_yolo_rock_focus"

# Detect acceleration hardware (Apple Silicon GPU -> mps, CUDA -> cuda, or cpu)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"🚀 Initializing Training Pipeline on Device: {DEVICE.upper()}")


# ==========================================
# 2. TRAINING PHASE (ROCK-OPTIMIZED)
# ==========================================
def run_training():
    model = YOLO(BASE_MODEL)

    print("\n--- Starting Model Training (120 Epochs with Rock Focus) ---")
    results = model.train(
        # --- Core Settings ---
        data=DATA_YAML,
        epochs=120,  # Full convergence window
        patience=30,  # Early stopping checkpointing
        imgsz=256,  # Native input size for VectorBlox
        batch=32,  # Stabilizes BatchNorm on M1 Max GPU
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        # --- Loss Weights (Rock Detection Fix) ---
        box=9.5,  # Penalizes tiny bounding box positioning errors
        cls=1.5,  # Increases penalty for misclassifying rare rock classes
        dfl=2.0,  # Distribution Focal Loss: sharpens object boundary edges
        # --- Augmentations for Small & Rare Objects ---
        copy_paste=0.5,  # Synthesizes extra rock instances across images
        mosaic=1.0,  # Combines 4 tiles to vary spatial scale
        scale=0.7,  # Zooms +/- 70% to handle varying camera distances
        mixup=0.15,  # Prevents overfitting on continuous bedrock textures
        degrees=10.0,  # Minor rotational variance
        fliplr=0.5,  # Horizontal mirroring
        # --- Validation Settings ---
        val=True,
        plots=True,
    )
    return model


# ==========================================
# 3. UNSEEN TEST SET EVALUATION
# ==========================================
def evaluate_test_set(best_weights_path):
    print("\n--- Evaluating Best Weights on Unseen Test Set ---")
    best_model = YOLO(best_weights_path)

    # conf=0.15 captures subtle small-object detections previously missed
    metrics = best_model.val(
        data=DATA_YAML,
        split="test",
        imgsz=256,
        batch=1,  # Simulates single-frame real-time FPGA inference
        device=DEVICE,
        conf=0.15,  # Optimal evaluation confidence threshold
        iou=0.6,  # NMS IoU threshold
    )

    print(f"\n✅ Unseen Test mAP@50:    {metrics.box.map50 * 100:.2f}%")
    print(f"✅ Unseen Test mAP@50-95: {metrics.box.map * 100:.2f}%")
    return best_model


# ==========================================
# 4. VECTORBLOX-COMPATIBLE ONNX EXPORT
# ==========================================
def export_to_onnx_vectorblox(model):
    print(
        "\n--- Exporting Best Model to VectorBlox-Compatible ONNX (Opset 12) ---"
    )
    try:
        onnx_path = model.export(
            format="onnx",
            imgsz=256,
            opset=12,  # Mandatory Opset 12 for VectorBlox / PolarFire SoC
            dynamic=False,  # Static shapes required for FPGA memory allocation
            simplify=True,  # Triggers onnxslim graph optimization
        )
        print(f"🎉 Export Successful! VectorBlox ONNX saved to: {onnx_path}")
    except Exception as e:
        print(f"⚠️ Export encountered an issue: {e}")


# ==========================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    # 1. Train the model
    trained_model = run_training()

    # 2. Identify best checkpoint weights
    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")

    # 3. Evaluate best model on unseen test set
    best_model = evaluate_test_set(best_weights)

    # 4. Export static ONNX model for Microchip VectorBlox
    export_to_onnx_vectorblox(best_model)