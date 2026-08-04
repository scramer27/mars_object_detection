import torch
from pathlib import Path
from ultralytics import YOLO

def train_and_export_yolo():
    BASE_DIR = Path(__file__).resolve().parent
    YAML_PATH = BASE_DIR / "data" / "yolo_mars" / "mars.yaml"
    OUTPUT_DIR = BASE_DIR / "output_2"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check Device (Apple Silicon Metal MPS)
    if torch.backends.mps.is_available():
        device = "mps"
        print(" Using Hardware Acceleration: Apple Silicon MPS (Metal)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(" Using Hardware Acceleration: NVIDIA CUDA GPU")
    else:
        device = "cpu"
        print(" Using Hardware Device: CPU")

    # Load VectorBlox-Compatible YOLOv8 Nano Pretrained Model
    model = YOLO("yolov8n.pt")

    print("\n=== Starting YOLOv8 Object Detection Training ===")
    
    # Train Detector
    results = model.train(
        data=str(YAML_PATH),
        epochs=20,
        imgsz=256,
        batch=32,
        device=device,
        project=str(OUTPUT_DIR),
        name="mars_yolo_run",
        exist_ok=True,
        verbose=True
    )

    print("\n=== Evaluating on Unseen Test Set ===")
    metrics = model.val(data=str(YAML_PATH), split="test", device=device)
    
    print(f"\n Overall Test Mean Average Precision (mAP@50):    {metrics.box.map50 * 100:.2f}%")
    print(f" Overall Test Mean Average Precision (mAP@50-95): {metrics.box.map * 100:.2f}%")

    # Export to ONNX Opset 18 for PolarFire VectorBlox Compiler
    print("\n=== Exporting Best YOLO Model to ONNX for VectorBlox ===")
    onnx_path = model.export(
        format="onnx",
        opset=18,
        imgsz=256,
        dynamic=False,
        simplify=True
    )

    print(f" Successfully exported object detection model to ONNX: {onnx_path}")


if __name__ == "__main__":
    train_and_export_yolo()