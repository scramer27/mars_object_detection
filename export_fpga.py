from ultralytics import YOLO

# 1. Load your trained PyTorch weights
model = YOLO("runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt")

# 2. Export directly to TFLite INT8 using your dataset calibration
model.export(
    format="tflite",
    int8=True,
    data="ai4mars.yaml",  # Path to your dataset yaml on Mac
    imgsz=256,
)
print("Export complete!")