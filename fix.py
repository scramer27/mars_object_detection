from ultralytics import YOLO

# Load your best weights
model = YOLO("runs/detect/output_new/mars_yolo_rock_focus/weights/best.pt")

# Run validation with verbose output and save predictions to see what it's doing
metrics = model.val(
    data="data/yolo_mars/mars.yaml",
    split="test",
    imgsz=256,
    conf=0.01,  # Lower confidence to see if anything pops up
    iou=0.6,
    save_json=False,
    plots=True,  # Generates confusion matrix and validation batch images
)

print(f"Loaded class names: {model.names}")