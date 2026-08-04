#!/usr/bin/env python3
"""
Mars Object Detection - Live Inference Viewer (Windows Version)

Displays side-by-side comparison of ground truth labels vs model predictions
for the Mars terrain detection task.

Classes: Soil, Bedrock, Sand, Big Rock
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. PATHS - Windows-compatible
# ==============================================================================
WEIGHTS_PATH = "fpga_payload/best.pt"
TEST_IMAGES_DIR = "test_images_reconstructed"
TEST_LABELS_DIR = "test_labels_reconstructed"

# ==============================================================================
# 2. CLASS DEFINITIONS & COLOR PALETTE
# ==============================================================================
class_names = ["Soil", "Bedrock", "Sand", "Big Rock"]

# BGR Colors (OpenCV format)
CLASS_COLORS = {
    0: (0, 165, 255),    # Soil: Orange
    1: (255, 255, 0),    # Bedrock: Cyan
    2: (0, 230, 255),    # Sand: Yellow
    3: (255, 0, 255),    # Big Rock: Magenta
}

# ==============================================================================
# 3. LOAD MODEL
# ==============================================================================
if not os.path.exists(WEIGHTS_PATH):
    print(f"❌ Model not found: {WEIGHTS_PATH}")
    print("   Expected: fpga_payload/best.pt (trained Mars YOLOv8 model)")
    exit(1)

if not os.path.exists(TEST_IMAGES_DIR):
    print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
    print("   Run 'python reconstruct_test_images.py' first to generate test images")
    exit(1)

print(f"🚀 Loading Mars YOLOv8 Model: {WEIGHTS_PATH}")
try:
    model = YOLO(WEIGHTS_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Load test image files
valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
image_files = sorted([
    os.path.join(TEST_IMAGES_DIR, f)
    for f in os.listdir(TEST_IMAGES_DIR)
    if f.endswith(valid_extensions)
])

if not image_files:
    print(f"❌ No images found in {TEST_IMAGES_DIR}")
    print("   Run 'python reconstruct_test_images.py' first")
    exit(1)

print(f"📸 Found {len(image_files)} test images")
print()

# ==============================================================================
# 4. VISUALIZATION HELPERS
# ==============================================================================
def draw_crisp_box(img, x1, y1, x2, y2, label_text, color):
    """Draws anti-aliased bounding boxes with filled text labels."""
    # Draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Measure text size for label background
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    # Position label above box if space available, otherwise inside
    pill_y1 = max(0, y1 - th - 10)
    pill_y2 = y1 if y1 - th - 10 >= 0 else y1 + th + 10

    # Draw filled label background
    cv2.rectangle(img, (x1, pill_y1), (x1 + tw + 10, pill_y2), color, -1, cv2.LINE_AA)

    # Draw text
    text_y = pill_y2 - 5 if y1 - th - 10 >= 0 else pill_y2 - 3
    cv2.putText(img, label_text, (x1 + 5, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def get_ground_truth_view(img_raw, label_path, target_size=(700, 700)):
    """Generate ground truth visualization panel."""
    img_disp = cv2.resize(img_raw, target_size, interpolation=cv2.INTER_CUBIC)
    h_disp, w_disp = img_disp.shape[:2]

    # Draw ground truth boxes from label file
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])

                    # Convert normalized YOLO format to pixel coordinates
                    x1 = int((xc - bw / 2) * w_disp)
                    y1 = int((yc - bh / 2) * h_disp)
                    x2 = int((xc + bw / 2) * w_disp)
                    y2 = int((yc + bh / 2) * h_disp)

                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                    label_str = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    draw_crisp_box(img_disp, x1, y1, x2, y2, label_str, color)
    else:
        # No ground truth available
        cv2.putText(img_disp, "No ground truth", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    return img_disp

def get_prediction_view(model, img_path, img_raw, target_size=(700, 700), conf_thresh=0.30):
    """Generate model prediction visualization panel."""
    img_disp = cv2.resize(img_raw, target_size, interpolation=cv2.INTER_CUBIC)
    h_disp, w_disp = img_disp.shape[:2]

    # Run inference at 256x256 (FPGA deployment resolution)
    results = model(img_path, conf=conf_thresh, imgsz=256, verbose=False)[0]

    # Draw predicted boxes
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            # Extract normalized coordinates
            xywhn = box.xywhn[0].cpu().numpy()
            xc, yc, bw, bh = xywhn

            # Extract class and confidence
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            # Convert to pixel coordinates
            x1 = int((xc - bw / 2) * w_disp)
            y1 = int((yc - bh / 2) * h_disp)
            x2 = int((xc + bw / 2) * w_disp)
            y2 = int((yc + bh / 2) * h_disp)

            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            c_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            label_str = f"{c_name} {conf:.2f}"
            draw_crisp_box(img_disp, x1, y1, x2, y2, label_str, color)
    else:
        # No detections
        cv2.putText(img_disp, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    return img_disp

# ==============================================================================
# 5. MAIN APPLICATION LOOP
# ==============================================================================
print("🎬 Starting live inference viewer...")
print("   Controls:")
print("   - D: Next image")
print("   - A: Previous image")
print("   - Q: Quit")
print()

idx = 0
window_name = "Mars Object Detection - Ground Truth vs Prediction"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1400, 750)

try:
    while True:
        # Load current image
        img_path = image_files[idx]
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(TEST_LABELS_DIR, base_name + ".txt")

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            print(f"⚠️  Failed to load: {filename}")
            idx = (idx + 1) % len(image_files)
            continue

        # Generate visualization panels
        gt_panel = get_ground_truth_view(raw_img, label_path, target_size=(700, 700))
        pred_panel = get_prediction_view(model, img_path, raw_img, target_size=(700, 700), conf_thresh=0.30)

        # Create display canvas
        header_h, footer_h, sep_w = 60, 35, 6
        panel_w, panel_h = 700, 700
        total_w = panel_w * 2 + sep_w
        total_h = panel_h + header_h + footer_h

        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)  # Dark background

        # Place panels
        canvas[header_h : header_h + panel_h, :panel_w] = gt_panel
        canvas[header_h : header_h + panel_h, panel_w + sep_w :] = pred_panel

        # Draw separator line
        cv2.line(canvas, (panel_w + sep_w // 2, 0),
                (panel_w + sep_w // 2, total_h), (45, 45, 45), 2)

        # Add header text
        cv2.putText(canvas, "GROUND TRUTH", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "MODEL PREDICTION (YOLOv8n)", (panel_w + sep_w + 20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 255, 100), 2, cv2.LINE_AA)

        # Add file info
        file_info = f"[{idx + 1}/{len(image_files)}]  {filename}"
        (tw, _), _ = cv2.getTextSize(file_info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(canvas, file_info, (total_w - tw - 20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # Add footer controls
        footer_text = "Controls: [D] Next Image  |  [A] Previous Image  |  [Q] Quit"
        cv2.putText(canvas, footer_text, (20, total_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

        # Display
        cv2.imshow(window_name, canvas)

        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            print("👋 Exiting...")
            break
        elif key in (ord("d"), ord("D")):  # Next
            idx = (idx + 1) % len(image_files)
        elif key in (ord("a"), ord("A")):  # Previous
            idx = (idx - 1) % len(image_files)

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")

finally:
    cv2.destroyAllWindows()
    print("✅ Viewer closed")
