import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ==============================================================================
# 1. PATHS - Updated to match your v8 FPGA Run
# ==============================================================================
# This path matches your recent git commit and training output
weights_path = "/Users/scramer/Documents/mars_object_detection/runs/detect/output_yolov8_fpga/mars_yolov8n_fpga/weights/best.pt"
test_images_dir = "/Users/scramer/Documents/mars_object_detection/data/yolo_mars/images/test"
test_labels_dir = "/Users/scramer/Documents/mars_object_detection/data/yolo_mars/labels/test"
yaml_path = "/Users/scramer/Documents/mars_object_detection/data/yolo_mars/mars.yaml"

# ==============================================================================
# 2. CLASS DEFINITIONS & COLOR PALETTE
# ==============================================================================
class_names = ["soil", "bedrock", "sand", "big_rock"]
if os.path.exists(yaml_path):
    try:
        with open(yaml_path, "r") as f:
            data_cfg = yaml.safe_load(f)
            if "names" in data_cfg:
                class_names = data_cfg["names"]
    except Exception:
        pass

# BGR Colors: 0: Soil (Orange), 1: Bedrock (Cyan), 2: Sand (Yellow), 3: Big Rock (Magenta)
CLASS_COLORS = {
    0: (0, 165, 255), 
    1: (255, 255, 0), 
    2: (0, 230, 255), 
    3: (255, 0, 255), 
}

print(f"🚀 Loading YOLOv8 Model: {weights_path}")
model = YOLO(weights_path)

# Load test files
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([
    os.path.join(test_images_dir, f)
    for f in os.listdir(test_images_dir)
    if f.lower().endswith(valid_extensions)
])

if not image_files:
    print(f"❌ No images found in {test_images_dir}")
    exit()

# ==============================================================================
# 3. VISUALIZATION HELPERS
# ==============================================================================
def draw_crisp_box(img, x1, y1, x2, y2, label_text, color):
    """Draws anti-aliased crisp bounding boxes and filled text labels."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pill_y1 = max(0, y1 - th - 10)
    pill_y2 = y1 if y1 - th - 10 >= 0 else y1 + th + 10
    
    cv2.rectangle(img, (x1, pill_y1), (x1 + tw + 10, pill_y2), color, -1, cv2.LINE_AA)
    text_y = pill_y2 - 5 if y1 - th - 10 >= 0 else pill_y2 - 3
    cv2.putText(img, label_text, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

def get_ground_truth_view(img_raw, label_path, target_size=(768, 768)):
    img_disp = cv2.resize(img_raw, target_size, interpolation=cv2.INTER_CUBIC)
    h_disp, w_disp = img_disp.shape[:2]
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = int((xc - bw / 2) * w_disp)
                    y1 = int((yc - bh / 2) * h_disp)
                    x2 = int((xc + bw / 2) * w_disp)
                    y2 = int((yc + bh / 2) * h_disp)
                    
                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                    label_str = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    draw_crisp_box(img_disp, x1, y1, x2, y2, label_str, color)
    return img_disp

def get_prediction_view(model, img_path, img_raw, target_size=(768, 768), conf_thresh=0.30):
    img_disp = cv2.resize(img_raw, target_size, interpolation=cv2.INTER_CUBIC)
    h_disp, w_disp = img_disp.shape[:2]
    
    # Standard VectorBlox resolution: 256
    results = model(img_path, conf=conf_thresh, imgsz=256, verbose=False)[0]
    
    if results.boxes is not None:
        for box in results.boxes:
            xywhn = box.xywhn[0].cpu().numpy()
            xc, yc, bw, bh = xywhn
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            x1 = int((xc - bw / 2) * w_disp)
            y1 = int((yc - bh / 2) * h_disp)
            x2 = int((xc + bw / 2) * w_disp)
            y2 = int((yc + bh / 2) * h_disp)
            
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            c_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            label_str = f"{c_name} {conf:.2f}"
            draw_crisp_box(img_disp, x1, y1, x2, y2, label_str, color)
    return img_disp

# ==============================================================================
# 4. MAIN APPLICATION LOOP
# ==============================================================================
idx = 0
window_name = "Mars Object Detection - Ground Truth vs Prediction"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1400, 750)

while True:
    img_path = image_files[idx]
    filename = os.path.basename(img_path)
    base_name, _ = os.path.splitext(filename)
    label_path = os.path.join(test_labels_dir, base_name + ".txt")
    
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        idx = (idx + 1) % len(image_files)
        continue
    
    gt_panel = get_ground_truth_view(raw_img, label_path, target_size=(700, 700))
    pred_panel = get_prediction_view(model, img_path, raw_img, target_size=(700, 700))
    
    # Layout Canvas
    header_h, footer_h, sep_w = 60, 35, 6
    panel_w, panel_h = 700, 700
    total_w = panel_w * 2 + sep_w
    total_h = panel_h + header_h + footer_h
    
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20) 
    
    canvas[header_h : header_h + panel_h, :panel_w] = gt_panel
    canvas[header_h : header_h + panel_h, panel_w + sep_w :] = pred_panel
    cv2.line(canvas, (panel_w + sep_w // 2, 0), (panel_w + sep_w // 2, total_h), (45, 45, 45), 2)
    
    cv2.putText(canvas, "GROUND TRUTH", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "MODEL PREDICTION (YOLOv8n)", (panel_w + sep_w + 20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 255, 100), 2, cv2.LINE_AA)
    
    file_info = f"[{idx + 1}/{len(image_files)}]  {filename}"
    (tw, _), _ = cv2.getTextSize(file_info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(canvas, file_info, (total_w - tw - 20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    
    footer_text = "Controls: [D] Next Image  |  [A] Previous Image  |  [Q] Quit"
    cv2.putText(canvas, footer_text, (20, total_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    
    cv2.imshow(window_name, canvas)
    key = cv2.waitKey(0) & 0xFF
    if key in (ord("q"), ord("Q"), 27): break
    elif key in (ord("d"), ord("D")): idx = (idx + 1) % len(image_files)
    elif key in (ord("a"), ord("A")): idx = (idx - 1) % len(image_files)

cv2.destroyAllWindows()