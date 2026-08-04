#!/usr/bin/env python3
"""
Model Verification Script

Tests the trained Mars YOLOv8 model on reconstructed test images
and reports detection statistics without requiring OpenCV visualization.
"""

import os
import sys
from pathlib import Path

# Check dependencies
try:
    from ultralytics import YOLO
    print("[OK] Ultralytics YOLO library loaded")
except ImportError:
    print("[ERROR] Ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

try:
    from PIL import Image
    print("[OK] PIL library loaded")
except ImportError:
    print("[ERROR] PIL not installed. Run: pip install Pillow")
    sys.exit(1)

# Configuration
WEIGHTS_PATH = "fpga_payload/best.pt"
TEST_IMAGES_DIR = "test_images_reconstructed"
TEST_LABELS_DIR = "test_labels_reconstructed"

class_names = ["Soil", "Bedrock", "Sand", "Big Rock"]

# Verify files exist
if not os.path.exists(WEIGHTS_PATH):
    print(f"[ERROR] Model not found: {WEIGHTS_PATH}")
    sys.exit(1)

if not os.path.exists(TEST_IMAGES_DIR):
    print(f"[ERROR] Test images not found: {TEST_IMAGES_DIR}")
    print("        Run: python reconstruct_test_images.py")
    sys.exit(1)

# Load model
print(f"\n[*] Loading model: {WEIGHTS_PATH}")
try:
    model = YOLO(WEIGHTS_PATH)
    print("[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# Find test images
image_files = sorted([
    os.path.join(TEST_IMAGES_DIR, f)
    for f in os.listdir(TEST_IMAGES_DIR)
    if f.upper().endswith(('.JPG', '.PNG', '.JPEG'))
])

if not image_files:
    print(f"[ERROR] No images found in {TEST_IMAGES_DIR}")
    sys.exit(1)

print(f"[*] Found {len(image_files)} test images\n")

# Run inference on all images
print("=" * 80)
print("RUNNING INFERENCE ON TEST SET")
print("=" * 80)

class_detections = {i: 0 for i in range(4)}
total_detections = 0
images_with_detections = 0

for idx, img_path in enumerate(image_files):
    filename = os.path.basename(img_path)

    # Run inference
    results = model(img_path, conf=0.30, imgsz=256, verbose=False)[0]

    num_detections = len(results.boxes) if results.boxes is not None else 0

    if num_detections > 0:
        images_with_detections += 1
        total_detections += num_detections

        # Count detections per class
        for box in results.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            class_detections[cls_id] += 1

    # Print progress
    if (idx + 1) <= 10 or (idx + 1) % 10 == 0:
        status = f"[{num_detections} detections]" if num_detections > 0 else "[no detections]"
        print(f"  {idx + 1:2d}/{len(image_files)}: {filename[:40]:<40} {status}")

print("\n" + "=" * 80)
print("INFERENCE RESULTS SUMMARY")
print("=" * 80)

print(f"\nTotal images tested:      {len(image_files)}")
print(f"Images with detections:   {images_with_detections}")
print(f"Images without detections: {len(image_files) - images_with_detections}")
print(f"Total detections:         {total_detections}")
print(f"Average detections/image: {total_detections / len(image_files):.2f}")

print(f"\nDetections per class:")
print(f"  {class_names[0]:<12} : {class_detections[0]:3d} ({100*class_detections[0]/max(total_detections,1):.1f}%)")
print(f"  {class_names[1]:<12} : {class_detections[1]:3d} ({100*class_detections[1]/max(total_detections,1):.1f}%)")
print(f"  {class_names[2]:<12} : {class_detections[2]:3d} ({100*class_detections[2]/max(total_detections,1):.1f}%)")
print(f"  {class_names[3]:<12} : {class_detections[3]:3d} ({100*class_detections[3]/max(total_detections,1):.1f}%)")

print("\n" + "=" * 80)

# Sample detailed output for first 3 images with detections
print("SAMPLE DETECTIONS (First 3 images)")
print("=" * 80)

sample_count = 0
for img_path in image_files:
    if sample_count >= 3:
        break

    filename = os.path.basename(img_path)
    results = model(img_path, conf=0.30, imgsz=256, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        sample_count += 1
        print(f"\n{filename}:")

        for box in results.boxes:
            xywhn = box.xywhn[0].cpu().numpy()
            xc, yc, bw, bh = xywhn
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            print(f"  - {class_names[cls_id]:<12} confidence: {conf:.3f}  "
                  f"bbox: [{xc:.3f}, {yc:.3f}, {bw:.3f}, {bh:.3f}]")

print("\n" + "=" * 80)
print("[SUCCESS] Model verification complete!")
print("\nNext steps:")
print("  1. Install OpenCV: pip install opencv-python")
print("  2. Run live viewer: python live_inference_windows.py")
print("  3. Deploy to PolarFire FPGA")
print("=" * 80)
