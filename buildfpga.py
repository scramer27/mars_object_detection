import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

# --- Paths Setup ---
BASE_DIR = Path(__file__).resolve().parent

# Source paths
ONNX_SRC = BASE_DIR / "runs" / "detect" / "output_clean" / "mars_yolo_fpga" / "weights" / "best.onnx"
TEST_IMG_DIR = BASE_DIR / "data" / "yolo_mars" / "images" / "test"
TEST_LBL_DIR = BASE_DIR / "data" / "yolo_mars" / "labels" / "test"

# Destination directory for Git tracking
PAYLOAD_DIR = BASE_DIR / "fpga_payload"
RAW_BIN_DIR = PAYLOAD_DIR / "raw_inputs"
LABELS_DIR = PAYLOAD_DIR / "labels"

# Create output folders
PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAW_BIN_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Starting FPGA Test Payload Builder...")

# 1. Copy ONNX model into payload directory
if ONNX_SRC.exists():
    onnx_dest = PAYLOAD_DIR / "best.onnx"
    shutil.copy2(ONNX_SRC, onnx_dest)
    print(f"✅ Copied ONNX model to: {onnx_dest}")
else:
    print(f"⚠️ Warning: Could not find ONNX file at {ONNX_SRC}")

# 2. Gather Test Images
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = sorted([f for f in TEST_IMG_DIR.glob("*.*") if f.suffix.lower() in valid_exts])

if not image_files:
    print(f"❌ No test images found in {TEST_IMG_DIR}")
    exit(1)

# Process up to 50 test samples for a lightweight bundle
sample_count = min(50, len(image_files))
print(f"📦 Converting {sample_count} test images to raw binaries...")

for img_path in image_files[:sample_count]:
    # Open image & ensure RGB at native 256x256
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    
    # Convert array to Planar CHW layout: Shape (3, 256, 256) uint8
    img_np = np.array(img, dtype=np.uint8)
    img_chw = np.transpose(img_np, (2, 0, 1))

    # Save as raw contiguous C-style binary file
    bin_path = RAW_BIN_DIR / f"{img_path.stem}.bin"
    with open(bin_path, "wb") as f:
        f.write(img_chw.tobytes())

    # Copy matching Ground Truth bounding box txt file if present
    lbl_path = TEST_LBL_DIR / f"{img_path.stem}.txt"
    if lbl_path.exists():
        shutil.copy2(lbl_path, LABELS_DIR / lbl_path.name)

print("\n🎉 Payload creation complete!")
print(f"📁 Payload Location: {PAYLOAD_DIR}")
print("   - Includes: best.onnx")
print("   - Includes: raw_inputs/*.bin")
print("   - Includes: labels/*.txt")