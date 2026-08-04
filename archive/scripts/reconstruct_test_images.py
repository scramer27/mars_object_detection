#!/usr/bin/env python3
"""
Reconstruct Test Images from Binary Files

Converts CHW binary format (used for FPGA) back to viewable JPG images
for live inference visualization and model verification.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path

# Configuration
RAW_INPUT_DIR = "fpga_payload/raw_inputs"
LABELS_DIR = "fpga_payload/labels"
OUTPUT_DIR = "test_images_reconstructed"
OUTPUT_LABELS_DIR = "test_labels_reconstructed"

# Image dimensions (from TECHNICAL_OVERVIEW.md - trained at 256x256)
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

def reconstruct_image_from_binary(bin_path: str) -> np.ndarray:
    """
    Load binary file in CHW format and convert to HWC for display.

    Binary format: [3, 256, 256] uint8 (planar RGB)
    Output format: [256, 256, 3] uint8 (interleaved RGB)
    """
    with open(bin_path, 'rb') as f:
        raw_data = f.read()

    expected_size = IMG_CHANNELS * IMG_HEIGHT * IMG_WIDTH
    if len(raw_data) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes, got {len(raw_data)} in {bin_path}")

    # Reshape to CHW format
    img_chw = np.frombuffer(raw_data, dtype=np.uint8).reshape((IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH))

    # Convert CHW → HWC (standard image format)
    img_hwc = np.transpose(img_chw, (1, 2, 0))

    return img_hwc

def main():
    """Reconstruct all test images from binary format."""

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    # Find all binary files
    bin_files = sorted([f for f in os.listdir(RAW_INPUT_DIR) if f.endswith('.bin')])

    if not bin_files:
        print(f"[ERROR] No .bin files found in {RAW_INPUT_DIR}")
        return

    print(f"[*] Reconstructing {len(bin_files)} test images...")
    print(f"   Input:  {RAW_INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print()

    successful = 0
    failed = 0

    for bin_file in bin_files:
        try:
            # Get base name (without .bin extension)
            base_name = bin_file.replace('.bin', '')

            # Reconstruct image
            bin_path = os.path.join(RAW_INPUT_DIR, bin_file)
            img_hwc = reconstruct_image_from_binary(bin_path)

            # Save as JPG
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.JPG")
            img_pil = Image.fromarray(img_hwc, mode='RGB')
            img_pil.save(output_path, quality=95)

            # Copy corresponding label file if it exists
            label_src = os.path.join(LABELS_DIR, f"{base_name}.txt")
            label_dst = os.path.join(OUTPUT_LABELS_DIR, f"{base_name}.txt")

            if os.path.exists(label_src):
                import shutil
                shutil.copy(label_src, label_dst)

            successful += 1
            if successful <= 5 or successful % 10 == 0:
                print(f"  [OK] {base_name}.JPG")

        except Exception as e:
            print(f"  [FAIL] Failed to process {bin_file}: {e}")
            failed += 1

    print()
    print(f"[SUCCESS] Reconstruction complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed:     {failed}")
    print()
    print(f"[*] Images saved to: {OUTPUT_DIR}/")
    print(f"[*] Labels saved to: {OUTPUT_LABELS_DIR}/")
    print()
    print("Next step: Run live_inference_windows.py to visualize predictions")

if __name__ == "__main__":
    main()
