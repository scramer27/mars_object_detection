import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# AI4Mars Class Mapping
CLASS_NAMES = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock"}

def convert_mask_to_yolo_boxes(mask_path, img_size=(256, 256), min_area_pixels=100):
    """
    Finds contiguous pixel regions for each terrain class and converts
    them to normalized YOLO bounding boxes: [class_id, x_center, y_center, width, height]
    """
    mask = Image.open(mask_path).resize(img_size, Image.NEAREST)
    mask_np = np.array(mask)
    img_h, img_w = mask_np.shape
    
    yolo_labels = []

    for class_id in range(4):
        # Create binary mask for specific class
        binary_mask = (mask_np == class_id).astype(np.uint8) * 255
        
        # Find object contours/blobs
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_pixels:  # Filter out tiny noise contours
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Normalize coordinates to [0.0, 1.0] for YOLO format
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return yolo_labels


def build_yolo_dataset():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "extracted" / "ai4mars"
    YOLO_DIR = BASE_DIR / "data" / "yolo_mars"

    print("=== Generating YOLO Bounding Box Dataset from AI4Mars Masks ===")

    all_files = list(DATA_DIR.rglob("*.png")) + list(DATA_DIR.rglob("*.JPG"))
    label_dict = {f.stem: f for f in all_files if "label" in str(f).lower()}
    image_list = [f for f in all_files if "label" not in str(f).lower() and f.stem in label_dict]

    # Limit sample count for quick iteration if desired
    random.seed(42)
    random.shuffle(image_list)
    image_list = image_list[:3000]

    # Split: 70% Train, 15% Val, 15% Test
    n_total = len(image_list)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    splits = {
        'train': image_list[:n_train],
        'val': image_list[n_train:n_train+n_val],
        'test': image_list[n_train+n_val:]
    }

    for split in ['train', 'val', 'test']:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

        print(f" Processing {split} set ({len(splits[split])} images)...")

        for img_path in splits[split]:
            mask_path = label_dict[img_path.stem]
            
            # Convert mask to YOLO bounding boxes
            boxes = convert_mask_to_yolo_boxes(mask_path, img_size=(256, 256))
            if not boxes:
                continue  # Skip empty images

            # Save Image
            dest_img_path = YOLO_DIR / "images" / split / f"{img_path.stem}.jpg"
            img = Image.open(img_path).convert("RGB").resize((256, 256))
            img.save(dest_img_path)

            # Save YOLO Annotation TXT File
            dest_txt_path = YOLO_DIR / "labels" / split / f"{img_path.stem}.txt"
            with open(dest_txt_path, "w") as f:
                f.write("\n".join(boxes))

    # Write YOLO mars.yaml Config File
    yaml_content = f"""path: {YOLO_DIR.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: Soil
  1: Bedrock
  2: Sand
  3: Big Rock
"""
    yaml_path = YOLO_DIR / "mars.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n Successfully generated YOLO dataset configuration at: {yaml_path.resolve()}")


if __name__ == "__main__":
    build_yolo_dataset()