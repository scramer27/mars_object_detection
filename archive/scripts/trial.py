import os
import zipfile
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import label, find_objects

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
EXTRACT_DIR = DATA_DIR / "extracted" / "ai4mars"
YOLO_DIR = DATA_DIR / "ai4mars_yolo"

ZIP_PATH = DOWNLOAD_DIR / "ai4mars-dataset-merged-0.6.zip"

def extract_dataset():
    """Extracts the downloaded AI4Mars zip file."""
    if not ZIP_PATH.exists():
        print(f"Error: Could not find '{ZIP_PATH}'. Please ensure Step 1 download completed.")
        return False

    extracted_flag = EXTRACT_DIR / ".extracted_complete"
    if not extracted_flag.exists():
        print("=== Extracting AI4Mars Dataset (~15.5 GB) ===")
        print("This may take 2-3 minutes...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        extracted_flag.touch()
        print("Extraction complete!")
    else:
        print("Dataset already extracted.")
    return True

def convert_masks_to_yolo_boxes():
    """Converts Big Rock terrain masks (Class 3) into YOLO bounding box annotations."""
    print("\n=== Generating YOLO Bounding Boxes for 'Big Rock' Detection ===")
    
    (YOLO_DIR / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels").mkdir(parents=True, exist_ok=True)

    all_files = list(EXTRACT_DIR.rglob("*.png")) + list(EXTRACT_DIR.rglob("*.JPG"))
    label_dict = {f.stem: f for f in all_files if "label" in str(f).lower()}
    
    pairs = []
    for f in all_files:
        if "label" not in str(f).lower() and f.stem in label_dict:
            pairs.append((f, label_dict[f.stem]))

    print(f"Mapped {len(pairs)} image-mask pairs. Processing bounding boxes...")
    
    saved_count = 0
    for img_path, mask_path in pairs:
        try:
            mask_arr = np.array(Image.open(mask_path))
            
            # Big Rock = Class 3
            rock_mask = (mask_arr == 3).astype(int)
            if rock_mask.sum() == 0:
                continue

            # Connected components to isolate individual rocks
            labeled_array, _ = label(rock_mask)
            objects = find_objects(labeled_array)

            h, w = mask_arr.shape
            yolo_lines = []

            for obj in objects:
                ymin, ymax = obj[0].start, obj[0].stop
                xmin, xmax = obj[1].start, obj[1].stop

                box_w = xmax - xmin
                box_h = ymax - ymin

                # Filter tiny noise pixels
                if box_w < 15 or box_h < 15:
                    continue

                # Convert to normalized YOLO coordinates: [class x_center y_center width height]
                x_center = (xmin + box_w / 2.0) / w
                y_center = (ymin + box_h / 2.0) / h
                norm_w = box_w / w
                norm_h = box_h / h

                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            if yolo_lines:
                target_img = YOLO_DIR / "images" / img_path.name
                target_label = YOLO_DIR / "labels" / f"{img_path.stem}.txt"

                if not target_img.exists():
                    img = Image.open(img_path)
                    img.save(target_img)

                with open(target_label, "w") as f:
                    f.write("\n".join(yolo_lines))

                saved_count += 1
                if saved_count % 250 == 0:
                    print(f"Generated {saved_count} YOLO training annotations...")

        except Exception:
            continue

    print(f"\nFinished! Created {saved_count} object detection samples inside '{YOLO_DIR}'!")

if __name__ == "__main__":
    if extract_dataset():
        convert_masks_to_yolo_boxes()