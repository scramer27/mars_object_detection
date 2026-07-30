import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

BASE_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = BASE_DIR / "data" / "extracted" / "ai4mars"
YOLO_DIR = BASE_DIR / "data" / "ai4mars_yolo"
IMG_DIR = YOLO_DIR / "images"
LABEL_DIR = YOLO_DIR / "labels"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_side_by_side(num_samples=5):
    print("=== Generating High-Contrast Before & After Visuals ===")
    
    yolo_labels = [f for f in LABEL_DIR.glob("*.txt") if f.stat().st_size > 0]
    if not yolo_labels:
        print("No YOLO labels found! Make sure setup script finished running.")
        return

    selected_labels = random.sample(yolo_labels, min(num_samples, len(yolo_labels)))

    # High-contrast neon colors
    # 0: Soil (Lime Green), 1: Bedrock (Cyan), 2: Sand (Yellow), 3: Big Rock (Magenta)
    colors = ['#00FF00', '#00FFFF', '#FFFF00', '#FF007F']
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(num_samples, 3, figsize=(16, 4.2 * num_samples))
    fig.suptitle("AI4Mars: Raw Image vs. Bounding Boxes vs. High-Contrast Segmentation", fontsize=16, fontweight='bold')

    all_extracted_files = list(EXTRACT_DIR.rglob("*.png")) + list(EXTRACT_DIR.rglob("*.JPG"))
    label_dict = {f.stem: f for f in all_extracted_files if "label" in str(f).lower()}

    for row_idx, yolo_txt_path in enumerate(selected_labels):
        img_stem = yolo_txt_path.stem
        img_path = next((f for f in IMG_DIR.glob(f"{img_stem}.*")), None)
        raw_mask_path = label_dict.get(img_stem)

        if not img_path or not img_path.exists():
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        # --- Column 1: RAW IMAGE ("BEFORE") ---
        ax_raw = axes[row_idx, 0] if num_samples > 1 else axes[0]
        ax_raw.imshow(img)
        if row_idx == 0:
            ax_raw.set_title("BEFORE\n(Raw Rover Frame)", fontsize=12, fontweight='bold')
        ax_raw.set_ylabel(f"Sample #{row_idx+1}", fontsize=10, fontweight='bold')
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])

        # --- Column 2: YOLO BOUNDING BOXES ("AFTER - DETECTION") ---
        ax_yolo = axes[row_idx, 1] if num_samples > 1 else axes[1]
        ax_yolo.imshow(img)
        
        box_count = 0
        with open(yolo_txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    box_count += 1
                    _, x_center, y_center, norm_w, norm_h = map(float, parts)
                    w_px = norm_w * img_w
                    h_px = norm_h * img_h
                    xmin = (x_center * img_w) - (w_px / 2.0)
                    ymin = (y_center * img_h) - (h_px / 2.0)

                    rect = patches.Rectangle(
                        (xmin, ymin), w_px, h_px,
                        linewidth=2.5, edgecolor="#FF007F", facecolor="none"
                    )
                    ax_yolo.add_patch(rect)

        if row_idx == 0:
            ax_yolo.set_title("AFTER\n(YOLO Big Rock BBoxes)", fontsize=12, fontweight='bold')
        ax_yolo.set_xlabel(f"{box_count} Big Rock target(s)", fontsize=9, fontweight='bold')
        ax_yolo.set_xticks([])
        ax_yolo.set_yticks([])

        # --- Column 3: SEGMENTATION MASK ("AFTER - SEGMENTATION") ---
        ax_seg = axes[row_idx, 2] if num_samples > 1 else axes[2]
        ax_seg.imshow(img)
        
        if raw_mask_path and raw_mask_path.exists():
            mask_arr = np.array(Image.open(raw_mask_path))
            # Mask out invalid values (>3 or <0)
            masked_data = np.ma.masked_where((mask_arr > 3) | (mask_arr < 0), mask_arr)
            ax_seg.imshow(masked_data, cmap=cmap, alpha=0.65, vmin=0, vmax=3)

        if row_idx == 0:
            ax_seg.set_title("AFTER\n(High-Contrast Terrain Segmentation)", fontsize=12, fontweight='bold')
        ax_seg.set_xticks([])
        ax_seg.set_yticks([])

    # High visibility legend
    legend_patches = [
        mpatches.Patch(color='#00FF00', label='Soil (Lime Green)'),
        mpatches.Patch(color='#00FFFF', label='Bedrock (Cyan)'),
        mpatches.Patch(color='#FFFF00', label='Sand / Dunes (Yellow)'),
        mpatches.Patch(color='#FF007F', label='Big Rock (Magenta)')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)
    
    out_file = OUTPUT_DIR / "preview_side_by_side_high_contrast.png"
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"\nSaved updated visual to: {out_file.resolve()}")

if __name__ == "__main__":
    generate_side_by_side()