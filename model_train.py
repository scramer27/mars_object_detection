import os
import sys
import random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# =====================================================================
# 1. MLP-MIXER ARCHITECTURE (FPGA-Optimized with Dropout)
# =====================================================================

class MLPBlock(nn.Module):
    """2-Layer MLP with ReLU (DSP/FPGA friendly) + Dropout for Regularization"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    """Alternates between Spatial/Token Mixing and Channel/Feature Mixing"""
    def __init__(self, num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mlp = MLPBlock(num_patches, tokens_mlp_dim, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mlp = MLPBlock(hidden_dim, channels_mlp_dim, dropout=dropout)

    def forward(self, x):
        # 1. Spatial/Token Mixing across patches
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        # 2. Channel Mixing within patches
        y = self.norm2(x)
        y = self.channel_mlp(y)
        return x + y


class MLPMixerTerrainClassifier(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, num_classes=4, hidden_dim=128, depth=4, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2  # 256 patches
        self.patch_size = patch_size

        # Linear Patch Embedding
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)

        # Stacked Mixer Blocks
        self.blocks = nn.ModuleList([
            MixerBlock(self.num_patches, hidden_dim, tokens_mlp_dim=64, channels_mlp_dim=256, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        
        # Raw Logits Head (NO Softmax layer attached - VectorBlox ready!)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        p = self.patch_size

        # Unfold image into non-overlapping 16x16 patches: [B, 256, Patch_Dim]
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, self.num_patches, -1)

        # Patch projection & Mixing
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.head(x)


# =====================================================================
# 2. AI4MARS DATASET LOADER WITH DATA AUGMENTATION
# =====================================================================

class AI4MarsDataset(Dataset):
    def __init__(self, root_dir, img_size=256, max_samples=3000, is_train=True):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.is_train = is_train
        
        # Gather matching image and label pairs
        all_files = list(self.root_dir.rglob("*.png")) + list(self.root_dir.rglob("*.JPG"))
        label_dict = {f.stem: f for f in all_files if "label" in str(f).lower()}
        image_list = [f for f in all_files if "label" not in str(f).lower() and f.stem in label_dict]

        if len(image_list) > max_samples:
            image_list = random.sample(image_list, max_samples)

        self.samples = [(img, label_dict[img.stem]) for img in image_list]

        # Training Augmentations vs Validation Transforms
        if self.is_train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        mask = Image.open(mask_path).resize((self.img_size, self.img_size), Image.NEAREST)
        mask_np = np.array(mask)
        valid_pixels = mask_np[(mask_np >= 0) & (mask_np <= 3)]

        if len(valid_pixels) > 0:
            counts = np.bincount(valid_pixels, minlength=4)
            label = int(np.argmax(counts))
        else:
            label = 0

        return img_tensor, torch.tensor(label, dtype=torch.long)


# =====================================================================
# 3. DEVICE CHECK, TRAINING LOOP & CHECKPOINTING
# =====================================================================

def get_compute_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(" Using Hardware Acceleration: NVIDIA CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" Using Hardware Acceleration: Apple Silicon MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print(" Using Hardware Device: CPU")
    return device


def train_and_export():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "extracted" / "ai4mars"
    OUTPUT_DIR = BASE_DIR / "output"
    CKPT_DIR = BASE_DIR / "checkpoints"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = CKPT_DIR / "best_mlp_mixer.pth"

    device = get_compute_device()

    # Load Train/Val Datasets
    full_dataset = AI4MarsDataset(root_dir=DATA_DIR, img_size=256, max_samples=3000, is_train=True)
    print(f" Loaded {len(full_dataset)} image-mask pairs from AI4Mars dataset.")

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0 if device.type == 'mps' else 2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = MLPMixerTerrainClassifier(image_size=256, num_classes=4, dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0

    print("\n=== Starting Enhanced MLP-Mixer Training Loop ===")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)

        print(f"\n--- Epoch [{epoch}/{epochs}] ---")

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Live Step Progress Printer
            step_loss = loss.item()
            step_acc = (correct / total) * 100
            progress_pct = ((batch_idx + 1) / num_batches) * 100
            
            sys.stdout.write(
                f"\rBatch [{batch_idx+1:02d}/{num_batches:02d}] ({progress_pct:5.1f}%) | "
                f"Step Loss: {step_loss:.4f} | Running Acc: {step_acc:6.2f}%"
            )
            sys.stdout.flush()

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        current_lr = scheduler.get_last_lr()[0]

        print(f"\n Epoch {epoch} Summary -> Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | LR: {current_lr:.6f}")

        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = (val_correct / val_total) * 100
        print(f" Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        # Checkpointing Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, best_ckpt_path)
            print(f" [CHECKPOINT] Saved new best model weights -> {best_ckpt_path.name} ({val_acc:.2f}%)")

    # =====================================================================
    # 4. EXPORT BEST CHECKPOINT TO ONNX
    # =====================================================================
    print("\n=== Exporting Best Model Checkpoint to ONNX ===")
    
    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f" Loaded best checkpoint from Epoch {checkpoint['epoch']} with Val Acc: {checkpoint['val_acc']:.2f}%")

    model.eval().cpu()
    dummy_input = torch.randn(1, 3, 256, 256)
    onnx_path = OUTPUT_DIR / "mlp_mixer_mars.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['logits'],
        dynamic_axes=None
    )

    print(f" Successfully exported optimal model to ONNX: {onnx_path.resolve()}")


if __name__ == "__main__":
    train_and_export()