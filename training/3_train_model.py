"""
PASO 3 — Entrenar modelo de segmentación de pies
==================================================
Arquitectura: MobileNetV2 (encoder) + U-Net decoder ligero
Entrada:  256×256 RGB
Salida:   256×256 máscara de confianza (0=fondo, 1=pie)
Tamaño final ONNX: ~5MB
Velocidad en browser: ~20-40ms en móvil

Requisitos:
    pip install torch torchvision tqdm scikit-learn Pillow

Uso:
    python 3_train_model.py
    # O con GPU: python 3_train_model.py --epochs 50
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from tqdm import tqdm

# ---- Argumentos ----
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",     type=int,   default=30)
parser.add_argument("--batch_size", type=int,   default=16)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--data_dir",   type=str,   default="data_augmented")
parser.add_argument("--output",     type=str,   default="foot_model.pth")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en: {device}")

# ---- Dataset ----
class FootDataset(Dataset):
    def __init__(self, data_dir, img_size=256):
        self.img_dir  = Path(data_dir) / "images"
        self.mask_dir = Path(data_dir) / "masks"
        self.paths    = sorted(self.img_dir.glob("*.jpg")) + sorted(self.img_dir.glob("*.png"))
        self.img_tf   = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path  = self.paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return self.img_tf(img), (self.mask_tf(mask) > 0.5).float()

# ---- Arquitectura: MobileNetV2 + Decoder U-Net ligero ----
class ConvBnRelu(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(cin + cskip, cout),
            ConvBnRelu(cout, cout),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class FootSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        mb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        features = mb.features

        # Encoder (capas de MobileNetV2)
        self.enc0 = features[0]          # 256→128, ch=32
        self.enc1 = features[1:4]        # 128→64,  ch=24
        self.enc2 = features[4:7]        # 64→32,   ch=32
        self.enc3 = features[7:11]       # 32→16,   ch=64
        self.enc4 = features[11:16]      # 16→8,    ch=96
        self.bottleneck = features[16:]  # 8→8,     ch=1280→320

        # Decoder ligero
        self.up4 = UpBlock(320, 96, 128)
        self.up3 = UpBlock(128, 64, 64)
        self.up2 = UpBlock(64,  32, 32)
        self.up1 = UpBlock(32,  24, 24)
        self.up0 = UpBlock(24,  32, 16)

        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e0 = self.enc0(x)         # 128×128×32
        e1 = self.enc1(e0)        # 64×64×24
        e2 = self.enc2(e1)        # 32×32×32
        e3 = self.enc3(e2)        # 16×16×64
        e4 = self.enc4(e3)        # 8×8×96
        b  = self.bottleneck(e4)  # 8×8×320

        d = self.up4(b,  e4)      # 16×16×128
        d = self.up3(d,  e3)      # 32×32×64
        d = self.up2(d,  e2)      # 64×64×32
        d = self.up1(d,  e1)      # 128×128×24
        d = self.up0(d,  e0)      # 256×256×16

        return torch.sigmoid(self.head(d))

# ---- Pérdida: BCE + Dice (mejor para segmentación desbalanceada) ----
def dice_loss(pred, target, eps=1e-6):
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - (2 * inter + eps) / (union + eps)

def combined_loss(pred, target):
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target).mean()
    return bce + dice

# ---- Entrenamiento ----
dataset = FootDataset(args.data_dir)
n_val   = max(1, int(len(dataset) * 0.1))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = FootSegNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

best_val_loss = float("inf")
print(f"\nDataset: {n_train:,} train / {n_val:,} val — {args.epochs} epochs\n")

for epoch in range(1, args.epochs + 1):
    # Train
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss  = combined_loss(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            val_loss += combined_loss(preds, masks).item()

    scheduler.step()
    tl = train_loss / len(train_loader)
    vl = val_loss  / len(val_loader)
    print(f"Epoch {epoch:3d} | train={tl:.4f} | val={vl:.4f}")

    if vl < best_val_loss:
        best_val_loss = vl
        torch.save(model.state_dict(), args.output)
        print(f"  💾 Guardado (val={vl:.4f})")

print(f"\n✅ Mejor modelo en: {args.output}  (val_loss={best_val_loss:.4f})")
