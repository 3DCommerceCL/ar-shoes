"""
PASO 1 — Augmentación de datos: 100 fotos → 100,000
=======================================================
Requisitos:
    pip install albumentations opencv-python Pillow tqdm

Estructura esperada antes de correr:
    training/data/images/   ← poner aquí las 100 fotos originales (.jpg/.png)
    training/data/masks/    ← poner aquí las máscaras binarias correspondientes
                               (blanco = pie/pierna, negro = fondo)
                               Mismo nombre que la foto: foto001.jpg → foto001.png

Resultado:
    training/data_augmented/images/   ← 100,000 imágenes aumentadas
    training/data_augmented/masks/    ← 100,000 máscaras correspondientes
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# ---- Configuración ----
INPUT_IMAGES = Path("data/images")
INPUT_MASKS  = Path("data/masks")
OUT_IMAGES   = Path("data_augmented/images")
OUT_MASKS    = Path("data_augmented/masks")
AUGMENTS_PER_IMAGE = 1000   # 100 fotos × 1000 = 100,000
IMG_SIZE = 256

OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_MASKS.mkdir(parents=True, exist_ok=True)

# ---- Pipeline de augmentación ----
# Cubre variaciones de: ángulo, luz, color, distancia, perspectiva, fondo
transform = A.Compose([
    A.RandomRotate90(p=0.3),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.ShiftScaleRotate(
        shift_limit=0.15, scale_limit=0.3, rotate_limit=40,
        border_mode=cv2.BORDER_REFLECT, p=0.8
    ),
    A.Perspective(scale=(0.05, 0.15), p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
        A.RandomGamma(gamma_limit=(70, 130)),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30),
    ], p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=7),
        A.MedianBlur(blur_limit=5),
    ], p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.CLAHE(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),
    A.Resize(IMG_SIZE, IMG_SIZE),
])

# ---- Augmentación de fondos sintéticos ----
def replace_background(image, mask, bg_color=None):
    """Reemplaza el fondo con un color aleatorio para diversificar."""
    if bg_color is None:
        # Color aleatorio (simula diferentes tipos de piso)
        bg_color = np.random.randint(50, 220, 3).tolist()
    result = image.copy()
    bg_mask = mask == 0  # píxeles de fondo
    result[bg_mask] = bg_color
    return result

# ---- Ejecutar ----
image_paths = sorted(INPUT_IMAGES.glob("*.jpg")) + sorted(INPUT_IMAGES.glob("*.png"))
print(f"Fotos encontradas: {len(image_paths)}")

total = 0
for img_path in tqdm(image_paths, desc="Procesando fotos"):
    mask_path = INPUT_MASKS / (img_path.stem + ".png")
    if not mask_path.exists():
        print(f"  [SKIP] No tiene máscara: {img_path.name}")
        continue

    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    # Binarizar máscara (>127 = pie)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    for i in range(AUGMENTS_PER_IMAGE):
        aug = transform(image=image, mask=mask)
        aug_img  = aug["image"]
        aug_mask = aug["mask"]

        # 30% de chance de reemplazar fondo con color aleatorio
        if np.random.random() < 0.3:
            aug_img = replace_background(aug_img, aug_mask)

        out_name = f"{img_path.stem}_aug_{i:04d}"
        cv2.imwrite(str(OUT_IMAGES / f"{out_name}.jpg"),
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(OUT_MASKS / f"{out_name}.png"), aug_mask)
        total += 1

print(f"\n✅ Generadas {total:,} imágenes en data_augmented/")
