"""
PASO 2 — Auto-etiquetado con SAM (Segment Anything Model)
===========================================================
Genera máscaras automáticamente para las 100 fotos originales.
Solo necesitas hacer clic en un punto del pie en cada foto.

Requisitos:
    pip install segment-anything opencv-python torch torchvision tqdm
    # Descargar checkpoint SAM (342MB):
    # wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Uso:
    python 2_sam_label.py --checkpoint sam_vit_b_01ec64.pth

El script abre cada imagen y espera que hagas clic en el pie.
Guarda la máscara automáticamente en data/masks/
"""

import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# ---- Argumentos ----
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth", help="Ruta al checkpoint SAM")
parser.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
args = parser.parse_args()

INPUT_IMAGES = Path("data/images")
OUTPUT_MASKS = Path("data/masks")
OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)

# ---- Cargar SAM ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando: {device.upper()}")
sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# ---- Variables de estado de la UI ----
click_points = []
click_labels = []
current_image = None
current_mask  = None

def on_mouse_click(event, x, y, flags, param):
    global click_points, click_labels, current_mask

    if event == cv2.EVENT_LBUTTONDOWN:  # clic izquierdo = pie (incluir)
        click_points.append([x, y])
        click_labels.append(1)
        run_prediction()

    elif event == cv2.EVENT_RBUTTONDOWN:  # clic derecho = fondo (excluir)
        click_points.append([x, y])
        click_labels.append(0)
        run_prediction()

def run_prediction():
    global current_mask
    if not click_points:
        return

    pts = np.array(click_points)
    lbs = np.array(click_labels)
    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbs,
        multimask_output=True,
    )
    # Tomar la máscara con mayor score
    best = np.argmax(scores)
    current_mask = (masks[best] * 255).astype(np.uint8)
    show_overlay()

def show_overlay():
    if current_image is None or current_mask is None:
        return
    overlay = current_image.copy()
    overlay[current_mask > 0] = overlay[current_mask > 0] * 0.5 + np.array([0, 180, 0]) * 0.5
    # Dibujar puntos de clic
    for i, (px, py) in enumerate(click_points):
        color = (0, 255, 0) if click_labels[i] == 1 else (0, 0, 255)
        cv2.circle(overlay, (px, py), 6, color, -1)
    cv2.imshow("SAM Labeler", overlay.astype(np.uint8))

# ---- Procesar imágenes ----
image_paths = sorted(INPUT_IMAGES.glob("*.jpg")) + sorted(INPUT_IMAGES.glob("*.png"))
print(f"\nInstrucciones:")
print("  Clic IZQUIERDO  → marcar pie (verde)")
print("  Clic DERECHO    → marcar fondo (rojo)")
print("  S               → guardar máscara y siguiente")
print("  R               → resetear clics en imagen actual")
print("  Q               → salir\n")

for idx, img_path in enumerate(image_paths):
    mask_path = OUTPUT_MASKS / (img_path.stem + ".png")
    if mask_path.exists():
        print(f"  [OK] Ya tiene máscara: {img_path.name} — saltando")
        continue

    print(f"[{idx+1}/{len(image_paths)}] {img_path.name}")

    current_image = cv2.imread(str(img_path))
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

    # Redimensionar para display si es muy grande
    h, w = current_image.shape[:2]
    if max(h, w) > 900:
        scale = 900 / max(h, w)
        current_image = cv2.resize(current_image, (int(w*scale), int(h*scale)))

    click_points = []
    click_labels = []
    current_mask  = None

    predictor.set_image(current_image)

    cv2.namedWindow("SAM Labeler", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM Labeler", on_mouse_click)
    cv2.imshow("SAM Labeler", cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s') and current_mask is not None:  # guardar
            # Redimensionar máscara al tamaño original si fue escalada
            orig = cv2.imread(str(img_path))
            if orig.shape[:2] != current_mask.shape[:2]:
                current_mask = cv2.resize(current_mask, (orig.shape[1], orig.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mask_path), current_mask)
            print(f"  ✅ Guardada: {mask_path.name}")
            break

        elif key == ord('r'):  # resetear
            click_points = []
            click_labels = []
            current_mask  = None
            cv2.imshow("SAM Labeler", cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))
            print("  🔄 Reseteado")

        elif key == ord('q'):  # salir
            print("\nDetenido por usuario.")
            cv2.destroyAllWindows()
            exit(0)

cv2.destroyAllWindows()
print(f"\n✅ Etiquetado completo. Máscaras en: {OUTPUT_MASKS}")
