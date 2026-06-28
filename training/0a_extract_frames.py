"""
PASO 0A — Extraer frames de videos para dataset
================================================
Toma videos grabados con el celular y extrae 1 frame cada N.
Resultado: imágenes listas para etiquetar con SAM2.

Requisitos:
    pip install opencv-python tqdm

Uso:
    python 0a_extract_frames.py --videos_dir videos/ --every_n 15
    # 20 videos × 30s × 30fps ÷ 15 = ~1,200 frames base
"""

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--videos_dir", default="videos",    help="Carpeta con videos .mp4/.mov")
parser.add_argument("--out_dir",    default="data/images", help="Carpeta de salida")
parser.add_argument("--every_n",    type=int, default=15,  help="Tomar 1 frame cada N")
parser.add_argument("--size",       type=int, default=256, help="Tamaño de salida")
args = parser.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

videos = list(Path(args.videos_dir).glob("*.mp4")) + \
         list(Path(args.videos_dir).glob("*.mov")) + \
         list(Path(args.videos_dir).glob("*.MP4"))

total = 0
for vid_path in tqdm(videos, desc="Videos"):
    cap = cv2.VideoCapture(str(vid_path))
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.every_n == 0:
            frame = cv2.resize(frame, (args.size, args.size))
            name  = f"{vid_path.stem}_f{frame_idx:05d}.jpg"
            cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            total += 1
        frame_idx += 1
    cap.release()

print(f"\n✅ {total} frames extraídos en {out}/")
print(f"   Siguiente paso: etiquetar con SAM2 → python 2_sam_label.py")
