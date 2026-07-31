"""Control de calidad de ETIQUETAS por desacuerdo con el modelo.

La deriva de SAM2 (que un objeto se pierda y otra clase se lo coma) es gradual y no se detecta
mirando áreas: en clip06 el zapato bajó de 7% a 3.5% sin cruzar ningún umbral, pero a ojo se ve
que el pie quedó pintado como pantalón.

Truco: usar el modelo ya entrenado como segundo opinador. Los frames mal etiquetados son minoría,
así que el modelo aprendió la regla correcta y DISCREPA justamente ahí. Un IoU bajo entre la
predicción y la etiqueta marca al frame como sospechoso.

    py -3 qc_labels.py --ckpt foot_v1.pth --data data
    py -3 qc_labels.py --ckpt foot_v1.pth --data data --drop      # borra los peores
    py -3 qc_labels.py --ckpt foot_v1.pth --data data --thr 0.45  # umbral de IoU

Genera <data>/qc_report.html con los peores casos (imagen | etiqueta | predicción) para revisar
a ojo antes de borrar nada.
"""
import sys
import html
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_model import _build_model, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

COLORS = {1: (255, 120, 0), 2: (0, 0, 255), 3: (0, 255, 0)}


def iou_of(pred, gt, cls):
    p, g = pred == cls, gt == cls
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else None


def main():
    ap = argparse.ArgumentParser(description="Detectar etiquetas malas por desacuerdo con el modelo")
    ap.add_argument("--ckpt", default="foot_v1.pth")
    ap.add_argument("--data", default="data")
    ap.add_argument("--thr", type=float, default=0.5, help="IoU(pie∪zapato) mínimo aceptable")
    ap.add_argument("--drop", action="store_true", help="borrar los frames por debajo del umbral")
    ap.add_argument("--top", type=int, default=60, help="cuántos mostrar en el informe")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model(small=ck.get("small", False))
    model.load_state_dict(ck["model"])
    model.eval().to(dev)
    print(f"modelo {args.ckpt} en {dev}")

    data = Path(args.data)
    imgs = sorted((data / "images").glob("*.jpg"))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    rows = []
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            batch = imgs[i:i + 16]
            xs, gts, keep = [], [], []
            for p in batch:
                im = cv2.imread(str(p))
                gt = cv2.imread(str(data / "masks" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
                if im is None or gt is None:
                    continue
                im = cv2.cvtColor(cv2.resize(im, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
                gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                t = torch.from_numpy(im.transpose(2, 0, 1).astype(np.float32) / 255.0)
                xs.append((t - mean) / std); gts.append(gt); keep.append(p)
            if not xs:
                continue
            seg, _ = model(torch.stack(xs).to(dev))
            pred = seg.argmax(1).cpu().numpy().astype(np.uint8)
            for p, gt, pr in zip(keep, gts, pred):
                fs_p, fs_g = pr >= 2, gt >= 2
                u = (fs_p | fs_g).sum()
                iou = float((fs_p & fs_g).sum() / u) if u else 1.0
                rows.append((p, iou, gt, pr))
            print(f"\r  {min(i+16, len(imgs))}/{len(imgs)}", end="")
    print()

    rows.sort(key=lambda r: r[1])
    bad = [r for r in rows if r[1] < args.thr]
    ious = [r[1] for r in rows]
    print(f"IoU etiqueta-vs-modelo: mediana {np.median(ious):.3f} | "
          f"por debajo de {args.thr}: {len(bad)}/{len(rows)}")

    by_clip = {}
    for p, iou, _, _ in bad:
        by_clip.setdefault(p.stem.split('_f')[0], []).append(p.stem)
    for c, lst in sorted(by_clip.items()):
        print(f"  {c}: {len(lst)} sospechosos  {lst[:4]}{'...' if len(lst) > 4 else ''}")

    # informe visual de los peores
    out = data / "qc_labels"
    out.mkdir(exist_ok=True)
    cards = []
    for p, iou, gt, pr in rows[:args.top]:
        im = cv2.resize(cv2.imread(str(p)), (IMG_SIZE, IMG_SIZE))
        a, b = im.copy(), im.copy()
        for c, col in COLORS.items():
            a[gt == c] = (0.5 * np.array(col) + 0.5 * a[gt == c]).astype(np.uint8)
            b[pr == c] = (0.5 * np.array(col) + 0.5 * b[pr == c]).astype(np.uint8)
        cv2.imwrite(str(out / f"{p.stem}.jpg"), np.hstack([im, a, b]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        cards.append((p.stem, iou))
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>QC etiquetas</title>"
        "<style>body{background:#111;color:#ddd;font-family:system-ui;margin:14px}"
        "figure{display:block;margin:10px 0}img{width:660px;border-radius:4px}"
        "figcaption{font-size:12px;color:#9aa}</style>"
        "<h1>Etiquetas sospechosas (peor IoU contra el modelo)</h1>"
        "<p style=color:#89a;font-size:13px>Izquierda: imagen · Centro: ETIQUETA (SAM2) · "
        "Derecha: PREDICCIÓN del modelo. Si la predicción es la correcta, la etiqueta está mal.</p>"
        + "".join(f'<figure><img src="{html.escape(s)}.jpg" loading=lazy>'
                  f'<figcaption>{html.escape(s)} — IoU {i:.2f}</figcaption></figure>'
                  for s, i in cards), encoding="utf-8")
    print(f"informe: {out / 'index.html'}")

    if args.drop:
        for p, _, _, _ in bad:
            p.unlink(missing_ok=True)
            (data / "masks" / f"{p.stem}.png").unlink(missing_ok=True)
        print(f"borrados {len(bad)} frames")


if __name__ == "__main__":
    main()
