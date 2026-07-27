"""Visor de máscaras: genera las imágenes CON LOS COLORES PINTADOS y un HTML para navegarlas.

Las máscaras que guarda el pipeline (data/masks/*.png) son "mapas de índices": cada píxel vale
0,1,2,3 (fondo/pierna/pie/zapato). Sobre 255 niveles posibles eso se ve casi NEGRO al abrirlas
en un visor común — están bien, sólo que no son imágenes para mirar. Este script produce la
versión coloreada para inspección humana.

    py -3 preview_masks.py                     # todo lo que haya en data/
    py -3 preview_masks.py --data data --out data/preview --alpha 0.5
    py -3 preview_masks.py --clip clip03       # sólo un clip

Salida: <out>/*.jpg (imagen + máscara superpuesta) y <out>/index.html para navegar todo.
El HTML marca en rojo los frames sospechosos (sin zapato, o con un área rara).
"""
import argparse
import html
from pathlib import Path

import cv2
import numpy as np

CLASSES = {0: ("fondo", None), 1: ("pierna", (255, 120, 0)),
           2: ("pie", (0, 0, 255)), 3: ("zapato", (0, 255, 0))}   # BGR


def overlay(img, mask, alpha=0.5):
    ov = img.copy()
    for cls, (_, color) in CLASSES.items():
        if color is None:
            continue
        sel = mask == cls
        if sel.any():
            ov[sel] = (alpha * np.array(color) + (1 - alpha) * ov[sel]).astype(np.uint8)
    return ov


def main():
    ap = argparse.ArgumentParser(description="Genera las máscaras pintadas + un HTML para verlas")
    ap.add_argument("--data", default="data", help="carpeta con images/ y masks/")
    ap.add_argument("--out", default=None, help="carpeta de salida (default: <data>/preview)")
    ap.add_argument("--clip", default=None, help="filtrar por prefijo de clip (ej. clip03)")
    ap.add_argument("--alpha", type=float, default=0.5, help="opacidad del color (0-1)")
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out) if args.out else data / "preview"
    out.mkdir(parents=True, exist_ok=True)

    pattern = f"{args.clip}_*" if args.clip else "*"
    imgs = sorted(list((data / "images").glob(f"{pattern}.jpg")) +
                  list((data / "images").glob(f"{pattern}.png")))
    if not imgs:
        raise SystemExit(f"No hay imágenes en {data}/images (filtro: {pattern})")

    cards = []
    for p in imgs:
        mp = data / "masks" / (p.stem + ".png")
        img = cv2.imread(str(p))
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / f"{p.stem}.jpg"), overlay(img, mask, args.alpha),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        pct = {c: 100 * float((mask == c).mean()) for c in CLASSES}
        # sospechoso: sin zapato, o el zapato ocupa casi todo (máscara desbordada)
        warn = pct[3] < 1 or pct[3] > 60 or pct[1] > 70
        cards.append((p.stem, pct, warn))

    rows = "".join(
        f'<figure class="{"warn" if w else ""}">'
        f'<img src="{html.escape(stem)}.jpg" loading="lazy">'
        f'<figcaption>{html.escape(stem)}<br>'
        f'<span class=l1>pierna {pct[1]:.1f}%</span> '
        f'<span class=l2>pie {pct[2]:.1f}%</span> '
        f'<span class=l3>zapato {pct[3]:.1f}%</span>'
        f'{"<br><b>⚠ revisar</b>" if w else ""}</figcaption></figure>'
        for stem, pct, w in cards)

    n_warn = sum(1 for _, _, w in cards if w)
    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Máscaras — {len(cards)} frames</title>
<style>
 body{{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:16px}}
 h1{{font-size:17px}} .leg span{{margin-right:14px;font-size:13px}}
 .sw{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;margin-right:4px}}
 figure{{display:inline-block;margin:5px;width:210px;vertical-align:top}}
 figure img{{width:210px;border-radius:5px;display:block;background:#000}}
 figcaption{{font-size:11px;color:#9aa;margin-top:3px;line-height:1.45}}
 figure.warn img{{outline:3px solid #e34}} figure.warn figcaption b{{color:#f66}}
 .l1{{color:#4af}} .l2{{color:#f66}} .l3{{color:#5e6}}
</style>
<h1>{len(cards)} frames etiquetados · {n_warn} para revisar</h1>
<p class=leg><span><i class=sw style="background:#08f"></i>pierna</span>
<span><i class=sw style="background:#f44"></i>pie</span>
<span><i class=sw style="background:#3d6"></i>zapato</span>
<span>el fondo queda sin pintar</span></p>
{rows}""", encoding="utf-8")

    print(f"✅ {len(cards)} imágenes coloreadas en {out}/")
    print(f"   Abrí: {out / 'index.html'}")
    if n_warn:
        print(f"   ⚠ {n_warn} frames marcados para revisar")


if __name__ == "__main__":
    main()
