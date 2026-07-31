"""Etiquetador RÁPIDO de keypoints sobre los frames REALES ya segmentados.

Por qué hace falta: la cabeza de keypoints hoy sólo se entrena con renders sintéticos, y encima
el PCK se mide sólo sobre sintético — o sea que no sabemos si transfiere al mundo real. Marcando
unos 100 frames reales conseguimos las dos cosas: supervisión real y una métrica honesta.

Las máscaras ya existen (SAM2), así que acá SÓLO se clickean los 6 puntos: ~10 segundos por pie.

    py -3 label_keypoints.py                      # 100 frames repartidos entre todos los clips
    py -3 label_keypoints.py --n 200 --clip clip04
    py -3 label_keypoints.py --review             # repasar lo ya etiquetado

⚠ REGLA IMPORTANTE — marcá TODOS los pies que se vean en el frame.
Si hay dos pies y marcás uno solo, el otro queda con heatmaps en cero, o sea que le enseñás al
modelo que "ese pie visible no tiene keypoints". Es peor que no usar el frame. El programa cuenta
los pies en la máscara y no te deja guardar si marcaste menos (podés saltear con S).

CONTROLES
    clic izq        marca el punto que pide arriba (en orden)
    O               cambiar al OTRO pie (para marcar el segundo)
    L / R           elegir explícitamente pie izquierdo / derecho
    ESPACIO         ese punto NO se ve (queda sin marcar, es válido)
    Z               deshacer el último punto
    ENTER           guardar (todos los pies completos) y pasar al siguiente
    S               saltear este frame
    Q               salir (lo guardado se conserva)

Los 6 puntos, EN ORDEN (mirá tu propio pie si dudás):
    1 heel       talón, lo más atrás
    2 toe        empeine, sobre la 2ª cabeza metatarsal (donde nace el dedo índice)
    3 ankle_in   hueso del tobillo INTERNO (el que mira al otro pie)
    4 ankle_out  hueso del tobillo EXTERNO
    5 ball       bola del pie (planta, bajo el dedo gordo)
    6 toe_tip    punta del dedo gordo
TAPADO vs FUERA DE CUADRO — se etiquetan DISTINTO:

  · TAPADO pero DENTRO de la imagen (lo cubre el zapato, el pantalón, el otro pie):
        --> CLICKEALO IGUAL, estimando dónde está.
    Es a propósito: el modelo tiene que inferir el pie DEBAJO del calzado; de ahí sale la pose
    para calzarle el zapato virtual. Casi todos los puntos (talón, bola, punta) están dentro del
    zapato y aun así se marcan. Los renders sintéticos usan exactamente esta misma regla.

  · FUERA DE CUADRO (el punto cae más allá del borde de la imagen):
        --> ESPACIO (no visible).
    Ahí no se puede marcar nada: el heatmap no tiene píxeles fuera de la imagen. Marcarlo en el
    borde sería una etiqueta falsa.

Regla corta: si el punto está dentro del rectángulo de la imagen, se clickea aunque no se vea.
Sólo ESPACIO cuando se salió del cuadro. Si está dentro pero no tenés ni idea de dónde cae,
ESPACIO también es válido — pero que sea la excepción, no la costumbre.

¿Cuál es el izquierdo y cuál el derecho? Pensá en el pie de la PERSONA, no el de la pantalla:
en una toma POV mirando tus propios pies, el que ves a tu izquierda ES el izquierdo.
"""
import sys
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

KP_NAMES = ["heel", "toe", "ankle_in", "ankle_out", "ball", "toe_tip"]
KP_HELP = {
    "heel": "TALON (lo mas atras)",
    "toe": "EMPEINE (base del dedo indice)",
    "ankle_in": "TOBILLO INTERNO (mira al otro pie)",
    "ankle_out": "TOBILLO EXTERNO",
    "ball": "BOLA del pie (bajo el dedo gordo)",
    "toe_tip": "PUNTA del dedo gordo",
}
KP_COLORS = [(60, 60, 255), (60, 200, 255), (60, 255, 60), (255, 200, 60),
             (255, 60, 200), (255, 255, 60)]
DISP = 760


def load_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                done[Path(r["file"]).stem] = r.get("kps")
    return done


def save_all(path, done):
    with open(path, "w", encoding="utf-8") as f:
        for stem, kps in done.items():
            f.write(json.dumps({"file": f"{stem}.jpg", "kps": kps}) + "\n")


def pick_frames(data, n, clip_filter):
    """Reparte los frames ENTRE los clips: 100 seguidos del mismo clip se parecen demasiado."""
    imgs = sorted((data / "images").glob("*.jpg"))
    if clip_filter:
        imgs = [p for p in imgs if p.stem.startswith(clip_filter)]
    by_clip = {}
    for p in imgs:
        by_clip.setdefault(p.stem.split("_f")[0], []).append(p)
    out = []
    per = max(1, n // max(len(by_clip), 1))
    for _clip, lst in sorted(by_clip.items()):
        step = max(1, len(lst) // per)
        out += lst[::step][:per]
    random.Random(0).shuffle(out)
    return out[:n]


def count_feet(mask):
    """Cuántos pies se ven, por componentes conectadas de la máscara pie+zapato."""
    if mask is None:
        return 1
    fs = (mask >= 2).astype(np.uint8)
    fs = cv2.morphologyEx(fs, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(fs, 8)
    big = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 400)
    return max(1, big)


def main():
    ap = argparse.ArgumentParser(description="Etiquetar keypoints en frames reales")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--clip", default=None)
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    kp_path = data / "keypoints.jsonl"
    done = load_done(kp_path)
    print(f"Ya etiquetados: {len(done)} frames")

    if args.review:
        frames = [data / "images" / f"{s}.jpg" for s in done]
    else:
        frames = [p for p in pick_frames(data, args.n + len(done), args.clip)
                  if p.stem not in done][:args.n]
    if not frames:
        print("Nada para etiquetar. Usá --review para repasar.")
        return
    print(f"A etiquetar: {len(frames)} frames")
    print("Marcá TODOS los pies visibles: O cambia al otro pie. ENTER guarda, S saltea, Q sale.\n")

    win = "keypoints"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISP, DISP)
    st = {"pts": {"right": [], "left": []}, "side": "right", "feet": 1, "disp": None}

    def cur():
        return st["pts"][st["side"]]

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(cur()) < len(KP_NAMES):
            cur().append((x, y, 1))
            redraw()

    def redraw():
        img = st["disp"].copy()
        other = "left" if st["side"] == "right" else "right"
        for (x, y, v) in st["pts"][other]:          # el otro pie, en gris
            if v:
                cv2.circle(img, (x, y), 5, (140, 140, 140), -1)
        for i, (x, y, v) in enumerate(cur()):
            if v:
                cv2.circle(img, (x, y), 6, KP_COLORS[i], -1)
                cv2.circle(img, (x, y), 6, (0, 0, 0), 1)
                cv2.putText(img, str(i + 1), (x + 9, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, KP_COLORS[i], 2)
        i = len(cur())
        cv2.rectangle(img, (0, 0), (img.shape[1], 78), (0, 0, 0), -1)
        if i < len(KP_NAMES):
            cv2.putText(img, f"pie {st['side'].upper()}  -  {i+1}/6  {KP_NAMES[i]}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, KP_COLORS[i], 2)
            cv2.putText(img, KP_HELP[KP_NAMES[i]], (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        else:
            cv2.putText(img, f"pie {st['side'].upper()} COMPLETO", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (80, 255, 80), 2)
            cv2.putText(img, "ENTER=guardar   O=marcar el OTRO pie si tambien se ve",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 120), 1)
        col = (120, 255, 255) if st["feet"] == 1 else (120, 200, 255)
        cv2.putText(img, f"der {len(st['pts']['right'])}/6   izq {len(st['pts']['left'])}/6"
                         f"   |  en la mascara veo {st['feet']} pie(s)",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(img, "TAPADO -> clickealo igual (estimando).  ESPACIO solo si quedo FUERA del cuadro",
                    (10, img.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 255, 200), 1)
        cv2.putText(img, "L/R=elegir pie  O=otro pie  ESPACIO=fuera de cuadro  Z=deshacer  S=saltear  Q=salir",
                    (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 0), 1)
        cv2.imshow(win, img)

    cv2.setMouseCallback(win, on_mouse)
    saved = 0
    for idx, p in enumerate(frames):
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = cv2.imread(str(data / "masks" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        disp = cv2.resize(img, (DISP, DISP))
        if m is not None:      # contorno de la máscara: ubica el pie sin taparlo
            mm = cv2.resize(m, (DISP, DISP), interpolation=cv2.INTER_NEAREST)
            for cls, colr in {1: (255, 120, 0), 3: (0, 255, 0)}.items():
                cnts, _ = cv2.findContours((mm == cls).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(disp, cnts, -1, colr, 1)
        st["disp"] = disp
        st["pts"] = {"right": [], "left": []}
        st["side"] = "right"
        st["feet"] = count_feet(m)
        cv2.setWindowTitle(win, f"[{idx+1}/{len(frames)}] {p.stem}   (guardados: {saved})")
        redraw()

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                save_all(kp_path, done)
                cv2.destroyAllWindows()
                print(f"\nGuardado. Total etiquetados: {len(done)} → {kp_path}")
                return
            if k == ord('s'):
                break
            if k == ord('z') and cur():
                cur().pop(); redraw()
            elif k == ord(' ') and len(cur()) < len(KP_NAMES):
                cur().append((0, 0, 0)); redraw()
            elif k in (ord('o'), ord('l'), ord('r')):
                if k == ord('o'):
                    st["side"] = "left" if st["side"] == "right" else "right"
                else:
                    st["side"] = "left" if k == ord('l') else "right"
                redraw()
            elif k in (13, 10):
                # Guarda TODOS los lados completos. Un pie visible sin marcar le enseñaría al
                # modelo "acá no hay keypoints" — peor que descartar el frame.
                rec = {"left": None, "right": None}
                for sd in ("left", "right"):
                    if len(st["pts"][sd]) == len(KP_NAMES):
                        rec[sd] = [[round(x / DISP, 5), round(y / DISP, 5), float(v)]
                                   for (x, y, v) in st["pts"][sd]]
                n_lab = sum(1 for v in rec.values() if v)
                if n_lab == 0:
                    continue
                if n_lab < st["feet"]:
                    print(f"  ⚠ {p.stem}: marcaste {n_lab} pie(s) y en la máscara veo {st['feet']}. "
                          f"Apretá O y marcá el otro (o S para saltear este frame).")
                    continue
                done[p.stem] = rec
                saved += 1
                save_all(kp_path, done)     # incremental: si cortás, no perdés nada
                break

    cv2.destroyAllWindows()
    save_all(kp_path, done)
    print(f"\n✅ {saved} frames etiquetados en esta sesión. Total: {len(done)} → {kp_path}")


if __name__ == "__main__":
    main()
