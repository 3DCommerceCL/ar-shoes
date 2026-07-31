"""Etiquetador RÁPIDO de keypoints sobre los frames REALES ya segmentados.

Por qué hace falta: la cabeza de keypoints hoy sólo se entrena con renders sintéticos, y encima
el PCK se mide sólo sobre sintético — o sea que no sabemos si transfiere al mundo real. Marcando
unos 100 frames reales conseguimos las dos cosas: supervisión real y una métrica honesta.

Las máscaras ya existen (SAM2), así que acá SÓLO se clickean los 6 puntos: ~10 segundos por frame.

    py -3 label_keypoints.py                      # 100 frames repartidos entre todos los clips
    py -3 label_keypoints.py --n 200 --clip clip04
    py -3 label_keypoints.py --review             # repasar lo ya etiquetado

CONTROLES
    clic izq        marca el punto que pide arriba (en orden)
    ESPACIO         ese punto NO se ve (queda sin marcar, es válido)
    L / R           lado del pie que estás marcando (izquierdo/derecho); por defecto derecho
    ENTER           guardar y pasar al siguiente
    Z               deshacer el último punto
    S               saltear este frame
    Q               salir (lo guardado se conserva)

Los 6 puntos, EN ORDEN (mirá tu propio pie si dudás):
    1 heel       talón, lo más atrás
    2 toe        empeine, sobre la 2ª cabeza metatarsal (donde nace el dedo índice)
    3 ankle_in   hueso del tobillo INTERNO (el que mira al otro pie)
    4 ankle_out  hueso del tobillo EXTERNO
    5 ball       bola del pie (planta, bajo el dedo gordo)
    6 toe_tip    punta del dedo gordo
Se marcan aunque el zapato los tape: el modelo debe inferir el pie DEBAJO del calzado
(por eso el zapato virtual puede calzarse). Estimá dónde están y clickeá ahí.
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
    """Reparte los frames a etiquetar ENTRE los clips (no 100 seguidos del mismo, que se parecen)."""
    imgs = sorted((data / "images").glob("*.jpg"))
    if clip_filter:
        imgs = [p for p in imgs if p.stem.startswith(clip_filter)]
    by_clip = {}
    for p in imgs:
        by_clip.setdefault(p.stem.split("_f")[0], []).append(p)
    out = []
    per = max(1, n // max(len(by_clip), 1))
    for clip, lst in sorted(by_clip.items()):
        step = max(1, len(lst) // per)
        out += lst[::step][:per]
    random.Random(0).shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description="Etiquetar keypoints en frames reales")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=100, help="cuántos frames etiquetar")
    ap.add_argument("--clip", default=None, help="filtrar por clip (ej. clip04)")
    ap.add_argument("--review", action="store_true", help="repasar los ya etiquetados")
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
        print("Nada para etiquetar (¿ya están todos?). Usá --review para repasar.")
        return
    print(f"A etiquetar: {len(frames)} frames\n{__doc__.split('CONTROLES')[1].split('Los 6 puntos')[0]}")

    win = "keypoints"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISP, DISP)
    state = {"pts": [], "side": "right"}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN and len(state["pts"]) < len(KP_NAMES):
            state["pts"].append((x, y, 1))
            redraw()

    def redraw():
        img = state["disp"].copy()
        for i, (x, y, v) in enumerate(state["pts"]):
            if v:
                cv2.circle(img, (x, y), 6, KP_COLORS[i], -1)
                cv2.circle(img, (x, y), 6, (0, 0, 0), 1)
                cv2.putText(img, str(i + 1), (x + 9, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, KP_COLORS[i], 2)
        i = len(state["pts"])
        cv2.rectangle(img, (0, 0), (img.shape[1], 54), (0, 0, 0), -1)
        if i < len(KP_NAMES):
            cv2.putText(img, f"{i+1}/6  {KP_NAMES[i]}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, KP_COLORS[i], 2)
            cv2.putText(img, KP_HELP[KP_NAMES[i]], (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        else:
            cv2.putText(img, "LISTO - ENTER guarda y sigue", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2)
        cv2.putText(img, f"pie: {state['side'].upper()}  [L/R]  ESPACIO=no se ve  Z=deshacer  S=saltear  Q=salir",
                    (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1)
        cv2.imshow(win, img)

    cv2.setMouseCallback(win, on_mouse)
    saved = 0
    for idx, p in enumerate(frames):
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = cv2.imread(str(data / "masks" / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        disp = cv2.resize(img, (DISP, DISP))
        if m is not None:   # contorno de la máscara: ayuda a ubicar el pie sin taparlo
            mm = cv2.resize(m, (DISP, DISP), interpolation=cv2.INTER_NEAREST)
            for cls, col in {1: (255, 120, 0), 3: (0, 255, 0)}.items():
                cnts, _ = cv2.findContours((mm == cls).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(disp, cnts, -1, col, 1)
        state["disp"] = disp
        state["pts"] = list(done.get(p.stem, {}).get(state["side"], []) and []) or []
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
            if k == ord('z') and state["pts"]:
                state["pts"].pop(); redraw()
            elif k == ord(' ') and len(state["pts"]) < len(KP_NAMES):
                state["pts"].append((0, 0, 0)); redraw()
            elif k in (ord('l'), ord('r')):
                state["side"] = "left" if k == ord('l') else "right"; redraw()
            elif k in (13, 10):
                if len(state["pts"]) < len(KP_NAMES):
                    continue                      # faltan puntos: no guardar a medias
                block = [[round(x / DISP, 5), round(y / DISP, 5), float(v)]
                         for (x, y, v) in state["pts"]]
                done[p.stem] = {"left": None, "right": None}
                done[p.stem][state["side"]] = block
                saved += 1
                save_all(kp_path, done)           # guardado incremental: si cortás, no perdés nada
                break
        state["pts"] = []

    cv2.destroyAllWindows()
    save_all(kp_path, done)
    print(f"\n✅ {saved} frames etiquetados en esta sesión. Total: {len(done)} → {kp_path}")


if __name__ == "__main__":
    main()
