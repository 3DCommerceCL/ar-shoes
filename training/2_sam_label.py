"""
PASO 2 — Etiquetado con SAM: máscara MULTICLASE + keypoints
============================================================
Etiqueta fotos reales en el MISMO formato que produce 0b_blender_render.py, para poder
mezclarlas en el entrenamiento / fine-tuning y como test set congelado (ver ROADMAP T3.6).

Salida:
    data/masks/<stem>.png   ← PNG uint8 con índices {0 fondo, 1 pierna, 2 pie, 3 zapato}
    data/keypoints.jsonl    ← una línea por imagen: {"file": "...jpg", "kps": [[x,y,vis]×6]}

Controles:
    1 / 2 / 3   → clase activa (pierna / pie / zapato)
    clic izq    → punto SAM positivo para la clase activa
    clic der    → punto SAM negativo para la clase activa
    K           → modo keypoints: 6 clics en orden heel, toe, ankle_in, ankle_out, ball, toe_tip
                  (ESPACIO salta el keypoint actual como NO visible)
    S           → guardar (máscara + keypoints) y siguiente
    R           → resetear la imagen actual
    Q           → salir

Requisitos:
    pip install segment-anything opencv-python torch torchvision numpy
    # checkpoint SAM: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
Uso:
    python 2_sam_label.py --checkpoint sam_vit_b_01ec64.pth
    python 2_sam_label.py --review        # recorre lo etiquetado para control de calidad
"""
import cv2
import json
import argparse
from pathlib import Path

import numpy as np

KP_NAMES = ["heel", "toe", "ankle_in", "ankle_out", "ball", "toe_tip"]
CLASS_NAMES = {1: "pierna", 2: "pie", 3: "zapato"}
CLASS_COLORS = {1: (255, 80, 80), 2: (80, 80, 255), 3: (80, 255, 80)}  # BGR
INPUT_IMAGES = Path("data/images")
OUTPUT_MASKS = Path("data/masks")
KP_PATH = Path("data/keypoints.jsonl")
MAX_DISP = 900


def load_kp_map():
    m = {}
    if KP_PATH.exists():
        for line in KP_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                m[r["file"]] = r["kps"]
    return m


def save_kp_map(m):
    with open(KP_PATH, "w", encoding="utf-8") as f:
        for file, kps in m.items():
            f.write(json.dumps({"file": file, "kps": kps}) + "\n")


def compose_index_mask(class_masks, shape):
    """class_masks: dict clase→máscara binaria. Compone por prioridad pierna<pie<zapato."""
    idx = np.zeros(shape, np.uint8)
    for cls in (1, 2, 3):
        m = class_masks.get(cls)
        if m is not None:
            idx[m > 0] = cls
    return idx


def draw_overlay(base, class_masks, points, kp_list, active_class, kp_mode, kp_index):
    ov = base.copy()
    for cls, m in class_masks.items():
        if m is not None:
            ov[m > 0] = (0.5 * np.array(CLASS_COLORS[cls]) + 0.5 * ov[m > 0]).astype(np.uint8)
    for cls, (pts, lbs) in points.items():
        for (px, py), lb in zip(pts, lbs):
            cv2.circle(ov, (px, py), 5, CLASS_COLORS[cls] if lb == 1 else (0, 0, 0), -1)
    for i, kp in enumerate(kp_list):
        if kp is not None:
            cv2.circle(ov, (kp[0], kp[1]), 6, (0, 220, 255), -1)
            cv2.putText(ov, str(i), (kp[0] + 6, kp[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    status = (f"KEYPOINT {kp_index}/6: {KP_NAMES[kp_index]} (ESPACIO=no visible)"
              if kp_mode and kp_index < 6 else
              ("KEYPOINTS completos" if kp_mode else f"Clase activa: {active_class} {CLASS_NAMES[active_class]}"))
    cv2.rectangle(ov, (0, 0), (ov.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(ov, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return ov


def review_mode():
    imgs = sorted(list(INPUT_IMAGES.glob("*.jpg")) + list(INPUT_IMAGES.glob("*.png")))
    kp_map = load_kp_map()
    for p in imgs:
        mp = OUTPUT_MASKS / (p.stem + ".png")
        if not mp.exists():
            continue
        img = cv2.imread(str(p))
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        ov = img.copy()
        for cls, col in CLASS_COLORS.items():
            ov[m == cls] = (0.5 * np.array(col) + 0.5 * ov[m == cls]).astype(np.uint8)
        h, w = img.shape[:2]
        for i, kp in enumerate(kp_map.get(p.name, [])):
            if kp[2] > 0:
                cv2.circle(ov, (int(kp[0] * w), int(kp[1] * h)), 6, (0, 220, 255), -1)
        cv2.imshow("review", ov)
        print(f"{p.name}: clases={np.unique(m).tolist()}  (cualquier tecla=siguiente, q=salir)")
        if (cv2.waitKey(0) & 0xFF) == ord('q'):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth")
    ap.add_argument("--model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()

    OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)
    if args.review:
        review_mode()
        return

    import torch
    from segment_anything import sam_model_registry, SamPredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {device.upper()}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    kp_map = load_kp_map()
    imgs = sorted(list(INPUT_IMAGES.glob("*.jpg")) + list(INPUT_IMAGES.glob("*.png")))

    state = {}
    def reset_state():
        state["points"] = {1: ([], []), 2: ([], []), 3: ([], [])}
        state["masks"] = {1: None, 2: None, 3: None}
        state["kp"] = [None] * 6
        state["active"] = 2
        state["kp_mode"] = False
        state["kp_index"] = 0

    def predict(cls):
        pts, lbs = state["points"][cls]
        if not pts:
            state["masks"][cls] = None
            return
        masks, scores, _ = predictor.predict(point_coords=np.array(pts),
                                             point_labels=np.array(lbs), multimask_output=True)
        state["masks"][cls] = (masks[int(np.argmax(scores))]).astype(np.uint8)

    def on_mouse(event, x, y, flags, param):
        if state["kp_mode"]:
            if event == cv2.EVENT_LBUTTONDOWN and state["kp_index"] < 6:
                state["kp"][state["kp_index"]] = (x, y)
                state["kp_index"] += 1
        else:
            cls = state["active"]
            if event == cv2.EVENT_LBUTTONDOWN:
                state["points"][cls][0].append([x, y]); state["points"][cls][1].append(1); predict(cls)
            elif event == cv2.EVENT_RBUTTONDOWN:
                state["points"][cls][0].append([x, y]); state["points"][cls][1].append(0); predict(cls)
        refresh()

    def refresh():
        cv2.imshow("SAM Labeler", draw_overlay(disp_bgr, state["masks"], state["points"],
                                               state["kp"], state["active"], state["kp_mode"], state["kp_index"]))

    print(__doc__.split("Requisitos")[0])
    cv2.namedWindow("SAM Labeler", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM Labeler", on_mouse)

    for idx, p in enumerate(imgs):
        mp = OUTPUT_MASKS / (p.stem + ".png")
        if mp.exists():
            print(f"  [OK] ya etiquetada: {p.name}")
            continue
        orig = cv2.imread(str(p))
        h0, w0 = orig.shape[:2]
        scale = min(1.0, MAX_DISP / max(h0, w0))
        disp_bgr = cv2.resize(orig, (int(w0 * scale), int(h0 * scale))) if scale < 1 else orig.copy()
        dh, dw = disp_bgr.shape[:2]
        predictor.set_image(cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB))
        reset_state(); refresh()
        print(f"[{idx+1}/{len(imgs)}] {p.name}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('1'), ord('2'), ord('3')):
                state["active"] = key - ord('0'); state["kp_mode"] = False; refresh()
            elif key == ord('k'):
                state["kp_mode"] = True; state["kp_index"] = 0; refresh()
            elif key == ord(' ') and state["kp_mode"] and state["kp_index"] < 6:
                state["kp"][state["kp_index"]] = None; state["kp_index"] += 1; refresh()
            elif key == ord('r'):
                reset_state(); refresh(); print("  reseteado")
            elif key == ord('q'):
                save_kp_map(kp_map); cv2.destroyAllWindows(); return
            elif key == ord('s'):
                idx_mask = compose_index_mask(state["masks"], (dh, dw))
                if scale < 1:
                    idx_mask = cv2.resize(idx_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(mp), idx_mask)
                kps = []
                for kp in state["kp"]:
                    if kp is None:
                        kps.append([0.0, 0.0, 0.0])
                    else:
                        kps.append([round(kp[0] / dw, 5), round(kp[1] / dh, 5), 1.0])
                kp_map[p.name] = kps
                save_kp_map(kp_map)
                print(f"  guardada: {mp.name}  (clases {np.unique(idx_mask).tolist()}, "
                      f"{sum(1 for k in kps if k[2] > 0)}/6 kps)")
                break

    save_kp_map(kp_map)
    cv2.destroyAllWindows()
    print(f"\nListo. Máscaras en {OUTPUT_MASKS}, keypoints en {KP_PATH}")


if __name__ == "__main__":
    main()
