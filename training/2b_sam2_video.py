"""
PASO 2B — Etiquetado MASIVO de videos con SAM2 (vía principal de datos reales)
==============================================================================
Un video de 30s = ~900 frames. Con SAM2 clickeás pierna/pie/zapato en el PRIMER frame
y las máscaras se propagan solas a todo el video → cientos de frames etiquetados por
minuto de trabajo humano (la estrategia de datos de Wanna/Kivisense).

Salida (formato del Dataset de train_model.py):
    <out>/images/<clip>_fNNNNN.jpg
    <out>/masks/<clip>_fNNNNN.png      ← índices {0 fondo, 1 pierna, 2 pie, 3 zapato}
    (keypoints.jsonl NO se genera aquí: estos frames entrenan solo la cabeza de segmentación,
     w_kp=0; los keypoints vienen del sintético de Blender y de las fotos de 2_sam_label.py)

El split de train_model.py agrupa <clip>_fNNNNN por <clip> (ver _base_stem), así los frames
de un mismo video nunca quedan repartidos entre train y val (sin fuga temporal).

Instalación (una vez):
    pip install torch torchvision            # ya instalado en este proyecto
    git clone https://github.com/facebookresearch/sam2 && cd sam2 && pip install -e .
    # checkpoint (elegir según GPU/CPU; tiny funciona en CPU, lento):
    #   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
    #   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
    #   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

Uso:
    python 2b_sam2_video.py --video data_videos/clip01.mp4
    python 2b_sam2_video.py --review --out data              # control de calidad de lo exportado

Rendimiento medido (checkpoint tiny, CPU de este equipo): ~9.5 s/frame a 320px — a 1024px es varias
veces más. Para CPU: usar --extract_stride 3 y --max_side 640 (un clip de 30s queda en ~100 frames
procesados). Para lotes grandes conviene GPU (Colab: sube el script + videos, cambia --device cuda).

Controles (ventana del primer frame):
    1 / 2 / 3    clase activa (pierna / pie / zapato)
    clic izq     punto positivo de la clase activa
    clic der     punto negativo de la clase activa
    ENTER        propagar a todo el video y exportar
    R            resetear clics    |    Q  salir sin exportar
En --review: cualquier tecla = siguiente frame, D = borrar el frame mostrado, Q = salir.
"""
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CLASS_NAMES  = {1: "pierna", 2: "pie", 3: "zapato"}
CLASS_COLORS = {1: (255, 80, 80), 2: (80, 80, 255), 3: (80, 255, 80)}  # BGR
MAX_DISP = 960


# ---------------------------------------------------------------------
# extracción de frames (SAM2 init_state espera un dir de JPEGs numerados)
# ---------------------------------------------------------------------
def extract_frames(video_path, tmp_dir, max_side=1024, extract_stride=1, square=True):
    """Extrae 1 de cada extract_stride frames, renumerados consecutivos (SAM2 los exige así).

    square=True hace RECORTE CUADRADO CENTRADO, igual que inference.js en la app: así el modelo
    entrena viendo exactamente el mismo encuadre que verá en producción (y coincide además con
    los renders sintéticos, que son cuadrados). Sin esto, un video vertical se aplastaba a
    cuadrado en el entrenamiento mientras la app recorta → distorsión distinta = peor precisión.
    IMPORTANTE: hay que grabar con los pies CENTRADOS, o el recorte se los come."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir el video: {video_path}")
    n = src = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src % extract_stride == 0:
            if square:
                h, w = frame.shape[:2]
                s = min(h, w)
                frame = frame[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
            h, w = frame.shape[:2]
            if max(h, w) > max_side:
                sc = max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * sc), int(h * sc)))
            cv2.imwrite(str(tmp_dir / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        src += 1
    cap.release()
    if n == 0:
        raise SystemExit("El video no tiene frames legibles")
    return n


# ---------------------------------------------------------------------
# UI de clics sobre el primer frame
# ---------------------------------------------------------------------
def collect_clicks(first_frame):
    state = {"active": 2, "points": {1: ([], []), 2: ([], []), 3: ([], [])}, "done": False}
    disp = first_frame.copy()
    h, w = disp.shape[:2]
    scale = min(1.0, MAX_DISP / max(h, w))
    if scale < 1:
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)))

    def refresh():
        ov = disp.copy()
        for cls, (pts, lbs) in state["points"].items():
            for (px, py), lb in zip(pts, lbs):
                cv2.circle(ov, (px, py), 6, CLASS_COLORS[cls] if lb == 1 else (0, 0, 0), -1)
        cv2.rectangle(ov, (0, 0), (ov.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(ov, f"Clase activa: {state['active']} {CLASS_NAMES[state['active']]}  "
                        f"(1/2/3 cambia, ENTER propaga, R resetea, Q sale)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("SAM2 - primer frame", ov)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["points"][state["active"]][0].append([x, y]); state["points"][state["active"]][1].append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["points"][state["active"]][0].append([x, y]); state["points"][state["active"]][1].append(0)
        else:
            return
        refresh()

    cv2.namedWindow("SAM2 - primer frame", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM2 - primer frame", on_mouse)
    refresh()
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('1'), ord('2'), ord('3')):
            state["active"] = k - ord('0'); refresh()
        elif k == ord('r'):
            for c in state["points"]:
                state["points"][c] = ([], [])
            refresh()
        elif k in (13, 10):   # ENTER
            cv2.destroyAllWindows()
            # devolver en coords del frame ORIGINAL (deshacer el scale de display)
            out = {}
            for cls, (pts, lbs) in state["points"].items():
                if pts:
                    out[cls] = (np.array(pts, np.float32) / scale, np.array(lbs, np.int32))
            if not out:
                raise SystemExit("Sin clics — nada que propagar")
            return out
        elif k == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelado por el usuario")


# ---------------------------------------------------------------------
# propagación con SAM2
# ---------------------------------------------------------------------
def propagate(frames_dir, clicks, checkpoint, config, device):
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(config, checkpoint, device=device)
    state = predictor.init_state(video_path=str(frames_dir))

    for cls, (pts, lbs) in clicks.items():
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0,
                                        obj_id=cls, points=pts, labels=lbs)

    per_frame = {}
    with torch.inference_mode():
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks = {}
            for i, oid in enumerate(obj_ids):
                masks[int(oid)] = (mask_logits[i] > 0.0).squeeze().cpu().numpy().astype(np.uint8)
            per_frame[frame_idx] = masks
    return per_frame


def compose_index_mask(masks, shape):
    idx = np.zeros(shape, np.uint8)
    for cls in (1, 2, 3):  # prioridad: zapato pisa a pie pisa a pierna
        if cls in masks:
            m = masks[cls]
            if m.shape != shape:
                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            idx[m > 0] = cls
    return idx


# ---------------------------------------------------------------------
# review
# ---------------------------------------------------------------------
def review(out_dir):
    out_dir = Path(out_dir)
    imgs = sorted((out_dir / "images").glob("*_f*.jpg"))
    print(f"{len(imgs)} frames exportados. Cualquier tecla=siguiente, D=borrar, Q=salir")
    for p in imgs:
        mp = out_dir / "masks" / (p.stem + ".png")
        img = cv2.imread(str(p))
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        ov = img.copy()
        if m is not None:
            for cls, col in CLASS_COLORS.items():
                ov[m == cls] = (0.5 * np.array(col) + 0.5 * ov[m == cls]).astype(np.uint8)
        cv2.imshow("review", ov)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('d'):
            p.unlink(missing_ok=True); mp.unlink(missing_ok=True)
            print(f"  borrado {p.name}")
        elif k == ord('q'):
            break
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Etiquetado masivo de videos con SAM2")
    ap.add_argument("--video", help="clip .mp4 a etiquetar")
    ap.add_argument("--out", default="data", help="carpeta de salida (images/ + masks/)")
    ap.add_argument("--stride", type=int, default=5, help="exportar 1 de cada N frames procesados")
    ap.add_argument("--extract_stride", type=int, default=1,
                    help="procesar 1 de cada N frames del video (subí a 3 en CPU)")
    ap.add_argument("--max_side", type=int, default=1024, help="lado máximo de frame (bajá a 640 en CPU)")
    ap.add_argument("--no_crop", action="store_true",
                    help="NO recortar cuadrado (sólo si los pies quedaron fuera del centro del cuadro)")
    ap.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml",
                    help="config del checkpoint (t/s/b+/l deben coincidir)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--review", action="store_true")
    args = ap.parse_args()

    if args.review:
        review(args.out)
        return
    if not args.video:
        raise SystemExit("Falta --video (o usa --review)")

    try:
        import torch  # noqa
        import sam2   # noqa
    except ImportError as e:
        raise SystemExit(f"Falta instalar SAM2 ({e}).\n"
                         "git clone https://github.com/facebookresearch/sam2 && cd sam2 && pip install -e .\n"
                         "y descargar el checkpoint (ver docstring).")
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}" + (" (CPU: lento — usa el checkpoint tiny)" if device == "cpu" else ""))

    video = Path(args.video)
    clip = video.stem.replace("_", "-")   # el stem no debe contener _f para no romper _base_stem
    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    try:
        n = extract_frames(video, tmp, max_side=args.max_side, extract_stride=args.extract_stride,
                           square=not args.no_crop)
        print(f"{n} frames extraídos de {video.name} (extract_stride {args.extract_stride}, "
              f"max {args.max_side}px, {'recorte cuadrado centrado' if not args.no_crop else 'SIN recorte'})")
        first = cv2.imread(str(tmp / "00000.jpg"))
        clicks = collect_clicks(first)
        print(f"Propagando clases {sorted(clicks.keys())} con SAM2 ({n} frames)…")
        per_frame = propagate(tmp, clicks, args.checkpoint, args.config, device)

        exported = 0
        for fi in range(0, n, args.stride):
            if fi not in per_frame:
                continue
            frame = cv2.imread(str(tmp / f"{fi:05d}.jpg"))
            idx = compose_index_mask(per_frame[fi], frame.shape[:2])
            if (idx > 0).sum() < 200:   # propagación perdida → no exportar
                continue
            name = f"{clip}_f{fi:05d}"
            cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(out_dir / "masks" / f"{name}.png"), idx)
            exported += 1
        print(f"✅ {exported} frames exportados a {out_dir}/ (stride {args.stride})")
        print(f"   Control de calidad: python 2b_sam2_video.py --review --out {out_dir}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
