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
def extract_frames(video_path, tmp_dir, max_side=1024, extract_stride=1, square=True,
                   force_rotate=None):
    """Extrae 1 de cada extract_stride frames, renumerados consecutivos (SAM2 los exige así).

    square=True hace RECORTE CUADRADO CENTRADO, igual que inference.js en la app: así el modelo
    entrena viendo exactamente el mismo encuadre que verá en producción (y coincide además con
    los renders sintéticos, que son cuadrados). Sin esto, un video vertical se aplastaba a
    cuadrado en el entrenamiento mientras la app recorta → distorsión distinta = peor precisión.
    IMPORTANTE: hay que grabar con los pies CENTRADOS, o el recorte se los come."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir el video: {video_path}")

    # Rotación: los teléfonos guardan el video en el sensor y anotan un flag de rotación en el
    # contenedor. Los reproductores lo respetan; OpenCV entrega el frame CRUDO (se ve "al revés"
    # o de costado). Leemos el flag y rotamos nosotros; --rotate lo fuerza a mano.
    meta_rot = 0
    try:
        meta_rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    except Exception:
        meta_rot = 0
    rot = meta_rot if force_rotate is None else int(force_rotate) % 360
    ROT_OPS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rot:
        print(f"  [rot] rotación {rot}° ({'metadato del video' if force_rotate is None else 'forzada'})")

    n = src = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rot in ROT_OPS:
            frame = cv2.rotate(frame, ROT_OPS[rot])
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
def parse_points(spec, w, h):
    """'3.1:0.4,0.58 3.2:0.6,0.55 -3.2:0.1,0.1' → {obj_id: (puntos_px, labels)}.

    Formato: 'clase[.instancia][@frame]:x,y', con x,y en fracción [0,1] del frame ya extraído y
    '-' delante para punto NEGATIVO. obj_id = clase*100 + instancia.

    '@frame' agrega el punto en ESE frame (por defecto el 0). Sirve para corregir la propagación
    donde se desvía, o para marcar partes que en el frame 0 están tapadas y se descubren después
    (p. ej. el cuello de la bota que el jean tapa al principio). Es el flujo previsto por SAM2.

    IMPORTANTE — usar una INSTANCIA POR OBJETO FÍSICO: dos puntos positivos sobre dos objetos
    separados (p. ej. los dos zapatos) hacen que SAM2 devuelva UNA sola máscara que los engloba,
    y termina agarrando el piso que hay entre medio. Verificado en clip02. Con instancias
    separadas cada zapato se sigue por su cuenta y después se funden en la misma clase.
    """
    per_obj = {}
    for tok in spec.split():
        head, coords = tok.split(":")
        neg = head.startswith("-")
        head = head.lstrip("-")
        head, _, frame_s = head.partition("@")          # clase[.inst][@frame]
        frame = int(frame_s) if frame_s else 0
        cls_s, _, inst_s = head.partition(".")
        cls = int(cls_s)
        inst = int(inst_s) if inst_s else 0
        if cls not in (1, 2, 3):
            raise SystemExit(f"Clase inválida en --points: {tok} (usar 1=pierna, 2=pie, 3=zapato)")
        fx, fy = (float(v) for v in coords.split(","))
        pts, lbs = per_obj.setdefault((cls * 100 + inst, frame), ([], []))
        pts.append([fx * w, fy * h])
        lbs.append(0 if neg else 1)
    return {k: (np.array(p, np.float32), np.array(l, np.int32)) for k, (p, l) in per_obj.items()}


def collect_clicks(first_frame):
    """Cada clic IZQUIERDO abre un objeto nuevo de la clase activa (dos zapatos = dos objetos);
    los clics DERECHOS agregan puntos negativos al último objeto abierto, para corregirlo.
    (Agrupar dos objetos separados bajo una sola máscara hace que SAM2 agarre el piso entre medio.)"""
    state = {"active": 2, "objs": {}, "count": {1: 0, 2: 0, 3: 0}, "last": None}
    disp = first_frame.copy()
    h, w = disp.shape[:2]
    scale = min(1.0, MAX_DISP / max(h, w))
    if scale < 1:
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)))

    def refresh():
        ov = disp.copy()
        for obj_id, (pts, lbs) in state["objs"].items():
            cls = obj_id // 100
            for (px, py), lb in zip(pts, lbs):
                cv2.circle(ov, (px, py), 6, CLASS_COLORS[cls] if lb == 1 else (0, 0, 0), -1)
                if lb == 1:
                    cv2.putText(ov, f"{cls}.{obj_id % 100}", (px + 8, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLASS_COLORS[cls], 1)
        cv2.rectangle(ov, (0, 0), (ov.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(ov, f"Clase activa: {state['active']} {CLASS_NAMES[state['active']]}  "
                        f"| izq=objeto nuevo  der=corrige el ultimo  (1/2/3, ENTER, R, Q)",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow("SAM2 - primer frame", ov)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cls = state["active"]
            state["count"][cls] += 1
            obj_id = cls * 100 + state["count"][cls]
            state["objs"][obj_id] = ([[x, y]], [1])
            state["last"] = obj_id
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["last"] is None:
                return
            pts, lbs = state["objs"][state["last"]]
            pts.append([x, y]); lbs.append(0)
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
            state["objs"] = {}; state["count"] = {1: 0, 2: 0, 3: 0}; state["last"] = None
            refresh()
        elif k in (13, 10):   # ENTER
            cv2.destroyAllWindows()
            # devolver en coords del frame ORIGINAL (deshacer el scale de display)
            out = {}
            for obj_id, (pts, lbs) in state["objs"].items():
                if pts:
                    out[(obj_id, 0)] = (np.array(pts, np.float32) / scale, np.array(lbs, np.int32))
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

    for key, (pts, lbs) in clicks.items():
        obj_id, frame_idx = key if isinstance(key, tuple) else (key, 0)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=frame_idx,
                                        obj_id=obj_id, points=pts, labels=lbs)

    per_frame = {}
    with torch.inference_mode():
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            masks = {}
            for i, oid in enumerate(obj_ids):
                masks[int(oid)] = (mask_logits[i] > 0.0).squeeze().cpu().numpy().astype(np.uint8)
            per_frame[frame_idx] = masks
    return per_frame


def compose_index_mask(masks, shape):
    """masks: {obj_id: máscara}. Todas las instancias de una clase se funden en el mismo índice.
    Prioridad de pintado: pierna < pie < zapato (el zapato tapa al pie que tapa a la pierna)."""
    idx = np.zeros(shape, np.uint8)
    for cls in (1, 2, 3):
        for obj_id, m in masks.items():
            if (obj_id // 100 if obj_id >= 100 else obj_id) != cls:
                continue
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
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270],
                    help="forzar rotación en grados (por defecto: la que indique el metadato del video)")
    ap.add_argument("--points", default=None,
                    help="modo NO interactivo: puntos como 'clase:x,y' separados por espacio, con "
                         "x,y en fracción 0-1 del frame ya recortado. Negativo: anteponer '-'. "
                         "Ej: '2:0.5,0.45 3:0.45,0.6 3:0.6,0.6 1:0.5,0.9 -3:0.1,0.1'")
    ap.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml",
                    help="config del checkpoint (t/s/b+/l deben coincidir)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="segmenta SÓLO el frame 0 y guarda un overlay para verificar los --points "
                         "antes de gastar minutos propagando todo el clip")
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
                           square=not args.no_crop, force_rotate=args.rotate)
        print(f"{n} frames extraídos de {video.name} (extract_stride {args.extract_stride}, "
              f"max {args.max_side}px, {'recorte cuadrado centrado' if not args.no_crop else 'SIN recorte'})")
        first = cv2.imread(str(tmp / "00000.jpg"))
        if args.points:
            fh, fw = first.shape[:2]
            clicks = parse_points(args.points, fw, fh)
            print(f"  [pts] modo no interactivo: " +
                  ", ".join(f"obj {k[0]//100}.{k[0]%100}@f{k[1]}: {len(p[0])} pts"
                            for k, p in sorted(clicks.items())))
        else:
            clicks = collect_clicks(first)
        if args.dry_run:
            # Sólo el frame 0: verificar que los puntos caen donde uno cree ANTES de propagar.
            # (Los fallos vistos venían siempre de un punto mal ubicado: en el piso o en el pantalón.)
            keep = {k: v for k, v in clicks.items() if (k[1] if isinstance(k, tuple) else 0) == 0}
            for f in sorted(tmp.glob("*.jpg"))[1:]:
                f.unlink()
            per_frame = propagate(tmp, keep, args.checkpoint, args.config, device)
            frame0 = cv2.imread(str(tmp / "00000.jpg"))
            idx = compose_index_mask(per_frame.get(0, {}), frame0.shape[:2])
            ov = frame0.copy()
            for cls, colr in CLASS_COLORS.items():
                ov[idx == cls] = (0.5 * np.array(colr) + 0.5 * ov[idx == cls]).astype(np.uint8)
            h, w = frame0.shape[:2]
            for key, (pts, lbs) in keep.items():
                oid = key[0] if isinstance(key, tuple) else key
                for (px, py), lb in zip(pts, lbs):
                    c = CLASS_COLORS.get(oid // 100, (255, 255, 255)) if lb == 1 else (0, 0, 0)
                    cv2.circle(ov, (int(px), int(py)), 7, c, -1)
                    cv2.circle(ov, (int(px), int(py)), 7, (255, 255, 255), 2)
            out_img = out_dir / f"_dryrun_{clip}.jpg"
            cv2.imwrite(str(out_img), np.hstack([frame0, ov]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  [dry-run] {out_img}")
            for cls, nm in CLASS_NAMES.items():
                print(f"     {nm:8} {100*float((idx==cls).mean()):5.1f}% del frame"
                      + ("   ⚠ vacío: el punto no cayó donde creías" if (idx == cls).sum() < 50 else "")
                      + ("   ⚠ enorme: agarró piso o ropa" if (idx == cls).mean() > 0.35 else ""))
            return

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
