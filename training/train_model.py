"""
PASO 3 — Entrenar el modelo de pie (multi-branch: keypoints + segmentación multiclase)
=======================================================================================
Reemplaza al viejo 3_train_model.py (segmentación binaria, encoder roto).

Arquitectura (receta ARShoe / Springer-2025) — UN solo modelo para ambos pies:
    Encoder MobileNetV2 compartido  →  dos cabezas:
      (a) SEGMENTACIÓN multiclase 256×256, 4 clases {0 fondo, 1 pierna/pantalón, 2 pie, 3 zapato}
          (sin lado: un pie es un pie; la máscara sirve para oclusión)
      (b) KEYPOINTS: 12 heatmaps 64×64 = 6 por lado {heel, toe, ankle_in, ankle_out, ball, toe_tip}
          × {left, right} — un solo forward detecta ambos pies Y su lado. Los modos de la app
          (ambos pies / izq / der) son filtros de render, NO modelos distintos.
    ankle_in = maléolo MEDIAL (anatómico, no relativo a pantalla): así el espejo de un pie derecho
    es un pie izquierdo válido y el flip de augmentación genera el lado contrario gratis.

Formato de datos v2 (idéntico para sintético de Blender y fotos reales SAM):
    <data_dir>/images/*.jpg|png
    <data_dir>/masks/<stem>.png        (PNG uint8 con índices {0,1,2,3})
    <data_dir>/keypoints.jsonl         (una línea por imagen:
                                        {"file": "x.jpg", "kps": {"left": [[x,y,vis]×6]|null,
                                                                  "right": [[x,y,vis]×6]|null}}
                                        x,y normalizados [0,1]; vis 0/1; lado ausente = null)

Normalización FIJA del proyecto (la misma que usa inference.js):
    mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]   (ImageNet)

Uso:
    python train_model.py --smoke                 # test de humo sin dataset (datos dummy)
    python train_model.py --data_dir data_synthetic --epochs 30
    python train_model.py --data_dir data_synthetic --small     # MobileNetV2 alpha=0.5

Requisitos:
    pip install torch torchvision albumentations opencv-python numpy pillow tqdm
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

try:  # consola Windows (cp1252) no imprime UTF-8 por defecto
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- Constantes del proyecto ----
IMG_SIZE      = 256
HEATMAP_SIZE  = 64
NUM_CLASSES   = 4          # 0 fondo, 1 pierna/pantalón, 2 pie, 3 zapato
KP_NAMES_ONE  = ["heel", "toe", "ankle_in", "ankle_out", "ball", "toe_tip"]  # ankle_in = MEDIAL
KP_PER_FOOT   = len(KP_NAMES_ONE)
SIDES         = ["left", "right"]   # orden de los bloques en el tensor: [left×6, right×6]
KP_NAMES      = [f"{s}_{n}" for s in SIDES for n in KP_NAMES_ONE]
NUM_KP        = len(KP_NAMES)       # 12
HEATMAP_SIGMA = 2.0        # px (en resolución de heatmap)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
FROZEN_DIR    = "data_test_frozen"   # jamás debe entrar al entrenamiento (ver ROADMAP T3.6)


# =====================================================================
# GENERACIÓN DE HEATMAPS
# =====================================================================
def kps_to_array(raw):
    """Formato v2 {'left': [...]|None, 'right': [...]|None} → array (12,3).
    Compat: una lista suelta (formato v1 de un solo pie) se asume pie DERECHO."""
    arr = np.zeros((NUM_KP, 3), np.float32)
    if raw is None:
        return arr
    if isinstance(raw, list):                      # legacy v1
        arr[KP_PER_FOOT:2 * KP_PER_FOOT] = np.asarray(raw, np.float32)[:KP_PER_FOOT]
        return arr
    for si, side in enumerate(SIDES):
        block = raw.get(side)
        if block:
            arr[si * KP_PER_FOOT:(si + 1) * KP_PER_FOOT] = np.asarray(block, np.float32)[:KP_PER_FOOT]
    return arr


def flip_lr(image, mask, kps):
    """Espejo horizontal: imagen+máscara con fliplr, kps x→1-x y bloques left↔right intercambiados.
    Válido porque ankle_in/out son anatómicos (medial/lateral): el espejo de un pie derecho ES un izquierdo."""
    image = np.ascontiguousarray(image[:, ::-1])
    mask  = np.ascontiguousarray(mask[:, ::-1])
    out = kps.copy()
    out[:, 0] = np.where(kps[:, 2] > 0, 1.0 - kps[:, 0], kps[:, 0])
    L, R = out[:KP_PER_FOOT].copy(), out[KP_PER_FOOT:2 * KP_PER_FOOT].copy()
    out[:KP_PER_FOOT], out[KP_PER_FOOT:2 * KP_PER_FOOT] = R, L
    return image, mask, out


def make_heatmaps(kps, size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA):
    """kps: array (NUM_KP, 3) con x,y normalizados [0,1] y vis. Devuelve (NUM_KP,size,size) float32."""
    hm = np.zeros((NUM_KP, size, size), dtype=np.float32)
    ax = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)  # (size,size)
    for i in range(NUM_KP):
        x, y, v = kps[i]
        if v <= 0:
            continue
        cx, cy = x * size, y * size
        if cx < 0 or cy < 0 or cx > size - 1 or cy > size - 1:
            continue
        hm[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return hm


# =====================================================================
# DATASET
# =====================================================================
class FootDataset:
    """Dataset perezoso; augmentación ON-THE-FLY (no se materializan 100k archivos).
    Devuelve tensores torch: image (3,H,W), mask (H,W) long, heatmaps (6,64,64), kps (6,3)."""

    def __init__(self, samples, augment, img_size=IMG_SIZE):
        import torch  # import diferido para permitir --smoke sin efectos colaterales
        self._torch = torch
        self.samples = samples          # lista de dicts {img, mask, kps}
        self.img_size = img_size
        self.augment = augment
        self.tf = _build_augmenter(img_size) if augment else _build_resize_only(img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        torch = self._torch
        s = self.samples[idx]

        # --- cargar imagen ---
        if isinstance(s["img"], np.ndarray):          # muestra dummy (smoke)
            image = s["img"]
            mask  = s["mask"]
            kps   = kps_to_array(s["kps"])
        else:
            image = cv2.cvtColor(cv2.imread(str(s["img"])), cv2.COLOR_BGR2RGB)
            m = cv2.imread(str(s["mask"]), cv2.IMREAD_GRAYSCALE)
            mask = m if m is not None else np.zeros(image.shape[:2], np.uint8)
            kps = kps_to_array(s["kps"])

        # Flip horizontal (50%): genera el pie del lado contrario gratis (swap de bloques L/R)
        if self.augment and np.random.random() < 0.5:
            image, mask, kps = flip_lr(image, mask, kps)

        h, w = image.shape[:2]
        # keypoints normalizados → píxeles para albumentations
        kp_px = [(float(kps[i, 0] * w), float(kps[i, 1] * h)) for i in range(NUM_KP)]
        vis   = kps[:, 2].copy()

        out = self.tf(image=image, mask=mask, keypoints=kp_px)
        image, mask = out["image"], out["mask"]
        kp_out = out["keypoints"]

        H = W = self.img_size
        kps_norm = np.zeros((NUM_KP, 3), np.float32)
        for i in range(NUM_KP):
            if i < len(kp_out):
                x, y = kp_out[i]
                inside = (0 <= x <= W - 1) and (0 <= y <= H - 1)
                kps_norm[i] = [x / W, y / H, 1.0 if (vis[i] > 0 and inside) else 0.0]

        heatmaps = make_heatmaps(kps_norm)

        # a tensores
        img_t = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_t = (img_t - mean) / std
        mask_t = torch.from_numpy(np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64))
        hm_t   = torch.from_numpy(heatmaps)
        kp_t   = torch.from_numpy(kps_norm)
        # has_kp: ¿esta muestra trae ANOTACIÓN de keypoints? Los frames de video (SAM2) traen
        # sólo máscaras → deben quedar FUERA de la pérdida de keypoints; si no, sus heatmaps
        # vacíos le enseñarían al modelo a no predecir keypoints nunca.
        # OJO: distinto de "ese pie no está en la imagen" — ahí el heatmap vacío SÍ es correcto
        # (así el modelo aprende qué pies están presentes) y la muestra sí participa.
        has_kp = torch.tensor(1.0 if s.get("kps") else 0.0, dtype=torch.float32)
        return img_t, mask_t, hm_t, kp_t, has_kp


def _build_augmenter(img_size):
    import albumentations as A
    # NOTA: el flip horizontal NO va acá — se hace en __getitem__ (flip_lr) porque además de
    # espejar la imagen hay que intercambiar los bloques de keypoints left↔right.
    return A.Compose(
        [
            A.Affine(scale=(0.75, 1.25), translate_percent=(-0.12, 0.12),
                     rotate=(-40, 40), fit_output=False, p=0.9),
            A.Perspective(scale=(0.03, 0.10), p=0.4),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35),
                A.RandomGamma(gamma_limit=(70, 130)),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=25),
            ], p=0.8),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.MotionBlur(blur_limit=7)], p=0.25),
            A.ImageCompression(quality_range=(40, 90), p=0.3),
            A.Resize(img_size, img_size),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def _build_resize_only(img_size):
    import albumentations as A
    return A.Compose(
        [A.Resize(img_size, img_size)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


# =====================================================================
# MODELO: MobileNetV2 encoder + U-Net decoder + 2 cabezas
# =====================================================================
def _build_model(small=False):
    import torch
    import torch.nn as nn
    from torchvision import models

    class ConvBnRelu(nn.Sequential):
        def __init__(self, cin, cout, k=3, s=1, p=1):
            super().__init__(nn.Conv2d(cin, cout, k, s, p, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class UpBlock(nn.Module):
        def __init__(self, cin, cskip, cout):
            super().__init__()
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = nn.Sequential(ConvBnRelu(cin + cskip, cout), ConvBnRelu(cout, cout))

        def forward(self, x, skip=None):
            x = self.up(x)
            if skip is not None:
                x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    class FootNet(nn.Module):
        def __init__(self, small=False):
            super().__init__()
            if small:
                mb = models.mobilenet_v2(width_mult=0.5)
            else:
                mb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            f = mb.features
            # Grupos por resolución (índices verificados: strides en features[0],[2],[4],[7],[14])
            self.enc1 = f[0:2]     # → @128
            self.enc2 = f[2:4]     # → @64
            self.enc3 = f[4:7]     # → @32
            self.enc4 = f[7:14]    # → @16
            self.bott = f[14:18]   # → @8  (excluye f[18], la conv 1x1 a 1280 solo-clasificación)

            # Detección DINÁMICA de canales (robusto a width_mult; evita el bug de canales hardcodeados)
            ch1, ch2, ch3, ch4, chb, res = self._probe()
            assert res == [128, 64, 32, 16, 8], f"resoluciones inesperadas: {res}"

            self.up4 = UpBlock(chb, ch4, 96)   # 8→16
            self.up3 = UpBlock(96,  ch3, 64)   # 16→32
            self.up2 = UpBlock(64,  ch2, 48)   # 32→64   ← rama de keypoints
            self.up1 = UpBlock(48,  ch1, 32)   # 64→128
            self.up0 = UpBlock(32,  0,   16)   # 128→256 (sin skip)

            self.seg_head = nn.Conv2d(16, NUM_CLASSES, 1)          # logits @256
            self.kp_head  = nn.Sequential(ConvBnRelu(48, 48),
                                          nn.Conv2d(48, NUM_KP, 1))  # heatmaps @64

        def _probe(self):
            was_training = self.training
            self.eval()
            with torch.no_grad():
                x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
                e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
                e4 = self.enc4(e3); b = self.bott(e4)
            self.train(was_training)
            res = [e1.shape[-1], e2.shape[-1], e3.shape[-1], e4.shape[-1], b.shape[-1]]
            return e1.shape[1], e2.shape[1], e3.shape[1], e4.shape[1], b.shape[1], res

        def forward(self, x):
            e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
            e4 = self.enc4(e3); b = self.bott(e4)
            d16 = self.up4(b, e4)
            d32 = self.up3(d16, e3)
            d64 = self.up2(d32, e2)
            d128 = self.up1(d64, e1)
            d256 = self.up0(d128, None)
            seg = self.seg_head(d256)                 # logits (B,4,256,256)
            kp  = torch.sigmoid(self.kp_head(d64))    # heatmaps (B,6,64,64) en [0,1]
            return seg, kp

    return FootNet(small=small)


# =====================================================================
# PÉRDIDAS
# =====================================================================
def dice_loss_multiclass(logits, target, eps=1e-6):
    import torch
    import torch.nn.functional as F
    probs = F.softmax(logits, dim=1)                          # (B,C,H,W)
    onehot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + onehot.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(seg_logits, kp_pred, mask, hm, has_kp=None, w_kp=10.0):
    """has_kp: (B,) 1.0 si la muestra trae anotación de keypoints, 0.0 si es sólo-segmentación
    (frames de video de SAM2). Las muestras sin anotación NO aportan a la pérdida de keypoints."""
    import torch.nn.functional as F
    ce   = F.cross_entropy(seg_logits, mask)
    dice = dice_loss_multiclass(seg_logits, mask)
    if has_kp is None:
        mse = F.mse_loss(kp_pred, hm)
    else:
        per_sample = ((kp_pred - hm) ** 2).mean(dim=(1, 2, 3))       # (B,)
        w = has_kp.to(per_sample.dtype)
        mse = (per_sample * w).sum() / w.sum().clamp(min=1e-6)
    return ce + dice + w_kp * mse, {"ce": ce.item(), "dice": dice.item(), "mse": mse.item()}


# =====================================================================
# MÉTRICAS
# =====================================================================
def update_iou_stats(stats, seg_logits, mask):
    """Acumula intersección/unión por clase + para la unión pie∪zapato."""
    pred = seg_logits.argmax(dim=1)
    for c in range(NUM_CLASSES):
        p, g = (pred == c), (mask == c)
        stats["inter"][c] += (p & g).sum().item()
        stats["union"][c] += (p | g).sum().item()
    p_fs = (pred >= 2)          # pie(2) o zapato(3)
    g_fs = (mask >= 2)
    stats["fs_inter"] += (p_fs & g_fs).sum().item()
    stats["fs_union"] += (p_fs & g_fs).sum().item() + (p_fs ^ g_fs).sum().item()


def pck_batch(kp_pred, kps_gt, thresh=0.1):
    """PCK@thresh por lado: correcto si dist(pred,gt) < thresh * largo_del_pie DE ESE LADO
    (heel→toe del bloque). Devuelve (correctos, total)."""
    import torch
    B = kp_pred.shape[0]
    S = kp_pred.shape[-1]
    correct = total = 0
    for b in range(B):
        gt = kps_gt[b]                         # (12,3): [left×6, right×6]
        for blk in range(len(SIDES)):
            o = blk * KP_PER_FOOT
            heel, toe = gt[o + 0], gt[o + 1]
            foot_len = float(torch.norm(heel[:2] - toe[:2])) if (heel[2] > 0 and toe[2] > 0) else 0.0
            if foot_len < 1e-3:
                foot_len = 0.25                # fallback: ~pie normalizado
            for i in range(o, o + KP_PER_FOOT):
                if gt[i, 2] <= 0:
                    continue
                hm = kp_pred[b, i]
                idx = int(torch.argmax(hm))
                py, px = divmod(idx, S)
                pred = torch.tensor([px / S, py / S], device=gt.device)
                if float(torch.norm(pred - gt[i, :2])) < thresh * foot_len:
                    correct += 1
                total += 1
    return correct, total


# =====================================================================
# CARGA DE MUESTRAS / SPLIT
# =====================================================================
def _base_stem(stem):
    """Agrupa por contenido base para el split sin fuga:
    - variantes augmentadas:      foto01_aug_0003   → foto01
    - frames del mismo video:     clip01_f0042      → clip01  (2b_sam2_video.py; fuga temporal si se separan)
    """
    import re
    stem = stem.split("_aug_")[0]
    return re.sub(r"_f\d+$", "", stem)


def load_samples(data_dir):
    data_dir = Path(data_dir)
    if FROZEN_DIR in data_dir.parts:
        raise SystemExit(f"ABORT: {FROZEN_DIR} es el test set congelado y NO debe entrenarse (ROADMAP T3.6).")
    img_dir, mask_dir = data_dir / "images", data_dir / "masks"
    kp_path = data_dir / "keypoints.jsonl"

    kp_map = {}
    if kp_path.exists():
        for line in kp_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kp_map[Path(rec["file"]).stem] = rec.get("kps")

    paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    samples = []
    for p in paths:
        mp = mask_dir / (p.stem + ".png")
        if not mp.exists():
            continue
        samples.append({"img": p, "mask": mp, "kps": kp_map.get(p.stem)})
    return samples


def split_by_base(samples, val_frac=0.1, seed=42):
    """Split POR IMAGEN BASE (sin fuga): todas las variantes de una base caen del mismo lado."""
    from collections import defaultdict
    groups = defaultdict(list)
    for s in samples:
        groups[_base_stem(Path(s["img"]).stem)].append(s)
    bases = sorted(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(bases)
    n_val = max(1, int(len(bases) * val_frac))
    val_bases = set(bases[:n_val])
    train = [s for b in bases if b not in val_bases for s in groups[b]]
    val   = [s for b in val_bases for s in groups[b]]
    return train, val


def make_dummy_samples(n=20, seed=0):
    """Muestras sintéticas en memoria para el smoke test (imagen ruido + rect pie/zapato + kps)."""
    rng = np.random.default_rng(seed)
    out = []
    for _i in range(n):
        img = (rng.random((IMG_SIZE, IMG_SIZE, 3)) * 255).astype(np.uint8)
        mask = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
        x0, y0 = rng.integers(20, 120), rng.integers(20, 120)
        w, h = rng.integers(60, 110), rng.integers(60, 110)
        mask[y0:y0 + h, x0:x0 + w] = 2                                # pie
        mask[y0 + h // 2:y0 + h, x0:x0 + w] = 3                       # zapato (mitad inferior)
        mask[max(0, y0 - 30):y0, x0:x0 + w] = 1                       # pierna
        block = []
        for _k in range(KP_PER_FOOT):
            kx = (x0 + rng.integers(0, w)) / IMG_SIZE
            ky = (y0 + rng.integers(0, h)) / IMG_SIZE
            block.append([float(kx), float(ky), 1.0])
        side = SIDES[int(rng.integers(0, 2))]   # un pie por muestra, lado aleatorio (v2)
        # 1 de cada 3 muestras sin keypoints: simula los frames de video de SAM2 (sólo-segmentación)
        kps = None if (_i % 3 == 2) else {side: block}
        out.append({"img": img, "mask": mask, "kps": kps})
    return out


# =====================================================================
# ENTRENAMIENTO
# =====================================================================
def run(args):
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if args.smoke:
        print("MODO SMOKE: datos dummy en memoria, 2 epochs, sin dataset real.")
        train_s = make_dummy_samples(20, seed=1)
        val_s   = make_dummy_samples(6, seed=2)
        args.epochs = min(args.epochs, 2)
        num_workers = 0
    else:
        samples = load_samples(args.data_dir)
        if not samples:
            raise SystemExit(f"Sin muestras en {args.data_dir}/images (+masks). ¿Corriste 0b_blender_render.py?")
        train_s, val_s = split_by_base(samples, val_frac=0.1, seed=42)
        print(f"Muestras: {len(samples)} → {len(train_s)} train / {len(val_s)} val (split por imagen base)")
        num_workers = args.workers

    train_ds = FootDataset(train_s, augment=not args.smoke)
    val_ds   = FootDataset(val_s,   augment=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))

    model = _build_model(small=args.small).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for img, mask, hm, _kp, has_kp in train_ld:
            img, mask, hm, has_kp = img.to(device), mask.to(device), hm.to(device), has_kp.to(device)
            seg, kp = model(img)
            loss, _ = combined_loss(seg, kp, mask, hm, has_kp=has_kp, w_kp=args.w_kp)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        sched.step()

        # --- validación ---
        model.eval()
        stats = {"inter": [0] * NUM_CLASSES, "union": [0] * NUM_CLASSES, "fs_inter": 0, "fs_union": 0}
        pck_c = pck_t = 0
        with torch.no_grad():
            for img, mask, hm, kp_gt, has_kp in val_ld:
                img, mask, kp_gt = img.to(device), mask.to(device), kp_gt.to(device)
                seg, kp = model(img)
                update_iou_stats(stats, seg, mask)
                # PCK sólo sobre muestras CON anotación de keypoints (los frames de video no la traen)
                sel = has_kp.to(kp_gt.device) > 0
                if sel.any():
                    c, t = pck_batch(kp[sel], kp_gt[sel])
                    pck_c += c; pck_t += t

        ious = [(stats["inter"][c] / stats["union"][c]) if stats["union"][c] > 0 else float("nan")
                for c in range(NUM_CLASSES)]
        miou = float(np.nanmean(ious))
        fs_iou = stats["fs_inter"] / stats["fs_union"] if stats["fs_union"] > 0 else 0.0
        pck = pck_c / pck_t if pck_t > 0 else 0.0
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={tl/max(1,len(train_ld)):.4f} | "
              f"mIoU={miou:.3f} | IoU(pie∪zapato)={fs_iou:.3f} | PCK@0.1={pck:.3f} | "
              f"IoU/clase={[f'{v:.2f}' for v in ious]}")

        if miou > best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict(), "small": args.small,
                        "classes": NUM_CLASSES, "kp_names": KP_NAMES,
                        "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                       args.output)
            print(f"  💾 Guardado (mIoU={miou:.3f}) → {args.output}")

    print(f"\n✅ Mejor mIoU={best_miou:.3f} en {args.output}")


def build_argparser():
    p = argparse.ArgumentParser(description="Entrena FootNet (keypoints + segmentación multiclase)")
    p.add_argument("--data_dir", default="data_synthetic")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w_kp", type=float, default=10.0, help="peso de la pérdida de keypoints")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--small", action="store_true", help="MobileNetV2 alpha=0.5 (sin preentrenar)")
    p.add_argument("--output", default="foot_model.pth")
    p.add_argument("--smoke", action="store_true", help="test de humo con datos dummy en memoria")
    return p


if __name__ == "__main__":
    # Guard obligatorio en Windows con num_workers>0 (spawn re-ejecuta el módulo).
    run(build_argparser().parse_args())
