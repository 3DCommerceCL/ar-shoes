"""
PASO 4 — Exportar FootNet a ONNX para onnxruntime-web
=====================================================
Genera <output>.onnx (o _int8.onnx) + un sidecar <output>.meta.json con la
normalización, clases y keypoints — para que inference.js no hardcodee nada.

Uso:
    python export_onnx.py --checkpoint foot_model.pth --output ../models/foot_net.onnx
    python export_onnx.py --checkpoint foot_model.pth --int8
"""
import argparse
import json
from pathlib import Path

import torch

from train_model import (_build_model, IMG_SIZE, HEATMAP_SIZE,
                         NUM_CLASSES, NUM_KP, KP_NAMES, IMAGENET_MEAN, IMAGENET_STD)


def main():
    ap = argparse.ArgumentParser(description="Exporta FootNet a ONNX")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="../models/foot_net.onnx")
    ap.add_argument("--int8", action="store_true", help="cuantización dinámica int8 (~4x más chico)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    small = ckpt.get("small", False)
    model = _build_model(small=small)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["seg", "heatmaps"],
        opset_version=args.opset,
        dynamo=False,  # exportador TorchScript clásico (no requiere onnxscript)
    )
    print(f"ONNX exportado: {out} ({out.stat().st_size / 1e6:.2f} MB)")

    final = out
    if args.int8:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = out.with_name(out.stem + "_int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8)
        print(f"ONNX int8: {q} ({q.stat().st_size / 1e6:.2f} MB)")
        final = q

    meta = {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "input_size": IMG_SIZE,
        "heatmap_size": HEATMAP_SIZE,
        "classes": ["background", "leg", "foot", "shoe"],
        "keypoints": KP_NAMES,
        "outputs": {
            "seg": [1, NUM_CLASSES, IMG_SIZE, IMG_SIZE],
            "heatmaps": [1, NUM_KP, HEATMAP_SIZE, HEATMAP_SIZE],
        },
        "checkpoint": Path(args.checkpoint).name,
    }
    meta_path = final.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
