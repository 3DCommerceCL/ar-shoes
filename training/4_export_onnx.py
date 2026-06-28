"""
PASO 4 — Exportar modelo a ONNX para usar en el browser
=========================================================
Genera foot_segmenter.onnx → copiar a models/ en el proyecto AR

Requisitos:
    pip install torch onnx onnxruntime

Uso:
    python 4_export_onnx.py --checkpoint foot_model.pth
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path

# Importar la arquitectura del paso anterior
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_model import FootSegNet

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="foot_model.pth")
parser.add_argument("--output",     default="../models/foot_segmenter.onnx")
args = parser.parse_args()

# ---- Cargar modelo ----
model = FootSegNet()
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
model.eval()

# ---- Export a ONNX ----
dummy = torch.randn(1, 3, 256, 256)  # batch=1, RGB, 256×256

torch.onnx.export(
    model,
    dummy,
    args.output,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11,
    do_constant_folding=True,
)
print(f"Exportado: {args.output}")

# ---- Validar ONNX ----
model_onnx = onnx.load(args.output)
onnx.checker.check_model(model_onnx)
print("✅ ONNX válido")

# ---- Test de inferencia ----
session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
dummy_np = dummy.numpy()
result = session.run(["output"], {"input": dummy_np})[0]
print(f"Forma de salida: {result.shape}")  # debe ser (1, 1, 256, 256)

# Tamaño del archivo
size_mb = Path(args.output).stat().st_size / 1_048_576
print(f"Tamaño: {size_mb:.1f} MB")
print(f"\n✅ Copiar {args.output} a la carpeta models/ del proyecto AR")
