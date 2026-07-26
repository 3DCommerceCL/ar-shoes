# Pipeline de entrenamiento — AR Shoe Try-On (v2)

Modelo objetivo: **red multi-branch** (encoder MobileNetV2 compartido) con dos cabezas —
(a) **keypoints** de AMBOS pies (12 heatmaps = 6 izq + 6 der — un solo modelo detecta ambos pies y su lado; los modos de la app ambos/izq/der son filtros de render, no modelos distintos) y (b) **segmentación multiclase** {fondo, pierna, pie, zapato}.
Los keypoints alimentan PnP (pose 6DoF, `solver.js`); la máscara sirve para oclusión. Es la receta de
ARShoe (arXiv 2108.10515) y Springer-2025. Ver [../ROADMAP.md](../ROADMAP.md) para el plan completo y los gates.

> Cambios vs. v1: ya **no** es segmentación binaria, ya **no** hay augmentación estática 100→100k
> (ahora es on-the-fly en el `Dataset`), y los scripts se renombraron a nombres importables.

```
Blender (0b) ─┐
              ├─► images/ + masks/{0,1,2,3} + keypoints.jsonl ─► train_model.py ─► export_onnx.py ─► models/foot_net_int8.onnx ─► inference.js (browser)
Fotos SAM (2) ┘                                                        ▲
                                                            augmentación on-the-fly
```

## Formato de datos (único para sintético y real)

```
<data_dir>/images/<stem>.jpg|png
<data_dir>/masks/<stem>.png        # PNG uint8 con índices {0 fondo, 1 pierna, 2 pie, 3 zapato}
<data_dir>/keypoints.jsonl         # {"file":"x.jpg","kps":{"left":[[x,y,vis]×6]|null,"right":...}}
                                   # (v2 por lado; ankle_in = maléolo MEDIAL, anatómico)
```
Keypoints en orden: `heel, toe, ankle_in, ankle_out, ball, toe_tip`.
Normalización fija: `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]` (la misma que usa `inference.js`).

## Fase 0b — Render sintético (Blender 5.1)

```bash
"C:/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background --python 0b_blender_render.py -- \
    --foot models/foot.glb --shoe ../models/shoe.glb --out data_synthetic --count 5000 --seed 42 \
    --shoe_dir models/shoes --hdri_dir hdri --floor_dir floor_textures
# preview de 20 con contact-sheet para inspección visual:
... --preview
```
- Pie + zapato se transforman JUNTOS (Empty raíz, sin reescalar GLBs — regla dura #1).
- Máscara por emisión de color (pierna=R, pie=G, zapato=B) + view transform `Raw` → índices exactos, sin clases fantasma.
- Keypoints GT: usa los Empties `kp_*` del GLB si existen; si no, los coloca por bbox. Proyección con oclusión.
- Degradación tipo cámara móvil (JPEG q40-85, ruido, motion blur, viñeteo, balance de blancos) — numpy, in-Blender.
- HDRIs (Poly Haven CC0) y texturas de piso si se pasan las carpetas; si no, fallback a luces/colores.

## Fase 2 — Fotos reales (SAM, multiclase + keypoints)

```bash
pip install segment-anything opencv-python torch torchvision numpy
# checkpoint: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python 2_sam_label.py --checkpoint sam_vit_b_01ec64.pth   # 1/2/3 clase, K keypoints, S guardar
python 2_sam_label.py --review                            # control de calidad
```
Ver [PROTOCOLO_FOTOS.md](PROTOCOLO_FOTOS.md) para qué fotos sacar. **50 se congelan** como test set (T3.6).

## Fase 3 — Entrenamiento

```bash
pip install torch torchvision albumentations opencv-python numpy pillow tqdm
python train_model.py --smoke                              # test de humo sin dataset (datos dummy)
python train_model.py --data_dir data_synthetic --epochs 30
```
- Arquitectura: MobileNetV2 (canales detectados dinámicamente) + decoder U-Net + 2 cabezas.
- Pérdida: CrossEntropy + Dice (seg) + MSE de heatmaps (kp). Métricas: mIoU, IoU(pie∪zapato), PCK@0.1.
- Split **por imagen base** (sin fuga); augmentación **on-the-fly**; mejor checkpoint por mIoU.

## Fase 4 — Export a ONNX

```bash
pip install onnx onnxruntime
python export_onnx.py --checkpoint foot_model.pth --output ../models/foot_net.onnx --int8
# genera foot_net_int8.onnx (~2.7 MB) + foot_net_int8.meta.json
```

## Fase 5 — Probar en el navegador

Abrir `../test_inference.html` (servido por HTTP), cargar el `.onnx`: corre con onnxruntime-web
(WebGPU con fallback WASM) y dibuja la máscara + keypoints. La integración en la app AR es la Fase 5 del ROADMAP.

## Métricas objetivo (medir SOBRE el test set real congelado)

| Métrica | Gate |
|---------|------|
| IoU (pie ∪ zapato) | > 0.85 (G4b) / > 0.6-0.7 en el piloto (G4a) |
| PCK@0.1 keypoints | > 0.85 |
| Inferencia móvil (WebGPU) | < 80 ms |
