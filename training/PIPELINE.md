# Pipeline de entrenamiento — AR Shoe Try-On

## Resumen del flujo

```
100 fotos → SAM auto-label → 100,000 augmentadas → Entrenar → ONNX → Browser
                                                                         ↑
                              Users try-on → capturas → re-entrenar ────┘
```

---

## Fase 1: Recolección de fotos (100 fotos base)

### Ángulos requeridos (distribuir equitativamente):
| Ángulo | % del dataset | Descripción |
|--------|--------------|-------------|
| Cenital (desde arriba) | 40% | Cámara apuntando directo al suelo |
| 45° diagonal | 30% | Ángulo típico de selfie de pies |
| Frontal | 20% | Cámara a altura de rodilla mirando los pies |
| Lateral | 10% | Vista desde el costado |

### Diversidad necesaria:
- **Personas:** min. 5 personas distintas (diferente tono de piel, tamaño de pie)
- **Calzado:** zapatillas, zapatos, botas, sandalias, medias, pie descalzo
- **Pisos:** madera, cerámico, alfombra, cemento, exterior
- **Iluminación:** natural, artificial, mixta, contraluz suave
- **Distancia:** 30cm, 60cm, 90cm, 120cm desde la cámara

### Herramienta de etiquetado:
```bash
cd training/
pip install segment-anything opencv-python torch torchvision tqdm
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
python 2_sam_label.py --checkpoint sam_vit_b_01ec64.pth
```
- Clic izquierdo en el pie → máscara automática
- `S` para guardar, `R` para resetear, `Q` para salir
- Tiempo estimado: **2-3 horas para 100 fotos**

---

## Fase 2: Augmentación (100 → 100,000)

```bash
pip install albumentations opencv-python Pillow tqdm
python 1_augment.py
```

- Tiempo: **~30-60 minutos** en CPU
- Disco: ~15GB para 100,000 imágenes JPEG
- Variaciones: rotación, flip, brillo, contraste, perspectiva, blur, fondo sintético

---

## Fase 3: Entrenamiento

```bash
pip install torch torchvision tqdm scikit-learn Pillow
python 3_train_model.py --epochs 30
```

| Hardware | Tiempo 30 epochs | Costo |
|---------|-----------------|-------|
| CPU local | ~40 horas | $0 |
| Google Colab T4 (gratis) | ~6-8 horas | $0 |
| Google Colab A100 (Pro) | ~2-3 horas | ~$5 |
| Runpod A100 | ~1-2 horas | ~$3 |

**Recomendación: Google Colab** (subir scripts + data_augmented/ a Google Drive)

---

## Fase 4: Export a ONNX

```bash
pip install onnx onnxruntime
python 4_export_onnx.py --checkpoint foot_model.pth
# Genera: models/foot_segmenter.onnx (~5MB)
```

---

## Fase 5: Integrar en la app AR

Una vez generado `models/foot_segmenter.onnx`, reemplazar el background 
subtraction en `pose.js` con el modelo ONNX:

```js
// pose.js — con modelo ONNX propio
import { InferenceSession, Tensor } from 'onnxruntime-web';

let session = null;
export async function initPose() {
  session = await InferenceSession.create('./models/foot_segmenter.onnx');
}

export async function detectPose(videoEl) {
  // Preprocesar video → tensor 1×3×256×256
  const tensor = preprocessVideo(videoEl);
  const result = await session.run({ input: tensor });
  const mask   = result.output.data; // Float32Array 256×256
  return { data: mask, width: 256, height: 256 };
  // Usar exactamente igual que el background subtraction actual
}
```

**Sin cambios en renderer.js ni app.js** — el mismo `extractFootLandmarks()` funciona.

---

## Fase 6: Aprendizaje continuo

Agregar en `app.js`:
```js
import { initCapture, captureFrame, downloadDataset } from './training/5_capture_dataset.js';
initCapture();

// En detectionLoop, cuando detección es exitosa:
captureFrame(videoEl, footLms); // guarda en IndexedDB del browser
```

Agregar botón en la UI:
```html
<button onclick="downloadDataset()">Exportar dataset (para re-entrenar)</button>
```

**Ciclo de mejora:**
1. App en producción captura frames automáticamente (cada 2s durante try-on)
2. Mensualmente: usuario descarga ZIP con todos los frames
3. Añadir esas imágenes a `data_augmented/` (ya vienen con landmarks como labels)
4. Re-entrenar: `python 3_train_model.py --epochs 10` (fine-tuning rápido)
5. Exportar nuevo ONNX y subir a GitHub Pages

---

## Métricas objetivo

| Métrica | Objetivo |
|---------|---------|
| IoU (Intersection over Union) | > 0.85 |
| Inference time en móvil | < 40ms |
| Falsos positivos (pared/mueble) | < 5% |
| Detección desde ángulo cenital | > 90% |

---

## Costo total estimado

| Item | Costo |
|------|-------|
| Fotos (tiempo propio) | $0 |
| SAM labeling (Colab) | $0 |
| Augmentación (local) | $0 |
| Entrenamiento (Colab Pro 1 mes) | $10 |
| Storage GitHub LFS para ONNX | $0 (< 100MB) |
| **Total** | **~$10** |
