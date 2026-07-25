# Protocolo de recolección de fotos y VIDEOS reales

## ⭐ VIDEOS (vía principal desde 2026-07-25 — etiquetado masivo con SAM2)

Un video de 30s = ~900 frames. Con SAM2 (`2b_sam2_video.py`) se clickea pie/zapato/pierna **una vez**
y la máscara se propaga sola a todo el video → cientos de frames etiquetados por minuto de trabajo.
Es la misma estrategia de datos de los SDKs comerciales (Wanna/Kivisense).

**Qué grabar (clips de 20-40s, cámara de teléfono, 1080p, 30fps):**
- La persona **de pie**, cámara en mano apuntando a los propios pies (POV real del try-on): caminar
  en el lugar, girar el pie, puntas adentro/afuera, levantar el talón, un paso adelante/atrás.
- Otra persona filmando desde ~1m: cenital, diagonal y lateral, moviendo la cámara lento.
- Variar TODO entre clips (no dentro del clip): persona, calzado (y descalzo/calcetines), piso, luz,
  pantalón largo/corto/arremangado.
- **Objetivo inicial: 20-30 clips** (≥5 personas). Eso son ~4.000-8.000 frames etiquetables.

**Reglas:** movimientos LENTOS (motion blur arruina la propagación de SAM2), el pie siempre dentro del
frame, un solo pie protagonista por clip (v1 del modelo es de una instancia), sin rostros en cuadro.
Guardar los .mp4 en `training/data_videos/`. Luego: `python 2b_sam2_video.py --video data_videos/clip01.mp4`.

Las FOTOS siguen siendo necesarias para el **test set congelado** (T3.6) y para keypoints de calidad —
el protocolo de abajo sigue vigente.

---

Objetivo: **300-500 fotos reales** diversas para (a) fine-tuning del modelo tras el pre-training sintético y (b) el **test set congelado** (ver [ROADMAP.md](../ROADMAP.md) T3.6). Sin estas fotos no hay forma de medir el éxito real del modelo — el sintético solo no basta.

> **Regla de oro:** 50 de estas fotos, etiquetadas con `2_sam_label.py`, se **congelan** en `training/data_test_frozen/` y **nunca** entran al entrenamiento. La métrica del proyecto (IoU, PCK) se mide siempre ahí.

## Distribución de ángulos (repartir el total)

| Ángulo | % | Descripción |
|--------|---|-------------|
| Cenital (desde arriba) | 40% | Cámara apuntando casi vertical al piso, pie abajo |
| Diagonal 45° | 30% | Ángulo típico de selfie de pies |
| Frontal | 20% | Cámara a la altura de la rodilla mirando los pies |
| Lateral | 10% | Vista desde el costado del pie |

## Diversidad obligatoria

- **Personas:** mínimo **5 distintas** — variar tono de piel, tamaño y forma de pie.
- **Calzado:** zapatillas, zapatos, botas, sandalias, **medias/calcetines** y **pie descalzo** (el modelo debe segmentar el pie tanto vestido como descalzo).
- **Pierna/pantalón:** mitad de las fotos **con pantalón** cubriendo el tobillo (largo y corto), mitad con la pierna descubierta. Es la frontera pierna↔zapato la que más cuesta segmentar.
- **Pisos (≥6 tipos):** madera, cerámico, alfombra, cemento, exterior (pasto/vereda), oscuro.
- **Iluminación:** natural, artificial cálida, mixta, contraluz suave, sombra dura del propio pie.
- **Distancia:** variar 30 / 60 / 90 / 120 cm desde la cámara.

## Requisitos técnicos

- **Encuadre:** como una selfie de pies real — incluir **parte de la pierna hasta ~la rodilla** cuando el ángulo lo permita (así el modelo aprende la frontera pierna/zapato y coincide con el uso real de la app).
- **Resolución mínima:** 720×720 (idealmente 1080+). Cuadrada o vertical; el pipeline hace center-crop cuadrado.
- **Cámara:** teléfono real (no captura de webcam de laptop) — queremos el ruido/compresión de móvil real.
- **Formato:** JPG. Nombre libre; el `stem` debe ser único.
- **Un pie por foto** por ahora (el modelo v1 asume una instancia). Fotos con dos pies: guardarlas aparte para v2.

## Flujo

1. Sacar las fotos siguiendo la distribución de arriba. Guardarlas en `training/data/images/`.
2. Etiquetar con `python 2_sam_label.py --checkpoint sam_vit_b_01ec64.pth`:
   clase pierna/pie/zapato (teclas 1/2/3) + 6 keypoints (tecla K: heel, toe, ankle_in, ankle_out, ball, toe_tip).
   Genera `data/masks/<stem>.png` (índices 0-3) + `data/keypoints.jsonl`.
3. Elegir **50** representativas (cubriendo todos los ángulos y pisos) y moverlas a `training/data_test_frozen/`
   (imágenes + máscaras + su línea de keypoints). Ese set queda **congelado** — ver T3.6.
4. El resto (`data/`) es para fine-tuning en T4.3.

## Consentimiento y privacidad

Son fotos de partes del cuerpo de personas reales. Antes de usarlas:
- Consentimiento explícito de cada persona para uso en entrenamiento de un modelo comercial.
- No incluir rostros ni datos identificatorios en el encuadre.
- Guardar las fotos localmente / en almacenamiento privado; no subirlas a repos públicos.
