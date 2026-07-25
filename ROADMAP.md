# ROADMAP MAESTRO — AR Shoe Try-On (plugin web embebible)

> **Documento de ejecución por agentes.** Cada tarea tiene: modelo asignado (`[OPUS]` = razonamiento complejo / arquitectura / CV / ML, `[SONNET]` = tarea mecánica bien especificada, `[HUMANO]` = solo la puede hacer el dueño del proyecto), dependencias, entregable, criterio de aceptación verificable, y un prompt listo para copiar (las tareas `[HUMANO]` llevan checklist en lugar de prompt).
> **Cómo usarlo:** para cada tarea, abrir una sesión del agente indicado y pegar `PROMPT MAESTRO DE CONTEXTO` + el prompt de la tarea. No ejecutar tareas fuera de orden sin respetar las dependencias y los GATES.
> Auditoría de origen: 2026-07-23 (workflow de 7 agentes: lectura de código completa, investigación de mercado con fuentes, 3 evaluadores independientes).

---

## PROMPT MAESTRO DE CONTEXTO (pegar al inicio de CADA sesión de agente)

```
CONTEXTO DEL PROYECTO — AR Shoe Try-On (c:\Users\Iga\Documents\ar-shoes, repo git, Windows 11)

QUÉ ES: try-on virtual de zapatillas 100% en navegador (móvil), sin backend, servido como página estática
(GitHub Pages). Objetivo de producto: plugin/widget embebible en e-commerce de calzado (Shopify/WooCommerce),
apuntando a tiendas pequeñas que no pueden pagar SDKs como Wanna (~$5.000/mes).

ARQUITECTURA ACTUAL (vanilla JS + Three.js 0.158 por importmap CDN, sin bundler):
- index.html: UI de calibración en 2 pasos + video + canvas.
- app.js: orquestador. Dos loops desacoplados: render 60fps (requestAnimationFrame) y detección ~5Hz
  (while + sleep(200)). API que consume: initPose(), detectPose(videoEl) -> {data,width,height},
  extractFootLandmarks(seg, side) -> {heel,toe,ankle,bboxW,bboxH,side}, detectDominantFoot(seg).
- pose.js: detección por SUSTRACCIÓN DE FONDO pura (sin ML): captura frame de referencia del piso vacío
  a 256x256, diff RGB umbral 0.10, landmarks SINTÉTICOS por centroide+bbox+PCA (heel/toe a ±45% del bbox,
  "ankle" = centroide). Ambigüedad de 180° en el PCA. Sin protección contra movimiento de cámara.
- renderer.js: cámara ORTOGRÁFICA -1..1, video como plano de fondo, zapato GLB anclado al centroide con
  rotation.set(-PI/2, 0, anguloPCA+PI) => SOLO yaw 2D, sin pitch/roll, sin perspectiva. Escala no métrica:
  clamp 20-38% de pantalla / constante hardcodeada 0.262. buildOccluder()/updateMask() son NO-OPS (sin oclusión).
- filter.js: OneEuroFilter correcto (minCutoff=1.0, beta=0.1), 3 filtros (heel/toe/ankle).
- CÓDIGO MUERTO (no importado por app.js): segmenter.js (MediaPipe selfie multiclass), depth.js (MiDaS),
  solver.js — OJO: solver.js NO es un PnP terminado: su solvePose() retroproyecta 3 puntos usando el
  depthMap de MiDaS (que se elimina) y construye ejes por productos cruzados; lo REUTILIZABLE son su
  svd3x3() y la constante FOOT_MODEL_3D (modelo canónico de pie ~26cm). En Fase 5 se escribe un PnP real
  (DLT/EPnP + refinamiento) sobre esas piezas.
- models/shoe.glb (18MB, pie+pierna+zapato juntos, asset de pie de Sketchfab "Human Foot in Blender"),
  models/shoe2.glb (8.5MB), models/midas_v21_small.onnx (66MB, huérfano).

PIPELINE DE ENTRENAMIENTO (training/, NUNCA ejecutado end-to-end, con bugs bloqueantes documentados en el roadmap):
- 0a_extract_frames.py (frames de video), 0b_blender_render.py (render sintético Cycles imagen+máscara),
  1_augment.py (augmentación estática 1000x — A ELIMINAR), 2_sam_label.py (etiquetado con SAM por clic),
  3_train_model.py (MobileNetV2 encoder + U-Net decoder, 256x256, BCE+Dice — slicing del encoder ROTO),
  4_export_onnx.py (import roto), 5_capture_dataset.js (captura continua — DESACTIVADO por diseño inviable).
- training/models/foot.glb (47KB placeholder cono rosado), training/previews/ (renders del 28-jun con
  domain gap severo), training/data/ VACÍO (0 fotos reales).

REGLAS DURAS (no negociables):
1. NUNCA reescalar/reposicionar los GLBs en scripts (ni Blender ni JS): el usuario los alinea a mano en
   Blender. El encuadre se ajusta SOLO con distancia/posición de cámara.
2. Plataforma Windows: todo script Python multiproceso necesita guard if __name__ == "__main__".
   Blender instalado en: C:\Program Files\Blender Foundation\Blender 5.1\blender.exe (API 4.x+:
   'Specular IOR Level', no 'Specular').
3. Sin bundler ni backend: la app debe seguir siendo estática (ES modules + CDN), desplegable en GitHub Pages.
4. Español para UI y docs; código con comentarios mínimos y en el estilo existente.
5. No lanzar el batch de 5000 renders sin pasar el GATE G4a (piloto con IoU real > 0.6-0.7).
6. Commits: mensajes cortos en inglés como los existentes (git log), trabajo en la rama indicada por la tarea.

DECISIÓN ARQUITECTÓNICA CENTRAL (auditoría 2026-07-23):
El target final del modelo NO es segmentación binaria: es una red multi-branch (encoder compartido) con
(a) heatmaps de keypoints anatómicos del pie y (b) máscara MULTI-CLASE {fondo, pierna/pantalón, pie, zapato}.
Los keypoints alimentan PnP (solver.js) contra un modelo canónico del pie => pose 6DoF real (R|t) del pie;
la máscara sirve para OCLUSIÓN (pierna tapa la caña) y a futuro shoe-erasing. Es la receta de ARShoe
(arXiv 2108.10515), Springer 2025 (s40747-025-02188-x) y de los SDKs comerciales (Wanna/Vyking/Snap).
Referencia open source funcional: WebAR.rocks.hand (MIT) con demo "Shoes VTO". No existe checkpoint público
de pose 6DoF de pie: por eso se entrena modelo propio con renders sintéticos de Blender + fine-tuning real.

SECUENCIA Y GATES: Fase 0 (rescate) -> Fase 1 (demo MediaPipe) -> GATE G2 (validación de demanda: >=3 tiendas
interesadas) -> Fase 3 (arreglar pipeline de entrenamiento) -> GATE G3 (smoke test end-to-end) -> Fase 4
(dataset sintético + entrenamiento, con GATE G4a piloto) -> Fase 5 (runtime 6DoF) -> Fase 6 (productización plugin).
```

---

## Mapa de dependencias

```
FASE 0 (rescate repo, 1 día)
  └─> FASE 1 (demo MediaPipe + GLB ligero + deploy, 1-2 semanas)
        ├─> T1.4 benchmark DeepAR (paralelo, 1-2 días)
        └─> FASE 2 (validación de demanda, 2 semanas)  ──> GATE G2 (>=3 tiendas)
              └─> FASE 3 (fix training pipeline, 1 semana)  ──> GATE G3 (smoke test e2e → merge a main)
                    ├─ [HUMANO] GLBs reales del experto Blender + ≥10 zapatos ALINEADOS (bloqueo de T4.1)
                    ├─ [HUMANO] 300-500 fotos reales (T3.5, puede empezar antes)
                    ├─ T3.6 test set: 50 fotos etiquetadas y CONGELADAS (bloqueo de T4.2)
                    └─> FASE 4 (piloto ──GATE G4a──> batch 5000 + training)
                          └─> FASE 5 (runtime 6DoF: ONNX + PnP + oclusión)
                                └─> FASE 6 (widget embebible + piloto con tiendas)
```

Nota: Fase 3 puede adelantarse en paralelo a Fase 2 si hay capacidad — es trabajo de escritorio sin costo de
datos. Lo único estrictamente bloqueado por G2 es el GASTO grande: batch de 5000, sesiones de fotos masivas
y semanas de entrenamiento.

---

# FASE 0 — Rescate del repo (1 día)

### T0.1 — Commit de rescate + Git LFS `[SONNET]`
**Depende de:** nada. **Entregable:** rama `rescue/jun28-work` con todo commiteado.
**Criterio de aceptación:** `git status` limpio; GLBs en LFS; push al remoto si existe.

**Prompt:**
```
Tarea T0.1 del ROADMAP.md. En c:\Users\Iga\Documents\ar-shoes:
1. Crea la rama rescue/jun28-work desde main.
2. Git LFS — SOLO para artefactos de entrenamiento que la web NUNCA sirve: *.pth y training/**/*.onnx.
   NO metas en LFS los assets servidos por la app (models/*.glb, models/*.onnx): GitHub Pages y jsDelivr
   NO resuelven punteros LFS y la demo quedaría rota sirviendo archivos de texto. Los GLB de la web se
   commitean normales (T1.2 los baja a <4MB, tamaño aceptable en git). Si más adelante se despliega Pages
   vía GitHub Actions con checkout lfs:true esta restricción puede revisarse — déjalo anotado en el commit.
3. Commitea en commits separados y descriptivos: (a) models/shoe.glb modificado + models/shoe2.glb,
   (b) training/: check_gpu.py, test_angles.py, test_render.py, test_color.jpg, test_mask.png,
   training/models/foot.glb, training/previews/*, (c) .claude/settings.local.json solo si tiene cambios
   relevantes del proyecto (si es ruido de sesión, restáuralo).
4. NO borres nada todavía (eso es T0.2). NO toques la rama main.
Verifica con git log --oneline y git status, y reporta el resultado.
```

### T0.2 — Limpieza de código muerto `[SONNET]`
**Depende de:** T0.1. **Entregable:** repo sin código/assets muertos, en la misma rama.
**Criterio de aceptación:** la app sigue cargando (sin errores de import en consola); `solver.js` CONSERVADO.

**Prompt:**
```
Tarea T0.2 del ROADMAP.md. En la rama rescue/jun28-work:
1. Elimina: segmenter.js, depth.js, models/midas_v21_small.onnx (66MB, huérfano).
   CONSERVA solver.js — añade un comentario de 1 línea al inicio:
   "// NO BORRAR: svd3x3 y FOOT_MODEL_3D se reutilizan para el PnP de Fase 5 (ver ROADMAP.md T5.2)".
2. En app.js: elimina imports/símbolos muertos (computeScaleFactor y hasBgData se importan y nunca se usan;
   lastFootLms se asigna y nunca se lee). Elimina también la llamada a measureGLBLength en app.js
   (solo alimenta a computeScaleFactor, que es muerto) y borra scaler.js con su import.
3. En renderer.js: el canvas de debug con los 3 puntos (drawDebugDots) debe quedar detrás de un flag:
   actívalo solo si location.search incluye "debug=1"; en producción no debe crearse el canvas.
4. En style.css: elimina la regla #permission-error (elemento inexistente).
5. En index.html/app.js: sanea el innerHTML del error fatal (usa textContent para err.message).
6. Prueba que no haya referencias rotas: grep de cada símbolo eliminado. Commitea con mensaje tipo
   "remove dead code (segmenter/depth/midas), gate debug overlay behind ?debug=1".
```

### T0.3 — Mitigaciones de estabilidad en la app actual `[SONNET]`
**Depende de:** T0.2. **Entregable:** app actual más robusta mientras llega MediaPipe.
**Criterio de aceptación:** con la cámara movida bruscamente tras calibrar, el zapato se OCULTA (no salta); al recalibrar o cambiar de pie, los filtros se resetean.

**Prompt:**
```
Tarea T0.3 del ROADMAP.md. En la rama rescue/jun28-work:
1. pose.js: detección de cámara movida. CONTEXTO: en 104b305 existió un rechazo "si >30% del frame
   difiere → null" (vivía en extractFootLandmarks, revisa `git show 104b305`), y b0dd106 lo eliminó
   A PROPÓSITO porque pierna+pantalón superan legítimamente el 30% en encuadres normales. NO restaures
   ese umbral tal cual: usa una señal mejor — mide el % de píxeles cambiados SOLO en el tercio SUPERIOR
   del frame (los pies viven abajo; si el tercio superior cambió >40%, la cámara se movió). En ese caso
   detectPose retorna null y app.js muestra "Recalibrando…" — opcionalmente recaptura el fondo
   automáticamente tras 1s de frame estable.
2. app.js: al pulsar Recalibrar y al cambiar de pie (toggleFoot), recrea los filtros
   (filters = createLandmarkFilters(3, 30)) para que el OneEuro no arrastre el estado anterior.
3. app.js/detectionLoop: fija el lado dominante con histéresis: solo cambia currentSide si
   detectDominantFoot devuelve el mismo lado nuevo durante 5 ciclos seguidos.
4. pose.js: corrige la distorsión de aspecto 1280x720 -> 256x256: haz center-crop cuadrado del video
   antes de reescalar (drawImage con recorte centrado), tanto en captureBackground como en detectPose.
   CRÍTICO: renderer.js sigue estirando el frame COMPLETO a pantalla, así que los landmarks calculados
   en espacio-crop deben REMAPEARSE a coordenadas normalizadas del frame completo antes de devolverlos
   (x' = (x_crop * cropW + offsetX) / frameW, análogo en y) — si no, el zapato queda desplazado.
Prueba mental de regresión: la firma pública de pose.js no cambia. Commitea.
```

### T0.4 — Verificación de licencia del asset de Sketchfab `[HUMANO]`
**Depende de:** nada. **Entregable:** confirmación escrita de que el modelo de pie ("Human Foot in Blender") permite uso comercial y generación de datasets de ML; si no, encargar al experto Blender un reemplazo propio o buscar asset CC0.
**Por qué importa:** un dataset comercial generado con un asset sin licencia contamina legalmente el modelo entrenado.

---

# FASE 1 — Demo pública estable con MediaPipe (1-2 semanas)

> **⚠️ RESUELTO 2026-07-25 — MediaPipe DESCARTADO para el caso de uso real (verificado empíricamente).**
> Experimento controlado (Pose Landmarker lite 0.10.35, umbrales mínimos 0.25, chromium headless):
> con **persona completa** detecta con visibilidad de pies 0.84; con recorte a rodillas cae a 0.33;
> con **solo pies NO detecta**, y con una foto real **POV cintura-abajo** (el encuadre del try-on,
> jeans+zapatillas nítidas) **NO detecta nada** en ningún recorte. Causa estructural: el person-detector
> ancla en cara/torso. Coincide con la QA en dispositivo del usuario y con el histórico del repo
> (`4f56d70`→`0d809a7`). **Decisión del usuario: proceder directo con el modelo propio (Fases 3-4),
> que es la receta Wanna/Kivisense.** La demo de Pages queda desplegada como demo técnica limitada
> (funciona con encuadre de cuerpo casi completo) pero NO es el producto. T1.4/F2: el outreach se
> hará con la demo del modelo propio o material grabado, no con la demo MediaPipe.

### T1.1 — Sustituir sustracción de fondo por MediaPipe Pose Landmarker `[OPUS]`
**Depende de:** T0.3. **Entregable:** `pose.js` reescrito sobre MediaPipe manteniendo la API; sin pantalla de calibración.
**Criterio de aceptación:** el zapato sigue al pie SIN calibrar fondo, sobrevive a movimiento de cámara, funciona con la app servida por `npx serve` o GitHub Pages en un móvil real.

**Prompt:**
```
Tarea T1.1 del ROADMAP.md — la más importante de la fase. Rama: feature/mediapipe-pose desde rescue/jun28-work.

OBJETIVO: reemplazar la sustracción de fondo (pose.js) por MediaPipe Pose Landmarker (tasks-vision),
manteniendo INTACTA la API que consume app.js: initPose(), detectPose(videoEl), extractFootLandmarks(seg, side),
detectDominantFoot(seg). renderer.js y filter.js NO se tocan en esta tarea.

IMPLEMENTACIÓN:
1. Carga @mediapipe/tasks-vision por CDN (jsdelivr) como ES module, modelo pose_landmarker_lite.task
   (float16) desde el storage oficial de Google. delegate: "GPU" con fallback a "CPU" si falla.
   runningMode: "VIDEO", numPoses: 1 (dos pies del MISMO cuerpo vienen en una sola pose).
2. initPose(): crea el PoseLandmarker (async). Elimina captureBackground/hasBgData y TODO el flujo de
   calibración: en index.html/app.js quita el paso 1 (calibrar fondo) — la app pasa directo a
   "pon tu pie en la cámara". Simplifica showStep en consecuencia.
3. detectPose(videoEl): llama detectForVideo(videoEl, performance.now()). De result.landmarks[0] usa los
   índices: 27=tobillo izq, 28=tobillo der, 29=talón izq, 30=talón der, 31=punta (foot_index) izq,
   32=punta der. Devuelve una estructura {left:{ankle,heel,toe}, right:{...}} con x,y normalizados y
   visibility. Mantén el contrato: si no hay pose o la visibility media de un pie es < 0.5, ese pie es null;
   si ambos null, retorna null.
4. extractFootLandmarks(seg, side): ahora trivial — devuelve el pie pedido con heel/toe/ankle REALES.
   Calcula bboxW/bboxH del triángulo heel-toe-ankle expandido ~40% (renderer.js los usa para escala).
5. detectDominantFoot(seg): el pie con mayor visibility media (o el más grande en pantalla si empatan).
6. app.js/detectionLoop: sube la frecuencia de detección a ~15Hz (sleep(66)) — MediaPipe lite lo permite;
   los filtros OneEuro existentes se siguen aplicando tal cual.
7. UX de encuadre: MediaPipe Pose está entrenado con cuerpo entero y falla en encuadres de SOLO pies.
   Añade al paso inicial el texto "Apunta la cámara a tus pies incluyendo parte de la pierna (hasta la
   rodilla)". Si no hay detección durante 3s, muestra ese hint en el status.
8. Mantén el flag ?debug=1 mostrando ahora los 6 landmarks reales de ambos pies.

VALIDACIÓN: sirve la app (npx serve o python -m http.server), ábrela, verifica en consola que no hay
errores de carga del wasm/modelo. Documenta en el commit qué versión de tasks-vision fijaste (pin de
versión exacta en la URL del CDN, nunca @latest).

CONTEXTO CRÍTICO 1: MediaPipe da 3 puntos REALES por pie pero NO da roll/pitch del pie — la rotación del
zapato seguirá siendo el yaw 2D actual de renderer.js. Eso es ACEPTABLE en esta fase: el objetivo es
estabilidad y eliminar la calibración, no pose 6DoF (eso llega en Fase 5 con el modelo propio).

CONTEXTO CRÍTICO 2 — HISTORIA: Pose Landmarker YA fue la implementación original de este proyecto y se
abandonó (commits 4f56d70 inicial, 5fa15b4 cambio a pose_landmarker_full + guía de encuadre, 0d809a7
eliminación de umbrales de visibilidad por mala detección). Revisa esos diffs ANTES de empezar para no
repetir los mismos errores. Qué debe ser distinto esta vez: (a) el problema de entonces era encuadre de
solo-pies con el modelo full — usa lite + delegate GPU y OBLIGA el encuadre pie+pierna desde la UX (hint
persistente, no solo inicial); (b) hoy existen OneEuro + histéresis que entonces no existían — el jitter
que motivó el abandono ahora se filtra; (c) criterio de aceptación NUEVO: prueba explícita con encuadre
de SOLO pies (sin rodilla): si la detección cae, el hint de encuadre debe recuperar al usuario en <5s.
Si tras implementar bien sigue siendo inutilizable en móvil real, documenta el fallo con evidencia y
propone el plan B (WebAR.rocks como tracker provisional) ANTES de descartar — no vuelvas a background
subtraction.
```

### T1.2 — Optimizar el GLB para web `[SONNET]`
**Depende de:** T1.1 (misma rama). **Entregable:** `models/shoe_web.glb` < 4MB usado por la app.
**Criterio de aceptación:** carga visualmente idéntica (a resolución móvil), sin cambiar transforms/escala del modelo; tiempo de carga en 4G < 3s.

**Prompt:**
```
Tarea T1.2 del ROADMAP.md. Rama: feature/mediapipe-pose (créala desde rescue/jun28-work si aún no existe).
models/shoe.glb pesa 18MB — inviable en móvil.
1. Usa @gltf-transform/cli (npx, sin instalar globalmente): inspecciona primero (npx @gltf-transform/cli
   inspect models/shoe.glb) y reporta qué pesa: texturas vs geometría.
2. Genera models/shoe_web.glb aplicando: resize de texturas a máx 1024px, compresión de texturas
   (webp si el pipeline lo soporta; si no, jpeg quality 80), draco o meshopt para geometría, prune y dedup.
   PROHIBIDO: cualquier transform que cambie escala/posición/rotación de nodos (regla dura #1) — no uses
   comandos tipo "center" ni "resample" de transforms.
3. Actualiza GLB_PATH en app.js a ./models/shoe_web.glb.
4. Verifica el tamaño final (<4MB; ideal <2MB) y que la app lo carga sin errores.
5. Repite para shoe2.glb -> shoe2_web.glb (sin conectarlo a la app, es asset alternativo).
Commitea con los tamaños antes/después en el mensaje.
```

### T1.3 — Deploy público + QA en móviles `[SONNET]` + `[HUMANO]`
**Depende de:** T1.1, T1.2. **Entregable:** URL pública en GitHub Pages desde la rama de la demo.
**Criterio de aceptación (lo verifica el humano):** estable en ≥3 móviles reales (mín. 1 iPhone Safari, 1 Android Chrome), sin recalibrar nunca, zapato no "salta" al mover la cámara.

**Prompt:**
```
Tarea T1.3 del ROADMAP.md. Configura GitHub Pages para servir la app:
1. Verifica que feature/mediapipe-pose contiene T1.1 Y T1.2 (shoe_web.glb + GLB_PATH actualizado);
   si falta algo, intégralo primero.
2. Si no hay remoto GitHub, créalo con gh repo create (privado o público según diga el usuario; pregunta
   solo si no está definido). Push de la rama feature/mediapipe-pose.
3. Activa Pages (gh api o instrucciones) sirviendo desde esa rama (o desde main tras merge, según prefiera
   el usuario). Verifica que los paths relativos (./models/, importmap CDN) funcionan bajo el subpath
   /<repo>/ de Pages — si no, ajusta a rutas relativas. Recuerda: los assets servidos NO deben estar en
   Git LFS (Pages no resuelve punteros LFS — regla fijada en T0.1).
4. Añade al README la URL pública y un QR (generado con cualquier lib JS local o un data-URI SVG).
5. Checklist de QA manual para el humano (déjalo en el README): iPhone Safari, Android Chrome,
   luz baja, pantalón largo vs corto, encuadre de solo-pies, mover la cámara mientras se usa,
   pie izquierdo y derecho.
6. MERGE: cuando el humano apruebe el QA, mergea feature/mediapipe-pose (que incluye rescue/jun28-work)
   a main y apunta Pages a main. A partir de aquí main es la demo pública estable.
```

### T1.4 — Benchmark DeepAR (paralelo, 1-2 días) `[SONNET]`
**Depende de:** nada (paralelo a T1.1). **Entregable:** `benchmark/deepar/` con demo funcional del shoe try-on de DeepAR + informe comparativo corto.
**Criterio de aceptación:** demo corriendo con licencia gratuita; `benchmark/COMPARATIVA.md` con capturas y conclusiones honestas (latencia, calidad de tracking, oclusión) vs nuestra demo.

**Prompt:**
```
Tarea T1.4 del ROADMAP.md. DeepAR (docs.deepar.ai/use-cases/shoe-try-on, desde $25/mes con tier gratuito
para proyectos pequeños) es el benchmark comercial barato.
1. Crea benchmark/deepar/ con su demo web oficial de shoe try-on adaptada (necesita API key gratuita:
   deja instrucciones claras de dónde la pone el usuario, NO la commitees).
2. Escribe benchmark/COMPARATIVA.md: tabla nuestra-demo vs DeepAR en: setup requerido, tracking con
   cámara en movimiento, rotación 3D del zapato, oclusión de pierna, latencia percibida, tamaño de assets.
3. Sé honesto: este benchmark define el listón de calidad que el modelo propio debe superar para
   justificar su existencia. Si DeepAR es claramente suficiente para el caso de uso, dilo — es señal
   de "buy" como plan B, no un fracaso.
```

---

# FASE 2 — Validación de demanda (2 semanas) — **GATE G2**

> **Nota 2026-07-25:** con MediaPipe descartado (ver Fase 1), la demo en vivo para outreach será la del
> modelo propio (post-G4a). El usuario decidió asumir el build sin esperar G2; el gate se mantiene como
> límite del GASTO GRANDE (batch 5000 / GPU) pero el trabajo de datos con videos+SAM2 (T3.7) procede ya.

### T2.1 — Landing + widget de ejemplo embebido `[SONNET]`
**Depende de:** T1.3. **Entregable:** `landing/index.html` estática: propuesta de valor, demo embebida (iframe del try-on), CTA de contacto.
**Criterio de aceptación:** desplegada en Pages; el try-on funciona embebido en un iframe con `allow="camera"`.

**Prompt:**
```
Tarea T2.1 del ROADMAP.md. Crea landing/ (estática, español, mobile-first):
1. Propuesta de valor para tiendas de calzado pequeñas: "try-on AR en tu tienda online sin apps,
   un script y listo" — precio de referencia sugerido en la página: desde $29/mes (editable).
2. Embebe la demo real en un iframe con allow="camera; fullscreen" y un botón "Pruébalo ahora".
   Verifica que getUserMedia funciona dentro del iframe (necesita allow y HTTPS).
3. Sección "cómo se integra": snippet <script src=".../ar-shoes-widget.js" data-model="URL_DEL_GLB">
   (aspiracional, el widget real es Fase 6 — márcalo como "beta").
4. CTA: mailto o formulario simple (formspree u otro sin backend) a aritos.3d@gmail.com.
5. Despliega junto a la app en Pages (subcarpeta /landing).
```

### T2.2 — Kit de outreach a tiendas `[SONNET]` + ejecución `[HUMANO]`
**Depende de:** T2.1. **Entregable:** `outreach/` con lista de 15 tiendas objetivo (Shopify/WooCommerce de calzado, hispanohablantes primero), plantilla de email/DM personalizable, y guion de demo de 5 min.

**Prompt:**
```
Tarea T2.2 del ROADMAP.md. Crea outreach/ con:
1. TIENDAS.md: 15 tiendas online de calzado pequeñas/medianas (usa WebSearch: Shopify/WooCommerce,
   hispanohablantes primero, luego EU/US), con URL, plataforma detectada, contacto visible y una línea
   de personalización por tienda (qué venden, por qué el try-on les aporta).
2. EMAIL.md: plantilla corta (max 120 palabras) con hueco de personalización, link a la landing y CTA
   de 15 min de llamada; variante DM de Instagram. Sin humo: la demo es beta y se ofrece piloto gratis
   a cambio de feedback.
3. DEMO_GUION.md: guion de 5 min (problema → demo en su propio producto si es posible → qué incluye el
   piloto → siguiente paso) + tabla para registrar respuestas (tienda, fecha, respuesta, interés 0-3).
El envío y las llamadas son [HUMANO]. El GATE G2 se evalúa con la tabla rellena tras ~2 semanas.
```
**GATE G2 (lo evalúa el humano tras ~2 semanas de outreach):** **≥3 tiendas piden probarlo en su web o firman intención → continuar a Fase 3/4 con gasto. 0-1 → parar y repensar (pivotar de nicho, o plan B DeepAR como integrador).** Las Fases 3 (arreglos de código, costo $0) pueden avanzar en paralelo mientras corre el outreach.

---

# FASE 3 — Corrección del pipeline de entrenamiento (1 semana, costo $0) — **GATE G3**

> Puede ejecutarse en paralelo a Fase 2. Es trabajo de escritorio: arregla lo roto y rediseña el target del modelo. Ningún gasto en datos/GPU hasta G2+G3 aprobados.

### T3.1 — Rediseñar y arreglar el modelo: multi-branch keypoints + máscara multi-clase `[OPUS]`
**Depende de:** nada (paralelo a F2). **Entregable:** `training/train_model.py` (renombrado) funcional con arquitectura multi-branch.
**Criterio de aceptación:** `python training/train_model.py --smoke` corre en CPU Windows sin dataset real (datos dummy autogenerados): forward OK, 2 epochs OK, checkpoint guardado, métricas IoU y PCK impresas.

**Prompt:**
```
Tarea T3.1 del ROADMAP.md — corazón técnico del proyecto. Reescribe training/3_train_model.py como
training/train_model.py (nombre importable). El actual tiene bugs que impiden hasta un forward.

BUGS CONFIRMADOS A CORREGIR (verifícalos tú mismo antes):
a) Slicing de MobileNetV2 roto: features[11:16] termina en 160ch@8x8 (la capa 14 tiene stride 2), no
   96ch@16x16; features[16:] entrega 1280ch, no 320. Fix de slices: enc4 = features[11:14] (96ch@16x16),
   bottleneck = features[14:18] (320ch@8x8, excluyendo la conv final 1280 que es solo para clasificación).
   OJO: ese fix OBLIGA a reestructurar el decoder, porque e3 = features[7:11] (64ch) queda TAMBIÉN a
   16x16 — dos skips en la misma resolución. Estructura correcta de resoluciones: e0=32ch@128,
   e1=24ch@64, e2=32ch@32, e3=64ch@16, e4=96ch@16, b=320ch@8. Decoder sugerido: up(8→16) concatenando
   [e3;e4] juntos (64+96=160ch de skip), luego up(16→32) con e2, up(32→64) con e1, up(64→128) con e0,
   up(128→256) sin skip — una etapa reestructurada, no los 5 UpBlocks actuales tal cual.
   VERIFICA canales/resoluciones reales ejecutando un tensor dummy por cada slice antes de fijarlos,
   y un forward completo del modelo nuevo antes de commitear.
b) Falta guard if __name__ == "__main__" (Windows + num_workers>0 = crash). Estructura el script en
   funciones con el guard.
c) Fuga train/val: el split debe hacerse POR IMAGEN BASE (agrupando por stem antes del sufijo _aug_XXXX
   y, cuando exista metadato, por escena/persona), nunca sobre archivos augmentados mezclados.
d) Augmentación: elimina la dependencia del 1000x estático de 1_augment.py. Integra albumentations
   ON-THE-FLY en Dataset.__getitem__ (mismas transforms que 1_augment.py pero aplicadas al vuelo,
   sincronizadas imagen/máscara/keypoints con KeypointParams). 1_augment.py queda obsoleto: bórralo y
   anota en PIPELINE.md que la augmentación vive en el Dataset.

NUEVA ARQUITECTURA (decisión de la auditoría 2026-07-23 — receta ARShoe arXiv 2108.10515):
- Encoder MobileNetV2 compartido (width_mult 1.0; deja alpha=0.5 como flag --small para experimentar).
- Head 1 — SEGMENTACIÓN MULTI-CLASE: 4 clases {0:fondo, 1:pierna/pantalón, 2:pie, 3:zapato},
  salida 4 canales + softmax (CrossEntropy + Dice multiclase). Ya NO es binaria con sigmoid.
- Head 2 — KEYPOINTS: heatmaps gaussianos (sigma~2px a resolución de salida) para 6 puntos:
  heel, toe (2ª cabeza metatarsal), ankle_inner, ankle_outer, ball (bola del pie), toe_tip.
  Decoder propio ligero compartiendo skips, salida 6 canales a 64x64, loss MSE sobre heatmaps.
  Los keypoints del pie IZQUIERDO y DERECHO se manejan con flag de instancia: en esta versión, una sola
  instancia por imagen (el dataset sintético renderiza un pie); documenta la extensión a 2 pies como TODO.
- Loss total = CE_seg + Dice_seg + w_kp * MSE_heatmaps (w_kp=10 inicial, flag).
- Métricas de validación OBLIGATORIAS: mIoU por clase (y IoU de la unión pie+zapato para comparar con el
  objetivo histórico >0.85) y PCK@0.1 de keypoints (correcto si dista <10% del largo del pie).
  El "mejor checkpoint" se elige por mIoU, no por loss.
- Dataset loader: UN SOLO formato para sintético y real — images/*.jpg, masks/*.png (índices 0-3) y
  keypoints.jsonl (una línea por imagen: {file, kps:[[x,y,vis]x6]}); es el formato que emiten tanto
  0b_blender_render.py v2 (T3.2) como la herramienta de etiquetado real 2_sam_label.py v2 (T3.4).
  Tolerancias: keypoints con vis=0 no contribuyen a la loss; si una imagen no tiene entrada en
  keypoints.jsonl, w_kp=0 para esa muestra. Normalización FIJA del proyecto: mean=[0.485,0.456,0.406],
  std=[0.229,0.224,0.225] (ImageNet) — la misma que usará inference.js (T3.3); documéntala en PIPELINE.md.
- Modo --smoke: genera en memoria 20 muestras dummy (ruido + máscaras/keypoints sintéticos triviales),
  corre 2 epochs, imprime métricas y guarda checkpoint. Sirve de test de humo sin datos.

Actualiza 4_export_onnx.py -> export_onnx.py: importa el modelo desde train_model (ahora importable),
exporta con opset>=17, entrada dinámica batch=1 3x256x256, DOS salidas nombradas (seg, heatmaps),
añade cuantización dinámica int8 opcional (--int8, onnxruntime.quantization.quantize_dynamic), y emite
junto al .onnx un sidecar <nombre>.meta.json con {mean, std, input_size, clases, keypoints, checkpoint,
commit, fecha, métricas} para que inference.js lo lea en vez de hardcodear.
Al final: python train_model.py --smoke && python export_onnx.py --checkpoint <smoke.pth> deben pasar
en esta máquina (usa training/check_gpu.py para saber si hay CUDA; CPU vale para el smoke).
Actualiza PIPELINE.md con la nueva arquitectura. Commitea en rama feature/training-v2.
```

### T3.2 — Reescribir el render sintético de Blender `[OPUS]`
**Depende de:** nada (paralelo a T3.1; coordina el formato de salida con T3.1). **Entregable:** `training/0b_blender_render.py` v2 para Blender 5.1.
**Criterio de aceptación:** `blender --background --python 0b_blender_render.py -- --count 20 --out data_pilot_test` produce 20 tríos imagen/máscara-multiclase/keypoints correctos, inspeccionados visualmente; pie y zapato NUNCA desalineados; piso/luz distintos en cada render.

**Prompt:**
```
Tarea T3.2 del ROADMAP.md. Reescribe training/0b_blender_render.py para Blender 5.1
(C:\Program Files\Blender Foundation\Blender 5.1\blender.exe, API 4.x+: "Specular IOR Level").
El script actual tiene bugs que corromperían el dataset. training/test_angles.py ya resuelve bien el
encuadre por distancia de cámara proporcional al radio del modelo — reutiliza ese enfoque.

BUGS A ELIMINAR (todos confirmados en el script actual):
a) El piso se crea/randomiza UNA vez fuera del loop -> muévelo DENTRO: material nuevo por render.
b) Solo se rota/escala el pie, no el zapato -> desalineación acumulativa. Fix doble: (1) parenta
   pie+zapato a un Empty raíz y transforma SOLO el Empty (rotación Z aleatoria); (2) al inicio de cada
   iteración RESETEA la transform del Empty a identidad antes de aplicar la nueva (nada acumulativo).
c) PROHIBIDO reescalar los GLBs (regla dura #1): elimina todo obj.scale. La variación de tamaño
   aparente se logra SOLO con distancia de cámara (como test_angles.py).
d) Máscara por swap de materiales con doble render Cycles y view transform que da gris ~204 ->
   reemplaza por render de máscara vía OBJECT INDEX (vía PRINCIPAL, no opcional): asigna pass_index por
   grupo (1=pierna, 2=pie, 3=zapato), activa el pass Object Index en la view layer y emite el PNG de
   índices {0,1,2,3} desde el compositor. Fallback SOLO si el object-index resulta inviable: materiales
   emisivos 0/85/170/255 con film_transparent + view_transform='Standard' + filtro de píxel desactivado
   (filter_size mínimo), y cuantización posterior por valor más cercano — documenta cuál vía quedó activa.
   En ambos casos valida en el preview que la máscara contiene EXACTAMENTE los valores {0,1,2,3}.

NUEVAS CAPACIDADES:
1. KEYPOINTS GT: crea 6 Empties pegados a la malla del pie en: talón, 2ª cabeza metatarsal, maléolo
   interno, maléolo externo, bola del pie, punta del dedo gordo. Colócalos por bounding-box relativo del
   mesh del pie (documenta las proporciones usadas; cuando lleguen los GLBs del experto con Empties
   nombrados, úsalos directamente si existen: kp_heel, kp_toe, kp_ankle_in, kp_ankle_out, kp_ball,
   kp_toe_tip). Proyecta cada Empty con bpy_extras.object_utils.world_to_camera_view y emite
   keypoints.jsonl: {"file":"synth_00001.jpg","kps":[[x,y,vis]x6]} con x,y normalizados 0-1 y
   vis=0 si queda fuera de frame u ocluido (usa ray_cast desde la cámara para la oclusión).
2. PIERNA/PANTALÓN: si el GLB trae colecciones leg_bare / leg_pants_dark / leg_pants_light (brief del
   experto), activa UNA aleatoriamente por render (pass_index=1). Si no existen aún, genera un cilindro
   elíptico simple sobre el tobillo con material de tela (2-3 colores) como pierna provisional y márcalo
   en el log — mejor pierna fea que sin pierna (el domain gap principal es la frontera pierna/zapato).
3. ILUMINACIÓN: soporta carpeta training/hdri/ con .hdr/.exr (Poly Haven, CC0 — deja en el README de
   training/ los 10 nombres sugeridos de HDRIs de interior/exterior a descargar [HUMANO]); por render:
   HDRI aleatorio con rotación Z aleatoria y strength 0.3-1.5. Fallback si la carpeta está vacía: el
   sistema actual de luces aleatorias.
4. PISOS: soporta training/floor_textures/ con imágenes (madera/cerámica/alfombra/cemento); por render:
   textura aleatoria con escala/rotación UV aleatoria. Fallback: colores planos actuales.
5. DEGRADACIÓN DE CÁMARA MÓVIL (crítico para el domain gap — hazlo POST-render con PIL/numpy en el
   mismo script tras guardar: no requiere Cycles): JPEG quality aleatoria 40-85, ruido gaussiano
   sigma 1-4, motion blur direccional leve (p=0.3), viñeteo suave (p=0.3), desviación de balance de
   blancos ±10% por canal (p=0.5). La máscara y keypoints NO se degradan.
6. CLI: --count, --out, --size (default 256), --seed (reproducibilidad), --shoe_dir (carpeta con GLBs
   de calzado: elige UNO aleatorio por render; cada GLB de zapato debe venir ya alineado a mano al pie —
   regla dura #1, la alineación es entregable del experto, NO del script), y --preview (equivale a
   --count 20 y abre un contact-sheet HTML con imagen+máscara superpuesta+keypoints dibujados para
   inspección visual rápida). Registra en un manifest.json por batch: seed, zapato usado por imagen,
   HDRI/piso elegidos — imprescindible para diagnosticar domain gap en T4.2.
Actualiza test_angles.py solo si comparte utilidades (no dupliques código: extrae helpers a
training/blender_utils.py si hace falta). Commitea en feature/training-v2.
```

### T3.3 — Runner de inferencia web (esqueleto) `[SONNET]`
**Depende de:** T3.1 (necesita el ONNX del smoke test). **Entregable:** `inference.js` + página de prueba `test_inference.html`.
**Criterio de aceptación:** el ONNX del smoke test corre en el navegador con onnxruntime-web (WebGPU si está disponible, WASM fallback), mostrando máscara y keypoints sobre una imagen estática, con latencia medida en pantalla.

**Prompt:**
```
Tarea T3.3 del ROADMAP.md. Crea inference.js (ES module, sin bundler) y test_inference.html:
1. Carga onnxruntime-web por CDN (pin de versión exacta). Crea la sesión con
   executionProviders: ['webgpu','wasm'] y loguea cuál quedó activo.
2. API: initInference(modelUrl), runInference(videoElOrImage) -> {seg: Uint8Array 256x256 (argmax de
   4 clases), kps: [[x,y,conf]x6] (argmax por heatmap + refinamiento por centro de masa 3x3)}.
   Preproceso: center-crop cuadrado -> resize 256 -> normalize (mean/std de ImageNet, igual que el
   training) -> tensor NCHW float32.
3. test_inference.html: sube una imagen o usa la webcam, dibuja la máscara coloreada semitransparente
   (4 colores) y los 6 keypoints, muestra latencia media de las últimas 30 inferencias y el EP activo.
4. Objetivo de rendimiento a documentar (no bloquear): <80ms en móvil de gama media con WebGPU.
   El modelo del smoke test da resultados basura — lo que se valida aquí es la TUBERÍA, no la calidad.
Commitea en feature/training-v2.
```

### T3.4 — Actualizar SAM labeling a multi-clase + keypoints `[SONNET]`
**Depende de:** T3.1 (formato de datos). **Entregable:** `training/2_sam_label.py` v2.
**Criterio de aceptación:** sobre una foto de prueba se puede etiquetar máscara {pierna, pie, zapato} con clics SAM por clase + 6 clics de keypoints, guardando en el formato del Dataset de T3.1.

**Prompt:**
```
Tarea T3.4 del ROADMAP.md. Extiende training/2_sam_label.py:
1. Flujo por imagen: tecla 1/2/3 selecciona clase activa (pierna/pie/zapato); clic izquierdo añade
   punto positivo SAM para esa clase, derecho negativo; la máscara final es la composición por índice
   (0=fondo). Tecla K entra en modo keypoints: 6 clics en orden (heel, toe, ankle_in, ankle_out, ball,
   toe_tip; ESPACIO salta uno como no-visible). S guarda (masks/*.png con índices + línea en
   keypoints.jsonl), R resetea, Q sale.
2. Mantén compatibilidad con el checkpoint sam_vit_b existente y el flujo actual de carpetas.
3. Añade --review: recorre lo etiquetado mostrando overlay para control de calidad rápido.
Actualiza las instrucciones de PIPELINE.md (sección Fase 1). Commitea en feature/training-v2.
```

### T3.5 — Protocolo de captura de fotos reales `[SONNET]` (doc) + `[HUMANO]` (ejecución)
**Depende de:** nada. **Entregable:** `training/PROTOCOLO_FOTOS.md` + 300-500 fotos reales recolectadas por el humano a lo largo de las siguientes semanas.
**Criterio de aceptación del doc:** cubre distribución de ángulos (40% cenital, 30% diagonal, 20% frontal, 10% lateral), ≥5 personas con diversidad de tono de piel, calzado variado + descalzo + calcetines, ≥6 tipos de piso, luz variada, con/sin pantalón cubriendo el tobillo, resolución mínima y encuadre tipo selfie-de-pies.

### T3.6 — Etiquetar y CONGELAR el test set real `[HUMANO]` (etiquetado) + `[SONNET]` (estructura y validación)
**Depende de:** T3.4 (herramienta) + primeras ~60 fotos de T3.5. **Entregable:** `training/data_test_frozen/` con 50 fotos etiquetadas (máscara multi-clase + keypoints, formato del Dataset de T3.1).
**Regla de oro: este set se CONGELA — la métrica de éxito del proyecto (G4a, G4b) se mide SIEMPRE aquí, nunca en sintético ni augmentado, y estas 50 fotos JAMÁS entran al train ni al fine-tuning.** El humano etiqueta (~2-3h con la herramienta de T3.4); Sonnet valida el formato, verifica que las 50 cubren la distribución de ángulos/pisos del protocolo, escribe `training/data_test_frozen/FROZEN.md` con el hash de cada archivo, y añade un check en train_model.py que ABORTA si detecta archivos de data_test_frozen en el data_dir de entrenamiento.

### T3.7 — Etiquetado masivo de VIDEOS con SAM2 `[OPUS]` (script) + `[HUMANO]` (grabar videos y clics)
**Añadida 2026-07-25 tras descartar MediaPipe — es la vía principal de datos reales (receta Wanna/Kivisense que pidió el usuario).**
**Depende de:** T3.1 (formato de datos). **Entregable:** `training/2b_sam2_video.py` + sección de VIDEOS en PROTOCOLO_FOTOS.md.
**Idea:** un video de 30s a 30fps = ~900 frames; con SAM2 se clickea el pie/zapato/pierna en UN frame y la máscara se propaga sola a todo el video → cientos de frames etiquetados por minuto de trabajo humano, vs ~2 min/foto con SAM1. Los keypoints se clickean solo en 1 de cada N frames (o vienen del sintético — el modelo tolera muestras sin keypoints, w_kp=0).

**Prompt:**
```
Tarea T3.7 del ROADMAP.md. Crea training/2b_sam2_video.py sobre facebookresearch/sam2 (verificar
checkpoint/API/licencia vigentes antes): (1) toma un video (mp4) + N clics del usuario en el primer
frame por clase (pierna/pie/zapato, UI cv2 como 2_sam_label.py), (2) propaga las 3 máscaras con el
video predictor de SAM2 a todos los frames, (3) exporta 1 de cada --stride frames (default 5) al
formato del Dataset de T3.1: images/<video>_fNNNN.jpg + masks/*.png (índices {0,1,2,3}) +
keypoints.jsonl (vacío para estos frames — w_kp=0), (4) --review con overlay para descartar frames
donde la propagación falló, (5) modo CPU funcional en Windows (checkpoint tiny/small) aunque lento;
documentar tiempo estimado por video. El split por imagen base de train_model.py ya agrupa por stem:
verificar que <video>_fNNNN comparte stem base por video (ajustar _base_stem si hace falta) para que
frames del mismo video NUNCA queden repartidos entre train y val (fuga temporal).
```

### **GATE G3 — Smoke test end-to-end `[OPUS]`**
**Depende de:** T3.1, T3.2, T3.3. **Criterio: TODO verde o no se avanza a Fase 4.**

**Prompt:**
```
GATE G3 del ROADMAP.md. Ejecuta y reporta el pipeline completo con datos mínimos:
1. blender --background --python training/0b_blender_render.py -- --count 50 --out training/data_smoke --seed 42
2. Inspecciona 10 tríos al azar (Read de las imágenes): máscara alineada con la imagen, índices {0,1,2,3}
   correctos, keypoints sobre el pie, pie y zapato alineados entre sí, pisos/luces DISTINTOS entre renders.
3. python training/train_model.py --data_dir training/data_smoke --epochs 2
4. python training/export_onnx.py --checkpoint <el generado> --int8
5. Abre test_inference.html con el ONNX y una imagen de data_smoke: la tubería corre y dibuja algo
   (la calidad no importa con 50 imágenes y 2 epochs).
6. Mide y anota segundos/render del paso 1: se usará en T4.3 para estimar el costo del batch de 5000.
Reporta CADA paso con evidencia (logs, capturas). Si algo falla, arréglalo y repite. El gate solo se
aprueba con los pasos verdes. Al aprobar: mergea feature/training-v2 a main.
```

---

# FASE 4 — Dataset sintético + entrenamiento (2-3 semanas) — **GATES G4a/G4b**

> **Bloqueo duro de entrada:** GATE G2 aprobado (demanda validada) + GATE G3 verde + GLBs reales del experto Blender recibidos e integrados (pie/pierna con textura de piel realista, malla cerrada, colecciones leg_*, Empties kp_* nombrados; ver brief en memoria del proyecto). **Sin GLBs realistas no se renderiza: la pierna-cono actual garantiza domain gap.** También requerido: ≥10 modelos de calzado variados (del experto o assets CC0/licenciados con procedencia documentada), **cada uno ya ALINEADO A MANO al pie canónico en Blender** (regla dura #1: los scripts no reescalan ni reposicionan — la alineación es entregable del experto). `[HUMANO]` gestiona ambas entregas.

### T4.1 — Integración de GLBs definitivos + preview `[SONNET]`
**Depende de:** entrega del experto. **Entregable:** GLBs en `training/models/` (+ `training/models/shoes/` para los ≥10 de calzado), preview aprobado por el humano.

**Prompt:**
```
Tarea T4.1 del ROADMAP.md. Con los GLBs definitivos del experto ya copiados a training/models/:
1. Verifica el brief: escala real (1 unidad = 1m, pie ~0.26m — mide el bbox por script SIN modificarlo),
   colecciones leg_bare/leg_pants_dark/leg_pants_light presentes, Empties kp_heel/kp_toe/kp_ankle_in/
   kp_ankle_out/kp_ball/kp_toe_tip presentes, y que cada zapato de training/models/shoes/ carga alineado
   al pie sin ajuste alguno. CUALQUIER incumplimiento: repórtalo como lista de correcciones para el
   experto y detente — no lo "arregles" en el script (regla dura #1).
2. Ejecuta 0b_blender_render.py --preview con los assets nuevos (varios zapatos), genera el contact-sheet
   y preséntalo al humano para aprobación EXPLÍCITA. Sin su aprobación no se pasa a T4.2.
3. Documenta en training/models/ASSETS.md: procedencia y licencia de cada GLB.
```

### T4.2 — Piloto: 500 renders + entrenamiento corto + medición REAL `[OPUS]`
**Depende de:** T4.1 + T3.6 (test set congelado). **Entregable:** informe con métricas.
**GATE G4a: IoU (unión pie+zapato) > 0.6-0.7 y PCK@0.1 > 0.7 sobre las 50 fotos reales del test set congelado → aprobar batch grande. Por debajo → diagnosticar domain gap (comparar visualmente predicciones, ajustar degradación/HDRIs/texturas en T3.2) e iterar el piloto. NO escalar a 5000 sin pasar este gate.**

**Prompt:**
```
Tarea T4.2 + GATE G4a del ROADMAP.md.
1. Genera 500 renders (--count 500 --seed fijo --shoe_dir training/models/shoes) con los GLBs definitivos.
2. Entrena 15 epochs (Colab/GPU local según check_gpu.py; documenta tiempo y costo).
3. Evalúa SOLO sobre training/data_test_frozen/ (50 fotos, T3.6): mIoU por clase, IoU pie+zapato, PCK@0.1.
4. Genera un contact-sheet HTML de las 50 predicciones (imagen + máscara pred + keypoints pred) y
   revísalo tú mismo: describe los modos de fallo dominantes (¿frontera pierna/zapato? ¿pisos oscuros?
   ¿ángulo frontal?).
5. Veredicto G4a con números. Si falla, propone el ajuste concreto de mayor impacto en T3.2 y NO
   recomiendes escalar.
```

### T4.3 — Batch completo + entrenamiento definitivo `[OPUS]` (con `[SONNET]` para el babysitting)
**Depende de:** GATE G4a. **Entregable:** `models/foot_net_v1_int8.onnx` (<5MB) + `models/MODEL_CARD.md` + informe.
**GATE G4b (criterio de éxito del modelo):** sobre `data_test_frozen`: IoU pie+zapato > 0.85, PCK@0.1 > 0.85, latencia < 80ms en móvil de gama media con WebGPU (medida con test_inference.html).

**Prompt:**
```
Tarea T4.3 + GATE G4b del ROADMAP.md.
1. ANTES de lanzar nada: estima el costo del batch — usa los seg/render medidos en G3/T4.2 y extrapola
   a 5000 (¿horas locales? ¿conviene una instancia GPU puntual?). Presenta la estimación al humano y
   espera su OK explícito (es el gasto grande del proyecto).
2. Batch: 5000 renders con --seed nuevo y manifest.json. Inspecciona 30 al azar antes de entrenar.
3. Entrenamiento: pre-training completo en sintético (30-50 epochs, early stopping por mIoU en un split
   sintético de validación) → FINE-TUNING con las 300-500 fotos reales etiquetadas (T3.5/T3.4, excluyendo
   SIEMPRE data_test_frozen) con lr/10, 10-15 epochs. Documenta ambas curvas.
4. Export: export_onnx.py --int8 → models/foot_net_v1_int8.onnx (+ .meta.json). Versionado: el nombre
   lleva versión (v1, v2…); NUNCA sobrescribas una versión publicada.
5. models/MODEL_CARD.md: dataset (tamaños, seeds, manifest), commit de origen, métricas completas sobre
   data_test_frozen, latencias medidas (WebGPU y WASM, dispositivo concreto), limitaciones conocidas.
6. Veredicto G4b con números. Si falla por poco (<10% relativo), UNA iteración de ajuste dirigida por
   los modos de fallo del contact-sheet antes de re-evaluar; si falla por mucho, vuelve a T4.2 con
   diagnóstico — no fuerces el gate.
```

---

# FASE 5 — Runtime 6DoF en la app (1-2 semanas)

### T5.1 — Integrar el modelo propio en la app `[OPUS]`
**Depende de:** G4b. **Entregable:** `pose.js` v3 usando `inference.js` (modelo propio) en lugar de MediaPipe.
**Criterio de aceptación:** mismo contrato de API hacia app.js; a ≥10Hz de detección en móvil medio; MediaPipe queda como fallback por flag (`?tracker=mediapipe`).

**Prompt:**
```
Tarea T5.1 del ROADMAP.md. Rama feature/6dof desde main.
1. pose.js v3: initPose() carga inference.js con models/foot_net_v1_int8.onnx (lee el .meta.json para
   normalización/clases — nada hardcodeado). detectPose(videoEl) devuelve {seg, kps, side} del modelo.
   extractFootLandmarks adapta kps al contrato actual (heel/toe/ankle) para que renderer.js siga
   funcionando SIN cambios en esta tarea (el 6DoF llega en T5.2).
2. Arquitectura de fallback: un módulo tracker.js elige implementación por query param
   (?tracker=onnx | mediapipe; default onnx) — MediaPipe se conserva como plan B y para comparar.
3. El center-crop del preproceso debe remapear coordenadas al frame completo (misma regla que T0.3.4).
4. Mide en ?debug=1: latencia de inferencia, EP activo (webgpu/wasm), fps de detección. Verifica ≥10Hz
   en un móvil medio; si WASM queda por debajo, baja la frecuencia de detección y súbele el peso a la
   interpolación OneEuro entre inferencias.
```

### T5.2 — Pose 6DoF: resucitar solver.js (PnP) + cámara en perspectiva `[OPUS]`
**Depende de:** T5.1. **Entregable:** zapato con rotación 3D real y escala métrica.
**Criterio de aceptación:** al girar el pie (yaw/pitch/roll) el zapato lo sigue en los 3 ejes; la escala responde a la distancia (adiós clamp 20-38% y constante 0.262); OneEuro reemplazado por filtrado de pose (posición + slerp de cuaterniones).

**Prompt:**
```
Tarea T5.2 del ROADMAP.md — el salto de calidad visible.
1. PnP real: OJO, solver.js NO trae un PnP terminado (su solvePose dependía del depth de MiDaS, ya
   eliminado). Escribe un PnP 2D→3D de verdad — DLT/EPnP con refinamiento iterativo (Gauss-Newton sobre
   error de reproyección) — reutilizando de solver.js la svd3x3() y la idea de FOOT_MODEL_3D. El modelo
   canónico: los MISMOS 6 keypoints del entrenamiento con coordenadas 3D extraídas UNA vez de los Empties
   kp_* del GLB canónico en Blender (~0.26m de largo), hardcodeadas y documentadas.
2. renderer.js: sustituye la OrthographicCamera por PerspectiveCamera con FOV vertical estimado del
   móvil (~60°; deja flag de calibración fina). El plano de video pasa a fullscreen-quad correcto
   (escala por FOV o THREE.VideoTexture sobre un plano a distancia fija que llene el frustum).
3. Por frame: keypoints 2D (conf>0.5, mínimo 4) -> solvePnP -> R|t. Aplica R|t al GLB directamente
   (posición métrica, rotación completa). Elimina el clamp de escala y la constante 0.262.
4. Filtrado: posición con OneEuro 3D; rotación con slerp hacia la nueva pose (factor adaptativo por
   velocidad, estilo OneEuro). Rechaza poses con error de reproyección alto (umbral, usa la última buena).
5. Oclusión (por fin): usa la clase 1 (pierna/pantalón) de la máscara como occluder — mesh plano en
   el frustum con depthWrite=true + colorWrite=false a la profundidad del tobillo estimada por t, o
   composición 2D del video sobre el render donde la máscara sea pierna. Elige el enfoque más simple
   que funcione y documenta el trade-off.
6. QA con ?debug=1: dibuja keypoints, ejes de pose y error de reproyección.
```

### T5.3 — Dos pies + pulido `[OPUS]`
**Depende de:** T5.2. **Entregable:** try-on de ambos pies simultáneos, selector de lado eliminado de la UI.

**Prompt:**
```
Tarea T5.3 del ROADMAP.md.
1. Dos instancias del GLB del zapato (la izquierda espejada con scale.x*-1 en un Group contenedor —
   espejar el GROUP no viola la regla #1, que aplica al asset; cuidado con las normales/culling).
2. El modelo detecta una instancia por inferencia: corre la inferencia sobre crops izquierdo/derecho
   del frame o extiende el decode a 2 instancias (elige según lo que el modelo de T4.3 permita;
   documenta el trade-off). Cada pie lleva su propio filtro de pose.
3. Elimina el botón de cambio de pie de la UI. Si solo hay un pie visible, muestra solo ese.
4. Merge de feature/6dof a main tras QA en móviles reales (misma checklist de T1.3).
```

---

# FASE 6 — Productización del plugin (2 semanas, tras tracción)

### T6.1 — Widget embebible `[OPUS]`
**Depende de:** merge de Fase 5 (o de Fase 1 si se productiza la versión MediaPipe tras G2 fuerte). **Entregable:** `dist/ar-shoes-widget.js` autocontenido.

**Prompt:**
```
Tarea T6.1 del ROADMAP.md. Empaqueta el try-on como widget embebible:
1. dist/ar-shoes-widget.js: un solo <script type="module"> que el cliente pega con
   data-model="URL_DEL_GLB" (y data-key para futura licencia). Inyecta un botón "Probar en AR" que abre
   el try-on en un overlay fullscreen (shadow DOM para aislar CSS). Sin bundler obligatorio para el
   CLIENTE; si internamente conviene un build (esbuild) para inlinear módulos, el output debe ser un
   único archivo — la regla "sin bundler" aplica a la app de desarrollo, no prohíbe empaquetar dist/.
2. Assets pesados (ONNX, wasm, GLB de demo) servidos por CDN (jsdelivr sobre el repo público — recuerda:
   nada de LFS en esos archivos, límite jsdelivr 20MB/archivo).
3. Página de prueba examples/tienda-fake.html simulando una ficha de producto real.
4. Requisitos que DEBE cumplir: funciona dentro de iframe con allow="camera", HTTPS obligatorio,
   no rompe el CSS del sitio anfitrión, peso inicial del script <50KB (lazy-load del resto al abrir).
```

### T6.2 — Integración e-commerce `[SONNET]`
**Depende de:** T6.1. **Entregable:** guías de integración; la app oficial de Shopify se pospone hasta ≥3 clientes pagando.

**Prompt:**
```
Tarea T6.2 del ROADMAP.md. Crea docs/integracion/:
1. SHOPIFY.md: integración manual vía theme.liquid / custom liquid block con el snippet del widget,
   capturas paso a paso, y cómo mapear data-model por producto (metafields).
2. WOOCOMMERCE.md: hook/shortcode equivalente.
3. Ambas guías probadas contra examples/tienda-fake.html como referencia de resultado esperado.
```

### T6.3 — Privacidad y captura de datos v2 `[OPUS]` (solo si se reactiva la captura)
**Reglas si se reactiva `5_capture_dataset.js`:** consentimiento opt-in explícito ANTES de guardar cualquier frame, política de retención visible, capturar casos de BAJA confianza (no solo detecciones buenas — lo contrario es sesgo de confirmación), etiquetar offline con SAM2 como maestro (los landmarks del propio tracker NO sirven como labels), minimización (crops, no frames completos). Hasta cumplir todo esto, la captura permanece desactivada.

---

## Resumen de asignación

| Tarea | Modelo | Por qué |
|---|---|---|
| T0.1, T0.2, T0.3, T1.2, T1.3, T1.4, T2.1, T2.2, T3.3, T3.4, T3.5 (doc), T3.6 (validación), T4.1, T6.2 | **Sonnet** | Mecánicas, bien especificadas, criterio de aceptación verificable |
| T1.1, T3.1, T3.2, G3, T4.2, T4.3, T5.1, T5.2, T5.3, T6.1, T6.3 | **Opus** | Arquitectura, CV/ML, integración con decisiones de diseño, diagnóstico de fallos |
| T0.4, fotos (T3.5), etiquetado (T3.6), outreach (T2.2), GLBs del experto, aprobación de gates | **Humano** | Legal, físico, comercial, criterio de negocio |

**Política de ramas:** `rescue/jun28-work` (F0) → `feature/mediapipe-pose` (F1, merge a main tras QA de T1.3) → `feature/training-v2` (F3, merge a main al aprobar G3) → `feature/6dof` (F5, merge a main tras QA de T5.3). Pages sirve siempre desde main una vez hecho el primer merge.

## Los 4 gates (memorizar)

1. **G2 — Demanda:** ≥3 tiendas interesadas tras 2 semanas de outreach con la demo MediaPipe. Sin esto, no hay gasto en dataset/GPU.
2. **G3 — Smoke test:** pipeline completo verde con 50 renders (render → train → ONNX → browser).
3. **G4a — Piloto:** IoU real > 0.6-0.7 y PCK > 0.7 con 500 renders, medido en las 50 fotos reales congeladas. Sin esto, no hay batch de 5000.
4. **G4b — Modelo final:** IoU > 0.85, PCK > 0.85, < 80ms móvil medio (WebGPU). Sin esto, no se reemplaza MediaPipe en producción.

## Referencias técnicas (para los agentes)

- ARShoe (receta multi-branch keypoints+seg): arxiv.org/abs/2108.10515 · Paper Springer 2025: link.springer.com/article/10.1007/s40747-025-02188-x
- MediaPipe Pose Landmarker (índices 27-32 = tobillos/talones/puntas): developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- onnxruntime-web WebGPU: onnxruntime.ai/docs/tutorials/web/ (WebGPU disponible en todos los navegadores móviles desde Safari 26, sept 2025)
- Benchmark comercial: docs.deepar.ai/use-cases/shoe-try-on (desde $25/mes) · Snap Foot Tracking: developers.snap.com/lens-studio/features/try-on/foot-tracking
- Open source de referencia: github.com/WebAR-rocks/WebAR.rocks.hand (MIT, demo Shoes VTO)
- HDRIs/texturas CC0: polyhaven.com
- Historia del propio repo (leer los diffs antes de repetir caminos): la implementación ORIGINAL fue MediaPipe Pose Landmarker (`4f56d70` inicial → `5fa15b4` cambio a modelo full + guía de encuadre → `0d809a7` quita umbrales de visibilidad) y se abandonó por mala detección en encuadres de solo-pies; después BodyPix (`1c3f30c`), MediaPipe selfie segmentation (`f7c5a2f` — modelo equivocado para pies), y finalmente background subtraction (`0c3b580`). El rechazo de cámara movida vivió en `104b305` (en extractFootLandmarks) y fue eliminado a propósito en `b0dd106` porque pierna+pantalón superan el 30% legítimamente. Lección: los fracasos fueron de encuadre/modelo elegido, no una prueba de que "ML no sirve".
