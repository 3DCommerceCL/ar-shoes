// Detección de pies via segmentación MediaPipe
// Estrategia: segmentar persona, encontrar píxeles inferiores = pies

let segmenter = null;
let segCanvas = null;
let segCtx    = null;

async function initPose(wasmPath = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm') {
  const { ImageSegmenter, FilesetResolver } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js'
  );
  const vision = await FilesetResolver.forVisionTasks(wasmPath);

  segmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
      delegate: 'CPU',
    },
    runningMode:           'VIDEO',
    outputCategoryMask:    false,
    outputConfidenceMasks: true,
  });

  segCanvas     = document.createElement('canvas');
  segCtx        = segCanvas.getContext('2d');
  console.log('[pose] Segmentador listo');
}

let _processing = false;
let _lastTs = 0;

// Retorna objeto segmentation con pixeles de persona, o null
async function detectPose(videoEl) {
  if (!segmenter || !videoEl.videoWidth || _processing) return null;
  _processing = true;
  try {
    const ts = performance.now();
    if (ts <= _lastTs) { _processing = false; return null; }
    _lastTs = ts;

    const result = segmenter.segmentForVideo(videoEl, ts);
    if (!result?.confidenceMasks?.[0]) { _processing = false; return null; }

    const mask = result.confidenceMasks[0];
    const data = mask.getAsFloat32Array();
    const W    = mask.width;
    const H    = mask.height;
    mask.close();

    _processing = false;
    return { data, width: W, height: H };
  } catch (e) {
    console.warn('[pose] segmentForVideo error:', e.message);
    _processing = false;
    return null;
  }
}

// Detecta qué pie tiene más píxeles en su mitad de pantalla
function detectDominantFoot(seg) {
  if (!seg) return 'right';
  const { data, width, height } = seg;
  const midX = width / 2;
  const yMin = Math.floor(height * 0.5);
  let L = 0, R = 0;

  for (let y = yMin; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[y * width + x] > 0.5) {
        x < midX ? L++ : R++;
      }
    }
  }
  return L > R ? 'left' : 'right';
}

// Extrae heel/toe/ankle desde los píxeles inferiores del segmento de persona
function extractFootLandmarks(seg, side = 'right') {
  if (!seg) return null;
  const { data, width, height } = seg;

  // Diagnóstico: calcular max confianza y cantidad de pixels por umbral
  let maxConf = 0, px15 = 0, px30 = 0, px50 = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v > maxConf) maxConf = v;
    if (v > 0.15) px15++;
    if (v > 0.30) px30++;
    if (v > 0.50) px50++;
  }
  // Mostrar en status para diagnóstico
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `max:${maxConf.toFixed(2)} px15:${px15} px30:${px30} px50:${px50}`;

  // Threshold muy bajo — aceptar cualquier señal del modelo
  const person = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[y * width + x] > 0.15) person.push([x, y]);
    }
  }
  if (person.length < 100) return null;

  // Tomar el 35% inferior del cuerpo segmentado (zona de pies)
  const ys   = person.map(([, y]) => y);
  const maxY = Math.max(...ys);
  const minY = Math.min(...ys);
  const yThresh = minY + (maxY - minY) * 0.65;

  const footArea = person.filter(([, y]) => y >= yThresh);
  if (footArea.length < 30) return null;

  // Filtrar por lado (izquierdo/derecho)
  const xs    = footArea.map(([x]) => x);
  const midX  = (Math.min(...xs) + Math.max(...xs)) / 2;
  const pixels = side === 'left'
    ? footArea.filter(([x]) => x < midX)
    : footArea.filter(([x]) => x >= midX);

  const src = pixels.length >= 20 ? pixels : footArea;

  return landmarksFromPixels(src, width, height, side);
}

// Calcula heel/toe/ankle usando PCA sobre los píxeles del pie
function landmarksFromPixels(pixels, width, height, side) {
  let sumX = 0, sumY = 0;
  for (const [x, y] of pixels) { sumX += x; sumY += y; }
  const cx = sumX / pixels.length;
  const cy = sumY / pixels.length;

  let cxx = 0, cxy = 0, cyy = 0;
  for (const [x, y] of pixels) {
    const dx = x - cx, dy = y - cy;
    cxx += dx * dx;
    cxy += dx * dy;
    cyy += dy * dy;
  }

  const angle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  const cos = Math.cos(angle), sin = Math.sin(angle);

  let minP = Infinity, maxP = -Infinity;
  let heelPt = [cx, cy], toePt = [cx, cy];
  for (const [x, y] of pixels) {
    const p = (x - cx) * cos + (y - cy) * sin;
    if (p < minP) { minP = p; heelPt = [x, y]; }
    if (p > maxP) { maxP = p; toePt  = [x, y]; }
  }

  return {
    heel:  { x: heelPt[0] / width, y: heelPt[1] / height, visibility: 1 },
    toe:   { x: toePt[0]  / width, y: toePt[1]  / height, visibility: 1 },
    ankle: { x: cx / width,        y: cy / height,         visibility: 1 },
    side,
  };
}

export { initPose, detectPose, extractFootLandmarks, detectDominantFoot };
