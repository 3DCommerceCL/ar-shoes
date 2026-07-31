// pose.js — detección de pie con NUESTRO modelo (FootNet ONNX), reemplaza a MediaPipe.
//
// El modelo devuelve dos cosas por frame:
//   · seg: máscara 4 clases {fondo, pierna, pie, zapato}  ← entrenada con video real (SAM2)
//   · heatmaps: 12 keypoints (6 por pie: heel, toe, ankle_in, ankle_out, ball, toe_tip)
//
// ESTRATEGIA DE DOS VÍAS (a propósito): los keypoints hoy sólo se entrenan con renders
// sintéticos, así que pueden no transferir a fotos reales. La máscara sí tiene datos reales.
// Por eso: si los keypoints vienen con confianza suficiente se usan (dan orientación anatómica
// real, sin la ambigüedad de 180° del PCA); si no, se derivan los landmarks de la máscara del
// zapato con centroide+PCA. Así el visor funciona igual mientras la cabeza de keypoints madura.
import { initInference, runInference, getActiveEP, getMeta } from './inference.js';

const MODEL_URL = './models/foot_net_v2.onnx';   // fp32: en WASM es ~6x más rápido que int8 (medido)
const KP_MIN_CONF = 0.35;      // confianza mínima del heatmap para fiarse del keypoint
const MIN_SHOE_PX = 120;       // píxeles mínimos de zapato/pie para dar el frame por válido
const INPUT = 256;

let ready = false;
let lastMode = 'none';

export function getTrackerMode() { return lastMode; }
export { getActiveEP };

export async function initPose() {
  const url = new URLSearchParams(location.search).get('model') || MODEL_URL;
  const ep = await initInference(url, url.replace(/\.onnx$/, '.meta.json'));
  ready = true;
  console.log('[pose] FootNet listo:', url, '| EP:', ep);
}

// Devuelve el resultado crudo del modelo (o null). app.js lo pasa a extractFootLandmarks.
export async function detectPose(videoEl) {
  if (!ready || !videoEl.videoWidth) return null;
  let r;
  try {
    r = await runInference(videoEl);
  } catch (e) {
    console.warn('[pose] inferencia falló:', e.message);
    return null;
  }
  // ¿hay pie/zapato suficiente en la máscara?
  let n = 0;
  for (let i = 0; i < r.seg.length; i++) if (r.seg[i] >= 2) n++;
  r.shoePixels = n;
  return n >= MIN_SHOE_PX ? r : null;
}

// Píxeles de zapato+pie del lado pedido, en coordenadas del recorte (0..INPUT)
function maskPixels(seg, side) {
  const pts = [];
  for (let y = 0; y < INPUT; y++) {
    for (let x = 0; x < INPUT; x++) {
      if (seg[y * INPUT + x] >= 2) pts.push([x, y]);
    }
  }
  if (pts.length < MIN_SHOE_PX) return pts;
  // Con dos pies visibles, quedarse con la mitad correspondiente al lado pedido
  const xs = pts.map(p => p[0]);
  const midX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const half = side === 'left' ? pts.filter(p => p[0] < midX) : pts.filter(p => p[0] >= midX);
  return half.length >= MIN_SHOE_PX * 0.4 ? half : pts;
}

// Landmarks desde la MÁSCARA (respaldo): centroide + eje principal por PCA + bbox
function landmarksFromMask(seg, side, crop) {
  const pts = maskPixels(seg, side);
  if (pts.length < MIN_SHOE_PX * 0.4) return null;

  let sx = 0, sy = 0;
  for (const [x, y] of pts) { sx += x; sy += y; }
  const cx = sx / pts.length, cy = sy / pts.length;

  let cxx = 0, cxy = 0, cyy = 0, minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  for (const [x, y] of pts) {
    const dx = x - cx, dy = y - cy;
    cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  let angle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);

  // DESAMBIGUAR TALÓN/PUNTA con la máscara de PIERNA. El PCA da un eje sin sentido: el zapato
  // podía aparecer dado vuelta 180° entre frames. Pero la pierna sale del TOBILLO, así que el
  // extremo del pie más cercano a la pierna es el talón y el otro la punta. Es información que
  // el modelo ya predice (IoU 0.97) y que la sustracción de fondo nunca tuvo.
  let lx = 0, ly = 0, ln = 0;
  for (let y = 0; y < INPUT; y++) {
    for (let x = 0; x < INPUT; x++) {
      if (seg[y * INPUT + x] === 1) { lx += x; ly += y; ln++; }
    }
  }
  if (ln > 40) {
    const legCx = lx / ln, legCy = ly / ln;
    // si el eje +PCA apunta HACIA la pierna, invertirlo (queremos que apunte a la punta)
    if (Math.cos(angle) * (legCx - cx) + Math.sin(angle) * (legCy - cy) > 0) {
      angle += Math.PI;
    }
  }
  const halfLen = Math.max(maxX - minX, maxY - minY) * 0.45;
  const toFrame = (px, py) => ({
    x: crop ? (crop.sx + (px / INPUT) * crop.s) / crop.w : px / INPUT,
    y: crop ? (crop.sy + (py / INPUT) * crop.s) / crop.h : py / INPUT,
    visibility: 1,
  });
  const fx = crop ? crop.s / crop.w : 1;
  const fy = crop ? crop.s / crop.h : 1;
  return {
    heel:  toFrame(cx - Math.cos(angle) * halfLen, cy - Math.sin(angle) * halfLen),
    toe:   toFrame(cx + Math.cos(angle) * halfLen, cy + Math.sin(angle) * halfLen),
    ankle: toFrame(cx, cy),
    bboxW: ((maxX - minX) / INPUT) * fx,
    bboxH: ((maxY - minY) / INPUT) * fy,
    side, source: 'mask',
  };
}

// Landmarks desde los KEYPOINTS del modelo (preferido: anatómicos, sin ambigüedad de 180°)
function landmarksFromKeypoints(byFoot, side) {
  const block = byFoot?.[side];
  if (!block) return null;
  const [heel, toe, ankleIn, ankleOut, , toeTip] = block;
  const conf = (heel[2] + toeTip[2] + ankleIn[2] + ankleOut[2]) / 4;
  if (conf < KP_MIN_CONF) return null;

  const ankle = { x: (ankleIn[0] + ankleOut[0]) / 2, y: (ankleIn[1] + ankleOut[1]) / 2, visibility: 1 };
  const len = Math.hypot(toeTip[0] - heel[0], toeTip[1] - heel[1]);
  return {
    heel:  { x: heel[0], y: heel[1], visibility: 1 },
    toe:   { x: toeTip[0], y: toeTip[1], visibility: 1 },
    ankle,
    // bbox aproximado desde el largo del pie (renderer.js lo usa para la escala)
    bboxW: len, bboxH: len,
    side, source: 'kp', conf,
  };
}

export function extractFootLandmarks(seg, side = 'right') {
  if (!seg) return null;
  const kp = landmarksFromKeypoints(seg.byFoot, side);
  if (kp) { lastMode = 'keypoints'; return kp; }
  const mk = landmarksFromMask(seg.seg, side, seg.crop);
  lastMode = mk ? 'mascara' : 'none';
  return mk;
}

export function detectDominantFoot(seg) {
  if (!seg) return 'right';
  // 1) por confianza de keypoints, si la hay
  if (seg.byFoot) {
    const c = (b) => b.reduce((a, k) => a + k[2], 0) / b.length;
    const l = c(seg.byFoot.left), r = c(seg.byFoot.right);
    if (Math.max(l, r) >= KP_MIN_CONF) return l > r ? 'left' : 'right';
  }
  // 2) por masa de la máscara a cada lado
  let L = 0, R = 0;
  for (let y = 0; y < INPUT; y++) {
    for (let x = 0; x < INPUT; x++) {
      if (seg.seg[y * INPUT + x] >= 2) (x < INPUT / 2 ? L++ : R++);
    }
  }
  return L > R ? 'left' : 'right';
}

// La máscara de pierna/pantalón sirve de OCLUSOR (que el pantalón tape la caña del zapato).
// Se expone para que renderer.js la use cuando se implemente la oclusión (Fase 5).
export function getLegMask(seg) {
  if (!seg) return null;
  const m = new Uint8Array(INPUT * INPUT);
  for (let i = 0; i < m.length; i++) m[i] = seg.seg[i] === 1 ? 1 : 0;
  return { data: m, width: INPUT, height: INPUT, crop: seg.crop };
}
