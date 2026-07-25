// Detección de pies vía MediaPipe Pose Landmarker (reemplaza la sustracción de fondo).
// Landmarks del pie, normalizados al frame completo [0,1]:
//   27/28 tobillos, 29/30 talones, 31/32 puntas (foot_index) — izquierdo/derecho.
// Ventajas sobre background subtraction: puntos anatómicos REALES, sin calibrar el piso,
// robusto a movimiento de cámara y a cambios de iluminación.
import { PoseLandmarker, FilesetResolver }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';

const WASM_URL  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';

const MIN_VIS = 0.3; // visibilidad media mínima para dar un pie por válido (bajo: encuadres de pies)

let landmarker    = null;
let lastVideoTime = -1;
let lastSeg       = null;

async function initPose() {
  const vision   = await FilesetResolver.forVisionTasks(WASM_URL);
  const forceCpu = new URLSearchParams(location.search).get('cpu') === '1';
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numPoses: 1, // los dos pies del mismo cuerpo llegan en una sola pose
    // Umbrales bajos: MediaPipe Pose está entrenado para cuerpo entero; con encuadres de
    // piernas+pies hay que ser permisivo para que acepte una detección parcial.
    minPoseDetectionConfidence: 0.25,
    minPosePresenceConfidence: 0.25,
    minTrackingConfidence: 0.25,
  });

  // GPU por defecto; en iOS puede conflictuar con el WebGL de Three.js → fallback a CPU
  // (o forzar CPU con ?cpu=1). Ver historial: "fix iOS WebGL conflict: use CPU delegate".
  if (!forceCpu) {
    try {
      landmarker = await PoseLandmarker.createFromOptions(vision, opts('GPU'));
      console.log('[pose] MediaPipe Pose Landmarker listo (GPU)');
      return;
    } catch (e) {
      console.warn('[pose] Delegate GPU falló, usando CPU:', e.message);
    }
  }
  landmarker = await PoseLandmarker.createFromOptions(vision, opts('CPU'));
  console.log('[pose] MediaPipe Pose Landmarker listo (CPU)');
}

// Construye la estructura de un pie a partir de los 33 landmarks de la pose
function buildFoot(lm, ankleI, heelI, toeI) {
  const a = lm[ankleI], h = lm[heelI], t = lm[toeI];
  if (!a || !h || !t) return null;
  const vis = ((a.visibility ?? 0) + (h.visibility ?? 0) + (t.visibility ?? 0)) / 3;
  if (vis < MIN_VIS) return null;
  return {
    ankle: { x: a.x, y: a.y, visibility: a.visibility ?? 1 },
    heel:  { x: h.x, y: h.y, visibility: h.visibility ?? 1 },
    toe:   { x: t.x, y: t.y, visibility: t.visibility ?? 1 },
    vis,
  };
}

// Devuelve { left, right } (cada pie puede ser null), o null si no hay pose útil.
// Contrato hacia app.js: síncrono, un objeto o null.
function detectPose(videoEl) {
  if (!landmarker || !videoEl.videoWidth) return null;
  // No reprocesar el mismo frame de vídeo (MediaPipe requiere timestamps crecientes)
  if (videoEl.currentTime === lastVideoTime) return lastSeg;
  lastVideoTime = videoEl.currentTime;

  let result;
  try {
    result = landmarker.detectForVideo(videoEl, performance.now());
  } catch (e) {
    return null;
  }

  if (!result || !result.landmarks || result.landmarks.length === 0) {
    lastSeg = null;
    return null;
  }

  const lm    = result.landmarks[0];
  const left  = buildFoot(lm, 27, 29, 31);
  const right = buildFoot(lm, 28, 30, 32);
  lastSeg = (left || right) ? { left, right } : null;
  return lastSeg;
}

// El pie más visible (con override manual, app.js ignora esto)
function detectDominantFoot(seg) {
  if (!seg) return 'right';
  if (seg.left && !seg.right) return 'left';
  if (seg.right && !seg.left) return 'right';
  return (seg.left?.vis ?? 0) > (seg.right?.vis ?? 0) ? 'left' : 'right';
}

// Landmarks del pie pedido en el contrato que espera renderer.js
// { heel, toe, ankle (normalizados), bboxW, bboxH, side }
function extractFootLandmarks(seg, side = 'right') {
  if (!seg) return null;
  const f = seg[side] || seg.left || seg.right;
  if (!f) return null;

  const xs = [f.heel.x, f.toe.x, f.ankle.x];
  const ys = [f.heel.y, f.toe.y, f.ankle.y];
  const footLen = Math.hypot(f.toe.x - f.heel.x, f.toe.y - f.heel.y);
  // bbox del triángulo talón-punta-tobillo expandido ~40%; nunca menor que el largo del pie
  // (evita que la escala colapse cuando el pie apunta hacia la cámara)
  const bboxW = Math.max((Math.max(...xs) - Math.min(...xs)) * 1.4, footLen);
  const bboxH = Math.max((Math.max(...ys) - Math.min(...ys)) * 1.4, footLen);

  return {
    heel:  { x: f.heel.x,  y: f.heel.y,  visibility: 1 },
    toe:   { x: f.toe.x,   y: f.toe.y,   visibility: 1 },
    ankle: { x: f.ankle.x, y: f.ankle.y, visibility: 1 },
    bboxW, bboxH, side,
  };
}

export { initPose, detectPose, extractFootLandmarks, detectDominantFoot };
