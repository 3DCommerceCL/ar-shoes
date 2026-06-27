// MediaPipe Pose Landmarker — detección de landmarks del pie
// Landmarks usados:
//   27 = left_heel   28 = right_heel
//   29 = left_foot_index  30 = right_foot_index
//   31 = left_toe_big     32 = right_toe_big (no en todos los modelos)

const FOOT_LANDMARKS = {
  left:  { heel: 29, toe: 31, ankle: 27 },
  right: { heel: 30, toe: 32, ankle: 28 },
};

// Índices MediaPipe Pose 33-landmark
const LEFT_HEEL        = 29;
const RIGHT_HEEL       = 30;
const LEFT_FOOT_INDEX  = 31;
const RIGHT_FOOT_INDEX = 32;
const LEFT_ANKLE       = 27;
const RIGHT_ANKLE      = 28;
const LEFT_KNEE        = 25;
const RIGHT_KNEE       = 26;

let poseLandmarker = null;
let lastResult     = null;

async function initPose(wasmPath = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm') {
  const { PoseLandmarker, FilesetResolver } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js'
  );

  const vision = await FilesetResolver.forVisionTasks(wasmPath);

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode:        'VIDEO',
    numPoses:           1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence:  0.5,
    minTrackingConfidence:      0.5,
  });

  console.log('[pose] MediaPipe PoseLandmarker listo');
}

// Retorna landmarks crudos del pie para el lado indicado ('left' | 'right')
// Coordenadas normalizadas [0,1] relativas al frame
function detectPose(videoEl, timestampMs) {
  if (!poseLandmarker) return null;

  const result = poseLandmarker.detectForVideo(videoEl, timestampMs);
  lastResult = result;

  if (!result.landmarks || result.landmarks.length === 0) return null;
  return result.landmarks[0]; // 33 landmarks del primer cuerpo detectado
}

// Extrae los 6 landmarks clave del pie (heel, toe, ankle de cada pie)
function extractFootLandmarks(allLandmarks, side = 'left') {
  if (!allLandmarks) return null;

  const idx = side === 'left'
    ? [LEFT_HEEL, LEFT_FOOT_INDEX, LEFT_ANKLE, LEFT_KNEE]
    : [RIGHT_HEEL, RIGHT_FOOT_INDEX, RIGHT_ANKLE, RIGHT_KNEE];

  const lms = idx.map(i => allLandmarks[i]);

  // Verificar visibilidad mínima
  const minVis = Math.min(...lms.map(l => l.visibility ?? 0));
  if (minVis < 0.4) return null;

  return {
    heel:   allLandmarks[idx[0]],
    toe:    allLandmarks[idx[1]],
    ankle:  allLandmarks[idx[2]],
    knee:   allLandmarks[idx[3]],
    side,
  };
}

// Detecta qué pie está más visible para hacer try-on automático
function detectDominantFoot(allLandmarks) {
  if (!allLandmarks) return 'right';

  const leftVis  = (allLandmarks[LEFT_HEEL]?.visibility  ?? 0) +
                   (allLandmarks[LEFT_FOOT_INDEX]?.visibility ?? 0);
  const rightVis = (allLandmarks[RIGHT_HEEL]?.visibility ?? 0) +
                   (allLandmarks[RIGHT_FOOT_INDEX]?.visibility ?? 0);

  return leftVis > rightVis ? 'left' : 'right';
}

export { initPose, detectPose, extractFootLandmarks, detectDominantFoot, FOOT_LANDMARKS };
