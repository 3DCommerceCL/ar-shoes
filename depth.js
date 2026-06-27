// MiDaS v2.1 Small — estimación de profundidad monocular via ONNX Runtime Web
// Ejecutar cada N frames para no saturar el hilo principal

const MIDAS_INPUT_SIZE = 256;
const MIDAS_MEAN = [0.485, 0.456, 0.406];
const MIDAS_STD  = [0.229, 0.224, 0.225];

let session = null;
let offscreen = null;
let offCtx   = null;

async function initDepth(modelPath = './models/midas_v21_small.onnx') {
  session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ['wasm'],
  });

  offscreen = document.createElement('canvas');
  offscreen.width  = MIDAS_INPUT_SIZE;
  offscreen.height = MIDAS_INPUT_SIZE;
  offCtx = offscreen.getContext('2d');

  console.log('[depth] MiDaS listo');
}

// Retorna Float32Array [videoH x videoW] con valores de profundidad relativos [0,1]
// 1 = más cerca, 0 = más lejos (MiDaS invierte la convención — normalizamos aquí)
async function estimateDepth(videoEl) {
  if (!session) return null;

  const W = videoEl.videoWidth;
  const H = videoEl.videoHeight;

  // Escalar frame a 256x256
  offCtx.drawImage(videoEl, 0, 0, MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE);
  const imgData = offCtx.getImageData(0, 0, MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE).data;

  // Construir tensor CHW float32 normalizado
  const tensor = new Float32Array(3 * MIDAS_INPUT_SIZE * MIDAS_INPUT_SIZE);
  for (let i = 0; i < MIDAS_INPUT_SIZE * MIDAS_INPUT_SIZE; i++) {
    const r = imgData[i * 4]     / 255;
    const g = imgData[i * 4 + 1] / 255;
    const b = imgData[i * 4 + 2] / 255;
    tensor[i]                                          = (r - MIDAS_MEAN[0]) / MIDAS_STD[0];
    tensor[i + MIDAS_INPUT_SIZE * MIDAS_INPUT_SIZE]    = (g - MIDAS_MEAN[1]) / MIDAS_STD[1];
    tensor[i + 2 * MIDAS_INPUT_SIZE * MIDAS_INPUT_SIZE] = (b - MIDAS_MEAN[2]) / MIDAS_STD[2];
  }

  const inputTensor = new ort.Tensor('float32', tensor, [1, 3, MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE]);
  const results = await session.run({ input: inputTensor });
  const raw = results.output.data; // Float32Array 256x256

  // Normalizar a [0,1] e invertir (MiDaS: alto = lejos, queremos alto = cerca)
  let minVal = Infinity, maxVal = -Infinity;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] < minVal) minVal = raw[i];
    if (raw[i] > maxVal) maxVal = raw[i];
  }
  const range = maxVal - minVal || 1;

  const normalized = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    normalized[i] = 1.0 - (raw[i] - minVal) / range; // invertido
  }

  // Escalar de 256x256 al tamaño real del video usando interpolación bilinear
  return upscaleDepthMap(normalized, MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE, W, H);
}

// Muestrea el mapa de profundidad en coordenadas normalizadas [0,1]
function sampleDepth(depthMap, videoW, videoH, nx, ny) {
  if (!depthMap) return 0.5;
  const px = Math.round(nx * (videoW - 1));
  const py = Math.round(ny * (videoH - 1));
  return depthMap[py * videoW + px] ?? 0.5;
}

// Interpola el mapa de 256x256 al tamaño del video (bilinear)
function upscaleDepthMap(src, srcW, srcH, dstW, dstH) {
  const dst = new Float32Array(dstW * dstH);
  const xRatio = srcW / dstW;
  const yRatio = srcH / dstH;

  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      const sx = dx * xRatio;
      const sy = dy * yRatio;
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, srcW - 1);
      const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, srcH - 1);
      const fx = sx - x0, fy = sy - y0;

      const v00 = src[y0 * srcW + x0];
      const v10 = src[y0 * srcW + x1];
      const v01 = src[y1 * srcW + x0];
      const v11 = src[y1 * srcW + x1];

      dst[dy * dstW + dx] =
        v00 * (1 - fx) * (1 - fy) +
        v10 * fx * (1 - fy) +
        v01 * (1 - fx) * fy +
        v11 * fx * fy;
    }
  }
  return dst;
}

export { initDepth, estimateDepth, sampleDepth };
