// Detección de pies via sustracción de fondo (background subtraction)
// No requiere modelo ML — compara frame actual vs frame de referencia (piso vacío)

let bgCanvas = null, bgCtx = null;
let bgData   = null; // ImageData del fondo capturado

function initBgSubtraction() {
  bgCanvas = document.createElement('canvas');
  bgCanvas.width  = 256;
  bgCanvas.height = 256;
  bgCtx = bgCanvas.getContext('2d', { willReadFrequently: true });
}

// Captura el frame actual como referencia de fondo (piso sin pie)
function captureBackground(videoEl) {
  if (!bgCtx) initBgSubtraction();
  bgCtx.drawImage(videoEl, 0, 0, 256, 256);
  const raw = bgCtx.getImageData(0, 0, 256, 256).data;
  bgData = new Uint8ClampedArray(raw); // copia independiente
  console.log('[pose] Fondo capturado');
  return true;
}

function hasBgData() {
  return bgData !== null;
}

// Retorna máscara de diferencia: 0 = igual al fondo, 1 = muy diferente (= pie)
function detectPose(videoEl) {
  if (!bgData || !bgCtx || !videoEl.videoWidth) return null;

  bgCtx.drawImage(videoEl, 0, 0, 256, 256);
  const current = bgCtx.getImageData(0, 0, 256, 256).data;

  const W = 256, H = 256;
  const diff = new Float32Array(W * H);

  for (let i = 0; i < W * H; i++) {
    const r = Math.abs(current[i * 4]     - bgData[i * 4]);
    const g = Math.abs(current[i * 4 + 1] - bgData[i * 4 + 1]);
    const b = Math.abs(current[i * 4 + 2] - bgData[i * 4 + 2]);
    diff[i] = (r + g + b) / (255 * 3); // 0-1
  }

  return { data: diff, width: W, height: H };
}

// Detecta qué pie (izquierdo/derecho) tiene más masa en la mitad inferior
function detectDominantFoot(seg) {
  if (!seg) return 'right';
  const { data, width, height } = seg;
  const midX = width / 2;
  const yMin = Math.floor(height * 0.5);
  let L = 0, R = 0;

  for (let y = yMin; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[y * width + x] > 0.10) {
        x < midX ? L++ : R++;
      }
    }
  }
  return L > R ? 'left' : 'right';
}

// Extrae heel/toe/ankle desde los píxeles que difieren del fondo
function extractFootLandmarks(seg, side = 'right') {
  if (!seg) return null;
  const { data, width, height } = seg;

  // Solo el 55% inferior del frame — pies siempre están abajo cuando cámara apunta al suelo
  const yStart = Math.floor(height * 0.45);
  const foreground = [];
  for (let y = yStart; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[y * width + x] > 0.10) foreground.push([x, y]);
    }
  }
  if (foreground.length < 100) return null;

  // Tomar la mitad inferior del blob → zona de zapatos
  const ys      = foreground.map(([, y]) => y);
  const maxY    = Math.max(...ys);
  const minY    = Math.min(...ys);
  const yThresh = minY + (maxY - minY) * 0.65;
  const footArea = foreground.filter(([, y]) => y >= yThresh);
  if (footArea.length < 40) return null;

  // Filtrar por lado
  const xs   = footArea.map(([x]) => x);
  const midX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const half = side === 'left'
    ? footArea.filter(([x]) => x < midX)
    : footArea.filter(([x]) => x >= midX);
  const src = half.length >= 25 ? half : footArea;

  return landmarksFromPixels(src, width, height, side);
}

// Calcula posición, orientación y tamaño del pie desde los píxeles detectados
function landmarksFromPixels(pixels, width, height, side) {
  // Centroide
  let sumX = 0, sumY = 0;
  for (const [x, y] of pixels) { sumX += x; sumY += y; }
  const cx = sumX / pixels.length;
  const cy = sumY / pixels.length;

  // Bounding box
  let minX = Infinity, maxX = -Infinity, minY2 = Infinity, maxY2 = -Infinity;
  for (const [x, y] of pixels) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY2) minY2 = y; if (y > maxY2) maxY2 = y;
  }
  const bboxW = maxX - minX;
  const bboxH = maxY2 - minY2;

  // PCA para ángulo de orientación
  let cxx = 0, cxy = 0, cyy = 0;
  for (const [x, y] of pixels) {
    const dx = x - cx, dy = y - cy;
    cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
  }
  const angle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  const cos = Math.cos(angle), sin = Math.sin(angle);

  // Heel y toe simétricos alrededor del centroide en la dirección PCA
  const halfLen = Math.max(bboxW, bboxH) * 0.45;
  return {
    heel:  { x: (cx - cos * halfLen) / width,  y: (cy - sin * halfLen) / height,  visibility: 1 },
    toe:   { x: (cx + cos * halfLen) / width,  y: (cy + sin * halfLen) / height,  visibility: 1 },
    ankle: { x: cx / width,                    y: cy / height,                     visibility: 1 },
    bboxW: bboxW / width,
    bboxH: bboxH / height,
    side,
  };
}

// initPose ahora es no-op (sin modelo ML)
async function initPose() {
  initBgSubtraction();
  console.log('[pose] Sustracción de fondo lista');
}

export { initPose, detectPose, captureBackground, hasBgData, extractFootLandmarks, detectDominantFoot };
