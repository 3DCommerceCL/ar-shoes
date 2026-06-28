// BodyPix — detección de pies desde vista cenital (top-down)
// Partes: left_feet=22, right_feet=23

const PART_LEFT_FOOT  = 22;
const PART_RIGHT_FOOT = 23;

let net = null;

async function initPose() {
  net = await bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2,
  });
  console.log('[pose] BodyPix listo');
}

// Retorna la segmentación de partes del cuerpo (async)
async function detectPose(videoEl) {
  if (!net) return null;
  try {
    return await net.segmentPersonParts(videoEl, {
      flipHorizontal:      false,
      internalResolution:  'medium',
      segmentationThreshold: 0.5,
      maxDetections:       2,
      scoreThreshold:      0.3,
      nmsRadius:           20,
    });
  } catch (e) {
    console.warn('[pose] error en detección:', e.message);
    return null;
  }
}

// Extrae landmarks del pie desde la máscara de segmentación
// Retorna { heel, toe, ankle } con coords normalizadas [0,1] o null
function extractFootLandmarks(segmentation, side = 'right') {
  if (!segmentation?.data) return null;

  const partId = side === 'left' ? PART_LEFT_FOOT : PART_RIGHT_FOOT;
  const { data, width, height } = segmentation;

  // Recolectar píxeles del pie
  const pixels = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[y * width + x] === partId) pixels.push([x, y]);
    }
  }

  // Fallback: si no hay píxeles de pie, usar la mitad inferior del torso
  if (pixels.length < 30) {
    return extractFromPersonBottom(segmentation, side);
  }

  return landmarksFromPixels(pixels, width, height, side);
}

// Fallback: buscar la región inferior de la persona cuando BodyPix no detecta el pie específico
function extractFromPersonBottom(segmentation, side) {
  const { data, width, height } = segmentation;

  // Recolectar todos los píxeles de persona (cualquier parte != -1)
  const allPerson = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[y * width + x] !== -1) allPerson.push([x, y]);
    }
  }

  if (allPerson.length < 100) return null;

  // Filtrar solo el 30% inferior de la imagen (zona de pies)
  const yThreshold = height * 0.7;
  const halfW = width / 2;
  const footPixels = allPerson.filter(([x, y]) => {
    if (y < yThreshold) return false;
    return side === 'left' ? x < halfW : x >= halfW;
  });

  if (footPixels.length < 30) {
    // Sin filtro de lado
    const bottomPixels = allPerson.filter(([, y]) => y >= yThreshold);
    if (bottomPixels.length < 30) return null;
    return landmarksFromPixels(bottomPixels, width, height, side);
  }

  return landmarksFromPixels(footPixels, width, height, side);
}

// Calcula heel/toe/ankle desde un conjunto de píxeles usando PCA
function landmarksFromPixels(pixels, width, height, side) {
  // Centroide
  let sumX = 0, sumY = 0;
  for (const [x, y] of pixels) { sumX += x; sumY += y; }
  const cx = sumX / pixels.length;
  const cy = sumY / pixels.length;

  // Covarianza para PCA (eje principal del pie)
  let cxx = 0, cxy = 0, cyy = 0;
  for (const [x, y] of pixels) {
    const dx = x - cx, dy = y - cy;
    cxx += dx * dx;
    cxy += dx * dy;
    cyy += dy * dy;
  }

  const angle = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  // Proyectar sobre eje principal → encontrar extremos (heel y toe)
  let minProj = Infinity,  maxProj = -Infinity;
  let heelPt  = [cx, cy], toePt   = [cx, cy];

  for (const [x, y] of pixels) {
    const proj = (x - cx) * cos + (y - cy) * sin;
    if (proj < minProj) { minProj = proj; heelPt = [x, y]; }
    if (proj > maxProj) { maxProj = proj; toePt  = [x, y]; }
  }

  return {
    heel:  { x: heelPt[0] / width, y: heelPt[1] / height, visibility: 1 },
    toe:   { x: toePt[0]  / width, y: toePt[1]  / height, visibility: 1 },
    ankle: { x: cx / width,        y: cy / height,         visibility: 1 },
    side,
  };
}

// Detecta qué pie tiene más píxeles (dominante)
function detectDominantFoot(segmentation) {
  if (!segmentation?.data) return 'right';

  let leftCount = 0, rightCount = 0;
  for (const v of segmentation.data) {
    if (v === PART_LEFT_FOOT)  leftCount++;
    if (v === PART_RIGHT_FOOT) rightCount++;
  }

  if (leftCount === 0 && rightCount === 0) return 'right';
  return leftCount > rightCount ? 'left' : 'right';
}

export { initPose, detectPose, extractFootLandmarks, detectDominantFoot };
