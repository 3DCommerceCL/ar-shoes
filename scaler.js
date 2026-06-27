// Estimación del tamaño real del pie → factor de escala para el GLB
// Usa la distancia píxel heel→toe + profundidad MiDaS para estimar largo en cm

const FOV_Y_DEG = 60;
const AVG_FOOT_CM = 26; // largo promedio pie adulto (usado como fallback)

// shoeModelLengthCm: largo del GLB medido con Box3 al cargar
let shoeModelLengthCm = AVG_FOOT_CM;

function setShoeModelLength(cm) {
  shoeModelLengthCm = cm;
}

// Estima el largo real del pie en cm y retorna el factor de escala para el GLB
// footLandmarks: { heel, toe } con coords normalizadas [0,1]
// depthMap: Float32Array del tamaño del video
// videoW, videoH: dimensiones en píxeles
function computeScaleFactor(footLandmarks, depthMap, videoW, videoH) {
  if (!footLandmarks?.heel || !footLandmarks?.toe) {
    return 1.0;
  }

  // Distancia píxel entre heel y toe
  const hx = footLandmarks.heel.x * videoW;
  const hy = footLandmarks.heel.y * videoH;
  const tx = footLandmarks.toe.x  * videoW;
  const ty = footLandmarks.toe.y  * videoH;
  const pixelLength = Math.sqrt((tx - hx) ** 2 + (ty - hy) ** 2);

  if (pixelLength < 5) return 1.0; // pie no visible o demasiado pequeño

  // Profundidad promedio en la región del pie (MiDaS, [0,1])
  let depthValue = 0.5; // fallback
  if (depthMap) {
    const samples = [
      footLandmarks.heel,
      footLandmarks.toe,
      footLandmarks.ankle,
    ].filter(Boolean);

    let sum = 0, count = 0;
    for (const pt of samples) {
      const px = Math.min(Math.round(pt.x * videoW), videoW - 1);
      const py = Math.min(Math.round(pt.y * videoH), videoH - 1);
      sum += depthMap[py * videoW + px];
      count++;
    }
    depthValue = sum / count;
  }

  // Convertir profundidad MiDaS [0,1] a distancia relativa de escena
  // MiDaS invierte profundidad: 1 = muy cerca, 0 = lejos
  // Rango típico 30cm – 200cm desde la cámara, escalado lineal
  const MIN_DIST_CM = 30;
  const MAX_DIST_CM = 200;
  const estimatedDistCm = MAX_DIST_CM - depthValue * (MAX_DIST_CM - MIN_DIST_CM);

  // Longitud real del pie usando geometría de cámara pinhole
  const fovRad = (FOV_Y_DEG * Math.PI) / 180;
  const focalPx = videoH / (2 * Math.tan(fovRad / 2));
  const realLengthCm = (pixelLength * estimatedDistCm) / focalPx;

  // Factor de escala: cuánto hay que escalar el GLB para que mida realLengthCm
  const scaleFactor = realLengthCm / shoeModelLengthCm;

  // Clamp para evitar escalas extremas (el pie nunca debería ser < 18cm o > 35cm)
  return Math.max(0.5, Math.min(2.0, scaleFactor));
}

// Mide el largo del modelo GLB cargado usando Three.js Box3
// Llamar una sola vez después de cargar el GLB
function measureGLBLength(gltfScene, THREE) {
  const box = new THREE.Box3().setFromObject(gltfScene);
  const size = new THREE.Vector3();
  box.getSize(size);

  // El eje más largo del zapato es la longitud (asumimos Y o Z)
  const maxDim = Math.max(size.x, size.y, size.z);

  // Si el modelo está en metros (convención glTF), convertir a cm
  // Heurística: si maxDim < 2, asumimos metros
  shoeModelLengthCm = maxDim < 2 ? maxDim * 100 : maxDim;

  console.log(`[scaler] GLB largo medido: ${shoeModelLengthCm.toFixed(1)} cm`);
  return shoeModelLengthCm;
}

export { computeScaleFactor, measureGLBLength, setShoeModelLength };
