// Segmentación del pie para oclusión realista
// Usa MediaPipe Image Segmenter (modelo selfie_multiclass)
// + bounding box de landmarks como fallback rápido

let imageSegmenter = null;
let segCanvas      = null;
let segCtx         = null;
let lastMask       = null;

async function initSegmenter(wasmPath = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm') {
  try {
    const { ImageSegmenter, FilesetResolver } = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.js'
    );

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
        delegate: 'GPU',
      },
      runningMode:       'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });

    segCanvas        = document.createElement('canvas');
    segCanvas.width  = 256;
    segCanvas.height = 256;
    segCtx           = segCanvas.getContext('2d');

    console.log('[segmenter] ImageSegmenter listo');
    return true;
  } catch (e) {
    console.warn('[segmenter] No disponible, usando fallback bbox:', e.message);
    return false;
  }
}

// Retorna ImageData (RGBA) con la máscara del pie escalada al tamaño del canvas destino
// category 3 = left leg/foot, 4 = right leg/foot en selfie_multiclass
async function getFootMask(videoEl, footLandmarks, timestampMs, targetW, targetH) {
  if (imageSegmenter && footLandmarks) {
    try {
      const result = imageSegmenter.segmentForVideo(videoEl, timestampMs);
      const mask   = result.categoryMask;

      // Construir ImageData RGBA desde la máscara de categorías
      const maskData = mask.getAsUint8Array();
      const imgData  = new ImageData(256, 256);

      for (let i = 0; i < maskData.length; i++) {
        const cat = maskData[i];
        // Categorías 3 y 4 corresponden a piernas/pies
        const isfoot = (cat === 3 || cat === 4) ? 255 : 0;
        imgData.data[i * 4]     = isfoot;
        imgData.data[i * 4 + 1] = isfoot;
        imgData.data[i * 4 + 2] = isfoot;
        imgData.data[i * 4 + 3] = isfoot;
      }

      mask.close();

      // Escalar a targetW x targetH
      segCtx.putImageData(imgData, 0, 0);
      const out = document.createElement('canvas');
      out.width  = targetW;
      out.height = targetH;
      out.getContext('2d').drawImage(segCanvas, 0, 0, targetW, targetH);

      lastMask = out;
      return out;
    } catch (e) {
      // fallback
    }
  }

  // Fallback: bounding box convexa alrededor de los landmarks del pie
  return buildBBoxMask(footLandmarks, targetW, targetH);
}

// Máscara simple basada en bounding box de landmarks del pie (fallback)
function buildBBoxMask(footLandmarks, w, h) {
  if (!footLandmarks) return null;

  const canvas = document.createElement('canvas');
  canvas.width  = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  const pts = [footLandmarks.heel, footLandmarks.toe, footLandmarks.ankle].filter(Boolean);
  if (pts.length === 0) return null;

  const xs = pts.map(p => p.x * w);
  const ys = pts.map(p => p.y * h);
  const minX = Math.min(...xs) - 20;
  const maxX = Math.max(...xs) + 20;
  const minY = Math.min(...ys) - 10;
  const maxY = Math.max(...ys) + 30; // incluir dedos debajo

  ctx.fillStyle = 'white';
  ctx.beginPath();
  ctx.ellipse(
    (minX + maxX) / 2,
    (minY + maxY) / 2,
    (maxX - minX) / 2,
    (maxY - minY) / 2,
    0, 0, Math.PI * 2
  );
  ctx.fill();

  lastMask = canvas;
  return canvas;
}

// Retorna la última máscara calculada (para usarla como textura stencil en Three.js)
function getLastMask() {
  return lastMask;
}

export { initSegmenter, getFootMask, getLastMask };
