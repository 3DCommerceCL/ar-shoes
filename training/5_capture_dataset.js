/**
 * PASO 5 — Captura continua de datos durante el try-on
 * ======================================================
 * Este módulo va en el app.js de producción.
 * Captura frames cuando la detección es buena → dataset para re-entrenamiento.
 *
 * Los frames se guardan en IndexedDB del navegador.
 * Cada N días, el usuario puede descargar un zip con todos los frames capturados.
 *
 * Para integrar en app.js:
 *   import { initCapture, captureFrame, downloadDataset } from './training/5_capture_dataset.js';
 *   initCapture();
 *   // En el loop de detección, cuando hay detección buena:
 *   captureFrame(videoEl, footLandmarks);
 */

const DB_NAME    = "ar-shoes-dataset";
const STORE_NAME = "frames";
const MAX_FRAMES = 5000; // Limitar a 5000 frames en local

let db = null;
let captureCount = 0;
let lastCaptureTime = 0;
const CAPTURE_INTERVAL_MS = 2000; // Capturar máximo 1 frame cada 2 segundos

// ---- Inicializar IndexedDB ----
export function initCapture() {
  const request = indexedDB.open(DB_NAME, 1);
  request.onupgradeneeded = (e) => {
    const db = e.target.result;
    if (!db.objectStoreNames.contains(STORE_NAME)) {
      db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
    }
  };
  request.onsuccess = (e) => {
    db = e.target.result;
    countFrames().then(n => {
      captureCount = n;
      console.log(`[capture] Dataset local: ${n} frames`);
    });
  };
  request.onerror = () => console.warn("[capture] IndexedDB no disponible");
}

// ---- Capturar un frame cuando la detección es buena ----
export function captureFrame(videoEl, footLandmarks) {
  if (!db || !footLandmarks) return;
  const now = Date.now();
  if (now - lastCaptureTime < CAPTURE_INTERVAL_MS) return;
  if (captureCount >= MAX_FRAMES) return;

  lastCaptureTime = now;

  // Capturar video en canvas 256×256
  const canvas = document.createElement("canvas");
  canvas.width  = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, 256, 256);

  canvas.toBlob((blob) => {
    if (!blob) return;
    const record = {
      timestamp: now,
      landmarks: {
        heel:  { x: footLandmarks.heel.x,  y: footLandmarks.heel.y  },
        toe:   { x: footLandmarks.toe.x,   y: footLandmarks.toe.y   },
        ankle: { x: footLandmarks.ankle.x, y: footLandmarks.ankle.y },
        side:  footLandmarks.side,
      },
      imageBlob: blob,
    };
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).add(record);
    captureCount++;
  }, "image/jpeg", 0.85);
}

// ---- Contar frames en DB ----
function countFrames() {
  return new Promise((resolve) => {
    if (!db) { resolve(0); return; }
    const tx  = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).count();
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => resolve(0);
  });
}

// ---- Descargar dataset como ZIP ----
// Requiere: <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
export async function downloadDataset() {
  if (!db) { alert("Base de datos no disponible"); return; }

  const tx      = db.transaction(STORE_NAME, "readonly");
  const store   = tx.objectStore(STORE_NAME);
  const request = store.getAll();

  request.onsuccess = async () => {
    const records = request.result;
    if (records.length === 0) { alert("No hay frames capturados todavía"); return; }

    const zip = new JSZip();
    const meta = [];

    for (let i = 0; i < records.length; i++) {
      const r    = records[i];
      const name = `frame_${String(i).padStart(5, "0")}`;
      zip.file(`images/${name}.jpg`, r.imageBlob);
      meta.push({ file: `${name}.jpg`, ...r.landmarks, timestamp: r.timestamp });
    }
    zip.file("landmarks.json", JSON.stringify(meta, null, 2));

    const blob = await zip.generateAsync({ type: "blob", compression: "DEFLATE" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `ar_shoes_dataset_${new Date().toISOString().slice(0,10)}.zip`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`[capture] Descargando ${records.length} frames`);
  };
}

// ---- Obtener estadísticas ----
export async function getCaptureStats() {
  const count = await countFrames();
  return { frames: count, maxFrames: MAX_FRAMES, percentage: Math.round(count / MAX_FRAMES * 100) };
}
