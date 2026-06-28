// app.js — orquestador principal del loop AR
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { initPose, detectPose, extractFootLandmarks, detectDominantFoot } from './pose.js';
import { initDepth, estimateDepth }                                        from './depth.js';
import { initSegmenter, getFootMask }                                      from './segmenter.js';
import { createLandmarkFilters, applyFilters }                             from './filter.js';
import { solvePose }                                                        from './solver.js';
import { computeScaleFactor, measureGLBLength }                            from './scaler.js';
import {
  initRenderer, loadShoeGLB, buildOccluder,
  updateShoeTransform, updateMask, renderFrame, setShoeOpacity,
} from './renderer.js';

// ---- Configuración ----
const GLB_PATH    = './models/shoe.glb';
const DEPTH_EVERY = 3;
const STATUS_EL   = document.getElementById('status');

function withTimeout(promise, ms, label) {
  const timer = new Promise((_, reject) =>
    setTimeout(() => reject(new Error(`${label} timeout (${ms / 1000}s)`)), ms)
  );
  return Promise.race([promise, timer]);
}

let videoEl, canvasEl;
let currentSide   = 'right';
let depthMap      = null;
let frameCount    = 0;
let filters       = null;
let isRunning     = false;
let noFootFrames  = 0;
const NO_FOOT_THRESHOLD = 60; // ~2s a 30fps

// ---- Bootstrap ----
async function init() {
  setStatus('Iniciando cámara…');

  videoEl  = document.getElementById('video');
  canvasEl = document.getElementById('canvas');

  // 1. Cámara
  await startCamera();

  // 2. Three.js ya importado estáticamente arriba

  // 3. Inicializar módulos — MiDaS y segmentador son opcionales
  setStatus('Cargando segmentador…');
  await withTimeout(initPose(), 30000, 'Segmentador')
    .catch(e => console.warn('[app] segmentador falló:', e.message));
  setStatus('Listo — apunta la cámara a tus pies');

  // 4. Renderer + GLB
  setStatus('Cargando zapato 3D…');
  initRenderer(canvasEl, videoEl, THREE, GLTFLoader);
  const shoe = await loadShoeGLB(GLB_PATH, THREE, GLTFLoader);
  measureGLBLength(shoe, THREE);
  buildOccluder(THREE);

  // 5. Filtros para heel/toe/ankle (3 puntos × {x,y})
  filters = createLandmarkFilters(3, 30);

  // 6. UI events
  document.getElementById('btn-switch-foot').addEventListener('click', toggleFoot);
  document.getElementById('slider-opacity').addEventListener('input', e => {
    setShoeOpacity(parseFloat(e.target.value));
  });

  setStatus('Apunta la cámara hacia tus pies');
  window.dispatchEvent(new Event('ar-ready'));
  isRunning = true;
  requestAnimationFrame(loop);
}

// ---- Loop principal ----
async function loop(timestamp) {
  if (!isRunning) return;

  frameCount++;
  const now = performance.now();

  // A. Segmentación BodyPix (async)
  const segmentation = await detectPose(videoEl);

  if (!segmentation) {
    noFootFrames++;
    if (noFootFrames > NO_FOOT_THRESHOLD) {
      setStatus('Apunta la cámara hacia tus pies');
    }
    try { renderFrame(); } catch(e) {}
    requestAnimationFrame(loop);
    return;
  }

  noFootFrames = 0;

  // B. Detectar pie dominante
  if (!window._footManualOverride) {
    currentSide = detectDominantFoot(segmentation);
  }

  // C. Extraer landmarks del pie desde segmentación
  const rawLms = extractFootLandmarks(segmentation, currentSide);
  if (!rawLms) {
    setStatus('Muestra tus pies en la cámara');
    renderFrame();
    requestAnimationFrame(loop);
    return;
  }

  // D. Suavizar landmarks con OneEuroFilter
  const lmArray = [rawLms.heel, rawLms.toe, rawLms.ankle];
  const smoothed = applyFilters(filters, lmArray, now / 1000);
  const footLms = { heel: smoothed[0], toe: smoothed[1], ankle: smoothed[2], side: rawLms.side };

  setStatus(`Pie ${currentSide} detectado ✓`);

  // E. Profundidad MiDaS (cada N frames)
  if (frameCount % DEPTH_EVERY === 0) {
    depthMap = await estimateDepth(videoEl);
  }

  // F. Máscara del pie (oclusión)
  const maskCanvas = await getFootMask(videoEl, footLms, now, canvasEl.width, canvasEl.height);
  updateMask(maskCanvas);

  // G. Escala del zapato
  const scale = computeScaleFactor(footLms, depthMap, videoEl.videoWidth, videoEl.videoHeight);

  // H. Actualizar posición del GLB directo desde landmarks 2D
  updateShoeTransform(footLms, scale);

  // J. Render
  try { renderFrame(); } catch(e) { console.warn('[render]', e.message); }

  requestAnimationFrame(loop);
}


// ---- Cámara ----
async function startCamera() {
  const constraints = {
    video: {
      facingMode: { ideal: 'environment' },
      width:  { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoEl.srcObject = stream;
    await new Promise(res => videoEl.addEventListener('loadedmetadata', res, { once: true }));
    await videoEl.play();
  } catch (err) {
    if (err.name === 'NotAllowedError') {
      setStatus('Permiso de cámara denegado. Por favor permite el acceso.');
    } else if (err.name === 'NotFoundError') {
      setStatus('No se encontró cámara en este dispositivo.');
    } else {
      setStatus('Error de cámara: ' + err.message);
    }
    throw err;
  }
}

// ---- UI helpers ----
function toggleFoot() {
  currentSide = currentSide === 'right' ? 'left' : 'right';
  window._footManualOverride = true;
  document.getElementById('btn-switch-foot').textContent =
    currentSide === 'right' ? 'Pie izquierdo' : 'Pie derecho';
}

function setStatus(msg) {
  if (STATUS_EL) STATUS_EL.textContent = msg;
  const loadingMsg = document.getElementById('loading-msg');
  if (loadingMsg) loadingMsg.textContent = msg;
}

// ---- Arrancar ----
window.addEventListener('DOMContentLoaded', () => {
  init().catch(err => {
    console.error('[app] Error fatal:', err);
    setStatus('Error al iniciar: ' + err.message);
    // Mostrar error visible en la pantalla de carga
    const loadingScreen = document.getElementById('loading-screen');
    if (loadingScreen) {
      loadingScreen.innerHTML = `
        <div style="padding:24px;text-align:center;color:#fff;max-width:320px">
          <p style="font-size:16px;margin-bottom:12px">Error al iniciar</p>
          <p style="font-size:13px;opacity:0.7;word-break:break-all">${err.message}</p>
          <button onclick="location.reload()" style="margin-top:20px;padding:10px 24px;border-radius:20px;border:none;background:#fff;color:#000;font-size:15px;font-weight:600">Reintentar</button>
        </div>`;
    }
  });
});
