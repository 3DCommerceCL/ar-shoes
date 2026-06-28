// app.js — orquestador principal del loop AR
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { initPose, detectPose, captureBackground, hasBgData,
         extractFootLandmarks, detectDominantFoot } from './pose.js';
import { createLandmarkFilters, applyFilters }      from './filter.js';
import { computeScaleFactor, measureGLBLength }     from './scaler.js';
import {
  initRenderer, loadShoeGLB, buildOccluder,
  updateShoeTransform, updateMask, renderFrame, setShoeOpacity,
} from './renderer.js';

const GLB_PATH  = './models/shoe.glb';
const STATUS_EL = document.getElementById('status');

let videoEl, canvasEl;
let currentSide   = 'right';
let filters       = null;
let isRunning     = false;
let noFootFrames  = 0;
let firstDetected = false;
const NO_FOOT_THRESHOLD = 15;

// ---- Bootstrap ----
async function init() {
  videoEl  = document.getElementById('video');
  canvasEl = document.getElementById('canvas');

  setLoadingMsg('Iniciando cámara…');
  await startCamera();

  setLoadingMsg('Cargando zapato 3D…');
  initRenderer(canvasEl, videoEl, THREE, GLTFLoader);
  await initPose();
  const shoe = await loadShoeGLB(GLB_PATH, THREE, GLTFLoader);
  measureGLBLength(shoe, THREE);
  buildOccluder(THREE);

  filters = createLandmarkFilters(3, 30);

  // UI events
  document.getElementById('btn-switch-foot').addEventListener('click', toggleFoot);
  document.getElementById('slider-opacity').addEventListener('input', e => {
    setShoeOpacity(parseFloat(e.target.value));
  });
  document.getElementById('btn-calibrate').addEventListener('click', onCalibrate);
  document.getElementById('btn-recalibrate').addEventListener('click', onRecalibrate);

  // Ocultar loading, mostrar pantalla de calibración paso 1
  document.getElementById('loading-screen').style.display = 'none';
  showStep(1);
}

// ---- Calibración ----
function onCalibrate() {
  captureBackground(videoEl);
  showStep(2);

  // Iniciar loops
  isRunning = true;
  requestAnimationFrame(renderLoop);
  detectionLoop();
}

function onRecalibrate() {
  isRunning    = false;
  firstDetected = false;
  updateShoeTransform(null);
  setTimeout(() => {
    isRunning = false;
    showStep(1);
  }, 100);
}

function showStep(n) {
  document.getElementById('step-1').style.display = n === 1 ? 'flex' : 'none';
  document.getElementById('step-2').style.display = n === 2 ? 'flex' : 'none';
  // El UI normal solo se ve en paso 2
  document.getElementById('ui').style.display = n === 2 ? 'flex' : 'none';
}

// ---- Loop de render — 60fps ----
function renderLoop() {
  if (!isRunning) return;
  if (videoEl && videoEl.paused) videoEl.play().catch(() => {});
  try { renderFrame(); } catch(e) {}
  requestAnimationFrame(renderLoop);
}

// ---- Loop de detección — ~5fps ----
async function detectionLoop() {
  while (isRunning) {
    const now = performance.now();
    const seg = detectPose(videoEl);

    if (!seg) {
      noFootFrames++;
      if (noFootFrames > NO_FOOT_THRESHOLD) {
        setStatus('Pon tu pie en la cámara');
        updateShoeTransform(null);
      }
      await sleep(200);
      continue;
    }

    // Contar píxeles significativos para feedback
    let pixCount = 0;
    for (let i = 0; i < seg.data.length; i++) {
      if (seg.data[i] > 0.10) pixCount++;
    }

    if (pixCount < 150) {
      noFootFrames++;
      if (noFootFrames > NO_FOOT_THRESHOLD) {
        setStatus('Pon tu pie en la cámara ↓');
        updateShoeTransform(null);
      }
      await sleep(200);
      continue;
    }

    // Si demasiado cambio → cámara se movió
    if (pixCount > 256 * 256 * 0.30) {
      setStatus('Recalibra si moviste la cámara');
      updateShoeTransform(null);
      await sleep(200);
      continue;
    }

    noFootFrames = 0;

    if (!window._footManualOverride) {
      currentSide = detectDominantFoot(seg);
    }

    const rawLms = extractFootLandmarks(seg, currentSide);
    if (!rawLms) {
      setStatus('Centra el pie en la cámara');
      await sleep(200);
      continue;
    }

    const lmArray  = [rawLms.heel, rawLms.toe, rawLms.ankle];
    const smoothed = applyFilters(filters, lmArray, now / 1000);
    const footLms  = { heel: smoothed[0], toe: smoothed[1], ankle: smoothed[2], side: rawLms.side };

    setStatus(`Pie ${currentSide === 'right' ? 'derecho' : 'izquierdo'} detectado ✓`);

    // Ocultar instrucción "Paso 2" al primer pie detectado
    if (!firstDetected) {
      firstDetected = true;
      document.getElementById('step-2').style.display = 'none';
    }

    updateShoeTransform(footLms, 1);

    await sleep(200);
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---- Cámara ----
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    videoEl.srcObject = stream;
    await new Promise(res => videoEl.addEventListener('loadedmetadata', res, { once: true }));
    await videoEl.play();
  } catch (err) {
    const msg = err.name === 'NotAllowedError'
      ? 'Permiso de cámara denegado'
      : 'Error de cámara: ' + err.message;
    setLoadingMsg(msg);
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
}

function setLoadingMsg(msg) {
  const el = document.getElementById('loading-msg');
  if (el) el.textContent = msg;
}

// ---- Arrancar ----
window.addEventListener('DOMContentLoaded', () => {
  init().catch(err => {
    console.error('[app] Error fatal:', err);
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
