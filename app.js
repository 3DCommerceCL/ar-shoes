// app.js — orquestador principal del loop AR
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { initPose, detectPose, extractFootLandmarks, detectDominantFoot,
         getTrackerMode, getActiveEP } from './pose.js';
import { createLandmarkFilters, applyFilters }      from './filter.js';
import {
  initRenderer, loadShoeGLB, buildOccluder,
  updateShoeTransform, updateMask, renderFrame, setShoeOpacity,
} from './renderer.js';

const GLB_PATH  = './models/shoe_web.glb';
const STATUS_EL = document.getElementById('status');

let videoEl, canvasEl;
let currentSide    = 'right';
let filters        = null;
let isRunning      = false;
let noFootFrames   = 0;
let firstDetected  = false;
let sideCandidate      = null; // lado propuesto para el cambio con histéresis
let sideCandidateCount = 0;
const NO_FOOT_THRESHOLD = 8;   // ciclos sin pie antes de ocultar el zapato
const DETECT_INTERVAL_MS = 80; // el modelo tarda; no encolar inferencias
const SIDE_SWITCH_FRAMES = 5;  // ciclos consecutivos para confirmar cambio de pie

// ---- Bootstrap ----
async function init() {
  videoEl  = document.getElementById('video');
  canvasEl = document.getElementById('canvas');

  setLoadingMsg('Iniciando cámara…');
  await startCamera();

  setLoadingMsg('Cargando zapato 3D…');
  initRenderer(canvasEl, videoEl, THREE, GLTFLoader);
  await initPose();
  await loadShoeGLB(GLB_PATH, THREE, GLTFLoader);
  buildOccluder(THREE);

  filters = createLandmarkFilters(3, 30);

  // UI events
  document.getElementById('btn-switch-foot').addEventListener('click', toggleFoot);
  document.getElementById('slider-opacity').addEventListener('input', e => {
    setShoeOpacity(parseFloat(e.target.value));
  });
  document.getElementById('btn-recalibrate').addEventListener('click', onRecalibrate);

  // Sin calibración: el modelo detecta el pie directamente
  document.getElementById('loading-screen').style.display = 'none';
  showStep(2);
  isRunning = true;
  requestAnimationFrame(renderLoop);
  detectionLoop();
}

function onRecalibrate() {
  firstDetected = false;
  filters = createLandmarkFilters(3, 30); // no arrastrar el estado del filtro tras recalibrar
  sideCandidate = null; sideCandidateCount = 0;
  window._footManualOverride = false;
  updateShoeTransform(null);
  showStep(2);
}

function showStep(n) {
  const s2 = document.getElementById('step-2');
  if (s2) s2.style.display = n === 2 ? 'flex' : 'none';
  const ui = document.getElementById('ui');
  if (ui) ui.style.display = 'flex';
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
    const seg = await detectPose(videoEl);

    if (!seg) {
      noFootFrames++;
      if (noFootFrames > NO_FOOT_THRESHOLD) {
        setStatus('Apuntá la cámara a tu pie ↓');
        updateShoeTransform(null);
      }
      await sleep(DETECT_INTERVAL_MS);
      continue;
    }

    // Cambio de pie con histéresis: solo cambia tras varios ciclos consecutivos coincidentes
    if (!window._footManualOverride) {
      const detected = detectDominantFoot(seg);
      if (detected === currentSide) {
        sideCandidate = null; sideCandidateCount = 0;
      } else if (detected === sideCandidate) {
        if (++sideCandidateCount >= SIDE_SWITCH_FRAMES) {
          currentSide = detected;
          filters = createLandmarkFilters(3, 30); // el pie cambió: reiniciar el filtro
          sideCandidate = null; sideCandidateCount = 0;
        }
      } else {
        sideCandidate = detected; sideCandidateCount = 1;
      }
    }

    const rawLms = extractFootLandmarks(seg, currentSide);
    if (!rawLms) {
      noFootFrames++;
      if (noFootFrames > NO_FOOT_THRESHOLD) {
        setStatus('Apuntá la cámara a tu pie ↓');
        updateShoeTransform(null);
      }
      await sleep(DETECT_INTERVAL_MS);
      continue;
    }

    noFootFrames = 0;

    const lmArray  = [rawLms.heel, rawLms.toe, rawLms.ankle];
    const smoothed = applyFilters(filters, lmArray, now / 1000);
    const footLms  = {
      heel: smoothed[0], toe: smoothed[1], ankle: smoothed[2],
      bboxW: rawLms.bboxW, bboxH: rawLms.bboxH,
      side: rawLms.side,
    };

    setStatus(`Pie ${currentSide === 'right' ? 'derecho' : 'izquierdo'} ✓  (${getTrackerMode()}, ${getActiveEP()})`);

    if (!firstDetected) {
      firstDetected = true;
      document.getElementById('step-2').style.display = 'none';
    }

    updateShoeTransform(footLms, 1);

    await sleep(DETECT_INTERVAL_MS);
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
  filters = createLandmarkFilters(3, 30); // no arrastrar el filtro del pie anterior
  sideCandidate = null; sideCandidateCount = 0;
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
          <p id="err-detail" style="font-size:13px;opacity:0.7;word-break:break-all"></p>
          <button onclick="location.reload()" style="margin-top:20px;padding:10px 24px;border-radius:20px;border:none;background:#fff;color:#000;font-size:15px;font-weight:600">Reintentar</button>
        </div>`;
      const detail = document.getElementById('err-detail');
      if (detail) detail.textContent = err.message; // textContent evita inyección de HTML
    }
  });
});
