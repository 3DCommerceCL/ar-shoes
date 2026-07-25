// inference.js — runner de FootNet (ONNX) en el navegador con onnxruntime-web.
// Salidas del modelo: "seg" (1×4×256×256 logits) y "heatmaps" (1×6×64×64).
// WebGPU con fallback a WASM. Sin bundler (ES module + CDN, versión pineada).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.bundle.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
ort.env.wasm.numThreads = 1; // GitHub Pages no envía COOP/COEP → sin SharedArrayBuffer/threads

const INPUT = 256, HM = 64, NUM_CLASSES = 4, NUM_KP = 6;
const DEFAULT_MEAN = [0.485, 0.456, 0.406];
const DEFAULT_STD  = [0.229, 0.224, 0.225];

let session = null;
let activeEP = 'none';
let meta = null;
let _canvas = null;

export function getActiveEP() { return activeEP; }
export function getMeta() { return meta; }

export async function initInference(modelUrl, metaUrl = null) {
  if (metaUrl) {
    try { meta = await (await fetch(metaUrl)).json(); }
    catch (e) { console.warn('[inference] meta.json no cargó, usando defaults:', e.message); }
  }
  try {
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu'] });
    activeEP = 'webgpu';
  } catch (e) {
    console.warn('[inference] WebGPU no disponible, usando WASM:', e.message);
    session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    activeEP = 'wasm';
  }
  console.log('[inference] EP activo:', activeEP, '| outputs:', session.outputNames);
  return activeEP;
}

// center-crop cuadrado + resize 256 + normalize → Float32 NCHW
function preprocess(src) {
  _canvas = _canvas || document.createElement('canvas');
  _canvas.width = INPUT; _canvas.height = INPUT;
  const ctx = _canvas.getContext('2d', { willReadFrequently: true });
  const w = src.videoWidth || src.naturalWidth || src.width;
  const h = src.videoHeight || src.naturalHeight || src.height;
  const s = Math.min(w, h), sx = (w - s) / 2, sy = (h - s) / 2;
  ctx.drawImage(src, sx, sy, s, s, 0, 0, INPUT, INPUT);
  const { data } = ctx.getImageData(0, 0, INPUT, INPUT);

  const mean = meta?.mean || DEFAULT_MEAN;
  const std  = meta?.std  || DEFAULT_STD;
  const plane = INPUT * INPUT;
  const arr = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    arr[i]           = (data[i * 4]     / 255 - mean[0]) / std[0];
    arr[plane + i]   = (data[i * 4 + 1] / 255 - mean[1]) / std[1];
    arr[2 * plane + i] = (data[i * 4 + 2] / 255 - mean[2]) / std[2];
  }
  return { tensor: new ort.Tensor('float32', arr, [1, 3, INPUT, INPUT]), crop: { sx, sy, s, w, h } };
}

// argmax de las 4 clases → Uint8Array 256*256 (en espacio del recorte)
function segArgmax(seg) {
  const plane = INPUT * INPUT;
  const out = new Uint8Array(plane);
  for (let p = 0; p < plane; p++) {
    let best = 0, bv = seg[p];
    for (let c = 1; c < NUM_CLASSES; c++) {
      const v = seg[c * plane + p];
      if (v > bv) { bv = v; best = c; }
    }
    out[p] = best;
  }
  return out;
}

// argmax + refino por centro de masa 3×3 de cada heatmap → [[x,y,conf]×6] en coords del FRAME completo
function kpsFromHeatmaps(hm, crop) {
  const plane = HM * HM;
  const kps = [];
  for (let k = 0; k < NUM_KP; k++) {
    const base = k * plane;
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < plane; i++) { const v = hm[base + i]; if (v > bv) { bv = v; bi = i; } }
    let px = bi % HM, py = Math.floor(bi / HM);
    let sx = 0, sy = 0, sw = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const x = px + dx, y = py + dy;
        if (x < 0 || y < 0 || x >= HM || y >= HM) continue;
        const w = Math.max(0, hm[base + y * HM + x]);
        sx += x * w; sy += y * w; sw += w;
      }
    }
    if (sw > 0) { px = sx / sw; py = sy / sw; }
    const nx = px / HM, ny = py / HM; // normalizado dentro del recorte
    const fx = crop ? (crop.sx + nx * crop.s) / crop.w : nx;
    const fy = crop ? (crop.sy + ny * crop.s) / crop.h : ny;
    kps.push([fx, fy, bv]);
  }
  return kps;
}

export async function runInference(src) {
  if (!session) throw new Error('Llamá initInference() primero');
  const { tensor, crop } = preprocess(src);
  const t0 = performance.now();
  const out = await session.run({ [session.inputNames[0]]: tensor });
  const latencyMs = performance.now() - t0;
  const seg = (out.seg || out[session.outputNames[0]]).data;
  const hm  = (out.heatmaps || out[session.outputNames[1]]).data;
  return { seg: segArgmax(seg), kps: kpsFromHeatmaps(hm, crop), latencyMs, ep: activeEP, crop };
}
