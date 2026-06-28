// Three.js — render del GLB sobre el feed de cámara (sin stencil por ahora)

let renderer, scene, camera;
let videoMesh, shoeModel;
let videoTexture;
let THREE_ref;
let debugCanvas, debugCtx;

function initRenderer(canvas, videoEl, THREE, GLTFLoader) {
  THREE_ref = THREE;

  renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: false,
    antialias: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: false,
    stencil: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  renderer.autoClear = true;

  scene  = new THREE.Scene();
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  camera.position.z = 5;

  // Fondo: feed de cámara en un plano en z=0
  videoTexture = new THREE.VideoTexture(videoEl);
  videoTexture.colorSpace = THREE.SRGBColorSpace;
  const bgGeo = new THREE.PlaneGeometry(2, 2);
  const bgMat = new THREE.MeshBasicMaterial({ map: videoTexture });
  videoMesh   = new THREE.Mesh(bgGeo, bgMat);
  videoMesh.position.z = 0;
  videoMesh.renderOrder = 0;
  scene.add(videoMesh);

  // Luces
  const ambLight = new THREE.AmbientLight(0xffffff, 1.0);
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
  dirLight.position.set(1, 2, 3);
  scene.add(ambLight, dirLight);

  // Canvas 2D para debug de landmarks
  debugCanvas = document.createElement('canvas');
  debugCanvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:5';
  document.body.appendChild(debugCanvas);
  debugCtx = debugCanvas.getContext('2d');

  window.addEventListener('resize', () => onResize(canvas));

  return { renderer, scene, camera };
}

async function loadShoeGLB(path, THREE, GLTFLoader) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      path,
      (gltf) => {
        shoeModel = gltf.scene;

        // Centrar modelo en bounding box
        const box    = new THREE.Box3().setFromObject(shoeModel);
        const center = new THREE.Vector3();
        box.getCenter(center);
        shoeModel.position.sub(center);

        // Materiales simplificados para iOS WebGL
        shoeModel.traverse(child => {
          if (child.isMesh) {
            const oldMat = child.material;
            // Reemplazar con MeshLambertMaterial simple para máxima compatibilidad
            child.material = new THREE_ref.MeshLambertMaterial({
              color: oldMat.color ?? 0xffffff,
              map:   oldMat.map   ?? null,
              depthTest:  true,
              depthWrite: true,
              side: THREE_ref.FrontSide,
            });
          }
        });

        shoeModel.renderOrder = 2;
        shoeModel.visible     = true; // visible desde el inicio para verificar render
        scene.add(shoeModel);

        console.log('[renderer] GLB cargado:', path);
        resolve(shoeModel);
      },
      (progress) => {
        const pct = Math.round((progress.loaded / (progress.total || 1)) * 100);
        console.log(`[renderer] GLB cargando: ${pct}%`);
      },
      (err) => {
        console.error('[renderer] Error cargando GLB:', err);
        reject(err);
      }
    );
  });
}

// Posicionar zapato usando landmarks 2D directamente
function updateShoeTransform(footLandmarks, scaleFactor = 1) {
  if (!shoeModel) return;

  if (!footLandmarks) {
    shoeModel.visible = false;
    clearDebug();
    return;
  }

  shoeModel.visible = true;

  // Landmarks a NDC [-1, 1]
  const hx = footLandmarks.heel.x * 2 - 1;
  const hy = -(footLandmarks.heel.y * 2 - 1);
  const tx = footLandmarks.toe.x  * 2 - 1;
  const ty = -(footLandmarks.toe.y  * 2 - 1);
  const ax = footLandmarks.ankle.x * 2 - 1;
  const ay = -(footLandmarks.ankle.y * 2 - 1);

  // Centrar en el ancla (centroide del blob del pie)
  const cx = ax;
  const cy = ay;

  // Ángulo del eje principal (heel → toe)
  const angle = Math.atan2(ty - hy, tx - hx);

  // El modelo mide ~0.262 unidades locales en largo
  // La pantalla NDC = 2 unidades → para llenar ~35% de pantalla necesitamos:
  // 0.262 * scale = 0.7  →  scale ≈ 2.7
  const footLen = Math.sqrt((tx - hx) ** 2 + (ty - hy) ** 2);
  // Cuando footLen es confiable (> 0.1) usarlo; si no, usar 0.7 NDC como tamaño
  const targetLen = Math.max(footLen, 0.7);
  const scale     = (targetLen * scaleFactor * 1.2) / 0.262;

  shoeModel.position.set(cx, cy, 1);
  shoeModel.rotation.set(-Math.PI / 2, 0, angle + Math.PI);
  shoeModel.scale.setScalar(scale);

  // Dibujar puntos de debug
  drawDebugDots(footLandmarks);
}

// Dibujar puntos de landmarks sobre el canvas de debug
function drawDebugDots(footLandmarks) {
  if (!debugCtx || !debugCanvas) return;

  const W = window.innerWidth;
  const H = window.innerHeight;
  debugCanvas.width  = W;
  debugCanvas.height = H;
  debugCtx.clearRect(0, 0, W, H);

  const pts = [
    { lm: footLandmarks.heel,  color: '#ff3333', label: 'heel' },
    { lm: footLandmarks.toe,   color: '#33ff33', label: 'toe'  },
    { lm: footLandmarks.ankle, color: '#3399ff', label: 'ankle'},
  ];

  for (const { lm, color, label } of pts) {
    if (!lm) continue;
    const x = lm.x * W;
    const y = lm.y * H;
    debugCtx.beginPath();
    debugCtx.arc(x, y, 10, 0, Math.PI * 2);
    debugCtx.fillStyle = color;
    debugCtx.fill();
    debugCtx.fillStyle = '#fff';
    debugCtx.font = 'bold 12px sans-serif';
    debugCtx.fillText(label, x + 14, y + 4);
  }
}

function clearDebug() {
  if (debugCtx && debugCanvas) {
    debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  }
}

// no-op — stencil eliminado
function buildOccluder() {}
function updateMask() {}

function setShoeOpacity(value) {
  if (!shoeModel) return;
  shoeModel.traverse(child => {
    if (child.isMesh) {
      child.material.opacity     = value;
      child.material.transparent = value < 1;
    }
  });
}

function renderFrame() {
  if (!renderer) return;
  renderer.render(scene, camera);
}

function onResize(canvas) {
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
}

export {
  initRenderer, loadShoeGLB, buildOccluder,
  updateShoeTransform, updateMask, renderFrame, setShoeOpacity,
};
