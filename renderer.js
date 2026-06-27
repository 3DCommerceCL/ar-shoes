// Three.js — render del GLB sobre el feed de cámara con oclusión stencil

let renderer, scene, camera;
let videoMesh, shoeModel;
let videoTexture, maskTexture;
let occluderMesh;
let THREE_ref;

function initRenderer(canvas, videoEl, THREE, GLTFLoader) {
  THREE_ref = THREE;

  renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: false,
    antialias: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: false,
    stencil: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  renderer.autoClear = false;

  scene  = new THREE.Scene();
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10);
  camera.position.z = 1;

  // Plano de fondo con feed de cámara
  videoTexture = new THREE.VideoTexture(videoEl);
  videoTexture.colorSpace = THREE.SRGBColorSpace;

  const bgGeo  = new THREE.PlaneGeometry(2, 2);
  const bgMat  = new THREE.MeshBasicMaterial({ map: videoTexture, depthWrite: false });
  videoMesh    = new THREE.Mesh(bgGeo, bgMat);
  videoMesh.renderOrder = 0;
  scene.add(videoMesh);

  // Textura de máscara (para oclusión stencil)
  maskTexture = new THREE.CanvasTexture(document.createElement('canvas'));

  // Luz ambiente + direccional para que el zapato se vea bien
  const ambLight  = new THREE.AmbientLight(0xffffff, 0.8);
  const dirLight  = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(0, 5, 5);
  scene.add(ambLight, dirLight);

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

        // Centrar el modelo en su bounding box
        const box    = new THREE.Box3().setFromObject(shoeModel);
        const center = new THREE.Vector3();
        box.getCenter(center);
        shoeModel.position.sub(center);

        // Stencil: el zapato solo se renderiza donde stencil = 0 (no hay pie)
        shoeModel.traverse(child => {
          if (child.isMesh) {
            child.material = child.material.clone();
            child.material.stencilWrite = true;
            child.material.stencilFunc  = THREE.NotEqualStencilFunc;
            child.material.stencilRef   = 1;
            child.material.stencilFail  = THREE.KeepStencilOp;
            child.material.stencilZFail = THREE.KeepStencilOp;
            child.material.stencilZPass = THREE.KeepStencilOp;
          }
        });

        shoeModel.renderOrder = 2;
        shoeModel.visible     = false;
        scene.add(shoeModel);

        console.log('[renderer] GLB cargado:', path);
        resolve(shoeModel);
      },
      undefined,
      reject
    );
  });
}

// Malla invisible que escribe 1 en el stencil donde está el pie
// Se llama con la máscara actualizada cada frame
function buildOccluder(THREE) {
  const geo = new THREE.PlaneGeometry(2, 2);
  const mat = new THREE.MeshBasicMaterial({
    map: maskTexture,
    transparent: true,
    alphaTest: 0.5,
    colorWrite: false,
    depthWrite: false,
    stencilWrite: true,
    stencilFunc:  THREE.AlwaysStencilFunc,
    stencilRef:   1,
    stencilFail:  THREE.ReplaceStencilOp,
    stencilZFail: THREE.ReplaceStencilOp,
    stencilZPass: THREE.ReplaceStencilOp,
  });

  occluderMesh = new THREE.Mesh(geo, mat);
  occluderMesh.renderOrder = 1;
  scene.add(occluderMesh);
}

// Actualizar posición y escala del zapato usando landmarks 2D directamente
// footLandmarks: { heel, toe, ankle } con coords normalizadas [0,1]
function updateShoeTransform(footLandmarks, scaleFactor = 1) {
  if (!shoeModel || !footLandmarks) {
    if (shoeModel) shoeModel.visible = false;
    return;
  }

  shoeModel.visible = true;

  // Convertir landmarks normalizados [0,1] a NDC [-1,1]
  const hx = footLandmarks.heel.x * 2 - 1;
  const hy = -(footLandmarks.heel.y * 2 - 1);
  const tx = footLandmarks.toe.x  * 2 - 1;
  const ty = -(footLandmarks.toe.y  * 2 - 1);

  // Centro entre talón y punta
  const cx = (hx + tx) / 2;
  const cy = (hy + ty) / 2;

  // Ángulo del pie en pantalla
  const angle = Math.atan2(ty - hy, tx - hx);

  // Longitud del pie en NDC → escala del zapato
  const footLen = Math.sqrt((tx - hx) ** 2 + (ty - hy) ** 2);
  const scale   = footLen * 1.1 * scaleFactor;

  shoeModel.position.set(cx, cy, 0.5);

  // Rotar zapato: -90° en X para acostarlo, luego ángulo del pie en Z
  shoeModel.rotation.set(-Math.PI / 2, 0, angle + Math.PI);
  shoeModel.scale.setScalar(Math.max(scale, 0.05));
}

// Actualizar máscara de oclusión
function updateMask(maskCanvas) {
  if (!maskCanvas || !maskTexture) return;
  maskTexture.image   = maskCanvas;
  maskTexture.needsUpdate = true;
}

// Render completo del frame
function renderFrame() {
  if (!renderer) return;

  renderer.clear(true, true, true);

  // 1. Fondo (video)
  renderer.clearStencil();
  renderer.render(scene, camera);
}

function setShoeOpacity(value) {
  if (!shoeModel) return;
  shoeModel.traverse(child => {
    if (child.isMesh) {
      child.material.opacity    = value;
      child.material.transparent = value < 1;
    }
  });
}

function onResize(canvas) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h);
}

export {
  initRenderer,
  loadShoeGLB,
  buildOccluder,
  updateShoeTransform,
  updateMask,
  renderFrame,
  setShoeOpacity,
};
