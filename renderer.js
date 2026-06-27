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

// Actualizar posición y escala del zapato en cada frame
// pose: { R, t, scaleFactor } de solver.js
// videoW, videoH: dimensiones del video
function updateShoeTransform(pose, videoW, videoH) {
  if (!shoeModel || !pose) {
    if (shoeModel) shoeModel.visible = false;
    return;
  }

  shoeModel.visible = true;

  const { t, R, scaleFactor = 1 } = pose;

  // Convertir traslación de coordenadas de cámara a NDC [-1,1]
  const nx = (t[0] / videoW) * 2 - 1;
  const ny = -(t[1] / videoH) * 2 + 1;
  const nz = -t[2] / 200; // profundidad normalizada

  shoeModel.position.set(nx, ny, nz);

  // Aplicar rotación desde la matriz R
  const m = new THREE_ref.Matrix4().set(
    R[0][0], R[0][1], R[0][2], 0,
    R[1][0], R[1][1], R[1][2], 0,
    R[2][0], R[2][1], R[2][2], 0,
    0,       0,       0,       1
  );
  shoeModel.quaternion.setFromRotationMatrix(m);

  // Escala: ajustar al tamaño del pie detectado
  const baseScale = 0.3; // ajuste visual base
  shoeModel.scale.setScalar(baseScale * scaleFactor);
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
