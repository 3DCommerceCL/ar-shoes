// solvePnP simplificado via DLT + SVD 3x3
// Estima pose 6DOF del pie (posición + rotación) a partir de:
//   - landmarks 2D de MediaPipe (coordenadas normalizadas [0,1])
//   - profundidad Z de MiDaS (normalizada [0,1])
//   - modelo 3D canónico del pie en cm

// Modelo 3D canónico del pie (centrado en el talón, en cm)
// Asume pie adulto promedio ~26cm largo, ~9cm ancho
const FOOT_MODEL_3D = {
  heel:       [  0,  0,  0],
  toe:        [  0, 26,  0],
  inner:      [ -4, 10,  0],
  outer:      [  4, 10,  0],
  ankle_l:    [ -3,  2,  7],
  ankle_r:    [  3,  2,  7],
};

// FOV aproximado cámara móvil (en radianes)
const FOV_Y = (60 * Math.PI) / 180;

// ---- Álgebra lineal mínima inline ----

function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function norm3(v) {
  return Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
}

function normalize3(v) {
  const n = norm3(v) || 1;
  return v.map(x => x / n);
}

// SVD de matriz 3x3 — implementación Jacobi
function svd3x3(A) {
  // Retorna { U, S, V } donde A = U * diag(S) * V^T
  // Método iterativo Jacobi (suficiente para 3x3)
  const n = 3;
  let U = [[1,0,0],[0,1,0],[0,0,1]];
  let V = [[1,0,0],[0,1,0],[0,0,1]];
  let B = A.map(r => [...r]);

  for (let iter = 0; iter < 20; iter++) {
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const bpp = B[p][p], bqq = B[q][q], bpq = B[p][q];
        const theta = 0.5 * Math.atan2(2 * bpq, bpp - bqq);
        const c = Math.cos(theta), s = Math.sin(theta);

        // Rotar B: B = J^T B J
        const newB = B.map(r => [...r]);
        for (let i = 0; i < n; i++) {
          newB[i][p] =  c * B[i][p] + s * B[i][q];
          newB[i][q] = -s * B[i][p] + c * B[i][q];
        }
        for (let i = 0; i < n; i++) {
          B[p][i] =  c * newB[p][i] + s * newB[q][i];
          B[q][i] = -s * newB[p][i] + c * newB[q][i];
        }

        // Acumular en V
        for (let i = 0; i < n; i++) {
          const vip = V[i][p], viq = V[i][q];
          V[i][p] =  c * vip + s * viq;
          V[i][q] = -s * vip + c * viq;
        }
      }
    }
  }

  // Singulares = diagonal de B, vectores izq = A * V / sigma
  const S = [B[0][0], B[1][1], B[2][2]];

  // Calcular U = A * V * diag(1/S)
  for (let j = 0; j < n; j++) {
    const sigma = S[j] || 1e-10;
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += A[i][k] * V[k][j];
      U[i][j] = sum / sigma;
    }
  }

  return { U, S, V };
}

// Estima R (rotación 3x3) y t (traslación [x,y,z]) del pie
// footLandmarks: { heel, toe, ankle } con coordenadas norm [0,1]
// depthMap: Float32Array del frame completo
// videoW, videoH: dimensiones del video
function solvePose(footLandmarks, depthMap, videoW, videoH, scaleFactor = 1.0) {
  if (!footLandmarks) return null;

  const focalLen = videoH / (2 * Math.tan(FOV_Y / 2));
  const cx = videoW / 2, cy = videoH / 2;

  // Convertir landmarks 2D norm a píxeles
  const pts2D = [
    { x: footLandmarks.heel.x  * videoW, y: footLandmarks.heel.y  * videoH },
    { x: footLandmarks.toe.x   * videoW, y: footLandmarks.toe.y   * videoH },
    { x: footLandmarks.ankle.x * videoW, y: footLandmarks.ankle.y * videoH },
  ];

  // Obtener profundidad MiDaS en cada punto (convertir a unidades relativas de escena)
  const DEPTH_SCALE = 200; // escala empírica para convertir MiDaS→cm de escena
  const pts3D_obs = pts2D.map((p, i) => {
    let z;
    if (depthMap) {
      const px = Math.min(Math.round(p.x), videoW - 1);
      const py = Math.min(Math.round(p.y), videoH - 1);
      z = depthMap[py * videoW + px] * DEPTH_SCALE;
    } else {
      z = 80; // fallback si no hay MiDaS
    }
    return [
      (p.x - cx) * z / focalLen,
      (p.y - cy) * z / focalLen,
      z,
    ];
  });

  // Vectores del pie en espacio de cámara
  const heelPt  = pts3D_obs[0];
  const toePt   = pts3D_obs[1];
  const anklePt = pts3D_obs[2];

  // Eje longitudinal del pie (heel → toe)
  const axisZ = normalize3([
    toePt[0] - heelPt[0],
    toePt[1] - heelPt[1],
    toePt[2] - heelPt[2],
  ]);

  // Eje vertical aproximado (hacia arriba desde el tobillo)
  const upApprox = normalize3([
    anklePt[0] - heelPt[0],
    anklePt[1] - heelPt[1],
    anklePt[2] - heelPt[2],
  ]);

  // Eje lateral (perpendicular al plano pie)
  const axisX = normalize3(cross3(upApprox, axisZ));

  // Recalcular eje vertical ortogonal
  const axisY = normalize3(cross3(axisZ, axisX));

  // Matriz de rotación 3x3 (columnas = ejes)
  const R = [
    [axisX[0], axisY[0], axisZ[0]],
    [axisX[1], axisY[1], axisZ[1]],
    [axisX[2], axisY[2], axisZ[2]],
  ];

  // Traslación = posición del talón en espacio de cámara
  const t = heelPt;

  return { R, t, pts3D_obs, scaleFactor };
}

// Convierte R,t a THREE.Matrix4 para aplicar al objeto GLB
function poseToMatrix4(pose, THREE) {
  if (!pose) return null;

  const { R, t } = pose;
  const m = new THREE.Matrix4();

  m.set(
    R[0][0], R[0][1], R[0][2], t[0],
    R[1][0], R[1][1], R[1][2], t[1],
    R[2][0], R[2][1], R[2][2], t[2],
    0,       0,       0,       1
  );

  return m;
}

export { solvePose, poseToMatrix4, FOOT_MODEL_3D };
