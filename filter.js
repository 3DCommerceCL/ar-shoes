// OneEuroFilter — smoothing de landmarks para eliminar jitter
class OneEuroFilter {
  constructor(freq, minCutoff = 1.0, beta = 0.1, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }

  _alpha(cutoff) {
    const te = 1.0 / this.freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }

  filter(x, timestamp) {
    if (this.tPrev !== null) {
      this.freq = 1.0 / (timestamp - this.tPrev);
    }
    this.tPrev = timestamp;

    const dx = this.xPrev === null ? 0.0 : (x - this.xPrev) * this.freq;
    const alphaDx = this._alpha(this.dCutoff);
    const dxHat = alphaDx * dx + (1 - alphaDx) * this.dxPrev;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const alpha = this._alpha(cutoff);
    const xHat = this.xPrev === null ? x : alpha * x + (1 - alpha) * this.xPrev;

    this.xPrev = xHat;
    this.dxPrev = dxHat;
    return xHat;
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }
}

// Crea filtros para todos los landmarks del pie (X e Y por landmark)
function createLandmarkFilters(count, freq = 30) {
  return Array.from({ length: count }, () => ({
    x: new OneEuroFilter(freq, 1.0, 0.1, 1.0),
    y: new OneEuroFilter(freq, 1.0, 0.1, 1.0),
    z: new OneEuroFilter(freq, 0.5, 0.05, 1.0),
  }));
}

function applyFilters(filters, landmarks, timestamp) {
  return landmarks.map((lm, i) => ({
    x: filters[i].x.filter(lm.x, timestamp),
    y: filters[i].y.filter(lm.y, timestamp),
    z: filters[i].z.filter(lm.z ?? 0, timestamp),
    visibility: lm.visibility,
  }));
}

export { OneEuroFilter, createLandmarkFilters, applyFilters };
