/** One Euro filter (Casiez et al. 2012): low lag when moving, steady when still. */

const TAU = 2 * Math.PI;
const alpha = (cutoff: number, dt: number) => {
  const r = TAU * cutoff * dt;
  return r / (r + 1);
};

export class OneEuro {
  private x: Float64Array | null = null;
  private dx: Float64Array;
  constructor(
    private dim: number,
    public minCutoff = 1.5,
    public beta = 0.05,
    public dCutoff = 1.0,
  ) {
    this.dx = new Float64Array(dim);
  }

  reset() { this.x = null; this.dx.fill(0); }

  /** @param dt seconds since the previous sample */
  filter(v: ArrayLike<number>, dt: number, out: number[] | Float32Array | Float64Array = new Array(this.dim)) {
    if (!this.x || !(dt > 0) || dt > 0.5) {
      this.x = Float64Array.from(v as ArrayLike<number>);
      this.dx.fill(0);
      for (let i = 0; i < this.dim; i++) out[i] = v[i];
      return out;
    }
    // One shared cutoff from the speed of the whole vector keeps directions coherent.
    const ad = alpha(this.dCutoff, dt);
    let speed2 = 0;
    for (let i = 0; i < this.dim; i++) {
      const d = (v[i] - this.x[i]) / dt;
      this.dx[i] += ad * (d - this.dx[i]);
      speed2 += this.dx[i] * this.dx[i];
    }
    const a = alpha(this.minCutoff + this.beta * Math.sqrt(speed2), dt);
    for (let i = 0; i < this.dim; i++) {
      this.x[i] += a * (v[i] - this.x[i]);
      out[i] = this.x[i];
    }
    return out;
  }
}

/** One Euro for unit quaternions (w,x,y,z): filters in the tangent space of the current estimate. */
export class OneEuroQuat {
  private q: [number, number, number, number] | null = null;
  private w = new Float64Array(3); // filtered angular velocity
  constructor(public minCutoff = 1.5, public beta = 0.6, public dCutoff = 1.0) {}

  reset() { this.q = null; this.w.fill(0); }

  filter(qIn: ArrayLike<number>, dt: number): [number, number, number, number] {
    let q: [number, number, number, number] = [qIn[0], qIn[1], qIn[2], qIn[3]];
    if (!this.q || !(dt > 0) || dt > 0.5) {
      this.q = q; this.w.fill(0);
      return [...q];
    }
    const p = this.q;
    if (p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3] < 0) q = [-q[0], -q[1], -q[2], -q[3]];
    // delta = q * conj(p); rotation vector = 2·vec(delta) for small angles
    const dw = q[0] * p[0] + q[1] * p[1] + q[2] * p[2] + q[3] * p[3];
    const dv = [
      -q[0] * p[1] + q[1] * p[0] - q[2] * p[3] + q[3] * p[2],
      -q[0] * p[2] + q[1] * p[3] + q[2] * p[0] - q[3] * p[1],
      -q[0] * p[3] - q[1] * p[2] + q[2] * p[1] + q[3] * p[0],
    ];
    const sinHalf = Math.hypot(dv[0], dv[1], dv[2]);
    const angle = 2 * Math.atan2(sinHalf, dw);
    const k = sinHalf > 1e-9 ? angle / sinHalf : 2;
    const rv = [dv[0] * k, dv[1] * k, dv[2] * k];
    const ad = alpha(this.dCutoff, dt);
    let speed2 = 0;
    for (let i = 0; i < 3; i++) {
      this.w[i] += ad * (rv[i] / dt - this.w[i]);
      speed2 += this.w[i] * this.w[i];
    }
    const a = alpha(this.minCutoff + this.beta * Math.sqrt(speed2), dt);
    // p ← exp(a·rv) * p
    const half = (a * angle) / 2;
    const s = sinHalf > 1e-9 ? Math.sin(half) / sinHalf : a;
    const ew = Math.cos(half), ex = dv[0] * s, ey = dv[1] * s, ez = dv[2] * s;
    const r: [number, number, number, number] = [
      ew * p[0] - ex * p[1] - ey * p[2] - ez * p[3],
      ew * p[1] + ex * p[0] + ey * p[3] - ez * p[2],
      ew * p[2] - ex * p[3] + ey * p[0] + ez * p[1],
      ew * p[3] + ex * p[2] - ey * p[1] + ez * p[0],
    ];
    const l = Math.hypot(r[0], r[1], r[2], r[3]) || 1;
    this.q = [r[0] / l, r[1] / l, r[2] / l, r[3] / l];
    return [...this.q];
  }
}
