/**
 * 3D thin-plate spline with kernel U(r) = r.
 *
 *   f(p) = A·[1, p] + Σ_i W_i · |p − C_i|
 *
 * The control points C are fixed (the template), so the system matrix is
 * factored once; refitting to new targets is a single matrix product. The same
 * C / W / A are uploaded to the vertex shader, which evaluates f for every
 * anatomy vertex on the GPU.
 */
import { invertMatrix } from "../math/linalg";

export class ThinPlateSpline {
  readonly n: number;
  readonly controls: Float32Array; // n×3
  /** n×3 kernel weights */
  readonly weights: Float32Array;
  /** 4×3 affine part, rows: [translation, x-coef, y-coef, z-coef], each a 3-vector */
  readonly affine: Float32Array;
  private inv: Float64Array; // (n+4)² inverse of the system matrix
  private m: number;

  /**
   * @param controls flat xyz control points
   * @param lambda   regularisation, per control point (length n) or one value. Units of length:
   *                 larger values let the surface miss a control point by more.
   */
  constructor(controls: ArrayLike<number>, lambda: number | ArrayLike<number> = 0) {
    const n = Math.floor(controls.length / 3);
    this.n = n;
    this.controls = Float32Array.from(controls);
    this.weights = new Float32Array(n * 3);
    this.affine = new Float32Array(12);
    const m = n + 4;
    this.m = m;
    const L = new Float64Array(m * m);
    for (let i = 0; i < n; i++) {
      const xi = controls[3 * i], yi = controls[3 * i + 1], zi = controls[3 * i + 2];
      for (let j = 0; j < n; j++) {
        L[i * m + j] = Math.hypot(xi - controls[3 * j], yi - controls[3 * j + 1], zi - controls[3 * j + 2]);
      }
      // U(r) = r is conditionally negative definite, so regularisation is subtracted.
      L[i * m + i] -= typeof lambda === "number" ? lambda : lambda[i];
      L[i * m + n] = 1; L[i * m + n + 1] = xi; L[i * m + n + 2] = yi; L[i * m + n + 3] = zi;
      L[n * m + i] = 1; L[(n + 1) * m + i] = xi; L[(n + 2) * m + i] = yi; L[(n + 3) * m + i] = zi;
    }
    this.inv = invertMatrix(L, m);
    // start as the identity map
    this.affine.set([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  }

  /** Solve for the warp sending each control point to `targets` (flat xyz, length n×3). */
  fit(targets: ArrayLike<number>): void {
    const { n, m, inv } = this;
    for (let d = 0; d < 3; d++) {
      for (let r = 0; r < m; r++) {
        let acc = 0;
        const row = r * m;
        for (let j = 0; j < n; j++) acc += inv[row + j] * targets[3 * j + d];
        if (r < n) this.weights[3 * r + d] = acc;
        else this.affine[3 * (r - n) + d] = acc;
      }
    }
  }

  /** Evaluate the warp at p. */
  apply(p: ArrayLike<number>, out: number[] | Float32Array | Float64Array = [0, 0, 0]) {
    const { n, controls: C, weights: W, affine: A } = this;
    const x = p[0], y = p[1], z = p[2];
    let ox = A[0] + A[3] * x + A[6] * y + A[9] * z;
    let oy = A[1] + A[4] * x + A[7] * y + A[10] * z;
    let oz = A[2] + A[5] * x + A[8] * y + A[11] * z;
    for (let i = 0; i < n; i++) {
      const r = Math.hypot(x - C[3 * i], y - C[3 * i + 1], z - C[3 * i + 2]);
      ox += W[3 * i] * r; oy += W[3 * i + 1] * r; oz += W[3 * i + 2] * r;
    }
    out[0] = ox; out[1] = oy; out[2] = oz;
    return out;
  }

  applyArray(pts: ArrayLike<number>, out: Float32Array | Float64Array): void {
    const tmp = [0, 0, 0];
    const k = Math.floor(pts.length / 3);
    for (let i = 0; i < k; i++) {
      this.apply([pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]], tmp);
      out[3 * i] = tmp[0]; out[3 * i + 1] = tmp[1]; out[3 * i + 2] = tmp[2];
    }
  }
}

/**
 * Per-axis scale + translation (no rotation, no shear) fitted from src → dst.
 * Used to carry points the tracker cannot see (cranial vault, neck) along with
 * the measured face proportions.
 */
export interface AxisScale { s: [number, number, number]; t: [number, number, number] }

export function fitAxisScale(src: ArrayLike<number>, dst: ArrayLike<number>, idx: ArrayLike<number>): AxisScale {
  const s: [number, number, number] = [1, 1, 1];
  const t: [number, number, number] = [0, 0, 0];
  const n = idx.length;
  for (let d = 0; d < 3; d++) {
    let ma = 0, mb = 0;
    for (let k = 0; k < n; k++) { ma += src[3 * idx[k] + d]; mb += dst[3 * idx[k] + d]; }
    ma /= n; mb /= n;
    let saa = 0, sab = 0;
    for (let k = 0; k < n; k++) {
      const a = src[3 * idx[k] + d] - ma;
      saa += a * a; sab += a * (dst[3 * idx[k] + d] - mb);
    }
    s[d] = saa > 1e-9 ? sab / saa : 1;
    t[d] = mb - s[d] * ma;
  }
  return { s, t };
}
