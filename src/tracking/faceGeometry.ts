/**
 * Screen landmarks → metric camera-space landmarks + head pose.
 *
 * This is a port of MediaPipe's face-geometry pipeline (geometry_pipeline.cc)
 * with two deliberate changes:
 *   1. the vertical field of view is a parameter (MediaPipe hard-codes 63°),
 *   2. the Procrustes weights only use the rigid upper face, so opening the jaw
 *      or smiling does not tilt the head pose.
 *
 * Conventions: camera at the origin looking down −Z, +Y up, +X right, units cm
 * (same as three.js). The head frame is MediaPipe's canonical face frame: +Y up,
 * +Z out of the face, subject's right at −X.
 */
import { fitSimilarity, invertSimilarity, applySimilarityArray, type Similarity } from "../math/similarity";
import { axisAngleMat3, invertMatrix, mat3Mul } from "../math/linalg";

export const NUM_FACE = 468;
export const NUM_LANDMARKS = 478;

/**
 * Landmarks that stay put on the skull under expression: nose bridge and tip,
 * eye corners, alar base, forehead, temples and cheekbones. No jaw, no mouth.
 */
const RIGID: Array<[number, number]> = [
  [4, 0.07], [6, 0.06], [168, 0.06], [197, 0.05], [195, 0.04], [5, 0.04],
  [33, 0.06], [263, 0.06], [133, 0.07], [362, 0.07],
  [129, 0.05], [358, 0.05], [198, 0.03], [420, 0.03],
  [9, 0.03], [151, 0.015], [10, 0.015], [67, 0.012], [297, 0.012], [54, 0.012], [284, 0.012],
  [109, 0.01], [338, 0.01],
  [127, 0.012], [356, 0.012], [234, 0.012], [454, 0.012],
  [117, 0.012], [346, 0.012], [119, 0.015], [348, 0.015], [121, 0.02], [350, 0.02],
  [143, 0.015], [372, 0.015],
];
export const RIGID_IDX = Int32Array.from(RIGID.map((r) => r[0]));
export const RIGID_W = Float64Array.from(RIGID.map((r) => r[1]));

export interface FaceSolve {
  /** canonical/head frame → camera space */
  pose: Similarity;
  /** 478 landmarks in camera space, cm. Each projects exactly onto its 2D landmark. */
  metric: Float32Array;
  /** 478 landmarks expressed in the head frame (pose⁻¹ · metric). */
  local: Float32Array;
}

export class FaceGeometry {
  private screen = new Float64Array(NUM_LANDMARKS * 3);
  private work = new Float64Array(NUM_LANDMARKS * 3);

  /** @param reference 468×3 reference shape in the head frame (the canonical face model). */
  constructor(private reference: ArrayLike<number>) {}

  /**
   * @param lm      flat normalised landmarks [x, y, z] × 478 (x right, y down, both 0..1; z in units of image width)
   * @param aspect  frame width / height
   * @param vfovDeg vertical field of view of the camera, degrees
   */
  solve(lm: ArrayLike<number>, aspect: number, vfovDeg: number, out?: FaceSolve): FaceSolve {
    const n = Math.min(NUM_LANDMARKS, Math.floor(lm.length / 3));
    const near = 1;
    const hNear = 2 * near * Math.tan((vfovDeg * Math.PI) / 360);
    const wNear = hNear * aspect;
    const scr = this.screen, tmp = this.work;

    // Project onto the near plane. z uses the x scale (MediaPipe convention).
    let depthOffset = 0;
    const nFace = Math.min(n, NUM_FACE);
    for (let i = 0; i < n; i++) {
      scr[3 * i] = (lm[3 * i] - 0.5) * wNear;
      scr[3 * i + 1] = (0.5 - lm[3 * i + 1]) * hNear;
      scr[3 * i + 2] = lm[3 * i + 2] * wNear;
      if (i < nFace) depthOffset += scr[3 * i + 2];
    }
    depthOffset /= nFace;

    // Pass 1: scale from the orthographic landmarks (handedness flipped: z toward viewer).
    for (let i = 0; i < n; i++) {
      tmp[3 * i] = scr[3 * i]; tmp[3 * i + 1] = scr[3 * i + 1]; tmp[3 * i + 2] = -scr[3 * i + 2];
    }
    const s1 = fitSimilarity(this.reference, tmp, RIGID_IDX, RIGID_W).s;

    // Pass 2: unproject with the first scale, re-estimate.
    this.unproject(scr, tmp, n, depthOffset, s1, near);
    const s2 = fitSimilarity(this.reference, tmp, RIGID_IDX, RIGID_W).s;

    // Final: unproject with the total scale; solve the pose.
    const metric = out?.metric ?? new Float32Array(NUM_LANDMARKS * 3);
    this.unproject(scr, tmp, n, depthOffset, s1 * s2, near);
    const pose = fitSimilarity(this.reference, tmp, RIGID_IDX, RIGID_W);
    for (let i = 0; i < n * 3; i++) metric[i] = tmp[i];

    const local = out?.local ?? new Float32Array(NUM_LANDMARKS * 3);
    applySimilarityArray(invertSimilarity(pose), metric, local);
    return { pose, metric, local };
  }

  private unproject(scr: Float64Array, dst: Float64Array, n: number, depthOffset: number, scale: number, near: number) {
    for (let i = 0; i < n; i++) {
      const z = (scr[3 * i + 2] - depthOffset + near) / scale;
      dst[3 * i] = (scr[3 * i] * z) / near;
      dst[3 * i + 1] = (scr[3 * i + 1] * z) / near;
      dst[3 * i + 2] = -z;
    }
  }
}

/**
 * Refine a head pose by minimising the 2D reprojection error of a known 3D shape
 * (perspective-n-point, Gauss-Newton with damping; scale is fixed at 1).
 *
 * The tracker's 2D landmarks are accurate to a pixel or two, while its depth
 * values are only a rough guess, and the pose that comes out of them tilts by
 * several degrees once the head turns. Fitting the user's own rigid face shape
 * to the 2D points alone gives a much steadier rotation.
 *
 * @param shape head-frame points (flat xyz), e.g. the user's neutral shape
 * @param idx   which points to use, with weights `w`
 * @param lm    normalised landmarks [x, y, z] (z is ignored)
 * @param init  starting pose (its scale is folded into the starting depth)
 */
export function refinePose(
  shape: ArrayLike<number>, idx: ArrayLike<number>, w: ArrayLike<number>,
  lm: ArrayLike<number>, aspect: number, vfovDeg: number, init: Similarity, iterations = 8,
): Similarity {
  const th = Math.tan((vfovDeg * Math.PI) / 360);
  const n = idx.length;
  let R = Float64Array.from(init.R);
  // a pose with scale s at translation t looks the same as scale 1 at t / s
  let t = [init.t[0] / init.s, init.t[1] / init.s, init.t[2] / init.s];
  const obsA = new Float64Array(n), obsB = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    obsA[k] = (lm[3 * i] - 0.5) * 2 * th * aspect;
    obsB[k] = (0.5 - lm[3 * i + 1]) * 2 * th;
  }
  const cost = (Rm: ArrayLike<number>, tv: number[]): number => {
    let e = 0;
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      const X = shape[3 * i], Y = shape[3 * i + 1], Z = shape[3 * i + 2];
      const px = Rm[0] * X + Rm[1] * Y + Rm[2] * Z + tv[0];
      const py = Rm[3] * X + Rm[4] * Y + Rm[5] * Z + tv[1];
      const d = -(Rm[6] * X + Rm[7] * Y + Rm[8] * Z + tv[2]);
      if (d < 1) return Infinity;
      e += w[k] * ((px / d - obsA[k]) ** 2 + (py / d - obsB[k]) ** 2);
    }
    return e;
  };
  let lambda = 1e-3;
  let best = cost(R, t);
  if (!isFinite(best)) return { s: 1, R, t: [t[0], t[1], t[2]] };
  const H = new Float64Array(36), g = new Float64Array(6);
  for (let it = 0; it < iterations; it++) {
    H.fill(0); g.fill(0);
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      const X = shape[3 * i], Y = shape[3 * i + 1], Z = shape[3 * i + 2];
      // q = R·X (rotation part only), p = q + t
      const qx = R[0] * X + R[1] * Y + R[2] * Z, qy = R[3] * X + R[4] * Y + R[5] * Z, qz = R[6] * X + R[7] * Y + R[8] * Z;
      const px = qx + t[0], py = qy + t[1], d = -(qz + t[2]);
      const ra = px / d - obsA[k], rb = py / d - obsB[k];
      // d(a)/dp = [1/d, 0, px/d²], d(b)/dp = [0, 1/d, py/d²];  dp/dω = −[q]×,  dp/dt = I
      const ax = 1 / d, az = px / (d * d), by = 1 / d, bz = py / (d * d);
      // rows of the Jacobian for (ωx, ωy, ωz, tx, ty, tz)
      // with ω × q = (ωy qz − ωz qy, ωz qx − ωx qz, ωx qy − ωy qx)
      const Ja = [az * qy, ax * qz - az * qx, -ax * qy, ax, 0, az];
      const Jb = [-by * qz + bz * qy, -bz * qx, by * qx, 0, by, bz];
      const wk = w[k];
      for (let r = 0; r < 6; r++) {
        g[r] += wk * (Ja[r] * ra + Jb[r] * rb);
        for (let c = 0; c < 6; c++) H[r * 6 + c] += wk * (Ja[r] * Ja[c] + Jb[r] * Jb[c]);
      }
    }
    let improved = false;
    for (let attempt = 0; attempt < 4 && !improved; attempt++) {
      const A = Float64Array.from(H);
      for (let r = 0; r < 6; r++) A[r * 6 + r] *= 1 + lambda;
      let inv: Float64Array;
      try { inv = invertMatrix(A, 6); } catch { lambda *= 10; continue; }
      const dx = new Float64Array(6);
      for (let r = 0; r < 6; r++) for (let c = 0; c < 6; c++) dx[r] -= inv[r * 6 + c] * g[c];
      const ang = Math.hypot(dx[0], dx[1], dx[2]);
      const Rn = ang > 1e-12 ? mat3Mul(axisAngleMat3([dx[0], dx[1], dx[2]], ang), R) : R;
      const tn = [t[0] + dx[3], t[1] + dx[4], t[2] + dx[5]];
      const c = cost(Rn, tn);
      if (c < best) { R = Float64Array.from(Rn); t = tn; best = c; lambda = Math.max(1e-6, lambda * 0.3); improved = true; }
      else lambda *= 10;
    }
    if (!improved) break;
  }
  return { s: 1, R, t: [t[0], t[1], t[2]] };
}

/** Project a camera-space point to normalised image coordinates (x right, y down, 0..1). */
export function projectToImage(p: ArrayLike<number>, aspect: number, vfovDeg: number): [number, number] {
  const t = Math.tan((vfovDeg * Math.PI) / 360);
  const z = -p[2];
  return [0.5 + p[0] / (z * 2 * t * aspect), 0.5 - p[1] / (z * 2 * t)];
}

/** Sensible default vertical field of view for a webcam frame of this shape. */
export function defaultVfov(width: number, height: number): number {
  const a = width / height;
  if (a < 1) return 62; // portrait phone front camera
  if (a < 1.5) return 50; // 4:3
  return 45; // 16:9 laptop / desktop webcams (≈72–78° diagonal)
}
