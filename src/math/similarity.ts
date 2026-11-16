import { jacobiEigen, quatToMat3, mat3MulVec, mat3Transpose, type Vec3 } from "./linalg";

/** p' = s · R · p + t.  R is row-major 3×3. */
export interface Similarity {
  s: number;
  R: Float64Array;
  t: Vec3;
}

export const identitySimilarity = (): Similarity => ({
  s: 1,
  R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
  t: [0, 0, 0],
});

/**
 * Weighted least-squares similarity (Horn's quaternion method) mapping src → dst.
 * Points are flat xyz arrays; `idx` selects which points take part, `w` gives a
 * weight per selected point (same length as idx). With `withScale=false` the
 * result is a rigid transform.
 */
export function fitSimilarity(
  src: ArrayLike<number>,
  dst: ArrayLike<number>,
  idx?: ArrayLike<number> | null,
  w?: ArrayLike<number> | null,
  withScale = true,
): Similarity {
  const n = idx ? idx.length : Math.floor(src.length / 3);
  let W = 0;
  const ca = [0, 0, 0], cb = [0, 0, 0];
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    const wk = w ? w[k] : 1;
    W += wk;
    for (let d = 0; d < 3; d++) {
      ca[d] += wk * src[3 * i + d];
      cb[d] += wk * dst[3 * i + d];
    }
  }
  if (W <= 0) return identitySimilarity();
  for (let d = 0; d < 3; d++) { ca[d] /= W; cb[d] /= W; }

  // Cross-covariance S[a][b] = Σ w · a_a · b_b  (centred)
  const S = new Float64Array(9);
  let varA = 0;
  for (let k = 0; k < n; k++) {
    const i = idx ? idx[k] : k;
    const wk = w ? w[k] : 1;
    const ax = src[3 * i] - ca[0], ay = src[3 * i + 1] - ca[1], az = src[3 * i + 2] - ca[2];
    const bx = dst[3 * i] - cb[0], by = dst[3 * i + 1] - cb[1], bz = dst[3 * i + 2] - cb[2];
    S[0] += wk * ax * bx; S[1] += wk * ax * by; S[2] += wk * ax * bz;
    S[3] += wk * ay * bx; S[4] += wk * ay * by; S[5] += wk * ay * bz;
    S[6] += wk * az * bx; S[7] += wk * az * by; S[8] += wk * az * bz;
    varA += wk * (ax * ax + ay * ay + az * az);
  }
  const [Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz] = S;
  const N = new Float64Array([
    Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
    Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
    Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
    Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz,
  ]);
  const { values, vectors } = jacobiEigen(N, 4);
  let best = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[best]) best = i;
  const q = [vectors[best], vectors[4 + best], vectors[8 + best], vectors[12 + best]];
  const ql = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  const R = quatToMat3([q[0] / ql, q[1] / ql, q[2] / ql, q[3] / ql]);

  let s = 1;
  if (withScale && varA > 1e-12) {
    // s = Σ w b·(R a) / Σ w |a|²  = trace(Rᵀ·Sᵀ)… computed directly from S
    let num = 0;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) num += R[r * 3 + c] * S[c * 3 + r];
    s = num / varA;
  }
  const Rca = mat3MulVec(R, ca);
  return { s, R, t: [cb[0] - s * Rca[0], cb[1] - s * Rca[1], cb[2] - s * Rca[2]] };
}

export function applySimilarity(T: Similarity, p: ArrayLike<number>): Vec3 {
  const r = mat3MulVec(T.R, p);
  return [T.s * r[0] + T.t[0], T.s * r[1] + T.t[1], T.s * r[2] + T.t[2]];
}

export function invertSimilarity(T: Similarity): Similarity {
  const Rt = mat3Transpose(T.R);
  const s = 1 / T.s;
  const rt = mat3MulVec(Rt, T.t);
  return { s, R: Rt, t: [-s * rt[0], -s * rt[1], -s * rt[2]] };
}

/** Apply to a flat xyz array, writing into `out` (may alias `pts`). */
export function applySimilarityArray(T: Similarity, pts: ArrayLike<number>, out: Float32Array | Float64Array): void {
  const R = T.R, s = T.s;
  const n = Math.floor(pts.length / 3);
  for (let i = 0; i < n; i++) {
    const x = pts[3 * i], y = pts[3 * i + 1], z = pts[3 * i + 2];
    out[3 * i] = s * (R[0] * x + R[1] * y + R[2] * z) + T.t[0];
    out[3 * i + 1] = s * (R[3] * x + R[4] * y + R[5] * z) + T.t[1];
    out[3 * i + 2] = s * (R[6] * x + R[7] * y + R[8] * z) + T.t[2];
  }
}

/** Column-major 4×4 (three.js layout). */
export function similarityToMat4(T: Similarity, out: Float32Array | Float64Array | number[] = new Float64Array(16)) {
  const R = T.R, s = T.s;
  out[0] = s * R[0]; out[1] = s * R[3]; out[2] = s * R[6]; out[3] = 0;
  out[4] = s * R[1]; out[5] = s * R[4]; out[6] = s * R[7]; out[7] = 0;
  out[8] = s * R[2]; out[9] = s * R[5]; out[10] = s * R[8]; out[11] = 0;
  out[12] = T.t[0]; out[13] = T.t[1]; out[14] = T.t[2]; out[15] = 1;
  return out;
}

/** Unit quaternion (w,x,y,z) from a row-major rotation matrix. */
export function mat3ToQuat(R: ArrayLike<number>): [number, number, number, number] {
  const tr = R[0] + R[4] + R[8];
  let w: number, x: number, y: number, z: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = s / 4; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    w = (R[7] - R[5]) / s; x = s / 4; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = s / 4; z = (R[5] + R[7]) / s;
  } else {
    const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = s / 4;
  }
  return [w, x, y, z];
}
