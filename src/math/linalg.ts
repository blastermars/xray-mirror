/** Small dense linear-algebra helpers. Plain arrays, no dependencies. */

export type Vec3 = [number, number, number];

export const v3 = {
  sub: (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  add: (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  scale: (a: ArrayLike<number>, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a: ArrayLike<number>, b: ArrayLike<number>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a: ArrayLike<number>, b: ArrayLike<number>): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ],
  len: (a: ArrayLike<number>): number => Math.hypot(a[0], a[1], a[2]),
  norm: (a: ArrayLike<number>): Vec3 => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
  lerp: (a: ArrayLike<number>, b: ArrayLike<number>, t: number): Vec3 => [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ],
  dist: (a: ArrayLike<number>, b: ArrayLike<number>): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]),
  at: (arr: ArrayLike<number>, i: number): Vec3 => [arr[3 * i], arr[3 * i + 1], arr[3 * i + 2]],
};

/**
 * Jacobi eigen-decomposition of a symmetric n×n matrix (row-major, modified in place).
 * Returns eigenvalues and eigenvectors (columns of V, row-major n×n).
 */
export function jacobiEigen(A: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 64; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i * n + i];
  return { values, vectors: V };
}

/** In-place inversion of an n×n row-major matrix by Gauss-Jordan with partial pivoting. */
export function invertMatrix(A: Float64Array, n: number): Float64Array {
  const M = new Float64Array(A);
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  for (let c = 0; c < n; c++) {
    let piv = c;
    let best = Math.abs(M[c * n + c]);
    for (let r = c + 1; r < n; r++) {
      const v = Math.abs(M[r * n + c]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-14) throw new Error("invertMatrix: singular matrix");
    if (piv !== c) {
      for (let k = 0; k < n; k++) {
        let t = M[c * n + k]; M[c * n + k] = M[piv * n + k]; M[piv * n + k] = t;
        t = I[c * n + k]; I[c * n + k] = I[piv * n + k]; I[piv * n + k] = t;
      }
    }
    const d = 1 / M[c * n + c];
    for (let k = 0; k < n; k++) { M[c * n + k] *= d; I[c * n + k] *= d; }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r * n + c];
      if (f === 0) continue;
      for (let k = 0; k < n; k++) { M[r * n + k] -= f * M[c * n + k]; I[r * n + k] -= f * I[c * n + k]; }
    }
  }
  return I;
}

/** 3×3 row-major rotation from unit quaternion (w, x, y, z). */
export function quatToMat3(q: ArrayLike<number>): Float64Array {
  const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
  return new Float64Array([
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ]);
}

/** Rotation (row-major 3×3) about a unit axis by angle (radians). */
export function axisAngleMat3(axis: ArrayLike<number>, angle: number): Float64Array {
  const [x, y, z] = v3.norm(axis);
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return new Float64Array([
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ]);
}

export function mat3MulVec(R: ArrayLike<number>, v: ArrayLike<number>): Vec3 {
  return [
    R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
    R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
    R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
  ];
}

export function mat3Mul(A: ArrayLike<number>, B: ArrayLike<number>): Float64Array {
  const o = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      o[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
  return o;
}

export function mat3Transpose(A: ArrayLike<number>): Float64Array {
  return new Float64Array([A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]);
}

/** Shortest-arc rotation (row-major 3×3) taking unit vector a onto unit vector b. */
export function rotationBetween(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const an = v3.norm(a), bn = v3.norm(b);
  const c = v3.dot(an, bn);
  if (c > 1 - 1e-12) return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  if (c < -1 + 1e-9) {
    // 180°: any axis perpendicular to a
    const helper: Vec3 = Math.abs(an[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    return axisAngleMat3(v3.cross(an, helper), Math.PI);
  }
  const axis = v3.cross(an, bn);
  return axisAngleMat3(axis, Math.atan2(v3.len(axis), c));
}
