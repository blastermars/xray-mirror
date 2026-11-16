// Affine registration between two releases of the same anatomy, using parts that exist in both.
// Correspondences are restricted to the SAME part (3.0 vertex <-> nearest 4.0 vertex of that part, and
// vice versa), and a 3x4 affine is re-solved by least squares each iteration (affine ICP).

class Grid {
  constructor(P, h) {
    this.P = P; this.h = h; this.cells = new Map();
    for (let i = 0; i < P.length / 3; i++) {
      const k = this.key(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
      let a = this.cells.get(k); if (!a) this.cells.set(k, a = []); a.push(i);
    }
  }
  key(x, y, z) { return Math.floor(x / this.h) + ',' + Math.floor(y / this.h) + ',' + Math.floor(z / this.h); }
  nearest(x, y, z) {
    const cx = Math.floor(x / this.h), cy = Math.floor(y / this.h), cz = Math.floor(z / this.h);
    let best = -1, bd = Infinity;
    for (let r = 0; r < 64; r++) {
      for (let i = cx - r; i <= cx + r; i++) for (let j = cy - r; j <= cy + r; j++) for (let k = cz - r; k <= cz + r; k++) {
        if (Math.max(Math.abs(i - cx), Math.abs(j - cy), Math.abs(k - cz)) !== r) continue;
        const a = this.cells.get(i + ',' + j + ',' + k); if (!a) continue;
        for (const v of a) { const d = (this.P[3 * v] - x) ** 2 + (this.P[3 * v + 1] - y) ** 2 + (this.P[3 * v + 2] - z) ** 2; if (d < bd) { bd = d; best = v; } }
      }
      if (best >= 0 && Math.sqrt(bd) <= r * this.h) break;
    }
    return [best, Math.sqrt(bd)];
  }
}

/** Nearest-vertex distances (same part) of parts[i].src to parts[i].dst, without any transform. */
export function residuals(parts) {
  return parts.map((p) => {
    const g = new Grid(p.dst, 2); const ds = [];
    for (let i = 0; i < p.src.length; i += 3) ds.push(g.nearest(p.src[i], p.src[i + 1], p.src[i + 2])[1]);
    ds.sort((a, b) => a - b);
    return { name: p.name, mean: ds.reduce((a, b) => a + b, 0) / ds.length, rms: Math.sqrt(ds.reduce((a, b) => a + b * b, 0) / ds.length), p95: ds[Math.floor(ds.length * 0.95)] };
  });
}

export function applyAffine(M, P) {
  const out = new Float64Array(P.length);
  for (let i = 0; i < P.length; i += 3) {
    const x = P[i], y = P[i + 1], z = P[i + 2];
    for (let k = 0; k < 3; k++) out[i + k] = M[k][0] * x + M[k][1] * y + M[k][2] * z + M[k][3];
  }
  return out;
}

function solve4(A, b) { // Gaussian elimination with partial pivoting
  const n = 4; const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  return M.map((r, i) => r[n] / r[i]);
}

function fitAffine(pairs /* [x,y,z, X,Y,Z] flat */) {
  // centre for conditioning
  const n = pairs.length / 6; const c = [0, 0, 0];
  for (let i = 0; i < pairs.length; i += 6) for (let k = 0; k < 3; k++) c[k] += pairs[i + k] / n;
  const AtA = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], Atb = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < pairs.length; i += 6) {
    const r = [pairs[i] - c[0], pairs[i + 1] - c[1], pairs[i + 2] - c[2], 1];
    for (let a = 0; a < 4; a++) { for (let b = 0; b < 4; b++) AtA[a][b] += r[a] * r[b]; for (let k = 0; k < 3; k++) Atb[k][a] += r[a] * pairs[i + 3 + k]; }
  }
  const rows = [0, 1, 2].map((k) => solve4(AtA, Atb[k]));
  // undo centring: y = L (x - c) + t  => translation t - L c
  return rows.map((r) => [r[0], r[1], r[2], r[3] - (r[0] * c[0] + r[1] * c[1] + r[2] * c[2])]);
}

/**
 * @param parts [{name, src: Float64Array (3.0 positions), dst: Float64Array (4.0 positions)}]
 * @returns {{M:number[][], perPart:{name, meanBefore, mean, rms, p95}[], rms:number}}
 */
export function affineICP(parts, iters = 40) {
  const grids = parts.map((p) => new Grid(p.dst, 2));
  // init: translation between overall centroids
  const cs = [0, 0, 0], cd = [0, 0, 0]; let ns = 0, nd = 0;
  for (const p of parts) {
    for (let i = 0; i < p.src.length; i += 3) { for (let k = 0; k < 3; k++) cs[k] += p.src[i + k]; ns++; }
    for (let i = 0; i < p.dst.length; i += 3) { for (let k = 0; k < 3; k++) cd[k] += p.dst[i + k]; nd++; }
  }
  let M = [[1, 0, 0, cd[0] / nd - cs[0] / ns], [0, 1, 0, cd[1] / nd - cs[1] / ns], [0, 0, 1, cd[2] / nd - cs[2] / ns]];
  const stats = (M) => parts.map((p, pi) => {
    const T = applyAffine(M, p.src); const ds = [];
    for (let i = 0; i < T.length; i += 3) ds.push(grids[pi].nearest(T[i], T[i + 1], T[i + 2])[1]);
    ds.sort((a, b) => a - b);
    return { name: p.name, mean: ds.reduce((a, b) => a + b, 0) / ds.length, rms: Math.sqrt(ds.reduce((a, b) => a + b * b, 0) / ds.length), p95: ds[Math.floor(ds.length * 0.95)] };
  });
  const before = stats([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]);
  for (let it = 0; it < iters; it++) {
    const pairs = [];
    parts.forEach((p, pi) => {
      const T = applyAffine(M, p.src);
      for (let i = 0; i < T.length; i += 3) { // 3.0 -> 4.0
        const [j] = grids[pi].nearest(T[i], T[i + 1], T[i + 2]);
        pairs.push(p.src[i], p.src[i + 1], p.src[i + 2], p.dst[3 * j], p.dst[3 * j + 1], p.dst[3 * j + 2]);
      }
      const g = new Grid(T, 2); // 4.0 -> transformed 3.0 (pairs expressed as src3 -> dst4)
      for (let j = 0; j < p.dst.length; j += 3) {
        const [i] = g.nearest(p.dst[j], p.dst[j + 1], p.dst[j + 2]);
        pairs.push(p.src[3 * i], p.src[3 * i + 1], p.src[3 * i + 2], p.dst[j], p.dst[j + 1], p.dst[j + 2]);
      }
    });
    M = fitAffine(pairs);
  }
  const after = stats(M);
  return { M, perPart: after.map((s, i) => ({ ...s, meanBefore: before[i].mean })) };
}
