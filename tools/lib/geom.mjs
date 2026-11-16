// Geometry helpers for indexed triangle meshes: {positions: Float64Array|Float32Array (xyz), indices: Uint32Array}.

export function triCount(m) { return m.indices.length / 3; }
export function vertCount(m) { return m.positions.length / 3; }

/** Drop unreferenced vertices; keeps vertex order of first reference. */
export function compact(m) {
  const n = m.positions.length / 3;
  const remap = new Int32Array(n).fill(-1);
  const pos = [];
  const idx = new Uint32Array(m.indices.length);
  for (let i = 0; i < m.indices.length; i++) {
    const v = m.indices[i];
    if (remap[v] < 0) {
      remap[v] = pos.length / 3;
      pos.push(m.positions[3 * v], m.positions[3 * v + 1], m.positions[3 * v + 2]);
    }
    idx[i] = remap[v];
  }
  return { positions: Float64Array.from(pos), indices: idx };
}

/**
 * Concatenate meshes WITHOUT welding across them (separate source elements stay separate closed shells;
 * welding coincident vertices of touching elements would create non-manifold edges).
 */
export function merge(meshes) {
  let nv = 0, ni = 0;
  for (const m of meshes) { nv += m.positions.length; ni += m.indices.length; }
  const positions = new Float64Array(nv);
  const indices = new Uint32Array(ni);
  let ov = 0, oi = 0;
  for (const m of meshes) {
    positions.set(m.positions, ov);
    const base = ov / 3;
    for (let i = 0; i < m.indices.length; i++) indices[oi + i] = m.indices[i] + base;
    ov += m.positions.length; oi += m.indices.length;
  }
  return { positions, indices };
}

/** Weld bit-identical positions (does not touch coordinates). */
export function weld(m) {
  const map = new Map();
  const n = m.positions.length / 3;
  const remap = new Uint32Array(n);
  const pos = [];
  for (let i = 0; i < n; i++) {
    const k = m.positions[3 * i] + ' ' + m.positions[3 * i + 1] + ' ' + m.positions[3 * i + 2];
    let j = map.get(k);
    if (j === undefined) { j = pos.length / 3; map.set(k, j); pos.push(m.positions[3 * i], m.positions[3 * i + 1], m.positions[3 * i + 2]); }
    remap[i] = j;
  }
  const idx = [];
  for (let t = 0; t < m.indices.length; t += 3) {
    const a = remap[m.indices[t]], b = remap[m.indices[t + 1]], c = remap[m.indices[t + 2]];
    if (a === b || b === c || a === c) continue;
    idx.push(a, b, c);
  }
  return { positions: Float64Array.from(pos), indices: Uint32Array.from(idx) };
}

/** Edge statistics: every undirected edge counted by number of incident triangles. */
export function edgeStats(m) {
  const n = m.positions.length / 3;
  const counts = new Map();
  const I = m.indices;
  for (let t = 0; t < I.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = I[t + e], b = I[t + ((e + 1) % 3)];
      const k = a < b ? a * n + b : b * n + a;
      counts.set(k, (counts.get(k) || 0) + 1);
    }
  }
  let boundary = 0, nonManifold = 0, edges = 0;
  for (const c of counts.values()) { edges++; if (c === 1) boundary++; else if (c > 2) nonManifold++; }
  return { edges, boundaryEdges: boundary, nonManifoldEdges: nonManifold, closed: boundary === 0 && nonManifold === 0 };
}

/** Connected components over shared vertices. Returns {count, triComp: Int32Array, sizes:number[]} */
export function components(m) {
  const n = m.positions.length / 3;
  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const I = m.indices;
  for (let t = 0; t < I.length; t += 3) {
    const a = find(I[t]), b = find(I[t + 1]), c = find(I[t + 2]);
    if (a !== b) parent[a] = b;
    const b2 = find(b);
    if (find(c) !== b2) parent[find(c)] = b2;
  }
  const label = new Map();
  const triComp = new Int32Array(I.length / 3);
  const sizes = [];
  for (let t = 0; t < I.length; t += 3) {
    const r = find(I[t]);
    let l = label.get(r);
    if (l === undefined) { l = sizes.length; label.set(r, l); sizes.push(0); }
    triComp[t / 3] = l; sizes[l]++;
  }
  return { count: sizes.length, triComp, sizes };
}

export function subsetTriangles(m, keep /* (triIndex)=>bool */) {
  const idx = [];
  for (let t = 0; t < m.indices.length; t += 3) if (keep(t / 3)) idx.push(m.indices[t], m.indices[t + 1], m.indices[t + 2]);
  return compact({ positions: m.positions, indices: Uint32Array.from(idx) });
}

export function bbox(m) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  const P = m.positions;
  for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { const v = P[i + k]; if (v < min[k]) min[k] = v; if (v > max[k]) max[k] = v; }
  return { min, max };
}

function triGeom(P, a, b, c) {
  const ax = P[3 * a], ay = P[3 * a + 1], az = P[3 * a + 2];
  const bx = P[3 * b], by = P[3 * b + 1], bz = P[3 * b + 2];
  const cx = P[3 * c], cy = P[3 * c + 1], cz = P[3 * c + 2];
  const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  return { area: 0.5 * Math.hypot(nx, ny, nz), a: [ax, ay, az], b: [bx, by, bz], c: [cx, cy, cz] };
}

export function surfaceArea(m) {
  let A = 0; const I = m.indices;
  for (let t = 0; t < I.length; t += 3) A += triGeom(m.positions, I[t], I[t + 1], I[t + 2]).area;
  return A;
}

/** Signed volume (positive when triangles are wound counter-clockwise seen from outside). */
export function signedVolume(m) {
  let V = 0; const I = m.indices, P = m.positions;
  // use local origin for numerical stability
  const o = [P[0], P[1], P[2]];
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const ax = P[a] - o[0], ay = P[a + 1] - o[1], az = P[a + 2] - o[2];
    const bx = P[b] - o[0], by = P[b + 1] - o[1], bz = P[b + 2] - o[2];
    const cx = P[c] - o[0], cy = P[c + 1] - o[1], cz = P[c + 2] - o[2];
    V += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
  }
  return V;
}

/** Area-weighted surface centroid and principal axes of the surface (uniform area distribution). */
export function centroidAndPCA(m) {
  const I = m.indices, P = m.positions;
  let A = 0; const s1 = [0, 0, 0];
  const tris = [];
  for (let t = 0; t < I.length; t += 3) {
    const g = triGeom(P, I[t], I[t + 1], I[t + 2]);
    A += g.area;
    for (let k = 0; k < 3; k++) s1[k] += g.area * (g.a[k] + g.b[k] + g.c[k]) / 3;
    tris.push(g);
  }
  const c = s1.map((v) => v / A);
  // second moment about centroid: for a triangle, E[x x^T] = (1/12)(sum vi vi^T + s s^T), s = sum vi
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const g of tris) {
    const vs = [g.a, g.b, g.c].map((v) => [v[0] - c[0], v[1] - c[1], v[2] - c[2]]);
    const s = [vs[0][0] + vs[1][0] + vs[2][0], vs[0][1] + vs[1][1] + vs[2][1], vs[0][2] + vs[1][2] + vs[2][2]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      let acc = s[i] * s[j];
      for (const v of vs) acc += v[i] * v[j];
      C[i][j] += g.area * acc / 12;
    }
  }
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] /= A;
  const { values, vectors } = jacobiEigen(C);
  const order = [0, 1, 2].sort((x, y) => values[y] - values[x]);
  const axes = order.map((k) => {
    const v = [vectors[0][k], vectors[1][k], vectors[2][k]];
    // deterministic sign: largest-magnitude component positive
    let mi = 0; for (let q = 1; q < 3; q++) if (Math.abs(v[q]) > Math.abs(v[mi])) mi = q;
    return v[mi] < 0 ? v.map((x) => -x) : v;
  });
  const pmin = [Infinity, Infinity, Infinity], pmax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) {
    const d = [P[i] - c[0], P[i + 1] - c[1], P[i + 2] - c[2]];
    for (let k = 0; k < 3; k++) {
      const p = d[0] * axes[k][0] + d[1] * axes[k][1] + d[2] * axes[k][2];
      if (p < pmin[k]) pmin[k] = p; if (p > pmax[k]) pmax[k] = p;
    }
  }
  return { area: A, centroid: c, pca: { axes, variances: order.map((k) => values[k]), min: pmin, max: pmax } };
}

function jacobiEigen(S) {
  const a = S.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) off += a[p][q] * a[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
      for (let k = 0; k < 3; k++) { // rotate columns
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = cs * akp - sn * akq; a[k][q] = sn * akp + cs * akq;
      }
      for (let k = 0; k < 3; k++) { // rotate rows
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = cs * apk - sn * aqk; a[q][k] = sn * apk + cs * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = cs * vkp - sn * vkq; v[k][q] = sn * vkp + cs * vkq;
      }
    }
  }
  return { values: [a[0][0], a[1][1], a[2][2]], vectors: v };
}

/**
 * Clip mesh to the half-space axis-coordinate >= c0 (exact triangle clipping; new vertices are shared
 * along cut edges so the result stays manifold), then cap each boundary loop lying in the cut plane
 * with a triangle fan to the loop centroid.
 * @returns {{mesh, loops: {vertices:number, centroid:number[], perimeter:number, area:number}[]}}
 */
export function clipAndCap(m, axis, c0) {
  const P = m.positions, I = m.indices;
  const n = P.length / 3;
  const pos = Array.from(P);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = P[3 * i + axis] - c0;
  const onPlane = new Set();
  const edgeCut = new Map();
  const cutVertex = (a, b) => {
    const k = a < b ? a * n + b : b * n + a;
    let v = edgeCut.get(k);
    if (v !== undefined) return v;
    const t = d[a] / (d[a] - d[b]);
    const p = [0, 1, 2].map((q) => P[3 * a + q] + t * (P[3 * b + q] - P[3 * a + q]));
    p[axis] = c0;
    v = pos.length / 3; pos.push(...p); onPlane.add(v); edgeCut.set(k, v);
    return v;
  };
  for (let i = 0; i < n; i++) if (d[i] === 0) onPlane.add(i);
  const out = [];
  for (let t = 0; t < I.length; t += 3) {
    const tri = [I[t], I[t + 1], I[t + 2]];
    const ins = tri.map((v) => d[v] >= 0);
    if (ins[0] && ins[1] && ins[2]) { out.push(...tri); continue; }
    if (!ins[0] && !ins[1] && !ins[2]) continue;
    const poly = [];
    for (let e = 0; e < 3; e++) {
      const a = tri[e], b = tri[(e + 1) % 3];
      const ia = d[a] >= 0, ib = d[b] >= 0;
      if (ia) poly.push(a);
      if (ia !== ib && d[a] !== 0 && d[b] !== 0) poly.push(cutVertex(a, b));
    }
    for (let k = 1; k + 1 < poly.length; k++) {
      const tr = [poly[0], poly[k], poly[k + 1]];
      if (tr[0] !== tr[1] && tr[1] !== tr[2] && tr[0] !== tr[2]) out.push(...tr);
    }
  }
  // boundary half-edges in the cut plane
  const nv = pos.length / 3;
  const ec = new Map();
  for (let t = 0; t < out.length; t += 3) for (let e = 0; e < 3; e++) {
    const a = out[t + e], b = out[t + ((e + 1) % 3)];
    const k = a < b ? a * nv + b : b * nv + a;
    const r = ec.get(k);
    if (r) r.count++; else ec.set(k, { count: 1, a, b });
  }
  const next = new Map(); // a -> b for boundary half-edge a->b on plane
  let skipped = 0;
  for (const r of ec.values()) {
    if (r.count !== 1) continue;
    if (!onPlane.has(r.a) || !onPlane.has(r.b)) continue;
    if (next.has(r.a)) { skipped++; continue; }
    next.set(r.a, r.b);
  }
  const loops = [];
  const used = new Set();
  for (const start of next.keys()) {
    if (used.has(start)) continue;
    const loop = [];
    let v = start, ok = false;
    while (!used.has(v)) {
      used.add(v); loop.push(v);
      const w = next.get(v);
      if (w === undefined) break;
      if (w === start) { ok = true; break; }
      v = w;
    }
    if (!ok || loop.length < 3) { skipped += loop.length; continue; }
    const c = [0, 0, 0];
    for (const u of loop) for (let q = 0; q < 3; q++) c[q] += pos[3 * u + q] / loop.length;
    c[axis] = c0;
    const ci = pos.length / 3; pos.push(...c);
    let per = 0, area = 0;
    for (let k = 0; k < loop.length; k++) {
      const a = loop[k], b = loop[(k + 1) % loop.length];
      out.push(b, a, ci); // reverse of boundary half-edge a->b
      per += Math.hypot(pos[3 * a] - pos[3 * b], pos[3 * a + 1] - pos[3 * b + 1], pos[3 * a + 2] - pos[3 * b + 2]);
      const g = triGeom(pos, b, a, ci); area += g.area;
    }
    loops.push({ vertices: loop.length, centroid: c, perimeter: per, area });
  }
  return { mesh: compact({ positions: Float64Array.from(pos), indices: Uint32Array.from(out) }), loops, skipped };
}

/** Remove connected components with fewer than `minTris` triangles or |volume| < minVol (mm^3). */
export function dropTinyComponents(m, minTris = 4, minVol = 1e-3) {
  const c = components(m);
  if (c.count <= 1) return { mesh: m, dropped: 0, droppedTris: 0 };
  const parts = Array.from({ length: c.count }, () => []);
  for (let t = 0; t < c.triComp.length; t++) parts[c.triComp[t]].push(m.indices[3 * t], m.indices[3 * t + 1], m.indices[3 * t + 2]);
  const keep = new Uint8Array(c.count);
  let dropped = 0, droppedTris = 0;
  for (let k = 0; k < c.count; k++) {
    const vol = Math.abs(signedVolume({ positions: m.positions, indices: Uint32Array.from(parts[k]) }));
    if (c.sizes[k] < minTris || vol < minVol) { dropped++; droppedTris += c.sizes[k]; } else keep[k] = 1;
  }
  if (!dropped) return { mesh: m, dropped: 0, droppedTris: 0 };
  return { mesh: subsetTriangles(m, (t) => keep[c.triComp[t]] === 1), dropped, droppedTris };
}

/** Unit icosphere (outward CCW winding). level 3 -> 642 vertices / 1280 triangles. */
export function icosphere(level) {
  const t = (1 + Math.sqrt(5)) / 2;
  let v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    .map((p) => { const l = Math.hypot(...p); return p.map((x) => x / l); });
  let f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  for (let l = 0; l < level; l++) {
    const cache = new Map(); const nf = [];
    const mid = (a, b) => {
      const k = a < b ? a + ',' + b : b + ',' + a;
      if (cache.has(k)) return cache.get(k);
      const p = [0, 1, 2].map((q) => (v[a][q] + v[b][q]) / 2); const len = Math.hypot(...p);
      v.push(p.map((x) => x / len)); cache.set(k, v.length - 1); return v.length - 1;
    };
    for (const [a, b, c] of f) { const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a); nf.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    f = nf;
  }
  return { dirs: v, faces: f };
}

/**
 * Star-shaped outer envelope: for each icosphere direction from `center`, take the FARTHEST intersection with
 * any triangle of `targets`. Produces a closed, manifold sphere-topology mesh whose vertices lie on the
 * original surfaces (same coordinate frame; nothing is moved/scaled).
 */
export function radialEnvelope(targets, center, level = 3) {
  const { dirs, faces } = icosphere(level);
  const tris = [];
  for (const m of targets) { const P = m.positions, I = m.indices; for (let t = 0; t < I.length; t += 3) tris.push(I[t] * 3, I[t + 1] * 3, I[t + 2] * 3, P); }
  const pos = new Float64Array(dirs.length * 3);
  let misses = 0; const rs = new Float64Array(dirs.length);
  for (let i = 0; i < dirs.length; i++) {
    const d = dirs[i]; let best = -1;
    for (let k = 0; k < tris.length; k += 4) {
      const P = tris[k + 3], a = tris[k], b = tris[k + 1], c = tris[k + 2];
      const e1x = P[b] - P[a], e1y = P[b + 1] - P[a + 1], e1z = P[b + 2] - P[a + 2];
      const e2x = P[c] - P[a], e2y = P[c + 1] - P[a + 1], e2z = P[c + 2] - P[a + 2];
      const px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-12) continue;
      const inv = 1 / det;
      const tx = center[0] - P[a], ty = center[1] - P[a + 1], tz = center[2] - P[a + 2];
      const u = (tx * px + ty * py + tz * pz) * inv; if (u < 0 || u > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const w = (d[0] * qx + d[1] * qy + d[2] * qz) * inv; if (w < 0 || u + w > 1) continue;
      const tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (tt > best) best = tt;
    }
    if (best < 0) misses++;
    rs[i] = best;
  }
  if (misses) { // fill misses with mean radius of hit directions (reported by caller)
    let s = 0, n = 0; for (const r of rs) if (r > 0) { s += r; n++; }
    for (let i = 0; i < rs.length; i++) if (rs[i] < 0) rs[i] = s / n;
  }
  for (let i = 0; i < dirs.length; i++) for (let q = 0; q < 3; q++) pos[3 * i + q] = center[q] + rs[i] * dirs[i][q];
  return { mesh: { positions: pos, indices: Uint32Array.from(faces.flat()) }, misses, rmin: Math.min(...rs), rmax: Math.max(...rs) };
}

/**
 * Resolve non-manifold edges/vertices produced when edge collapses pinch two sheets together:
 * half-edges around an edge shared by >2 triangles are paired geometrically (sorted by angle around the
 * edge; each triangle is paired with its angular neighbour on its interior side), then every vertex is
 * split into one copy per connected fan of incident triangles. Positions are duplicated, never moved.
 * @returns {{mesh, splitVertices:number, fixedEdges:number, unresolvedEdges:number}}
 */
export function makeManifold(m) {
  const I = m.indices, P = m.positions; const nT = I.length / 3, nV = P.length / 3;
  const groups = new Map();
  for (let he = 0; he < I.length; he++) {
    const a = I[he], b = I[3 * Math.floor(he / 3) + ((he % 3) + 1) % 3];
    const k = a < b ? a * nV + b : b * nV + a;
    let g = groups.get(k); if (!g) groups.set(k, g = []); g.push(he);
  }
  const twin = new Int32Array(I.length).fill(-1);
  const heFrom = (he) => I[he], heTo = (he) => I[3 * Math.floor(he / 3) + ((he % 3) + 1) % 3];
  const third = (he) => I[3 * Math.floor(he / 3) + ((he % 3) + 2) % 3];
  let fixedEdges = 0, unresolvedEdges = 0;
  for (const g of groups.values()) {
    if (g.length === 2) { if (heFrom(g[0]) === heTo(g[1])) { twin[g[0]] = g[1]; twin[g[1]] = g[0]; } continue; }
    if (g.length < 2) continue;
    const a = Math.min(heFrom(g[0]), heTo(g[0])), b = Math.max(heFrom(g[0]), heTo(g[0]));
    const e = norm3([P[3 * b] - P[3 * a], P[3 * b + 1] - P[3 * a + 1], P[3 * b + 2] - P[3 * a + 2]]);
    const u = norm3(Math.abs(e[0]) < 0.9 ? cross3(e, [1, 0, 0]) : cross3(e, [0, 1, 0])), w = cross3(e, u);
    const items = g.map((he) => {
      const p = third(he); const d = [P[3 * p] - P[3 * a], P[3 * p + 1] - P[3 * a + 1], P[3 * p + 2] - P[3 * a + 2]];
      const th = Math.atan2(dot3(d, w), dot3(d, u));
      // outward normal of the triangle (CCW): n = (to-from) x (p-from); interior is -n
      const f = heFrom(he), t = heTo(he);
      const n = cross3([P[3 * t] - P[3 * f], P[3 * t + 1] - P[3 * f + 1], P[3 * t + 2] - P[3 * f + 2]], [P[3 * p] - P[3 * f], P[3 * p + 1] - P[3 * f + 1], P[3 * p + 2] - P[3 * f + 2]]);
      const tangent = [-Math.sin(th) * u[0] + Math.cos(th) * w[0], -Math.sin(th) * u[1] + Math.cos(th) * w[1], -Math.sin(th) * u[2] + Math.cos(th) * w[2]];
      return { he, th, interiorPlus: dot3(n, tangent) < 0 };
    }).sort((x, y) => x.th - y.th);
    let ok = true; const pairs = [];
    for (let i = 0; i < items.length; i++) if (items[i].interiorPlus) {
      const j = (i + 1) % items.length;
      if (items[j].interiorPlus || heFrom(items[i].he) !== heTo(items[j].he)) { ok = false; break; }
      pairs.push([items[i].he, items[j].he]);
    }
    if (ok && pairs.length * 2 === items.length) { for (const [x, y] of pairs) { twin[x] = y; twin[y] = x; } fixedEdges++; } else unresolvedEdges++;
  }
  // split vertices into fans: union corners connected through twinned half-edges incident to the vertex
  const parent = new Int32Array(I.length); for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const union = (x, y) => { x = find(x); y = find(y); if (x !== y) parent[x] = y; };
  for (let he = 0; he < I.length; he++) {
    const tw = twin[he]; if (tw < 0) continue;
    // he: from->to in tri T; tw: to->from in tri T'. corner of `from` in T is he; corner of `from` in T' is next(tw)
    const nextTw = 3 * Math.floor(tw / 3) + ((tw % 3) + 1) % 3;
    union(he, nextTw);
  }
  const newIndex = new Map(); const vertOwner = new Int32Array(nV).fill(-1);
  const pos = Array.from(P); const idx = new Uint32Array(I.length); let splitVertices = 0;
  for (let c = 0; c < I.length; c++) {
    const v = I[c], r = find(c);
    let nv = newIndex.get(r);
    if (nv === undefined) {
      if (vertOwner[v] < 0) { vertOwner[v] = r; nv = v; } else { nv = pos.length / 3; pos.push(P[3 * v], P[3 * v + 1], P[3 * v + 2]); splitVertices++; }
      newIndex.set(r, nv);
    }
    idx[c] = nv;
  }
  return { mesh: compact({ positions: Float64Array.from(pos), indices: idx }), splitVertices, fixedEdges, unresolvedEdges };
}
function cross3(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm3(a) { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; }

/** Count pairs of triangles that intersect but share no vertex position (uniform-grid broad phase). */
export function selfIntersections(m) {
  const P = m.positions, I = m.indices, nT = I.length / 3;
  if (nT < 2) return 0;
  let el = 0; for (let t = 0; t < I.length; t += 3) el += Math.hypot(P[3 * I[t]] - P[3 * I[t + 1]], P[3 * I[t] + 1] - P[3 * I[t + 1] + 1], P[3 * I[t] + 2] - P[3 * I[t + 1] + 2]);
  const h = Math.max(1e-6, 2 * el / nT);
  const key = (P2) => P2[0] + ' ' + P2[1] + ' ' + P2[2];
  const vid = new Map(); const pid = new Int32Array(P.length / 3);
  for (let v = 0; v < pid.length; v++) { const k = key([P[3 * v], P[3 * v + 1], P[3 * v + 2]]); if (!vid.has(k)) vid.set(k, vid.size); pid[v] = vid.get(k); }
  const grid = new Map();
  const tb = new Float64Array(nT * 6);
  for (let t = 0; t < nT; t++) {
    for (let k = 0; k < 3; k++) { tb[6 * t + k] = Infinity; tb[6 * t + 3 + k] = -Infinity; }
    for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) { const v = P[3 * I[3 * t + c] + k]; if (v < tb[6 * t + k]) tb[6 * t + k] = v; if (v > tb[6 * t + 3 + k]) tb[6 * t + 3 + k] = v; }
    const lo = [0, 1, 2].map((k) => Math.floor(tb[6 * t + k] / h)), hi = [0, 1, 2].map((k) => Math.floor(tb[6 * t + 3 + k] / h));
    for (let x = lo[0]; x <= hi[0]; x++) for (let y = lo[1]; y <= hi[1]; y++) for (let z = lo[2]; z <= hi[2]; z++) {
      const k = x + ',' + y + ',' + z; let a = grid.get(k); if (!a) grid.set(k, a = []); a.push(t);
    }
  }
  const tested = new Set(); let count = 0;
  const V = (t, c) => { const i = 3 * I[3 * t + c]; return [P[i], P[i + 1], P[i + 2]]; };
  for (const cell of grid.values()) {
    for (let i = 0; i < cell.length; i++) for (let j = i + 1; j < cell.length; j++) {
      let a = cell[i], b = cell[j]; if (a > b) [a, b] = [b, a];
      const pk = a * nT + b; if (tested.has(pk)) continue; tested.add(pk);
      let overlap = true; for (let k = 0; k < 3; k++) if (tb[6 * a + k] > tb[6 * b + 3 + k] || tb[6 * b + k] > tb[6 * a + 3 + k]) { overlap = false; break; }
      if (!overlap) continue;
      const sa = [pid[I[3 * a]], pid[I[3 * a + 1]], pid[I[3 * a + 2]]];
      if (sa.includes(pid[I[3 * b]]) || sa.includes(pid[I[3 * b + 1]]) || sa.includes(pid[I[3 * b + 2]])) continue;
      const A = [V(a, 0), V(a, 1), V(a, 2)], B = [V(b, 0), V(b, 1), V(b, 2)];
      if (triTri(A, B)) count++;
    }
  }
  return count;
}
function segTri(p, q, T) {
  const d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
  const e1 = [T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]], e2 = [T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]];
  const pv = cross3(d, e2); const det = dot3(e1, pv);
  if (Math.abs(det) < 1e-14) return false;
  const inv = 1 / det; const tv = [p[0] - T[0][0], p[1] - T[0][1], p[2] - T[0][2]];
  const u = dot3(tv, pv) * inv; if (u <= 0 || u >= 1) return false;
  const qv = cross3(tv, e1); const v = dot3(d, qv) * inv; if (v <= 0 || u + v >= 1) return false;
  const t = dot3(e2, qv) * inv; return t > 0 && t < 1;
}
function triTri(A, B) {
  for (let e = 0; e < 3; e++) { if (segTri(A[e], A[(e + 1) % 3], B)) return true; if (segTri(B[e], B[(e + 1) % 3], A)) return true; }
  return false;
}

/** Remove duplicated triangles: back-to-back pairs (same vertices, opposite winding = zero-volume fins) are
 * removed entirely; same-winding duplicates keep one copy. */
export function removeDuplicateTriangles(m) {
  const I = m.indices; const groups = new Map();
  for (let t = 0; t < I.length; t += 3) {
    const k = [I[t], I[t + 1], I[t + 2]].sort((a, b) => a - b).join(',');
    let g = groups.get(k); if (!g) groups.set(k, g = []); g.push(t / 3);
  }
  const drop = new Set(); let removed = 0;
  const parity = (t) => { const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2]; return (a < b) + (b < c) + (c < a) === 2 ? 1 : -1; }; // cyclic orientation class
  for (const g of groups.values()) {
    if (g.length < 2) continue;
    const pos = g.filter((t) => parity(t) > 0), neg = g.filter((t) => parity(t) < 0);
    const pairs = Math.min(pos.length, neg.length);
    for (let i = 0; i < pairs; i++) { drop.add(pos[i]); drop.add(neg[i]); }
    for (const rest of [pos.slice(pairs), neg.slice(pairs)]) for (let i = 1; i < rest.length; i++) drop.add(rest[i]);
  }
  removed = drop.size;
  if (!removed) return { mesh: m, removed: 0 };
  return { mesh: subsetTriangles(m, (t) => !drop.has(t)), removed };
}

/** Clip to the half-space sign*(p[axis]-c0) >= 0 with exact triangle clipping (shared cut vertices); no capping. */
export function clipHalfSpace(m, axis, c0, sign = 1) {
  const P = m.positions, I = m.indices, n = P.length / 3;
  const pos = Array.from(P); const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = sign * (P[3 * i + axis] - c0);
  const edgeCut = new Map();
  const cutVertex = (a, b) => {
    const k = a < b ? a * n + b : b * n + a; let v = edgeCut.get(k);
    if (v !== undefined) return v;
    const t = d[a] / (d[a] - d[b]);
    const p = [0, 1, 2].map((q) => P[3 * a + q] + t * (P[3 * b + q] - P[3 * a + q])); p[axis] = c0;
    v = pos.length / 3; pos.push(...p); edgeCut.set(k, v); return v;
  };
  const out = [];
  for (let t = 0; t < I.length; t += 3) {
    const tri = [I[t], I[t + 1], I[t + 2]]; const ins = tri.map((v) => d[v] >= 0);
    if (ins[0] && ins[1] && ins[2]) { out.push(...tri); continue; }
    if (!ins[0] && !ins[1] && !ins[2]) continue;
    const poly = [];
    for (let e = 0; e < 3; e++) {
      const a = tri[e], b = tri[(e + 1) % 3]; const ia = d[a] >= 0, ib = d[b] >= 0;
      if (ia) poly.push(a);
      if (ia !== ib && d[a] !== 0 && d[b] !== 0) poly.push(cutVertex(a, b));
    }
    for (let k = 1; k + 1 < poly.length; k++) { const tr = [poly[0], poly[k], poly[k + 1]]; if (tr[0] !== tr[1] && tr[1] !== tr[2] && tr[0] !== tr[2]) out.push(...tr); }
  }
  return compact({ positions: Float64Array.from(pos), indices: Uint32Array.from(out) });
}

/** Per connected component: triangle count, area and the area-weighted mean of n·(p − axisPoint) where the axis
 * is the vertical line through (ax, ay): > 0 for surfaces facing away from that axis (outer skin), < 0 for inner. */
export function componentFacing(m, ax, ay) {
  const c = components(m); const P = m.positions, I = m.indices;
  const res = c.sizes.map((s) => ({ tris: s, area: 0, facing: 0 }));
  for (let t = 0; t < I.length; t += 3) {
    const a = 3 * I[t], b = 3 * I[t + 1], q = 3 * I[t + 2];
    const u = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]], v = [P[q] - P[a], P[q + 1] - P[a + 1], P[q + 2] - P[a + 2]];
    const nrm = cross3(u, v); const area = Math.hypot(...nrm) / 2;
    const cx = (P[a] + P[b] + P[q]) / 3 - ax, cy = (P[a + 1] + P[b + 1] + P[q + 1]) / 3 - ay;
    const r = Math.hypot(cx, cy) || 1;
    const k = c.triComp[t / 3];
    res[k].area += area; res[k].facing += (nrm[0] * cx + nrm[1] * cy) / r / 2; // = area * cos(angle to radial dir)
  }
  for (const r of res) r.facing /= r.area || 1;
  return { comps: c, res };
}
