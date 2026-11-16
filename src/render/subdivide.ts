/**
 * One step of Loop subdivision for an indexed triangle mesh.
 *
 * The source hand bones are very coarse (a fingertip bone is about a hundred
 * triangles), which shows as facets when a hand is held close to the camera.
 * Each triangle becomes four and the surface is rounded. Edges that do not have
 * exactly two faces are treated as creases: their midpoints are plain averages
 * and their end points stay put, so nothing tears.
 */
export function loopSubdivide(positions: ArrayLike<number>, indices: ArrayLike<number>): { positions: Float32Array; indices: Uint32Array } {
  const nv = positions.length / 3, nt = indices.length / 3;
  const key = (a: number, b: number) => (a < b ? a * nv + b : b * nv + a);

  // edge → [opposite vertex of first face, of second face, face count]
  const edges = new Map<number, { a: number; b: number; o1: number; o2: number; n: number; id: number }>();
  const addEdge = (a: number, b: number, opp: number) => {
    const k = key(a, b);
    const e = edges.get(k);
    if (!e) edges.set(k, { a, b, o1: opp, o2: -1, n: 1, id: -1 });
    else { if (e.n === 1) e.o2 = opp; e.n++; }
  };
  for (let t = 0; t < nt; t++) {
    const a = indices[3 * t], b = indices[3 * t + 1], c = indices[3 * t + 2];
    addEdge(a, b, c); addEdge(b, c, a); addEdge(c, a, b);
  }

  // neighbours of each original vertex, and whether it touches a crease
  const neigh: number[][] = Array.from({ length: nv }, () => []);
  const crease = new Uint8Array(nv);
  for (const e of edges.values()) {
    neigh[e.a].push(e.b); neigh[e.b].push(e.a);
    if (e.n !== 2) { crease[e.a] = 1; crease[e.b] = 1; }
  }

  const out = new Float32Array((nv + edges.size) * 3);
  // reposition the original vertices
  for (let v = 0; v < nv; v++) {
    const ns = neigh[v], n = ns.length;
    if (crease[v] || n < 3) { out[3 * v] = positions[3 * v]; out[3 * v + 1] = positions[3 * v + 1]; out[3 * v + 2] = positions[3 * v + 2]; continue; }
    const beta = n === 3 ? 3 / 16 : 3 / (8 * n);
    let sx = 0, sy = 0, sz = 0;
    for (const m of ns) { sx += positions[3 * m]; sy += positions[3 * m + 1]; sz += positions[3 * m + 2]; }
    const w = 1 - n * beta;
    out[3 * v] = w * positions[3 * v] + beta * sx;
    out[3 * v + 1] = w * positions[3 * v + 1] + beta * sy;
    out[3 * v + 2] = w * positions[3 * v + 2] + beta * sz;
  }
  // one new vertex per edge
  let next = nv;
  for (const e of edges.values()) {
    e.id = next++;
    for (let d = 0; d < 3; d++) {
      const pa = positions[3 * e.a + d], pb = positions[3 * e.b + d];
      out[3 * e.id + d] = e.n === 2
        ? 0.375 * (pa + pb) + 0.125 * (positions[3 * e.o1 + d] + positions[3 * e.o2 + d])
        : 0.5 * (pa + pb);
    }
  }
  const idx = new Uint32Array(nt * 12);
  for (let t = 0; t < nt; t++) {
    const a = indices[3 * t], b = indices[3 * t + 1], c = indices[3 * t + 2];
    const ab = edges.get(key(a, b))!.id, bc = edges.get(key(b, c))!.id, ca = edges.get(key(c, a))!.id;
    idx.set([a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca], 12 * t);
  }
  return { positions: out, indices: idx };
}
