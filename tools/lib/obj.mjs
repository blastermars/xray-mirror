// Minimal Wavefront OBJ reader for BodyParts3D element files.
// Reads only `v` and `f` records (normals/uvs ignored), triangulates polygons as fans,
// welds bit-identical positions, and drops degenerate triangles.
// Also parses the BodyParts3D comment header (File ID, Concept ID, English name, Bounds).

/** @param {string} text */
export function parseHeader(text) {
  const head = text.slice(0, 4096);
  const get = (k) => {
    const m = head.match(new RegExp('^# ' + k + ' : (.*)$', 'm'));
    return m ? m[1].trim() : '';
  };
  const b = head.match(/# Bounds\(mm\): \(([^)]*)\)-\(([^)]*)\)/);
  return {
    id: get('File ID'),
    fma: get('Concept ID'),
    name: get('English name'),
    representation: get('Representation ID'),
    buildUp: get('Build-up logic'),
    bounds: b ? { min: b[1].split(',').map(Number), max: b[2].split(',').map(Number) } : null,
  };
}

/**
 * @param {string} text
 * @returns {{positions: Float64Array, indices: Uint32Array, rawVertexCount: number, degenerate: number}}
 */
export function parseOBJ(text) {
  const raw = []; // raw positions as numbers
  const rawKey = []; // string key per raw vertex, for welding
  const tris = [];
  let pos = 0;
  const n = text.length;
  while (pos < n) {
    let end = text.indexOf('\n', pos);
    if (end < 0) end = n;
    const c0 = text.charCodeAt(pos);
    const c1 = text.charCodeAt(pos + 1);
    if (c0 === 118 /* v */ && (c1 === 32 || c1 === 9)) {
      const parts = text.slice(pos + 2, end).trim().split(/\s+/);
      const x = +parts[0], y = +parts[1], z = +parts[2];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) throw new Error('bad vertex line');
      raw.push(x, y, z);
      rawKey.push(x + ' ' + y + ' ' + z);
    } else if (c0 === 102 /* f */ && (c1 === 32 || c1 === 9)) {
      const parts = text.slice(pos + 2, end).trim().split(/\s+/);
      const vcount = raw.length / 3;
      const idx = parts.map((p) => {
        let i = parseInt(p, 10); // `a`, `a/b`, `a//c`, `a/b/c` -> leading integer
        if (!Number.isFinite(i) || i === 0) throw new Error('bad face index');
        i = i < 0 ? vcount + i : i - 1;
        return i;
      });
      for (let k = 1; k + 1 < idx.length; k++) tris.push(idx[0], idx[k], idx[k + 1]);
    }
    pos = end + 1;
  }
  // weld identical positions
  const map = new Map();
  const remap = new Uint32Array(raw.length / 3);
  const out = [];
  for (let i = 0; i < rawKey.length; i++) {
    let j = map.get(rawKey[i]);
    if (j === undefined) {
      j = out.length / 3;
      map.set(rawKey[i], j);
      out.push(raw[3 * i], raw[3 * i + 1], raw[3 * i + 2]);
    }
    remap[i] = j;
  }
  const idx = [];
  let degenerate = 0;
  for (let t = 0; t < tris.length; t += 3) {
    const a = remap[tris[t]], b = remap[tris[t + 1]], c = remap[tris[t + 2]];
    if (a === b || b === c || a === c) { degenerate++; continue; }
    idx.push(a, b, c);
  }
  return { positions: Float64Array.from(out), indices: Uint32Array.from(idx), rawVertexCount: raw.length / 3, degenerate };
}
