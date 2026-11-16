#!/usr/bin/env node
// Build anatomy assets for the X-ray mirror from BodyParts3D 4.0 (DBCLS, CC BY 4.0).
//   node tools/build-assets.mjs
// Downloads (cached) into assets-src/bodyparts3d/ (4.0) and assets-src/bodyparts3d_3.0/ (3.0, face + masticatory
// muscles only, affinely registered into the 4.0 frame), writes public/assets/{head.glb,hand.glb,muscles.glb,manifest.json}.
// All geometry stays in the original BodyParts3D frame and units (mm); GLB node transforms are identity
// and positions are stored as float32 (no quantization), compressed with EXT_meshopt_compression.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MeshoptSimplifier, MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import { Document, NodeIO } from '@gltf-transform/core';
import { EXTMeshoptCompression } from '@gltf-transform/extensions';
import { reorder } from '@gltf-transform/functions';
import { parseOBJ } from './lib/obj.mjs';
import * as G from './lib/geom.mjs';
import { ensureSource, elementIndex, elementParts, BASE_URL, ARCHIVE, ensureSource30, elementIndex30, BASE_URL_30, ARCHIVE_30 } from './lib/source.mjs';
import { HEAD, HAND, TORSO, NERVES, GROUP_BUDGETS, MUSCLES, FACE_MUSCLES, FACE_MUSCLE_BUDGET } from './lib/parts.mjs';
import { affineICP, applyAffine, residuals } from './lib/register.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SRC = path.join(ROOT, 'assets-src', 'bodyparts3d');
const SRC30 = path.join(ROOT, 'assets-src', 'bodyparts3d_3.0');
// BodyParts3D 3.0 licence, quoted from 20110915/README_e.html ("3. License").
const ATTRIBUTION_30 = 'BodyParts3D, Copyright© The Database Center for Life Science licensed by CC Attribution-Share Alike 2.1 Japan';
const LICENSE_URL_30 = 'http://creativecommons.org/licenses/by-sa/2.1/jp/deed.en_US';
// 3.0 skull parts used to register 3.0 -> 4.0 (small/remodelled bones such as vomer, ethmoid, nasal,
// lacrimal, palatine and conchae are left out). Mandible, teeth, hyoid and atlas are held out for validation.
const REG_FIT = ['Frontal bone', 'Left parietal bone', 'Right parietal bone', 'Occipital bone', 'Left temporal bone', 'Right temporal bone',
  'Sphenoid bone', 'Left maxilla', 'Right maxilla', 'Left zygomatic bone', 'Right zygomatic bone'];
const REG_TEST = ['Mandible', 'Left upper central secondary incisor tooth', 'Right lower first secondary molar tooth',
  'Left lower central secondary incisor tooth', 'Right upper first secondary molar tooth', 'Hyoid bone', 'Atlas'];
const OUT = path.join(ROOT, 'public', 'assets');
const ATTRIBUTION = 'BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International';
const LICENSE_URL = 'https://creativecommons.org/licenses/by/4.0/';
const SKIN_CUT_BELOW_C7_BODY_MM = 25;

const fmt = (v, d = 1) => v.toFixed(d);
const v3 = (a) => a.map((x) => +x.toFixed(3));
const sub = (a, b) => a.map((x, i) => x - b[i]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => { const l = Math.hypot(...a); return a.map((x) => x / l); };

async function main() {
  console.log('== Source');
  const objDir = ensureSource(SRC);
  const index = elementIndex(SRC, objDir);
  const parts = elementParts(SRC);
  const byName = new Map();
  for (const e of index) { const k = e.name.toLowerCase(); if (!byName.has(k)) byName.set(k, []); byName.get(k).push(e); }
  console.log(`  ${index.length} element files indexed`);

  await MeshoptSimplifier.ready; await MeshoptEncoder.ready; await MeshoptDecoder.ready;

  const notes = {};
  const note = (mesh, s) => { (notes[mesh] ||= []).push(s); };
  const meshCache = new Map();
  let M30 = null; // affine 3.0 -> 4.0 frame, set after registration
  function loadElement(e) {
    const ck = (e.release || '4.0') + ':' + e.id;
    if (meshCache.has(ck)) return meshCache.get(ck);
    const text = fs.readFileSync(path.join(e.release === '3.0' ? objDir30 : objDir, e.file), 'utf8');
    let m = parseOBJ(text);
    const geomKey = text.replace(/^#.*$/gm, '').length + ':' + m.indices.length + ':' + m.positions.slice(0, 9).join(',');
    let r;
    if (e.release === '3.0') {
      if (!M30) throw new Error('3.0 element requested before registration');
      // bake the 3.0 -> 4.0 transform; round to float32 so the stored GLB values are exactly these
      m = { positions: applyAffine(M30, m.positions).map(Math.fround), indices: m.indices };
      r = { ...G.dropTinyComponents(m), geomKey };
      const dup = G.removeDuplicateTriangles(r.mesh);
      if (dup.removed) { const d2 = G.dropTinyComponents(dup.mesh); r.mesh = d2.mesh; r.dupRemoved = dup.removed; }
      const s = G.edgeStats(r.mesh);
      if (s.nonManifoldEdges) { const fx = G.makeManifold(r.mesh); r.mesh = fx.mesh; r.srcNonManifold = s.nonManifoldEdges; r.srcSplit = fx.splitVertices; }
    } else r = { ...G.dropTinyComponents(m), geomKey };
    meshCache.set(ck, r);
    return r;
  }

  function resolve(def) {
    const out = []; const seen = new Map();
    for (const n of def.names) {
      const hits = (def.file === 'muscles' ? byName30 : byName).get(n.toLowerCase());
      if (!hits) throw new Error(`[${def.name}] no BodyParts3D element named "${n}"`);
      for (const e of hits) {
        if (def.exclude?.[e.id]) { note(def.name, `skipped ${e.id} ${e.name}: ${def.exclude[e.id]}`); continue; }
        if (e.release !== '3.0' && !parts.get(e.id)?.has(e.fma)) console.warn(`  warn: ${e.id} header concept ${e.fma} not in element_parts lists`);
        const m = loadElement(e);
        if (seen.has(m.geomKey)) { note(def.name, `skipped ${e.id} (geometrically identical duplicate of ${seen.get(m.geomKey)})`); continue; }
        seen.set(m.geomKey, e.id);
        if (m.dropped) note(def.name, `${e.id}: dropped ${m.dropped} degenerate/tiny disconnected component(s) (${m.droppedTris} tris)`);
        if (m.dupRemoved) note(def.name, `${e.id}: removed ${m.dupRemoved} duplicated (back-to-back) source triangles`);
        if (m.srcNonManifold) note(def.name, `${e.id}: source had ${m.srcNonManifold} non-manifold edge(s); resolved by splitting ${m.srcSplit} vertices`);
        out.push({ e, mesh: m.mesh });
      }
    }
    return out;
  }

  // ---------- resolve all sources
  const defs = [...HEAD.map((d) => ({ ...d, file: 'head' })), ...HAND.map((d) => ({ ...d, file: 'hand' })), ...TORSO.map((d) => ({ ...d, file: 'torso' })), ...NERVES.map((d) => ({ ...d, file: 'nerves' }))];
  for (const d of defs) d.src = resolve(d);

  // ---------- BodyParts3D 3.0 (face + masticatory muscles): register 3.0 frame onto 4.0 frame
  console.log('== BodyParts3D 3.0 -> 4.0 frame');
  const objDir30 = ensureSource30(SRC30);
  const index30 = elementIndex30(SRC30, objDir30).map((e) => ({ ...e, release: '3.0' }));
  const byName30 = new Map();
  for (const e of index30) { const k = e.name.toLowerCase(); if (!byName30.has(k)) byName30.set(k, []); byName30.get(k).push(e); }
  console.log(`  ${index30.length} BodyParts3D 3.0 files indexed`);
  const regPair = (name) => {
    const e4 = byName.get(name.toLowerCase())[0];
    const e3 = index30.find((x) => x.id === e4.fma);
    if (!e3) throw new Error(`registration part ${name} (${e4.fma}) missing from 3.0`);
    const m3 = G.dropTinyComponents(parseOBJ(fs.readFileSync(path.join(objDir30, e3.file), 'utf8'))).mesh;
    return { name, fma: e4.fma, id4: e4.id, src: m3.positions, dst: loadElement(e4).mesh.positions, m3, m4: loadElement(e4).mesh };
  };
  const regFit = REG_FIT.map(regPair), regTest = REG_TEST.map(regPair);
  const rawCompare = regTest.concat(regFit.slice(0, 1)).map((p) => {
    const c3 = G.centroidAndPCA(p.m3).centroid, c4 = G.centroidAndPCA(p.m4).centroid, b3 = G.bbox(p.m3), b4 = G.bbox(p.m4);
    return { name: p.name, fma: p.fma, centroidDelta: v3(sub(c4, c3)), bboxMinDelta: v3(sub(b4.min, b3.min)), bboxMaxDelta: v3(sub(b4.max, b3.max)),
      bboxSize3: v3(sub(b3.max, b3.min)), bboxSize4: v3(sub(b4.max, b4.min)) };
  });
  for (const r of rawCompare) console.log(`  raw 4.0-3.0 ${r.name.padEnd(42)} centroid Δ ${r.centroidDelta.map((x) => fmt(x, 2)).join(', ')}  bbox size 3.0 ${r.bboxSize3.map((x) => fmt(x)).join('x')} vs 4.0 ${r.bboxSize4.map((x) => fmt(x)).join('x')}`);
  const reg = affineICP(regFit, 30);
  M30 = reg.M;
  const det = M30[0][0] * (M30[1][1] * M30[2][2] - M30[1][2] * M30[2][1]) - M30[0][1] * (M30[1][0] * M30[2][2] - M30[1][2] * M30[2][0]) + M30[0][2] * (M30[1][0] * M30[2][1] - M30[1][1] * M30[2][0]);
  const regValid = residuals(regTest.map((p) => ({ ...p, src: applyAffine(M30, p.src) })));
  console.log(`  affine 3.0->4.0 (fit on ${REG_FIT.length} skull bones, 30 ICP iterations): ${M30.map((r) => '[' + r.map((x) => x.toFixed(5)).join(' ') + ']').join(' ')}; uniform scale ≈ ${Math.cbrt(det).toFixed(4)}`);
  for (const p of reg.perPart) console.log(`  fit   ${p.name.padEnd(42)} mean dist ${fmt(p.meanBefore, 2)} -> ${fmt(p.mean, 2)} mm (rms ${fmt(p.rms, 2)}, p95 ${fmt(p.p95, 2)})`);
  for (const p of regValid) console.log(`  check ${p.name.padEnd(42)} held-out mean dist ${fmt(p.mean, 2)} mm (rms ${fmt(p.rms, 2)}, p95 ${fmt(p.p95, 2)})`);
  const jawOK = regValid.filter((p) => !/hyoid|atlas/i.test(p.name)).every((p) => p.mean < 1.5);
  if (!jawOK) throw new Error('3.0 -> 4.0 registration does not reproduce held-out mandible/teeth within 1.5 mm; not exporting muscles.glb');
  const faceDefs = FACE_MUSCLES.map((d) => ({ ...d, file: 'muscles' }));
  for (const d of faceDefs) { d.src = resolve(d); defs.push(d); }

  // ---------- frame derivation from data
  console.log('== Frame');
  const cen = (names) => { const ms = names.flatMap((n) => byName.get(n.toLowerCase()).map((e) => loadElement(e).mesh)); return G.centroidAndPCA(G.merge(ms)).centroid; };
  const domAxis = (v) => { let k = 0; for (let i = 1; i < 3; i++) if (Math.abs(v[i]) > Math.abs(v[k])) k = i; return (v[k] > 0 ? '+' : '-') + 'xyz'[k]; };
  const skinFull = loadElement(byName.get('skin')[0]).mesh;
  const skinBB = G.bbox(skinFull);
  const skull = cen(['Frontal bone', 'Occipital bone', 'Left parietal bone', 'Right parietal bone']);
  const bodyMid = skinBB.min.map((x, i) => (x + skinBB.max[i]) / 2);
  const upDir = domAxis(sub(skull, bodyMid));
  const incisors = cen(['Left upper central secondary incisor tooth', 'Right upper central secondary incisor tooth']);
  const dens = cen(['Axis']);
  const antVec = sub(incisors, dens); antVec[2] = 0;
  const anteriorDir = domAxis(antVec);
  const leftDir = domAxis(sub(cen(['Left parietal bone', 'Left temporal bone', 'Left zygomatic bone', 'Left sclera']), cen(['Right parietal bone', 'Right temporal bone', 'Right zygomatic bone', 'Right sclera'])));
  const frame = { up: upDir, anterior: anteriorDir, left: leftDir };
  console.log(`  up ${upDir} (skull centroid vs body bbox centre), anterior ${anteriorDir} (incisors vs axis), subject-left ${leftDir} (Left* vs Right* skull/eye parts)`);
  if (upDir !== '+z' || anteriorDir !== '-y' || leftDir !== '+x') throw new Error('frame differs from BodyParts3D documentation (+z up, -y anterior, +x left)');

  // ---------- skin crop height: 25 mm below the inferior surface of the C7 vertebral BODY
  const c7 = loadElement(byName.get('seventh cervical vertebra')[0]).mesh;
  const c7bb = G.bbox(c7);
  let c7BodyMinZ = Infinity; // vertebral body = anterior part: vertices within 20 mm of the most anterior point
  for (let i = 0; i < c7.positions.length; i += 3) if (c7.positions[i + 1] < c7bb.min[1] + 20) c7BodyMinZ = Math.min(c7BodyMinZ, c7.positions[i + 2]);
  const cutZ = +(c7BodyMinZ - SKIN_CUT_BELOW_C7_BODY_MM).toFixed(2);
  console.log(`  C7 body inferior z=${fmt(c7BodyMinZ)} -> skin cut plane z=${cutZ}`);

  // ---------- budgets (area-proportional water-filling within a group) and simplification
  function allocate(items, budget) { // items: {tris, area}; returns targets
    const target = new Array(items.length).fill(0); const free = new Set(items.map((_, i) => i)); let B = budget;
    for (;;) {
      const A = [...free].reduce((s, i) => s + items[i].area, 0); let changed = false;
      for (const i of [...free]) { const share = B * items[i].area / A; if (items[i].tris <= share) { target[i] = items[i].tris; B -= items[i].tris; free.delete(i); changed = true; } }
      if (!changed) { for (const i of free) target[i] = Math.max(4, Math.floor(B * items[i].area / A)); return target; }
      if (!free.size) return target;
    }
  }
  function simplify(m, targetTris, errFrac, errMax, dedupe = false) {
    const tris = G.triCount(m);
    if (tris <= targetTris) return { mesh: m, error: 0, limit: 0 };
    const bb = G.bbox(m); const ext = Math.max(...sub(bb.max, bb.min));
    const limit = Math.max(0.02, Math.min(errFrac * ext, errMax));
    const local = new Float32Array(m.positions.length);
    for (let i = 0; i < local.length; i++) local[i] = m.positions[i] - bb.min[i % 3];
    // Edge collapses can pinch thin sheets together (edges shared by 4 triangles). Such pinches are split
    // topologically (duplicated vertices, no movement); if that cannot resolve them, retry with a larger target.
    const s0 = G.edgeStats(m);
    const good = (s) => s.nonManifoldEdges <= s0.nonManifoldEdges && s.boundaryEdges <= s0.boundaryEdges;
    // pinching depends chaotically on the exact collapse sequence, so first jitter the target slightly
    const factors = [1, 1.02, 0.98, 1.04, 0.96, 1.07, 0.93, 1.1, 0.9, 1.15, 1.25, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 6];
    // Attempts with factor <= 1.15 are all evaluated and the smallest valid result kept; beyond that the
    // first valid (larger) result wins.
    let retries = 0, best = null;
    for (const [f, flags] of factors.flatMap((f) => [[f, ['ErrorAbsolute']], [f, ['ErrorAbsolute', 'Regularize']]])) {
      if (best && f > 1.15) break;
      const target = Math.max(4, Math.round(targetTris * f));
      if (target >= tris) break;
      const [idx, err] = MeshoptSimplifier.simplify(m.indices, local, 3, target * 3, limit, flags);
      let res = null;
      let out = G.compact({ positions: m.positions, indices: idx });
      // (3.0 muscles only) collapses also leave back-to-back duplicate triangles (zero-volume fins): drop them
      if (dedupe) out = G.removeDuplicateTriangles(out).mesh;
      if (good(G.edgeStats(out))) res = { mesh: out, error: err, limit, split: 0 };
      else {
        const fix = G.makeManifold(out);
        if (good(G.edgeStats(fix.mesh))) res = { mesh: fix.mesh, error: err, limit, split: fix.splitVertices, pinched: fix.fixedEdges };
      }
      if (res && (!best || G.triCount(res.mesh) < G.triCount(best.mesh))) best = { ...res, retries };
      if (!res) retries++;
      if (best && f === 1 && best.retries === 0 && flags.length === 1) break; // plain first attempt succeeded
    }
    return best || { mesh: m, error: 0, limit, retries, split: 0 };
  }

  console.log('== Build meshes');
  // skin + eyes need special handling before budgeting
  for (const d of defs) {
    if (d.method === 'skin') {
      const r = G.clipAndCap(d.src[0].mesh, 2, cutZ);
      if (r.skipped) note(d.name, `${r.skipped} cut-plane boundary vertices could not be chained into loops`);
      note(d.name, `cropped at z=${cutZ} mm (${SKIN_CUT_BELOW_C7_BODY_MM} mm below C7 vertebral body), ${r.loops.length} cut loop(s) capped with fans: ` +
        r.loops.map((l) => `${l.vertices} verts, perimeter ${fmt(l.perimeter, 0)} mm, cap area ${fmt(l.area / 100, 0)} cm2`).join('; '));
      d.src[0] = { ...d.src[0], mesh: r.mesh };
      d.skinLoops = r.loops;
    }
    if (d.method === 'envelope') {
      const sclera = d.src.find((s) => /sclera/i.test(s.e.name)).mesh;
      const bb = G.bbox(sclera); const c = bb.min.map((x, i) => (x + bb.max[i]) / 2);
      const r = G.radialEnvelope(d.src.map((s) => s.mesh), c, 3);
      note(d.name, `radial outer envelope (icosphere level 3, 1280 tris) of ${d.src.length} eye elements around sclera bbox centre ${v3(c)}; radius ${fmt(r.rmin, 2)}–${fmt(r.rmax, 2)} mm` + (r.misses ? `; ${r.misses} rays missed (filled with mean radius)` : ''));
      d.final = r.mesh; d.simplifyError = 0;
    }
    if (d.method === 'skinTorso') {
      // open trunk skin envelope: z between the head-skin cut and 20 mm below the lowest 7th rib / xiphoid point,
      // sagittal cuts 35 mm lateral to each humeral-head centre; outer surface of the 2 mm skin layer only.
      const lowZ = Math.min(...['left seventh rib', 'right seventh rib', 'xiphoid process'].map((n) => G.bbox(loadElement(byName.get(n)[0]).mesh).min[2])) - 20;
      const headCentre = (n) => { const H = loadElement(byName.get(n)[0]).mesh; const hb = G.bbox(H); const pts = [];
        for (let i = 0; i < H.positions.length; i += 3) if (H.positions[i + 2] > hb.max[2] - 40) pts.push(H.positions[i], H.positions[i + 1], H.positions[i + 2]);
        const b = G.bbox({ positions: Float64Array.from(pts) }); return b.min.map((x, k) => (x + b.max[k]) / 2); };
      const xL = +(headCentre('left humerus')[0] + 35).toFixed(2), xR = +(headCentre('right humerus')[0] - 35).toFixed(2);
      const bz = +lowZ.toFixed(2);
      let m = d.src[0].mesh;
      m = G.clipHalfSpace(m, 2, cutZ, -1); m = G.clipHalfSpace(m, 2, bz, 1); m = G.clipHalfSpace(m, 0, xL, -1); m = G.clipHalfSpace(m, 0, xR, 1);
      const ay = G.centroidAndPCA(m).centroid[1];
      const { comps, res } = G.componentFacing(m, 0, ay);
      const outerK = res.map((r, k) => k).filter((k) => res[k].facing > 0).sort((a, b) => res[b].area - res[a].area)[0];
      note(d.name, `cropped z ${bz}..${cutZ} (bottom = 20 mm below lowest 7th-rib/xiphoid point), x ${xR}..${xL} (35 mm lateral to humeral-head centres); ${res.length} surface component(s) after cropping: ` +
        res.map((r) => `${r.tris} tris, ${fmt(r.area / 100, 0)} cm2, facing ${r.facing.toFixed(2)}`).join('; ') + ' -> kept the largest outward-facing one (outer skin surface), dropped the inner surface of the skin layer');
      let outer = G.subsetTriangles(m, (t) => comps.triComp[t] === outerK);
      // arm skin is fused to the trunk down to the level where the arms separate; below that, drop the arm strips
      const P = outer.positions, I = outer.indices; const tz = new Float64Array(I.length / 3);
      for (let t = 0; t < I.length; t += 3) tz[t / 3] = (P[3 * I[t] + 2] + P[3 * I[t + 1] + 2] + P[3 * I[t + 2] + 2]) / 3;
      let sepZ = null;
      for (let z = Math.floor(cutZ); z > bz; z -= 5) {
        const c = G.components(G.subsetTriangles(outer, (t) => tz[t] < z));
        if (c.sizes.filter((x) => x > 30).length >= 2) { sepZ = z; break; }
      }
      if (sepZ !== null) {
        const lowIdx = []; for (let t = 0; t < tz.length; t++) if (tz[t] < sepZ) lowIdx.push(t);
        const low = G.subsetTriangles(outer, (t) => tz[t] < sepZ); const lc = G.components(low);
        let big = 0; for (let k = 1; k < lc.count; k++) if (lc.sizes[k] > lc.sizes[big]) big = k;
        const drop = new Set(); lowIdx.forEach((t, j) => { if (lc.triComp[j] !== big) drop.add(t); });
        // subsetTriangles keeps triangle order, so triangle j of `low` is lowIdx[j] of `outer`
        note(d.name, `arm and trunk skin are fused down to z≈${sepZ}; below it ${lc.count - 1} detached arm strip(s) (${drop.size} tris) were dropped. Above z≈${sepZ} the medial upper-arm skin inside the sagittal cuts remains (fused to the trunk in the source).`);
        outer = G.subsetTriangles(outer, (t) => !drop.has(t));
      }
      const srcTris = G.triCount(outer);
      const bb = G.bbox(outer);
      const local = new Float32Array(outer.positions.length);
      for (let i = 0; i < local.length; i++) local[i] = outer.positions[i] - bb.min[i % 3];
      const [idx, err] = MeshoptSimplifier.simplify(outer.indices, local, 3, d.budget * 3, 0.5, ['ErrorAbsolute', 'LockBorder']);
      d.final = G.compact({ positions: outer.positions, indices: idx });
      d.simplifyError = err; d.sourceTris = srcTris; d.skinTorsoCuts = { zTop: cutZ, zBottom: bz, xLeft: xL, xRight: xR, armSeparationZ: sepZ };
    }
  }
  // budgets
  const groups = new Map();
  for (const d of defs) {
    if (d.final) continue;
    const key = d.group || d.name;
    if (!groups.has(key)) groups.set(key, { budget: d.group ? { ...GROUP_BUDGETS, face_muscles: FACE_MUSCLE_BUDGET }[d.group] : d.budget, items: [] });
    for (const s of d.src) groups.get(key).items.push({ d, s, tris: G.triCount(s.mesh), area: G.surfaceArea(s.mesh) });
  }
  for (const g of groups.values()) {
    const targets = allocate(g.items, g.budget);
    g.items.forEach((it, i) => {
      const r = simplify(it.s.mesh, targets[i], it.d.errFrac, it.d.errMax, it.d.file === 'muscles');
      it.s.out = r.mesh; it.s.err = r.error; it.s.target = targets[i];
      if (r.retries) note(it.d.name, `${it.s.e.id}: simplification retried ${r.retries}x with jittered/larger target to avoid unresolvable non-manifold edges: ${G.triCount(r.mesh)} tris vs target ${targets[i]}`);
      if (r.split) note(it.d.name, `${it.s.e.id}: simplification pinched ${r.pinched} edge(s); resolved by splitting ${r.split} vertices (duplicate positions, zero-thickness contact remains)`);
    });
  }
  for (const d of defs) {
    if (d.final) continue;
    d.final = G.merge(d.src.map((s) => s.out));
    d.simplifyError = Math.max(...d.src.map((s) => s.err));
    d.sourceTris = d.src.reduce((a, s) => a + G.triCount(s.mesh), 0);
  }
  console.log('== Self-intersection scan');
  for (const d of defs) {
    const total = G.selfIntersections(d.final);
    if (d.method === 'envelope' || d.method === 'skinTorso') { d.selfX = { sourceWithinElements: null, outputWithinElements: total, betweenElements: 0 }; continue; }
    const srcX = d.src.reduce((a, s) => a + G.selfIntersections(s.mesh), 0);
    const outX = d.src.reduce((a, s) => a + G.selfIntersections(s.out), 0);
    d.selfX = { sourceWithinElements: srcX, outputWithinElements: outX, betweenElements: total - outX };
    if (total) note(d.name, `intersecting triangle pairs: ${outX} within elements (source had ${srcX}), ${total - outX} between different source elements`);
  }

  // ---------- write GLBs
  const io = new NodeIO().registerExtensions([EXTMeshoptCompression]).registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });
  fs.mkdirSync(OUT, { recursive: true });
  async function writeGLB(fileKey) {
    const doc = new Document();
    doc.getRoot().getAsset().copyright = fileKey === 'muscles' ? ATTRIBUTION_30 : ATTRIBUTION;
    doc.getRoot().getAsset().generator = fileKey === 'muscles' ? 'tools/build-assets.mjs (BodyParts3D 3.0, registered to the 4.0 frame)' : 'tools/build-assets.mjs (BodyParts3D 4.0)';
    const buffer = doc.createBuffer();
    const scene = doc.createScene(fileKey);
    doc.getRoot().setDefaultScene(scene);
    for (const d of defs.filter((x) => x.file === fileKey)) {
      const m = d.final;
      const nv = G.vertCount(m);
      const pos = doc.createAccessor(d.name + '_position').setType('VEC3').setArray(Float32Array.from(m.positions)).setBuffer(buffer);
      const idx = doc.createAccessor(d.name + '_indices').setType('SCALAR').setArray(nv < 65536 ? Uint16Array.from(m.indices) : Uint32Array.from(m.indices)).setBuffer(buffer);
      const prim = doc.createPrimitive().setAttribute('POSITION', pos).setIndices(idx);
      const mesh = doc.createMesh(d.name).addPrimitive(prim);
      scene.addChild(doc.createNode(d.name).setMesh(mesh));
    }
    await doc.transform(reorder({ encoder: MeshoptEncoder, target: 'size' }));
    doc.createExtension(EXTMeshoptCompression).setRequired(true).setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
    const file = path.join(OUT, fileKey + '.glb');
    await io.write(file, doc);
    return file;
  }
  const files = { head: await writeGLB('head'), hand: await writeGLB('hand'), torso: await writeGLB('torso'), nerves: await writeGLB('nerves'), muscles: await writeGLB('muscles') };

  // ---------- read back + verify
  console.log('== Verify GLB round trip');
  const sortedTriples = (arr) => { const t = []; for (let i = 0; i < arr.length; i += 3) t.push([arr[i], arr[i + 1], arr[i + 2]]); t.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]); return t; };
  let worst = 0; const rows = [];
  for (const [key, file] of Object.entries(files)) {
    const doc = await io.read(file);
    for (const node of doc.getRoot().getDefaultScene().listChildren()) {
      const d = defs.find((x) => x.name === node.getName() && x.file === key);
      if (!d) throw new Error('unexpected node ' + node.getName());
      const M = node.getWorldMatrix();
      if (M.some((v, i) => Math.abs(v - (i % 5 === 0 ? 1 : 0)) > 0)) throw new Error(`node ${d.name} transform is not identity`);
      const prim = node.getMesh().listPrimitives()[0];
      const P = prim.getAttribute('POSITION').getArray();
      if (!(P instanceof Float32Array)) throw new Error('positions are not float32');
      const I = prim.getIndices().getArray();
      const a = sortedTriples(P), b = sortedTriples(Float32Array.from(d.final.positions)); // compare in float32 so near-tie sort order matches
      if (a.length !== b.length) throw new Error(`${d.name}: vertex count mismatch`);
      let md = 0; for (let i = 0; i < a.length; i++) for (let k = 0; k < 3; k++) md = Math.max(md, Math.abs(a[i][k] - b[i][k]));
      const back = { positions: Float64Array.from(P), indices: Uint32Array.from(I) };
      const bbA = G.bbox(back), bbB = G.bbox(d.final);
      const bbd = Math.max(...[0, 1, 2].flatMap((k) => [Math.abs(bbA.min[k] - bbB.min[k]), Math.abs(bbA.max[k] - bbB.max[k])]));
      worst = Math.max(worst, md, bbd);
      const es = G.edgeStats(back);
      if (I.length / 3 !== G.triCount(d.final)) throw new Error(`${d.name}: triangle count mismatch`);
      rows.push({ name: d.name, file: key, tris: I.length / 3, closed: es.closed, size: sub(bbA.max, bbA.min), md });
    }
  }
  console.log('  mesh                               file  tris    closed  bbox size (mm)             max|Δ| (mm)');
  for (const r of rows) console.log(`  ${r.name.padEnd(34)} ${r.file.padEnd(5)} ${String(r.tris).padStart(6)}  ${String(r.closed).padEnd(6)}  ${r.size.map((x) => fmt(x).padStart(6)).join(' x ')}   ${r.md.toExponential(1)}`);
  console.log(`  max decoded-vs-source deviation (vertices & bbox): ${worst.toExponential(2)} mm  ${worst < 0.1 ? 'OK (< 0.1 mm)' : 'FAIL'}`);
  if (worst >= 0.1) throw new Error('round-trip deviation too large');

  // ---------- manifest
  const meshes = {}; const info = {};
  for (const d of defs) {
    const m = d.final; const es = G.edgeStats(m); const cp = G.centroidAndPCA(m); const bb = G.bbox(m);
    info[d.name] = { ...cp, bb };
    meshes[d.name] = {
      file: d.file, triangles: G.triCount(m), vertices: G.vertCount(m), closed: es.closed, boundaryEdges: es.boundaryEdges, nonManifoldEdges: es.nonManifoldEdges,
      bbox: { min: v3(bb.min), max: v3(bb.max) }, centroid: v3(cp.centroid),
      pca: { axes: cp.pca.axes.map((a) => a.map((x) => +x.toFixed(5))), min: v3(cp.pca.min), max: v3(cp.pca.max) },
      sources: d.src.map((s) => ({ id: s.e.id, fma: s.e.fma, name: s.e.name })),
      label: d.label,
      ...(d.file === 'muscles' ? { release: 'BodyParts3D 3.0', frameTransform: 'BodyParts3D 3.0 positions mapped into the 4.0 frame with muscles.frameRegistration.matrix (baked into the vertices; node transform is identity)' } : {}),
      ...(d.sourceTris ? { sourceTriangles: d.sourceTris } : {}),
      ...(d.skinTorsoCuts ? { cuts: d.skinTorsoCuts } : {}),
      maxSimplifyError: +(d.simplifyError || 0).toFixed(3),
      intersectingTrianglePairs: d.selfX,
      ...(notes[d.name] ? { notes: notes[d.name] } : {}),
    };
  }
  const muscleNames = MUSCLES.map(([s]) => 'muscle_' + s);
  const manifest = {
    source: 'BodyParts3D 4.0', license: 'CC BY 4.0', licenseUrl: LICENSE_URL, attribution: ATTRIBUTION,
    sourceUrl: BASE_URL, sourceArchive: ARCHIVE,
    units: 'mm', frame: { ...frame, note: "signed axes in the original BodyParts3D frame: superior, anterior (nose direction) and the subject's left" },
    geometry: 'float32 positions in the original BodyParts3D frame; every GLB node transform is identity (no quantization, no re-centring); EXT_meshopt_compression (required) - three.js GLTFLoader needs setMeshoptDecoder(MeshoptDecoder). No normals/UVs/materials. Element meshes are concatenated, not welded to each other: a merged mesh is a set of separately closed shells.',
    files: { head: 'head.glb', hand: 'hand.glb', torso: 'torso.glb', nerves: 'nerves.glb', muscles: 'muscles.glb' },
    skinCutZ: cutZ,
    muscleInventory: {
      included: muscleNames,
      absentFromBodyParts3D: ['masseter', 'temporalis', 'medial/lateral pterygoid', 'buccinator', 'orbicularis oculi', 'orbicularis oris', 'occipitofrontalis/frontalis', 'zygomaticus major/minor', 'levator labii superioris', 'levator anguli oris', 'depressor anguli oris', 'depressor labii inferioris', 'mentalis', 'nasalis', 'procerus', 'corrugator supercilii', 'risorius', 'auricular muscles'],
      note: 'BodyParts3D 4.0 "muscle of face" (FMA46751) contains only the extra-ocular muscles; no muscles of facial expression or mastication exist in the dataset. Right longus colli is missing (left parts only). The facial-expression and masticatory muscles are provided separately in muscles.glb from BodyParts3D 3.0 (see "muscles").',
      inMusclesGlb: faceDefs.map((d) => d.name),
      notAvailableInEitherRelease: ['auricular muscles (anterior/superior/posterior auricular)', 'incisivus labii', 'depressor supercilii'],
    },
    muscles: {
      file: 'muscles.glb', source: 'BodyParts3D 3.0', releaseDate: '2011-09-15', sourceUrl: BASE_URL_30, sourceArchive: ARCHIVE_30,
      license: 'CC BY-SA 2.1 JP', licenseUrl: LICENSE_URL_30, attribution: ATTRIBUTION_30,
      licenseNote: 'Quoted from the 3.0 README_e.html (20110915/, file dated 2017-10-16): "The license for this database is the license specified in the Creative Commons Attribution-Share Alike 2.1 Japan" and "You must distribute any derivative work based on part or whole of the data from this database under this License." The current DBCLS licence page (dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html, last updated 2025/02/27) and LATEST/README_e.html state "The license for this database is specified in the Creative Commons Attribution 4.0 International" but neither mentions earlier releases, and the download page lists only the 4.0 files. Whether the CC BY 4.0 relicensing covers 3.0 is therefore NOT stated; muscles.glb is labelled with the 3.0 licence (share-alike) to be safe.',
      frameRegistration: {
        method: 'affine ICP (3x4, least squares, 30 iterations, symmetric nearest-vertex correspondences restricted to the same bone) fitted on BodyParts3D 3.0 vs 4.0 skull bones; applied to every 3.0 muscle vertex',
        fitParts: REG_FIT, matrix: M30.map((r) => r.map((x) => +x.toFixed(6))), uniformScale: +Math.cbrt(det).toFixed(5),
        fitResidualMm: reg.perPart.map((p) => ({ part: p.name, meanBefore: +p.meanBefore.toFixed(2), mean: +p.mean.toFixed(2), rms: +p.rms.toFixed(2), p95: +p.p95.toFixed(2) })),
        heldOutResidualMm: regValid.map((p) => ({ part: p.name, mean: +p.mean.toFixed(2), rms: +p.rms.toFixed(2), p95: +p.p95.toFixed(2) })),
        rawDifference4minus3: rawCompare,
        note: 'Residuals are nearest-VERTEX distances (upper bound of surface distance; vertex spacing ~1-2 mm). Hyoid and atlas are neck parts that moved independently between releases and are not used.',
      },
    },
    nerveInventory: {
      source: 'BodyParts3D 4.0 only (no BodyParts3D 3.0 parts used: 3.0 has a subset of the 4.0 nerves - optic nerves, chiasm, optic tracts - and nothing else)',
      exported: { nerve_head: 'optic nerves, optic chiasm, optic tracts, superior/inferior branches of the oculomotor nerves, trochlear nerves, ophthalmic nerves and branches (frontal, supra-orbital, supratrochlear, lacrimal, nasociliary, infratrochlear, anterior/posterior ethmoidal, long and short ciliary nerves, nasociliary-ciliary ganglion communicating branches), ciliary ganglia' },
      skippedEmpty: {
        nerve_cord: 'no spinal cord in either release: the BodyParts3D \"spinal cord\" concept (FMA7647) consists only of the element \"Central canal of spinal cord\" (FJ1737, a 2 mm x 13 mm x 35 mm cavity at C1-C2 level), which is not nerve tissue; no roots/rootlets',
        nerve_neck: 'none of cervical plexus, vagus, phrenic, accessory, hypoglossal, sympathetic trunk/ganglia, recurrent laryngeal exist in 4.0 or 3.0',
        nerve_chest: 'no thoracic spinal/intercostal nerves, thoracic sympathetic trunk, vagus or phrenic in 4.0 or 3.0',
        nerve_arm_l: 'no brachial plexus or arm nerves (median, ulnar, radial, musculocutaneous, axillary) in 4.0 or 3.0',
        nerve_arm_r: 'as nerve_arm_l',
      },
      alsoAbsentInHead: 'olfactory, abducens, trigeminal main trunk / maxillary / mandibular divisions, facial, vestibulocochlear, glossopharyngeal, and the head portions of vagus, accessory and hypoglossal nerves',
    },
    meshes,
  };
  fs.writeFileSync(path.join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 1) + '\n');

  // ---------- anatomical sanity checks
  console.log('== Anatomy checks');
  const C = (n) => info[n].centroid, BB = (n) => info[n].bb;
  const checks = [];
  const check = (ok, msg) => { checks.push(ok); console.log(`  ${ok ? 'PASS' : 'FAIL'} ${msg}`); };
  check(C('mandible')[2] < C('cranium')[2] && C('mandible')[1] < C('cranium')[1], `mandible centroid below (dz=${fmt(C('mandible')[2] - C('cranium')[2])}) and anterior (dy=${fmt(C('mandible')[1] - C('cranium')[1])}) of cranium centroid`);
  const inside = (p, b, pad = 0) => [0, 1, 2].every((k) => p[k] >= b.min[k] - pad && p[k] <= b.max[k] + pad);
  check(C('teeth_upper')[2] > C('teeth_lower')[2], `upper teeth centroid above lower (dz=${fmt(C('teeth_upper')[2] - C('teeth_lower')[2])} mm)`);
  check(inside(C('teeth_lower'), BB('mandible')) && inside(C('teeth_upper'), BB('cranium')), 'lower-teeth centroid inside mandible bbox, upper-teeth centroid inside cranium bbox');
  check(BB('teeth_upper').min[2] > BB('mandible').min[2] && BB('teeth_upper').max[2] < BB('cranium').max[2] && BB('teeth_lower').min[2] > BB('mandible').min[2], `teeth z-ranges [${fmt(BB('teeth_lower').min[2])}, ${fmt(BB('teeth_upper').max[2])}] lie between mandible bottom ${fmt(BB('mandible').min[2])} and cranium top ${fmt(BB('cranium').max[2])}`);
  const cz = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'].map((n) => C(n)[2]);
  check(cz.every((z, i) => i === 0 || z < cz[i - 1]), `c1..c7 centroid z descend monotonically: ${cz.map((z) => fmt(z)).join(' > ')}`);
  // C1 vs occiput: gap between top of atlas and lowest cranium point above the atlas footprint
  const cr = defs.find((d) => d.name === 'cranium').final; const c1bb = BB('c1');
  let occLow = Infinity; for (let i = 0; i < cr.positions.length; i += 3) { const x = cr.positions[i], y = cr.positions[i + 1], z = cr.positions[i + 2]; if (x > c1bb.min[0] && x < c1bb.max[0] && y > c1bb.min[1] && y < c1bb.max[1] && z > C('c1')[2]) occLow = Math.min(occLow, z); }
  check(occLow - c1bb.max[2] < 15 && C('c1')[2] < occLow, `C1 directly under occiput: atlas top z=${fmt(c1bb.max[2])}, lowest cranium point above atlas footprint z=${fmt(occLow)}`);
  const eyeD = Math.hypot(...sub(C('eye_l'), C('eye_r')));
  check(eyeD > 55 && eyeD < 70, `inter-eye centroid distance ${fmt(eyeD)} mm; eye_l x=${fmt(C('eye_l')[0])} (subject left = +x), eye_r x=${fmt(C('eye_r')[0])}`);
  const skullLen = BB('cranium').max[1] - BB('cranium').min[1];
  check(skullLen > 170 && skullLen < 230, `units: cranium antero-posterior length ${fmt(skullLen)} (expects ~180-200 for mm)`);
  // hand
  const handNames = HAND.map((d) => d.name);
  check(handNames.every((n) => C(n)[0] < 0) && defs.filter((d) => d.file === 'hand').every((d) => d.src.every((s) => /right/i.test(s.e.name))), `hand: all ${handNames.length} bones named "Right ..." and lie at x<0 (subject's right): x in [${fmt(Math.min(...handNames.map((n) => C(n)[0])))}, ${fmt(Math.max(...handNames.map((n) => C(n)[0])))}]`);
  const handAxis = norm(sub([1, 2, 3, 4, 5].map((i) => C('dp' + i)).reduce((a, b) => a.map((x, k) => x + b[k] / 5), [0, 0, 0]), C('capitate')));
  check(true, `hand long axis (capitate -> mean distal phalanx) = ${v3(handAxis)}`);
  for (let f = 1; f <= 5; f++) {
    const chain = f === 1 ? ['mc1', 'pp1', 'dp1'] : [`mc${f}`, `pp${f}`, `mp${f}`, `dp${f}`];
    const proj = chain.map((n) => dot(C(n), handAxis));
    check(proj.every((p, i) => i === 0 || p > proj[i - 1]), `ray ${f}: ${chain.join(' -> ')} advance along hand axis (${proj.map((p, i) => i ? '+' + fmt(p - proj[i - 1]) : '0').join(', ')} mm)`);
  }
  check(!byName.has('middle phalanx of right thumb') && !handNames.includes('mp1'), 'thumb has 2 phalanges (no middle phalanx in source or output)');
  check(C('radius')[0] < C('ulna')[0], `radius lateral to ulna in anatomical position (radius x=${fmt(C('radius')[0])} < ulna x=${fmt(C('ulna')[0])} on the right side)`);
  // face muscles (3.0, registered) vs 4.0 skeleton / eyes
  const sideMesh = (name, side) => { // side: 'left' (+x) | 'right' (-x), from the source element names
    const d = defs.find((x) => x.name === name);
    return G.merge(d.src.filter((s) => s.e.name.toLowerCase().includes(side)).map((s) => s.out));
  };
  const md = defs.find((d) => d.name === 'mandible').final; const mdbb = BB('mandible');
  const ramus = (sgn) => { // posterior 40 % of the mandible (by y) on one side = ramus + angle
    const yCut = mdbb.min[1] + 0.6 * (mdbb.max[1] - mdbb.min[1]); const pts = [];
    for (let i = 0; i < md.positions.length; i += 3) if (md.positions[i + 1] > yCut && Math.sign(md.positions[i]) === sgn) pts.push(md.positions[i], md.positions[i + 1], md.positions[i + 2]);
    return G.bbox({ positions: Float64Array.from(pts) });
  };
  const zyg = (side) => G.bbox(loadElement(byName.get(side.toLowerCase() + ' zygomatic bone')[0]).mesh);
  const overlap = (a, b) => [0, 1, 2].every((k) => a.min[k] <= b.max[k] && b.min[k] <= a.max[k]);
  for (const [side, sgn] of [['left', 1], ['right', -1]]) {
    const ms = G.bbox(sideMesh('muscle_masseter', side)); const r = ramus(sgn); const z = zyg(side[0].toUpperCase() + side.slice(1));
    check(overlap(ms, r) && ms.max[2] >= z.min[2] && ms.max[2] <= z.max[2] + 5 && Math.abs(ms.min[2] - r.min[2]) < 10,
      `${side} masseter z ${fmt(ms.min[2])}..${fmt(ms.max[2])} overlaps mandible ramus bbox (z ${fmt(r.min[2])}..${fmt(r.max[2])}); top within zygomatic bone z ${fmt(z.min[2])}..${fmt(z.max[2])}; bottom vs mandibular angle Δz=${fmt(ms.min[2] - r.min[2])}`);
    const tm = sideMesh('muscle_temporalis', side); const tc = G.centroidAndPCA(tm).centroid; const tb = G.bbox(tm); const cb = BB('cranium');
    check(sgn * tc[0] > 0.75 * (sgn > 0 ? cb.max[0] : -cb.min[0]) && tc[2] > z.max[2] && tb.min[2] > z.min[2] - 15,
      `${side} temporalis on the side of the cranium above the arch: centroid |x| ${fmt(Math.abs(tc[0]))} (cranium half-width ${fmt(sgn > 0 ? cb.max[0] : -cb.min[0])}), centroid z ${fmt(tc[2])} > zygomatic top ${fmt(z.max[2])}, lowest point z ${fmt(tb.min[2])}`);
    const oc = G.centroidAndPCA(sideMesh('muscle_orbicularis_oculi', side)).centroid; const ec = C(sgn > 0 ? 'eye_l' : 'eye_r');
    const dxz = Math.hypot(oc[0] - ec[0], oc[2] - ec[2]);
    check(dxz < 6, `${side} orbicularis oculi ring centre vs eyeball centroid: frontal-plane offset ${fmt(dxz)} mm (dx ${fmt(oc[0] - ec[0])}, dz ${fmt(oc[2] - ec[2])}), ring is ${fmt(ec[1] - oc[1])} mm anterior`);
  }
  // torso (4.0) checks
  const minDist = (A, B) => { // brute-force nearest distance between two vertex sets, with the closest pair
    let bd = Infinity, pa = null, pb = null;
    for (let i = 0; i < A.length; i += 3) for (let j = 0; j < B.length; j += 3) {
      const d = (A[i] - B[j]) ** 2 + (A[i + 1] - B[j + 1]) ** 2 + (A[i + 2] - B[j + 2]) ** 2;
      if (d < bd) { bd = d; pa = [A[i], A[i + 1], A[i + 2]]; pb = [B[j], B[j + 1], B[j + 2]]; }
    }
    return { d: Math.sqrt(bd), pa, pb };
  };
  const ends = (n) => { const P = info[n].pca; const c = info[n].centroid; const a = P.axes[0];
    return [c.map((x, k) => x + a[k] * P.min[0]), c.map((x, k) => x + a[k] * P.max[0])]; };
  const final = (n) => defs.find((d) => d.name === n).final;
  const srcOut = (n, re) => G.merge(defs.find((d) => d.name === n).src.filter((s2) => re.test(s2.e.name)).map((s2) => s2.out));
  const manubrium = srcOut('sternum', /manubrium/i);
  for (const sd of ['l', 'r']) {
    const cl = 'clavicle_' + sd, sc = 'scapula_' + sd, hu = 'humerus_' + sd;
    const [e0, e1] = ends(cl); const [med, lat] = Math.abs(e0[0]) < Math.abs(e1[0]) ? [e0, e1] : [e1, e0];
    const dm = minDist(Float64Array.from(med), manubrium.positions);
    const S = final(sc); const sb = G.bbox(S); const ac = []; // acromion region: top 25 mm of the scapula, lateral half
    const midX = (sb.min[0] + sb.max[0]) / 2;
    for (let i = 0; i < S.positions.length; i += 3) if (S.positions[i + 2] > sb.max[2] - 25 && Math.abs(S.positions[i]) > Math.abs(midX)) ac.push(S.positions[i], S.positions[i + 1], S.positions[i + 2]);
    const dl = minDist(Float64Array.from(lat), Float64Array.from(ac));
    check(dm.d < 15 && dl.d < 20, `${cl}: long-axis medial end ${v3(med)} is ${fmt(dm.d)} mm from the manubrium; lateral end ${v3(lat)} is ${fmt(dl.d)} mm from the ${sc} acromion region (top 25 mm, lateral half)`);
    const [h0, h1] = ends(hu); const [top, bot] = h0[2] > h1[2] ? [h0, h1] : [h1, h0];
    const ax = info[hu].pca.axes[0];
    const H = final(hu); const hb = G.bbox(H); const head = [];
    for (let i = 0; i < H.positions.length; i += 3) if (H.positions[i + 2] > hb.max[2] - 40) head.push(H.positions[i], H.positions[i + 1], H.positions[i + 2]);
    const gl = minDist(Float64Array.from(head), S.positions);
    check(Math.abs(ax[2]) > 0.9 && gl.d < 15 && Math.sign(top[0]) === Math.sign(info[sc].centroid[0]),
      `${hu}: long axis ${v3(ax)} (|z| ${fmt(Math.abs(ax[2]), 3)}), proximal end ${v3(top)} above distal end ${v3(bot)}; humeral head (top 40 mm) to ${sc} closest gap ${fmt(gl.d)} mm at scapula point ${v3(gl.pb)} (glenoid)`);
  }
  const spine = final('thoracic_spine');
  const ribGaps = [];
  for (const sd of ['ribs_l', 'ribs_r']) for (const s2 of defs.find((d) => d.name === sd).src) ribGaps.push([s2.e.name, minDist(s2.out.positions, spine.positions).d]);
  const worstRib = ribGaps.reduce((a, b) => (b[1] > a[1] ? b : a));
  check(worstRib[1] < 8, `ribs 1-7 reach the thoracic spine posteriorly: closest rib-to-vertebra gap per rib ${fmt(Math.min(...ribGaps.map((r) => r[1])), 2)}..${fmt(worstRib[1], 2)} mm (largest: ${worstRib[0]})`);
  const t1 = srcOut('thoracic_spine', /^first thoracic/i); const t1bb = G.bbox(t1); const t1c = G.centroidAndPCA(t1).centroid;
  const c7gap = BB('c7').min[2] - t1bb.max[2];
  check(Math.abs(t1c[0] - C('c7')[0]) < 5 && t1c[2] < C('c7')[2], `T1 under c7: c7 lowest z ${fmt(BB('c7').min[2])}, T1 highest z ${fmt(t1bb.max[2])} (gap ${fmt(c7gap)} mm; negative = bboxes interleave, as spinous processes overlap); centroid offset T1-c7 = ${v3(sub(t1c, C('c7')))}`);
  const lr = TORSO.filter((d) => /_[lr]$/.test(d.name)).map((d) => [d.name, C(d.name)[0]]);
  check(lr.every(([n, x]) => (n.endsWith('_l') ? x > 0 : x < 0)), `side check: ${lr.map(([n, x]) => n + ' x=' + fmt(x)).join(', ')}`);
  const headPt = (n) => { const H = final(n); const hb = G.bbox(H); const pts = []; for (let i = 0; i < H.positions.length; i += 3) if (H.positions[i + 2] > hb.max[2] - 40) pts.push(H.positions[i], H.positions[i + 1], H.positions[i + 2]); return G.bbox({ positions: Float64Array.from(pts) }); };
  const hc = (n) => { const b = headPt(n); return b.min.map((x, k) => (x + b.max[k]) / 2); };
  const hd = Math.hypot(...sub(hc('humerus_l'), hc('humerus_r')));
  check(hd > 300 && hd < 400, `humeral heads (centre of top 40 mm) ${fmt(hd)} mm apart: left ${v3(hc('humerus_l'))}, right ${v3(hc('humerus_r'))}`);
  // nerves: optic nerves vs eyeballs
  for (const [side, eye] of [['left', 'eye_l'], ['right', 'eye_r']]) {
    const on = srcOut('nerve_head', new RegExp(`^${side} optic nerve$`, 'i'));
    let a = null; for (let i = 0; i < on.positions.length; i += 3) if (!a || on.positions[i + 1] < a[1]) a = [on.positions[i], on.positions[i + 1], on.positions[i + 2]];
    const dC = Math.hypot(...sub(a, C(eye)));
    const ed = minDist(Float64Array.from(a), final(eye).positions);
    check(dC < 20, `${side} optic nerve: anterior end ${v3(a)} is ${fmt(dC)} mm from the ${eye} centroid (eyeball radius 12.3-14.9 mm; ${fmt(ed.d)} mm from the nearest ${eye} surface vertex)`);
  }
  const nh = defs.find((d) => d.name === 'nerve_head');
  const sideOK = nh.src.every((s2) => { const c = G.centroidAndPCA(s2.out).centroid[0]; return /^left|of left/i.test(s2.e.name) ? c > 0 : /^right|of right/i.test(s2.e.name) ? c < 0 : true; });
  check(sideOK, 'nerve_head: every Left/Right-named source element has a centroid on its own side (left = +x)');
  check(BB('nerve_head').min[2] > BB('c1').max[2], `nerve_head lies entirely above C1 (lowest z ${fmt(BB('nerve_head').min[2])} vs atlas top ${fmt(BB('c1').max[2])})`);
  const sizes = Object.fromEntries(Object.entries(files).map(([k, f]) => [k, fs.statSync(f).size]));
  console.log(`== Files: head.glb ${(sizes.head / 1048576).toFixed(2)} MB, hand.glb ${(sizes.hand / 1024).toFixed(0)} KB, torso.glb ${(sizes.torso / 1024).toFixed(0)} KB, nerves.glb ${(sizes.nerves / 1024).toFixed(0)} KB, muscles.glb ${(sizes.muscles / 1024).toFixed(0)} KB, manifest.json ${(fs.statSync(path.join(OUT, 'manifest.json')).size / 1024).toFixed(0)} KB`);
  const totals = Object.fromEntries(['head', 'hand', 'torso', 'nerves', 'muscles'].map((k) => [k, defs.filter((d) => d.file === k).reduce((a, d) => a + G.triCount(d.final), 0)]));
  console.log(`   triangles: head ${totals.head}, hand ${totals.hand}, torso ${totals.torso}, nerves ${totals.nerves}, muscles ${totals.muscles}`);
  for (const [n, ns] of Object.entries(notes)) for (const s of ns) console.log(`   note ${n}: ${s}`);
  if (checks.some((c) => !c)) { console.error('Some anatomy checks FAILED'); process.exitCode = 1; }
}

main().catch((e) => { console.error(e); process.exit(1); });
