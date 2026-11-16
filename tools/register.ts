/**
 * One-off template registration (dev tool, run in the browser via Vite):
 *
 *   1. render the template skin head,
 *   2. run the same face tracker the app uses on that render,
 *   3. ray-cast every 2D landmark back onto the skin,
 *
 * which records where the tracker itself places each landmark on the template.
 * From that it bakes public/assets/template.json: the transform into the head
 * frame, spline control points, cranial anchors, jaw hinge, neck joints, hand
 * joints and label anchors.
 */
import * as THREE from "three";
import canonical from "../src/data/canonicalFace.json";
import { loadGlbGeometries, type Manifest, type TemplateData } from "../src/render/anatomy";
import { createFaceLandmarker } from "../src/tracking/trackers";
import { RIGID_IDX, RIGID_W, NUM_FACE, FaceGeometry } from "../src/tracking/faceGeometry";
import { fitSimilarity, applySimilarity, invertSimilarity, type Similarity } from "../src/math/similarity";
import { headAngles } from "../src/fit/neutralShape";
import { buildHandTemplate } from "../src/fit/handTemplate";
import { v3, type Vec3 } from "../src/math/linalg";

const log = (msg: string) => {
  const el = document.getElementById("log")!;
  el.textContent += msg + "\n";
  console.log(msg);
};

const CANON = Float32Array.from(canonical.vertices);
const RENDER = 1024;
const VIEWS = [0, -25, 25, -45, 45]; // head yaw in degrees; the frontal view is preferred

const LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
const EYES = [7, 163, 144, 145, 153, 154, 155, 173, 157, 158, 159, 160, 161, 246, 249, 390, 373, 374, 380, 381, 382, 398, 384, 385, 386, 387, 388, 466];
/** Craniometric and outline landmarks that must be spline controls. */
const REQUIRED = [
  9, 168, 6, 197, 195, 4, 2, 164, 18, 199, 175, 152, 10, 151,
  33, 263, 133, 362, 105, 334, 52, 282, 118, 347, 129, 358,
  234, 454, 127, 356, 93, 323, 172, 397, 136, 365, 150, 379, 58, 288, 215, 435, 132, 361,
  54, 284, 67, 297, 109, 338, 21, 251, 162, 389, 148, 377, 205, 425, 50, 280, 123, 352, 116, 345, 138, 367,
];
const N_CONTROL = 78;
const N_ANCHOR = 20;

const axisVec = (s: string): Vec3 => {
  const sign = s.trim()[0] === "-" ? -1 : 1;
  const a = s.trim().slice(-1);
  return a === "x" ? [sign, 0, 0] : a === "y" ? [0, sign, 0] : [0, 0, sign];
};

function farthestPointSample(points: Vec3[], seed: Vec3[], count: number): number[] {
  const dist = points.map((p) => Math.min(...seed.map((s) => v3.dist(p, s)), Infinity));
  const picked: number[] = [];
  while (picked.length < count) {
    let best = -1, bd = -1;
    for (let i = 0; i < points.length; i++) if (dist[i] > bd) { bd = dist[i]; best = i; }
    if (best < 0 || bd <= 0) break;
    picked.push(best);
    for (let i = 0; i < points.length; i++) dist[i] = Math.min(dist[i], v3.dist(points[i], points[best]));
  }
  return picked;
}

function closestVertex(g: THREE.BufferGeometry, p: ArrayLike<number>): Vec3 {
  const a = g.getAttribute("position").array as Float32Array;
  let best = 0, bd = Infinity;
  for (let i = 0; i < a.length; i += 3) {
    const d = (a[i] - p[0]) ** 2 + (a[i + 1] - p[1]) ** 2 + (a[i + 2] - p[2]) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return [a[best], a[best + 1], a[best + 2]];
}

async function main() {
  const base = "/assets/";
  const manifest = (await (await fetch(base + "manifest.json")).json()) as Manifest;
  const head = await loadGlbGeometries(base + "head.glb");
  if (manifest.files?.muscles) {
    const extra = await loadGlbGeometries(base + manifest.files.muscles);
    for (const [name, g] of extra) if (!head.has(name)) head.set(name, g);
  }
  log(`loaded ${head.size} head meshes; frame ${JSON.stringify(manifest.frame)}`);

  // ── pre-alignment: BodyParts3D → (x = subject's left, y = up, z = anterior), cm, centred on the cranium
  const ax = axisVec(manifest.frame.left), ay = axisVec(manifest.frame.up), az = axisVec(manifest.frame.anterior);
  const det = v3.dot(ax, v3.cross(ay, az));
  log(`frame handedness det = ${det}`);
  if (det < 0) throw new Error("source frame is left-handed: mirror handling needed");
  const c0 = manifest.meshes.cranium.centroid;
  const pre = new THREE.Matrix4()
    .set(ax[0], ax[1], ax[2], 0, ay[0], ay[1], ay[2], 0, az[0], az[1], az[2], 0, 0, 0, 0, 1)
    .multiply(new THREE.Matrix4().makeTranslation(-c0[0], -c0[1], -c0[2]));
  pre.premultiply(new THREE.Matrix4().makeScale(0.1, 0.1, 0.1));
  for (const g of head.values()) { g.applyMatrix4(pre); g.computeVertexNormals(); g.computeBoundingSphere(); g.computeBoundingBox(); }
  const skinGeo = head.get("skin")!;

  // ── render
  const canvas = document.getElementById("view") as HTMLCanvasElement;
  canvas.width = canvas.height = RENDER;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(RENDER, RENDER, false);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x55606c);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x554433, 1.1));
  const key = new THREE.DirectionalLight(0xfff2e0, 2.2);
  key.position.set(-30, 40, 80);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xdde8ff, 0.8);
  fill.position.set(50, 0, 60);
  scene.add(fill);
  const pivot = new THREE.Group();
  scene.add(pivot);
  const skin = new THREE.Mesh(skinGeo, new THREE.MeshStandardMaterial({ color: 0xd9a688, roughness: 0.62, metalness: 0 }));
  pivot.add(skin);
  // eyes help the detector and are part of what it sees on a real face
  const targets: THREE.Mesh[] = [skin];
  for (const n of ["eye_l", "eye_r"]) {
    const g = head.get(n);
    if (!g) continue;
    // the eyelids are open slits in the skin: rays at the eye corners must stop on the eyeball
    const eye = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0xf2f0ea, roughness: 0.3 }));
    pivot.add(eye);
    targets.push(eye);
  }

  // outward direction of each canonical landmark, to decide which view sees it best
  const canonN = new Float32Array(NUM_FACE * 3);
  {
    const tri = canonical.triangles;
    for (let t = 0; t < tri.length; t += 3) {
      const a = v3.at(CANON, tri[t]), b = v3.at(CANON, tri[t + 1]), c = v3.at(CANON, tri[t + 2]);
      const n = v3.cross(v3.sub(b, a), v3.sub(c, a));
      for (const i of [tri[t], tri[t + 1], tri[t + 2]]) { canonN[3 * i] += n[0]; canonN[3 * i + 1] += n[1]; canonN[3 * i + 2] += n[2]; }
    }
    // make sure they point outward (the nose tip must face +Z)
    const flip = canonN[3 * 4 + 2] < 0 ? -1 : 1;
    for (let i = 0; i < NUM_FACE; i++) {
      const n = v3.norm(v3.at(canonN, i));
      canonN[3 * i] = n[0] * flip; canonN[3 * i + 1] = n[1] * flip; canonN[3 * i + 2] = n[2] * flip;
    }
  }
  // The registration camera matches a typical webcam view (the tracker's depth perception depends on it).
  const REG_FOV = 45;
  const camera = new THREE.PerspectiveCamera(REG_FOV, 1, 5, 400); // at the origin, looking down −Z

  const landmarker = await createFaceLandmarker("IMAGE");
  const overlay = document.getElementById("overlay") as HTMLCanvasElement;
  overlay.width = overlay.height = RENDER;
  const octx = overlay.getContext("2d")!;
  const shots = document.getElementById("shots")!;
  const raycaster = new THREE.Raycaster();
  const geo = new FaceGeometry(CANON);

  // ── run the tracker AND the app's own pose solve on a handful of near-frontal views.
  // The head frame is then defined the way the tracker perceives this head: with
  //   toHead = (perceived pose)⁻¹ · (true placement of the head in front of the camera)
  // the template lands exactly where it truly is whenever the tracker looks at it, including any
  // systematic tilt or depth flattening in the tracker's own estimate.
  const views: Array<{ yaw: number; pitch: number; dist: number }> = [
    { yaw: 0, pitch: 0, dist: 55 }, { yaw: 0, pitch: 0, dist: 47 }, { yaw: 0, pitch: 0, dist: 64 },
    { yaw: 7, pitch: 0, dist: 55 }, { yaw: -7, pitch: 0, dist: 55 },
    { yaw: 0, pitch: 6, dist: 55 }, { yaw: 0, pitch: -6, dist: 55 },
  ];
  // reference points of the pre-aligned frame, carried through every view's mapping and averaged
  const REF: Vec3[] = [[0, 0, 0], [10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0], [0, 0, 10], [0, 0, -10]];
  const refAcc = REF.map(() => [0, 0, 0] as Vec3);
  const T = new Float32Array(NUM_FACE * 3);
  const detected: string[] = [];
  const gauges: number[] = [];
  const trueAcc = new Float64Array(NUM_FACE * 3);
  const trueCount = new Uint16Array(NUM_FACE);
  /** mid-face landmarks: squarely visible, rays hit the skin cleanly */
  const CORE = [4, 5, 195, 197, 6, 168, 9, 151, 2, 164, 18, 199, 129, 358, 98, 327, 50, 280, 117, 346];
  let frontalLm: Array<{ x: number; y: number; z: number }> | null = null;
  for (const view of views) {
    pivot.rotation.set((view.pitch * Math.PI) / 180, (view.yaw * Math.PI) / 180, 0);
    pivot.position.set(0, 3, -view.dist); // the face, not the cranium centre, sits mid-picture
    pivot.updateMatrixWorld(true);
    renderer.render(scene, camera);
    const res = landmarker.detect(canvas);
    const lm = res.faceLandmarks?.[0];
    const tag = `yaw ${view.yaw}° pitch ${view.pitch}° at ${view.dist} cm`;
    if (!lm) { log(`${tag}: NO FACE DETECTED`); continue; }
    if (!frontalLm) frontalLm = lm;
    const flat = new Float32Array(478 * 3);
    for (let i = 0; i < Math.min(478, lm.length); i++) { flat[3 * i] = lm[i].x; flat[3 * i + 1] = lm[i].y; flat[3 * i + 2] = lm[i].z; }
    const solve = geo.solve(flat, 1, REG_FOV);
    const inv = invertSimilarity(solve.pose);
    // The tracker assumes a standard-sized face, so its world is the true one scaled about the
    // camera by some factor. Measure it: each lifted landmark lies on the ray through its pixel,
    // and that same ray hits the true skin at a known distance.
    const ratios: number[] = [];
    for (const i of CORE) {
      raycaster.setFromCamera(new THREE.Vector2(lm[i].x * 2 - 1, 1 - lm[i].y * 2), camera);
      const h = raycaster.intersectObjects(targets, false)[0];
      if (h) ratios.push(Math.hypot(solve.metric[3 * i], solve.metric[3 * i + 1], solve.metric[3 * i + 2]) / h.point.length());
    }
    ratios.sort((a, b) => a - b);
    const gauge = ratios[Math.floor(ratios.length / 2)];
    gauges.push(gauge);
    for (let k = 0; k < REF.length; k++) {
      // true camera-space position, expressed in the tracker's scale
      const cam = new THREE.Vector3(...REF[k]).applyMatrix4(pivot.matrixWorld).multiplyScalar(gauge);
      const h = applySimilarity(inv, [cam.x, cam.y, cam.z]);
      refAcc[k] = v3.add(refAcc[k], h);
    }
    for (let i = 0; i < NUM_FACE * 3; i++) T[i] += solve.local[i];
    // Where each landmark's pixel really meets the skin (pre-aligned frame). The tracker's own depth
    // for a landmark is tilted and flattened; the surface under its pixel is the true 3D point.
    const invPivot = pivot.matrixWorld.clone().invert();
    const NUDGE: Array<[number, number]> = [[0, 0]];
    for (const r of [3, 6, 10, 15]) for (let a = 0; a < 8; a++) NUDGE.push([Math.cos((a * Math.PI) / 4) * r, Math.sin((a * Math.PI) / 4) * r]);
    for (let i = 0; i < NUM_FACE; i++) {
      // the tracker's point for this landmark, brought back to true scale, says roughly where to expect it
      const expect = new THREE.Vector3(solve.metric[3 * i], solve.metric[3 * i + 1], solve.metric[3 * i + 2]).divideScalar(gauge);
      for (const [dx, dy] of NUDGE) {
        raycaster.setFromCamera(new THREE.Vector2((lm[i].x + dx / RENDER) * 2 - 1, 1 - (lm[i].y + dy / RENDER) * 2), camera);
        const h = raycaster.intersectObjects(targets, false)[0];
        // rays at the eye corners can slip through the eyelid slit and hit the far side: reject those
        if (!h || h.point.distanceTo(expect) > 2.2) continue;
        const local = h.point.clone().applyMatrix4(invPivot);
        trueAcc[3 * i] += local.x; trueAcc[3 * i + 1] += local.y; trueAcc[3 * i + 2] += local.z;
        trueCount[i]++;
        break;
      }
    }
    detected.push(tag);
    const ang = headAngles(solve.pose.R);
    log(`${tag}: perceived yaw ${ang.yaw.toFixed(1)}° pitch ${ang.pitch.toFixed(1)}°, tracker scale ×${gauge.toFixed(3)}`);

    octx.clearRect(0, 0, RENDER, RENDER);
    octx.fillStyle = "#39f";
    for (let i = 0; i < NUM_FACE; i++) octx.fillRect(lm[i].x * RENDER - 1.5, lm[i].y * RENDER - 1.5, 3, 3);
    const shot = document.createElement("canvas");
    shot.width = shot.height = 340;
    const sctx = shot.getContext("2d")!;
    sctx.drawImage(canvas, 0, 0, 340, 340);
    sctx.drawImage(overlay, 0, 0, 340, 340);
    shots.appendChild(shot);
  }
  if (detected.length < 4 || !frontalLm) throw new Error("the tracker did not find the face in enough views");
  for (let i = 0; i < NUM_FACE * 3; i++) T[i] /= detected.length;
  const refMean = refAcc.map((p) => v3.scale(p, 1 / detected.length));

  // pre-aligned frame → head frame
  const sim: Similarity = fitSimilarity(REF.flat(), refMean.flat());
  log(`pre-aligned → head frame: scale ${sim.s.toFixed(4)}, t = ${sim.t.map((x) => x.toFixed(2)).join(", ")}`);
  const simM = new THREE.Matrix4().set(
    sim.s * sim.R[0], sim.s * sim.R[1], sim.s * sim.R[2], sim.t[0],
    sim.s * sim.R[3], sim.s * sim.R[4], sim.s * sim.R[5], sim.t[1],
    sim.s * sim.R[6], sim.s * sim.R[7], sim.s * sim.R[8], sim.t[2],
    0, 0, 0, 1,
  );
  const toHead = simM.clone().multiply(pre);
  for (const g of head.values()) { g.applyMatrix4(simM); g.computeVertexNormals(); g.computeBoundingBox(); g.computeBoundingSphere(); }
  pivot.position.set(0, 0, 0);
  pivot.rotation.set(0, 0, 0);
  pivot.updateMatrixWorld(true);

  // true surface position of every landmark, in the head frame (falls back to the perceived point)
  const Ttrue = new Float32Array(NUM_FACE * 3);
  let nTrue = 0, depthGap = 0;
  for (let i = 0; i < NUM_FACE; i++) {
    if (trueCount[i] >= 3) {
      const p = applySimilarity(sim, [trueAcc[3 * i] / trueCount[i], trueAcc[3 * i + 1] / trueCount[i], trueAcc[3 * i + 2] / trueCount[i]]);
      Ttrue.set(p, 3 * i);
      depthGap += Math.abs(p[2] - T[3 * i + 2]);
      nTrue++;
    } else Ttrue.set(v3.at(T, i), 3 * i);
  }
  log(`true surface points for ${nTrue}/${NUM_FACE} landmarks; mean depth difference from the tracker's own depth: ${((depthGap / nTrue) * 10).toFixed(1)} mm`);

  // diagnostics: how far the perceived landmarks sit from the true skin surface, and from the canonical face
  const valid = new Uint8Array(NUM_FACE).fill(1);
  const nValid = NUM_FACE, nMissing = 0;
  let resid = 0, offSkin = 0;
  for (let i = 0; i < NUM_FACE; i++) {
    resid += v3.dist(v3.at(T, i), v3.at(CANON, i));
    offSkin += v3.dist(v3.at(T, i), closestVertex(skinGeo, v3.at(T, i)));
  }
  log(`mean perceived-landmark distance from the canonical face: ${((resid / nValid) * 10).toFixed(1)} mm`);
  log(`mean perceived-landmark distance from the true skin surface: ${((offSkin / nValid) * 10).toFixed(1)} mm (includes mesh vertex spacing)`);
  void RIGID_IDX; void RIGID_W; void VIEWS;

  // ── spline controls
  const excluded = new Set<number>([...LIPS, ...EYES]);
  const required = REQUIRED.filter((i) => valid[i]);
  const candidates: number[] = [];
  for (let i = 0; i < NUM_FACE; i++) if (valid[i] && !excluded.has(i) && !required.includes(i)) candidates.push(i);
  const extra = farthestPointSample(candidates.map((i) => v3.at(T, i)), required.map((i) => v3.at(T, i)), Math.max(0, N_CONTROL - required.length));
  const controlIdx = [...required, ...extra.map((k) => candidates[k])];

  // ── anchors on the skin away from the face (vault, back of head, neck)
  const sp = skinGeo.getAttribute("position").array as Float32Array;
  const far: Vec3[] = [];
  const lmPts: Vec3[] = [];
  for (let i = 0; i < NUM_FACE; i += 3) lmPts.push(v3.at(T, i));
  for (let i = 0; i < sp.length; i += 3 * 5) {
    const p: Vec3 = [sp[i], sp[i + 1], sp[i + 2]];
    let dmin = Infinity;
    for (const q of lmPts) { const d = v3.dist(p, q); if (d < dmin) dmin = d; }
    if (dmin > 4.5) far.push(p);
  }
  const anchorIdx = farthestPointSample(far, controlIdx.map((i) => v3.at(T, i)), N_ANCHOR);
  const anchors: number[] = [];
  for (const k of anchorIdx) anchors.push(...far[k].map((x) => +x.toFixed(3)));
  log(`controls: ${controlIdx.length} landmarks + ${anchorIdx.length} anchors`);

  // ── jaw hinge: top of the posterior border of each ramus (the condylar heads)
  const mp = head.get("mandible")!.getAttribute("position").array as Float32Array;
  const condyle = (side: number): Vec3 => {
    let minZ = Infinity;
    for (let i = 0; i < mp.length; i += 3) if (mp[i] * side > 2.5 && mp[i + 2] < minZ) minZ = mp[i + 2];
    const post: Vec3[] = [];
    for (let i = 0; i < mp.length; i += 3) if (mp[i] * side > 2.5 && mp[i + 2] < minZ + 1.8) post.push([mp[i], mp[i + 1], mp[i + 2]]);
    const maxY = Math.max(...post.map((p) => p[1]));
    const top = post.filter((p) => p[1] > maxY - 0.9);
    const c = top.reduce((a, p) => v3.add(a, p), [0, 0, 0] as Vec3).map((x) => x / top.length) as Vec3;
    return [c[0], c[1] - 0.25, c[2]];
  };
  const hingeL = condyle(1), hingeR = condyle(-1);
  log(`condyles: R ${hingeR.map((x) => x.toFixed(2))}  L ${hingeL.map((x) => x.toFixed(2))}`);

  // ── neck joints from vertebral centroids
  const cent = (name: string): Vec3 => {
    const g = head.get(name)!;
    const bb = g.boundingBox!;
    return [(bb.min.x + bb.max.x) / 2, (bb.min.y + bb.max.y) / 2, (bb.min.z + bb.max.z) / 2];
  };
  const vc = [1, 2, 3, 4, 5, 6, 7].map((k) => cent(`c${k}`));
  const neck: Vec3[] = [v3.add(vc[0], v3.scale(v3.sub(vc[0], vc[1]), 0.5))];
  for (let k = 0; k < 6; k++) neck.push(v3.lerp(vc[k], vc[k + 1], 0.5));
  neck.push(v3.add(vc[6], v3.scale(v3.sub(vc[6], vc[5]), 0.5)));

  // ── soft-tissue depth check against published means (mm): from the skin point nearest each
  // landmark to the nearest point of bone
  const depthAt = (i: number): number | null => {
    const onSkin = closestVertex(skinGeo, v3.at(T, i));
    let best = Infinity;
    for (const n of ["cranium", "mandible"]) {
      const q = closestVertex(head.get(n)!, onSkin);
      best = Math.min(best, v3.dist(q, onSkin));
    }
    // the meshes were scaled into the tracker's size convention; report true millimetres
    return isFinite(best) ? (best / sim.s) * 10 : null;
  };
  const published: Array<[string, number, number]> = [
    ["glabella", 9, 5.5], ["nasion", 168, 6.0], ["rhinion", 195, 3.0], ["mid-philtrum", 164, 11.5],
    ["supramentale", 18, 11.0], ["pogonion", 199, 11.0], ["menton", 152, 7.0],
    ["zygion R", 234, 7.5], ["zygion L", 454, 7.5], ["gonion R", 172, 12.5], ["gonion L", 397, 12.5],
    ["mid-supraorbital R", 105, 7.0], ["mid-infraorbital R", 118, 6.5],
  ];
  const depths: Record<string, { template: number | null; published: number }> = {};
  for (const [name, i, pub] of published) {
    const d = depthAt(i);
    depths[name] = { template: d === null ? null : +d.toFixed(1), published: pub };
    log(`tissue depth ${name.padEnd(20)} template ${d === null ? "  n/a" : d.toFixed(1).padStart(5)} mm   published ${pub} mm`);
  }

  // ── labels: nearest point on the named mesh to a hint position (head frame, cm)
  const labelSpecs: Array<{ id: string; text: string; layer: string; mesh: string; hint: Vec3 | null; jaw?: boolean }> = [
    { id: "frontal", text: "Frontal bone", layer: "bone", mesh: "cranium", hint: [2.5, 8.5, 4.5] },
    { id: "parietal", text: "Parietal bone", layer: "bone", mesh: "cranium", hint: [6, 12, -6] },
    { id: "temporal", text: "Temporal bone", layer: "bone", mesh: "cranium", hint: [-7.5, 3, -5] },
    { id: "occipital", text: "Occipital bone", layer: "bone", mesh: "cranium", hint: [-2, 4, -16] },
    { id: "nasal", text: "Nasal bone", layer: "bone", mesh: "cranium", hint: [0.3, 2.4, 5.6] },
    { id: "orbit", text: "Orbit", layer: "bone", mesh: "cranium", hint: [-4.6, 3.6, 3.2] },
    { id: "zygomatic", text: "Zygomatic bone", layer: "bone", mesh: "cranium", hint: [5.8, 0.3, 2.6] },
    { id: "maxilla", text: "Maxilla", layer: "bone", mesh: "cranium", hint: [-1.8, -1.8, 5.0] },
    { id: "mandible", text: "Mandible", layer: "bone", mesh: "mandible", hint: [3.8, -7.6, 2.4], jaw: true },
    { id: "ramus", text: "Ramus of mandible", layer: "bone", mesh: "mandible", hint: [-5.8, -3.5, -2.2], jaw: true },
    { id: "teeth_u", text: "Upper teeth", layer: "teeth", mesh: "teeth_upper", hint: [1.2, -3.2, 5.5] },
    { id: "teeth_l", text: "Lower teeth", layer: "teeth", mesh: "teeth_lower", hint: [-1.2, -5.0, 5.0], jaw: true },
    { id: "atlas", text: "Atlas (C1)", layer: "spine", mesh: "c1", hint: null },
    { id: "axis", text: "Axis (C2)", layer: "spine", mesh: "c2", hint: null },
    { id: "c5", text: "C5 vertebra", layer: "spine", mesh: "c5", hint: null },
    { id: "hyoid", text: "Hyoid bone", layer: "spine", mesh: "hyoid", hint: null },
    { id: "frontal_lobe", text: "Frontal lobe", layer: "brain", mesh: "cerebrum", hint: [3, 9, 2] },
    { id: "parietal_lobe", text: "Parietal lobe", layer: "brain", mesh: "cerebrum", hint: [-4, 12, -7] },
    { id: "temporal_lobe", text: "Temporal lobe", layer: "brain", mesh: "cerebrum", hint: [6.5, 3.5, -2] },
    { id: "occipital_lobe", text: "Occipital lobe", layer: "brain", mesh: "cerebrum", hint: [1.5, 6, -14] },
    { id: "cerebellum", text: "Cerebellum", layer: "brain", mesh: "cerebellum", hint: [-3, 1, -11] },
    { id: "brainstem", text: "Brainstem", layer: "brain", mesh: "brainstem", hint: null },
    { id: "eye", text: "Eyeball", layer: "eyes", mesh: "eye_l", hint: [3.2, 2.8, 4.5] },
  ];
  const labels: TemplateData["labels"] = [];
  for (const s of labelSpecs) {
    const g = head.get(s.mesh);
    if (!g) { log(`label ${s.id}: mesh ${s.mesh} missing`); continue; }
    const anchor = closestVertex(g, s.hint ?? cent(s.mesh));
    labels.push({ id: s.id, text: s.text, layer: s.layer, anchor: anchor.map((x) => +x.toFixed(3)) as Vec3, jaw: !!s.jaw });
  }
  // muscles: a readable handful, each labelled at the surface point nearest the centre of its left half
  const MUSCLE_LABELS: Array<[string, string]> = [
    ["muscle_temporalis", "Temporalis"], ["muscle_masseter", "Masseter"], ["muscle_frontalis", "Frontalis"],
    ["muscle_occipitofrontalis", "Occipitofrontalis"], ["muscle_orbicularis_oculi", "Orbicularis oculi"],
    ["muscle_orbicularis_oris", "Orbicularis oris"], ["muscle_zygomaticus_major", "Zygomaticus major"],
    ["muscle_buccinator", "Buccinator"], ["muscle_mentalis", "Mentalis"], ["muscle_nasalis", "Nasalis"],
    ["muscle_depressor_anguli_oris", "Depressor anguli oris"], ["muscle_levator_labii_superioris", "Levator labii superioris"],
    ["muscle_sternocleidomastoid", "Sternocleidomastoid"], ["muscle_platysma", "Platysma"],
  ];
  for (const [n, text] of MUSCLE_LABELS) {
    const g = head.get(n);
    if (!g) continue;
    const a = g.getAttribute("position").array as Float32Array;
    let cx = 0, cy = 0, cz = 0, k = 0;
    for (let i = 0; i < a.length; i += 3) if (a[i] > 0) { cx += a[i]; cy += a[i + 1]; cz += a[i + 2]; k++; }
    if (!k) continue;
    // prefer the outward-facing side of the muscle: nudge the search point away from the head axis
    const anchor = closestVertex(g, [(cx / k) * 1.15, cy / k, (cz / k) + 0.5]);
    const jaw = /mentalis|depressor/.test(n);
    labels.push({ id: n, text, layer: "muscles", anchor: anchor.map((x) => +x.toFixed(3)) as Vec3, jaw });
  }

  // ── hand
  const hand = buildHandTemplate(manifest.meshes as never);
  const round = (a: ArrayLike<number>, d = 4) => Array.from(a, (x) => +x.toFixed(d));

  const template: TemplateData = {
    version: 1,
    toHead: round(toHead.elements, 6),
    landmarks: round(T),
    landmarksTrue: round(Ttrue),
    controlIdx,
    anchors,
    jaw: { hingeR: round(hingeR) as Vec3, hingeL: round(hingeL) as Vec3 },
    jawRef: {
      mouthY: +((T[3 * 13 + 1] + T[3 * 14 + 1]) / 2).toFixed(3),
      chinY: +T[3 * 152 + 1].toFixed(3),
      hingeY: +((hingeL[1] + hingeR[1]) / 2).toFixed(3),
      hingeZ: +((hingeL[2] + hingeR[2]) / 2).toFixed(3),
    },
    neckJoints: round(neck.flat()),
    hand: {
      joints: round(hand.joints),
      bones: hand.bones.map((b) => ({ ...b, aT: round(b.aT) as Vec3, bT: round(b.bT) as Vec3 })),
      rigid: hand.rigid,
      forearm: hand.forearm,
      forearmAxis: round(hand.forearmAxis) as Vec3,
    },
    labels,
  };

  const report = {
    detectedViews: detected, landmarksFromTracker: nValid, landmarksFilled: nMissing,
    meanTemplateVsCanonicalMm: +((resid / nValid) * 10).toFixed(2),
    scale: sim.s, depths, controls: controlIdx.length, anchors: anchorIdx.length,
    hingeR, hingeL, jawRef: template.jawRef,
    faceWidthTemplateCm: v3.dist(v3.at(T, 234), v3.at(T, 454)), faceWidthCanonicalCm: v3.dist(v3.at(CANON, 234), v3.at(CANON, 454)),
  };
  (window as unknown as Record<string, unknown>).__template = template;
  (window as unknown as Record<string, unknown>).__report = report;

  const r = await fetch("/__save?name=template.json", { method: "POST", body: JSON.stringify(template) });
  log(r.ok ? "saved public/assets/template.json" : `SAVE FAILED: ${r.status}`);
  document.title = r.ok ? "registration done" : "registration failed";
}

main().catch((e) => {
  log("ERROR: " + (e?.stack ?? e));
  document.title = "registration failed";
  (window as unknown as Record<string, unknown>).__report = { error: String(e) };
});
