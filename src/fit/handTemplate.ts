/**
 * Derives the template hand's 21 joints and bone chain from the bone meshes
 * themselves (principal axis and extents of each bone, from the asset manifest).
 * Output is in the hand template frame: BodyParts3D coordinates × 0.1 (cm).
 */
import { v3, type Vec3 } from "../math/linalg";
import type { HandBoneDef, HandTemplate } from "./hand";

export interface BoneStats {
  centroid: Vec3;
  pca: { axes: Vec3[]; min: Vec3; max: Vec3 };
}

const CARPALS = ["scaphoid", "lunate", "triquetrum", "pisiform", "trapezium", "trapezoid", "capitate", "hamate"];
const MM_TO_CM = 0.1;

export function buildHandTemplate(meshes: Record<string, BoneStats>): HandTemplate & { forearmAxis: Vec3 } {
  const need = (n: string) => {
    const m = meshes[n];
    if (!m) throw new Error(`hand template: missing bone mesh "${n}"`);
    return m;
  };
  const cm = (p: ArrayLike<number>): Vec3 => [p[0] * MM_TO_CM, p[1] * MM_TO_CM, p[2] * MM_TO_CM];
  const centroid = (n: string) => cm(need(n).centroid);
  const carpus = CARPALS.map(centroid).reduce((a, p) => v3.add(a, p), [0, 0, 0] as Vec3).map((x) => x / CARPALS.length) as Vec3;

  /** [proximal, distal] ends of a long bone; proximal = the end nearer to `ref`. */
  const ends = (n: string, ref: Vec3): [Vec3, Vec3] => {
    const m = need(n);
    const c = cm(m.centroid), ax = m.pca.axes[0];
    const e0 = v3.add(c, v3.scale(ax, m.pca.min[0] * MM_TO_CM));
    const e1 = v3.add(c, v3.scale(ax, m.pca.max[0] * MM_TO_CM));
    return v3.dist(e0, ref) < v3.dist(e1, ref) ? [e0, e1] : [e1, e0];
  };
  const mid = (a: Vec3, b: Vec3) => v3.lerp(a, b, 0.5);

  const J = new Array<number>(63).fill(0);
  const set = (i: number, p: Vec3) => { J[3 * i] = p[0]; J[3 * i + 1] = p[1]; J[3 * i + 2] = p[2]; };
  const bones: HandBoneDef[] = [];

  // wrist: between the distal radius and the proximal carpal row
  const [radDist, radProx] = ends("radius", carpus);
  const proxRow = ["scaphoid", "lunate", "triquetrum"].map(centroid).reduce((a, p) => v3.add(a, p), [0, 0, 0] as Vec3).map((x) => x / 3) as Vec3;
  const wrist = mid(radDist, proxRow);
  set(0, wrist);
  const forearmAxis = v3.norm(v3.sub(radProx, radDist));

  for (let digit = 1; digit <= 5; digit++) {
    const names = digit === 1 ? ["mc1", "pp1", "dp1"] : [`mc${digit}`, `pp${digit}`, `mp${digit}`, `dp${digit}`];
    const seg = names.map((n, k) => ends(n, k === 0 ? carpus : centroid(names[k - 1])));
    const base = digit === 1 ? 1 : 5 + 4 * (digit - 2);
    // joints between consecutive bones
    const joints: Vec3[] = [];
    for (let k = 0; k < seg.length - 1; k++) joints.push(mid(seg[k][1], seg[k + 1][0]));
    const boneTip = seg[seg.length - 1][1];
    const lastJoint = joints[joints.length - 1];
    // the tracker's fingertip sits on the skin pad, a little beyond the bone
    const skinTip = v3.add(lastJoint, v3.scale(v3.sub(boneTip, lastJoint), 1 / 0.84));

    if (digit === 1) {
      const cmc = seg[0][0];
      set(1, cmc); set(2, joints[0]); set(3, joints[1]); set(4, skinTip);
      const mc = bones.push({ name: "mc1", a: 1, b: 2, parent: -1, aT: cmc, bT: joints[0], tip: false }) - 1;
      const pp = bones.push({ name: "pp1", a: 2, b: 3, parent: mc, aT: joints[0], bT: joints[1], tip: false }) - 1;
      bones.push({ name: "dp1", a: 3, b: 4, parent: pp, aT: joints[1], bT: boneTip, tip: true });
    } else {
      set(base, joints[0]); set(base + 1, joints[1]); set(base + 2, joints[2]); set(base + 3, skinTip);
      const mc = bones.push({ name: names[0], a: -1, b: base, parent: -1, aT: seg[0][0], bT: joints[0], tip: false }) - 1;
      const pp = bones.push({ name: names[1], a: base, b: base + 1, parent: mc, aT: joints[0], bT: joints[1], tip: false }) - 1;
      const mp = bones.push({ name: names[2], a: base + 1, b: base + 2, parent: pp, aT: joints[1], bT: joints[2], tip: false }) - 1;
      bones.push({ name: names[3], a: base + 2, b: base + 3, parent: mp, aT: joints[2], bT: boneTip, tip: true });
    }
  }
  return { joints: J, bones, rigid: CARPALS, forearm: ["radius", "ulna"], forearmAxis };
}
