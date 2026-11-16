/**
 * Shoulder girdle and upper chest placement from body-pose landmarks.
 *
 * The chest is treated as one rigid block (ribs, breastbone, thoracic spine,
 * collarbones, shoulder blades). Its orientation comes from the tracker's 3D
 * shoulder line; its size and position from where the two shoulders appear in
 * the picture. Two image points cannot separate size from distance, so the
 * distance is taken from the base of the neck, which the head and neck chain
 * already place in camera space. The upper arm bones are then aimed from each
 * shoulder at its elbow.
 */
import { applySimilarity, type Similarity } from "../math/similarity";
import { mat3Mul, mat3MulVec, rotationBetween, v3, type Vec3 } from "../math/linalg";

export interface TorsoTemplate {
  /** shoulder joint centres (head-frame units, template pose): subject's left and right */
  shoulderL: Vec3;
  shoulderR: Vec3;
  /** lower ends of the upper arm bones */
  elbowL: Vec3;
  elbowR: Vec3;
  /** where the neck chain's lowest joint meets the chest (below C7) */
  neckBase: Vec3;
}

export interface BodyObservation {
  /** normalised image points (x right, y down) */
  shoulderL: [number, number];
  shoulderR: [number, number];
  elbowL: [number, number] | null;
  elbowR: [number, number] | null;
  /** tracker's metric 3D points converted to camera axes (x right, y up, z toward the camera), metres */
  worldShoulderL: Vec3;
  worldShoulderR: Vec3;
  worldElbowL: Vec3 | null;
  worldElbowR: Vec3 | null;
  /** wrists, when seen (image point and tracker 3D point) */
  wristL?: [number, number] | null;
  wristR?: [number, number] | null;
  worldWristL?: Vec3 | null;
  worldWristR?: Vec3 | null;
  /** midpoint of the hips, if the tracker reported them */
  worldHipMid: Vec3 | null;
}

export interface TorsoPose {
  /** template (head-frame units) → camera space */
  pose: Similarity;
  /** shoulder joints in camera space */
  shoulderL: Vec3;
  shoulderR: Vec3;
  /** column-major 4×4 for each upper arm bone (template → camera) */
  humerusL: Float32Array;
  humerusR: Float32Array;
  /** elbows in camera space, when the tracker saw them */
  elbowL: Vec3 | null;
  elbowR: Vec3 | null;
}

/**
 * A joint is known to lie on the ray through its pixel and one bone length from the joint above it.
 * Of the (up to two) such points, return the one nearest the depth the tracker suggests; when the
 * ray passes further away than the bone is long, the nearest point of the ray.
 * @param wantDepth suggested distance from the camera along the view axis (cm), or null for "same as `from`"
 */
export function jointOnRay(from: Vec3, img: ArrayLike<number>, length: number, wantDepth: number | null, aspect: number, th: number): Vec3 {
  const ray = v3.norm(unproject(img, 1, aspect, th));
  const along = v3.dot(ray, from);
  const disc = along * along - v3.dot(from, from) + length * length;
  const want = Math.max(8, wantDepth ?? -from[2]) / -ray[2];
  let d = along;
  if (disc > 0) {
    const r = Math.sqrt(disc), d1 = along - r, d2 = along + r;
    d = d1 > 8 && Math.abs(d1 - want) < Math.abs(d2 - want) ? d1 : d2;
  }
  return v3.scale(ray, Math.max(8, d));
}

const unproject = (p: ArrayLike<number>, depth: number, aspect: number, th: number): Vec3 => [
  (p[0] - 0.5) * 2 * th * aspect * depth,
  (0.5 - p[1]) * 2 * th * depth,
  -depth,
];

/** Chest orientation (row-major, template axes → camera) from the 3D shoulder line and trunk direction. */
export function torsoRotation(obs: BodyObservation): Float64Array {
  const x = v3.norm(v3.sub(obs.worldShoulderL, obs.worldShoulderR));
  let up: Vec3 = [0, 1, 0];
  if (obs.worldHipMid) {
    const mid = v3.lerp(obs.worldShoulderL, obs.worldShoulderR, 0.5);
    const trunk = v3.norm(v3.sub(mid, obs.worldHipMid));
    // The hips are usually out of frame at a desk, so the tracker is guessing them: lean on its
    // answer only as far as it agrees with an upright trunk.
    if (trunk[1] > 0.6) up = v3.norm(v3.lerp([0, 1, 0], trunk, 0.6));
  }
  const z = v3.norm(v3.cross(x, up));
  const y = v3.cross(z, x);
  return new Float64Array([x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]);
}

/**
 * @param neckDepth distance from the camera (cm, positive) of the base of the neck, from the head and
 *                  neck chain; pass null to keep the previous depth
 * @param prev      previous solution (used when there is no neck depth)
 */
export function solveTorso(
  tpl: TorsoTemplate, obs: BodyObservation, neckDepth: number | null,
  aspect: number, vfovDeg: number, prev?: TorsoPose | null,
): TorsoPose {
  const th = Math.tan((vfovDeg * Math.PI) / 360);
  const R = torsoRotation(obs);
  const qM = v3.lerp(tpl.shoulderL, tpl.shoulderR, 0.5);
  const widthT = v3.dist(tpl.shoulderL, tpl.shoulderR);
  const rel = (q: Vec3) => mat3MulVec(R, v3.sub(q, qM)); // offset from mid-shoulder, rotated, unscaled

  // depth of the mid-shoulder point: neck base depth plus the (rotated) template offset between them
  let s = prev?.pose.s ?? 1;
  let PL: Vec3 = [0, 0, 0], PR: Vec3 = [0, 0, 0];
  const neckOff = rel(tpl.neckBase), offL = rel(tpl.shoulderL), offR = rel(tpl.shoulderR);
  const baseDepth = neckDepth ?? (prev ? -(prev.pose.t[2] + s * mat3MulVec(prev.pose.R, tpl.neckBase)[2]) : 60);
  for (let it = 0; it < 4; it++) {
    // camera z decreases with depth, so depth = −z
    const dM = baseDepth + s * neckOff[2];
    PL = unproject(obs.shoulderL, Math.max(8, dM - s * offL[2]), aspect, th);
    PR = unproject(obs.shoulderR, Math.max(8, dM - s * offR[2]), aspect, th);
    s = Math.min(1.6, Math.max(0.6, v3.dist(PL, PR) / widthT));
  }
  const M = v3.lerp(PL, PR, 0.5);
  const RqM = mat3MulVec(R, qM);
  const pose: Similarity = { s, R, t: [M[0] - s * RqM[0], M[1] - s * RqM[1], M[2] - s * RqM[2]] };

  const out: TorsoPose = {
    pose,
    shoulderL: applySimilarity(pose, tpl.shoulderL),
    shoulderR: applySimilarity(pose, tpl.shoulderR),
    humerusL: new Float32Array(16),
    humerusR: new Float32Array(16),
    elbowL: null,
    elbowR: null,
  };
  const arm = (m: Float32Array, shoulderT: Vec3, elbowT: Vec3, shoulder: Vec3, elbowImg: [number, number] | null, wS: Vec3, wE: Vec3 | null): Vec3 | null => {
    const uT = v3.norm(v3.sub(elbowT, shoulderT));
    const lenT = v3.dist(elbowT, shoulderT) || 1;
    let Rb: ArrayLike<number> = R, sl = s;
    let elbowCam: Vec3 | null = null;
    if (elbowImg && wE) {
      // The bone then ends exactly on the tracked elbow in the picture, which the tracker's rough
      // depth alone would not guarantee. Suggested depth: the shoulder's, plus the tracker's
      // shoulder→elbow depth difference (metres toward the camera → centimetres).
      const elbow = jointOnRay(shoulder, elbowImg, s * lenT, -shoulder[2] - (wE[2] - wS[2]) * 100, aspect, th);
      const dir = v3.sub(elbow, shoulder);
      const len = v3.len(dir);
      if (len > 1) {
        Rb = mat3Mul(rotationBetween(mat3MulVec(R, uT), v3.scale(dir, 1 / len)), R);
        // follow the measured length, but an upper arm does not stretch much
        sl = Math.min(1.35 * s, Math.max(0.8 * s, len / lenT));
        elbowCam = v3.add(shoulder, v3.scale(dir, (sl * lenT) / len));
      }
    }
    // p' = shoulder + Rb · D · (p − shoulderT),  D = s·I + (sl − s)·uT uTᵀ
    const D = new Float64Array(9);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) D[r * 3 + c] = (r === c ? s : 0) + (sl - s) * uT[r] * uT[c];
    const L = mat3Mul(Rb, D);
    const o = mat3MulVec(L, shoulderT);
    m[0] = L[0]; m[1] = L[3]; m[2] = L[6]; m[3] = 0;
    m[4] = L[1]; m[5] = L[4]; m[6] = L[7]; m[7] = 0;
    m[8] = L[2]; m[9] = L[5]; m[10] = L[8]; m[11] = 0;
    m[12] = shoulder[0] - o[0]; m[13] = shoulder[1] - o[1]; m[14] = shoulder[2] - o[2]; m[15] = 1;
    return elbowCam;
  };
  out.elbowL = arm(out.humerusL, tpl.shoulderL, tpl.elbowL, out.shoulderL, obs.elbowL, obs.worldShoulderL, obs.worldElbowL);
  out.elbowR = arm(out.humerusR, tpl.shoulderR, tpl.elbowR, out.shoulderR, obs.elbowR, obs.worldShoulderR, obs.worldElbowR);
  return out;
}

export interface LongBoneStats { centroid: Vec3; pca: { axes: Vec3[]; min: Vec3; max: Vec3 } }

/**
 * Joint positions of the template from its bone meshes (source millimetres, mapped by `toHead`).
 * The shoulder joint is taken a little inside the upper end of the humerus (the centre of its head),
 * the elbow at its lower end.
 */
export function buildTorsoTemplate(
  humerusL: LongBoneStats, humerusR: LongBoneStats, neckBase: Vec3,
  toHead: (p: Vec3) => Vec3, up: Vec3,
): TorsoTemplate {
  const ends = (b: LongBoneStats): [Vec3, Vec3] => {
    const ax = b.pca.axes[0];
    const e0 = v3.add(b.centroid, v3.scale(ax, b.pca.min[0]));
    const e1 = v3.add(b.centroid, v3.scale(ax, b.pca.max[0]));
    // the upper end is the one further along the body's "up" direction
    return v3.dot(e0, up) > v3.dot(e1, up) ? [e0, e1] : [e1, e0];
  };
  const joint = (b: LongBoneStats): [Vec3, Vec3] => {
    const [top, bottom] = ends(b);
    const head = v3.add(top, v3.scale(v3.norm(v3.sub(bottom, top)), 22)); // head radius ≈ 22 mm
    return [toHead(head), toHead(bottom)];
  };
  const [sL, eL] = joint(humerusL), [sR, eR] = joint(humerusR);
  return { shoulderL: sL, shoulderR: sR, elbowL: eL, elbowR: eR, neckBase };
}

/** Forearm (radius and ulna as one rigid pair) in the hand template frame: a RIGHT arm, true centimetres. */
export interface ForearmTemplate { elbow: Vec3; wrist: Vec3 }

/**
 * Place the forearm between a known elbow and the wrist seen in the picture.
 * @param baseR   orientation to start from (row-major, template → camera): the tracked hand's palm
 *                rotation when there is one, so the forearm turns with the hand; otherwise the chest's
 * @param left    mirror the template for the subject's left arm
 * @param scale   true-size scale of the arm (template cm → camera cm)
 * @param out     column-major 4×4, template → camera
 * @returns the wrist in camera space
 */
export function solveForearm(
  tpl: ForearmTemplate, elbow: Vec3, wristImg: ArrayLike<number>, wantDepth: number | null,
  baseR: ArrayLike<number>, left: boolean, scale: number, aspect: number, vfovDeg: number, out: Float32Array,
): Vec3 {
  const th = Math.tan((vfovDeg * Math.PI) / 360);
  const mx = left ? -1 : 1;
  const eT: Vec3 = [mx * tpl.elbow[0], tpl.elbow[1], tpl.elbow[2]];
  const wT: Vec3 = [mx * tpl.wrist[0], tpl.wrist[1], tpl.wrist[2]];
  const lenT = v3.dist(eT, wT) || 1;
  const uT = v3.scale(v3.sub(wT, eT), 1 / lenT);
  const wrist = jointOnRay(elbow, wristImg, scale * lenT, wantDepth, aspect, th);
  const dir = v3.sub(wrist, elbow);
  const len = v3.len(dir) || 1e-3;
  const R = mat3Mul(rotationBetween(mat3MulVec(baseR, uT), v3.scale(dir, 1 / len)), baseR);
  const sl = Math.min(1.3 * scale, Math.max(0.75 * scale, len / lenT));
  const D = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) D[r * 3 + c] = (r === c ? scale : 0) + (sl - scale) * uT[r] * uT[c];
  const RD = mat3Mul(R, D);
  const o = mat3MulVec(RD, eT);
  // linear part RD · Mirror (the mirror flips the template's x)
  out[0] = RD[0] * mx; out[1] = RD[3] * mx; out[2] = RD[6] * mx; out[3] = 0;
  out[4] = RD[1]; out[5] = RD[4]; out[6] = RD[7]; out[7] = 0;
  out[8] = RD[2]; out[9] = RD[5]; out[10] = RD[8]; out[11] = 0;
  out[12] = elbow[0] - o[0]; out[13] = elbow[1] - o[1]; out[14] = elbow[2] - o[2]; out[15] = 1;
  return v3.add(elbow, v3.scale(dir, (sl * lenT) / len));
}

/** Forearm template from the radius and ulna meshes (source millimetres) and the hand template's wrist (cm). */
export function buildForearmTemplate(radius: LongBoneStats, ulna: LongBoneStats, wrist: Vec3): ForearmTemplate {
  const far = (b: LongBoneStats): Vec3 => {
    const ax = b.pca.axes[0];
    const e0 = v3.scale(v3.add(b.centroid, v3.scale(ax, b.pca.min[0])), 0.1);
    const e1 = v3.scale(v3.add(b.centroid, v3.scale(ax, b.pca.max[0])), 0.1);
    return v3.dist(e0, wrist) > v3.dist(e1, wrist) ? e0 : e1;
  };
  // the elbow joint sits between the upper ends of the two bones (the ulna's tip reaches a little past it)
  return { elbow: v3.lerp(far(radius), far(ulna), 0.4), wrist };
}
