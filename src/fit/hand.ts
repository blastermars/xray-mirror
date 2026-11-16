/**
 * Hand skeleton placement.
 *
 * The tracker gives 21 image landmarks and 21 metric "world" landmarks (hand
 * shape in metres, oriented as seen by the camera, origin at the hand centre).
 * Only the translation to camera space is unknown, and it is linear:
 *   a_i · (Z_i + tz) = X_i + tx,   b_i · (Z_i + tz) = Y_i + ty.
 * Each joint is then put on its own image ray at that depth, so the bones
 * re-project exactly onto the landmarks while keeping true perspective.
 *
 * Every bone mesh is then carried from the template hand to the tracked joints:
 * carpals move rigidly with the palm, metacarpals and phalanges are aimed and
 * stretched between their two joints (thickness follows overall hand size).
 */
import { fitSimilarity, applySimilarity, type Similarity } from "../math/similarity";
import { mat3Mul, mat3MulVec, rotationBetween, v3, type Vec3 } from "../math/linalg";

export const HAND_JOINTS = 21;

/** Fingertip landmarks sit on the skin pad; the bone ends short of that. */
const TIP_BONE_FRACTION = 0.84;

/**
 * @param image flat [x, y] × 21, normalised (x right, y down)
 * @param world flat [x, y, z] × 21 in metres (x right, y down, z away from camera)
 * @returns 21 joints in camera space, cm (x right, y up, −z forward)
 */
export function liftHand(
  image: ArrayLike<number>,
  world: ArrayLike<number>,
  aspect: number,
  vfovDeg: number,
  out: Float32Array = new Float32Array(HAND_JOINTS * 3),
): Float32Array {
  const t = Math.tan((vfovDeg * Math.PI) / 360);
  // Normal equations for [tx, ty, tz]
  let s11 = 0, s13 = 0, s23 = 0, s33 = 0, r1 = 0, r2 = 0, r3 = 0;
  const A = new Float64Array(HAND_JOINTS), B = new Float64Array(HAND_JOINTS);
  for (let i = 0; i < HAND_JOINTS; i++) {
    const a = (image[2 * i] - 0.5) * 2 * t * aspect;
    const b = (image[2 * i + 1] - 0.5) * 2 * t;
    A[i] = a; B[i] = b;
    const X = world[3 * i] * 100, Y = world[3 * i + 1] * 100, Z = world[3 * i + 2] * 100;
    const ra = a * Z - X; // tx − a·tz = ra
    const rb = b * Z - Y; // ty − b·tz = rb
    s11 += 1; s13 += -a; s23 += -b; s33 += a * a + b * b;
    r1 += ra; r2 += rb; r3 += -a * ra - b * rb;
  }
  // [[s11,0,s13],[0,s11,s23],[s13,s23,s33]] · t = r   (eliminate tx, ty)
  const den = s33 - (s13 * s13 + s23 * s23) / s11;
  let tz = Math.abs(den) > 1e-9 ? (r3 - (s13 * r1 + s23 * r2) / s11) / den : 50;
  if (!(tz > 8)) tz = 8; // never closer than 8 cm
  for (let i = 0; i < HAND_JOINTS; i++) {
    const d = Math.max(4, world[3 * i + 2] * 100 + tz);
    out[3 * i] = A[i] * d;
    out[3 * i + 1] = -B[i] * d;
    out[3 * i + 2] = -d;
  }
  return out;
}

export interface HandBoneDef {
  name: string;
  /** joint index at the proximal end, or −1 when the base follows the palm */
  a: number;
  /** joint index at the distal end */
  b: number;
  /** index of the parent bone in the list (−1 = palm) */
  parent: number;
  /** template proximal / distal points (hand template frame, cm) */
  aT: Vec3;
  bT: Vec3;
  /** true for the last phalanx of a digit */
  tip: boolean;
}

export interface HandTemplate {
  /** 21 template joints, flat xyz (cm), for a RIGHT hand */
  joints: ArrayLike<number>;
  bones: HandBoneDef[];
  /** meshes that move rigidly with the palm (carpals) */
  rigid: string[];
  /** forearm bones, moved with the palm */
  forearm: string[];
}

/** Landmarks defining the rigid palm: wrist, thumb CMC, the four MCP joints. */
const PALM_IDX = [0, 1, 5, 9, 13, 17];

export interface HandPose {
  /** template (mirrored for a left hand) → camera space, uniform scale */
  palm: Similarity;
  /** column-major 4×4 per bone, template mesh space → camera space */
  bones: Float32Array[];
  /** column-major 4×4 for the rigid palm (carpals, forearm) */
  palmMatrix: Float32Array;
  scale: number;
}

/**
 * @param tpl    template (right hand)
 * @param joints 21 tracked joints in camera space (cm)
 * @param left   true when the tracked hand is a left hand (template is mirrored in X)
 */
export function solveHand(tpl: HandTemplate, joints: ArrayLike<number>, left: boolean, out?: HandPose): HandPose {
  const mx = left ? -1 : 1;
  const tj = new Float64Array(HAND_JOINTS * 3);
  for (let i = 0; i < HAND_JOINTS; i++) {
    tj[3 * i] = mx * tpl.joints[3 * i]; tj[3 * i + 1] = tpl.joints[3 * i + 1]; tj[3 * i + 2] = tpl.joints[3 * i + 2];
  }
  const palm = fitSimilarity(tj, joints, PALM_IDX);
  const pose: HandPose = out ?? {
    palm,
    bones: tpl.bones.map(() => new Float32Array(16)),
    palmMatrix: new Float32Array(16),
    scale: palm.s,
  };
  pose.palm = palm; pose.scale = palm.s;

  // palm matrix = Sim · Mirror
  const R = palm.R, s = palm.s;
  const pm = pose.palmMatrix;
  pm[0] = s * R[0] * mx; pm[1] = s * R[3] * mx; pm[2] = s * R[6] * mx; pm[3] = 0;
  pm[4] = s * R[1]; pm[5] = s * R[4]; pm[6] = s * R[7]; pm[7] = 0;
  pm[8] = s * R[2]; pm[9] = s * R[5]; pm[10] = s * R[8]; pm[11] = 0;
  pm[12] = palm.t[0]; pm[13] = palm.t[1]; pm[14] = palm.t[2]; pm[15] = 1;

  const rots: Float64Array[] = [];
  for (let k = 0; k < tpl.bones.length; k++) {
    const bone = tpl.bones[k];
    const aT: Vec3 = [mx * bone.aT[0], bone.aT[1], bone.aT[2]];
    const bT: Vec3 = [mx * bone.bT[0], bone.bT[1], bone.bT[2]];
    const a: Vec3 = bone.a >= 0 ? v3.at(joints, bone.a) : applySimilarity(palm, aT);
    let b: Vec3 = v3.at(joints, bone.b);
    if (bone.tip) b = v3.lerp(a, b, TIP_BONE_FRACTION);
    const uT = v3.norm(v3.sub(bT, aT));
    const lenT = v3.dist(aT, bT) || 1;
    const dir = v3.sub(b, a);
    const len = v3.len(dir) || 1e-3;
    const u = v3.scale(dir, 1 / len);

    // Minimal rotation from the parent's orientation: fingers bend without twisting.
    const Rp = bone.parent >= 0 ? rots[bone.parent] : palm.R;
    const Rk = mat3Mul(rotationBetween(mat3MulVec(Rp, uT), u), Rp);
    rots.push(Rk);

    // p' = a + R · D · M · (p − aT_unmirrored…)  with D = sr·I + (sl − sr)·uT uTᵀ, M = mirror
    const sr = s, sl = len / lenT;
    const D = new Float64Array(9);
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) D[r * 3 + c] = (r === c ? sr : 0) + (sl - sr) * uT[r] * uT[c];
    const RD = mat3Mul(Rk, D);
    // linear part L = RD · Mirror  (mirror scales column 0 by mx)
    const L = [RD[0] * mx, RD[1], RD[2], RD[3] * mx, RD[4], RD[5], RD[6] * mx, RD[7], RD[8]];
    // translation: a − RD · aT (aT already mirrored)
    const o = mat3MulVec(RD, aT);
    const m = pose.bones[k];
    m[0] = L[0]; m[1] = L[3]; m[2] = L[6]; m[3] = 0;
    m[4] = L[1]; m[5] = L[4]; m[6] = L[7]; m[7] = 0;
    m[8] = L[2]; m[9] = L[5]; m[10] = L[8]; m[11] = 0;
    m[12] = a[0] - o[0]; m[13] = a[1] - o[1]; m[14] = a[2] - o[2]; m[15] = 1;
  }
  return pose;
}
