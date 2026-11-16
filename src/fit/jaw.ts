/**
 * Mandible kinematics measured from the chin landmarks.
 *
 * The jaw is modelled as a rotation about the inter-condylar hinge axis, a
 * forward-down glide of the condyles along the articular eminence once the
 * opening passes ~12°, and a small side-to-side swing. The opening angle is
 * found by matching the user's neutral chin points, moved by the model, to the
 * chin points seen in the current frame (both in the head frame).
 */
import { axisAngleMat3, mat3Mul, mat3MulVec, v3, type Vec3 } from "../math/linalg";

/** Chin and lower-border skin landmarks that move with the mandible (no lips). */
export const CHIN_IDX = [152, 175, 199, 200, 148, 377, 176, 400, 149, 378, 150, 379, 171, 396, 208, 428, 32, 262, 18, 201, 421];

export interface JawRig {
  /** condyle centres in the head frame (subject's right, left) */
  hingeR: Vec3;
  hingeL: Vec3;
}

export interface JawState {
  /** opening angle, radians (0 = closed) */
  open: number;
  /** side swing, radians (+ = chin toward subject's left, +X) */
  swing: number;
  /** condyle glide, cm */
  glide: number;
}

export const JAW_MAX_OPEN = 0.62; // ≈ 35°
const GLIDE_START = 0.2; // ≈ 11.5°
const GLIDE_MAX = 1.2; // cm
/** Eminence slope: forward (+Z) and down (−Y), ≈ 35° below horizontal. */
const GLIDE_DIR: Vec3 = [0, -0.5736, 0.8192];

export function glideFor(open: number): number {
  const t = Math.max(0, Math.min(1, (open - GLIDE_START) / (JAW_MAX_OPEN - GLIDE_START)));
  return GLIDE_MAX * t * t * (3 - 2 * t);
}

/** Column-major 4×4 jaw transform in the head frame. */
export function jawMatrix(rig: JawRig, state: JawState, out: Float32Array | number[] = new Float32Array(16)) {
  const c = v3.lerp(rig.hingeR, rig.hingeL, 0.5);
  const axis = v3.norm(v3.sub(rig.hingeL, rig.hingeR)); // ≈ +X; positive angle opens
  const R = mat3Mul(axisAngleMat3([0, 1, 0], state.swing), axisAngleMat3(axis, state.open));
  const Rc = mat3MulVec(R, c);
  const g = state.glide;
  out[0] = R[0]; out[1] = R[3]; out[2] = R[6]; out[3] = 0;
  out[4] = R[1]; out[5] = R[4]; out[6] = R[7]; out[7] = 0;
  out[8] = R[2]; out[9] = R[5]; out[10] = R[8]; out[11] = 0;
  out[12] = c[0] - Rc[0] + GLIDE_DIR[0] * g;
  out[13] = c[1] - Rc[1] + GLIDE_DIR[1] * g;
  out[14] = c[2] - Rc[2] + GLIDE_DIR[2] * g;
  out[15] = 1;
  return out;
}

function applyMat(m: ArrayLike<number>, p: ArrayLike<number>): Vec3 {
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
}

/**
 * Measure the jaw state.
 * @param rig     hinge in the head frame (already warped to the user)
 * @param neutral user's neutral shape, 468×3, head frame
 * @param local   current landmarks in the head frame
 */
export function solveJaw(rig: JawRig, neutral: ArrayLike<number>, local: ArrayLike<number>, idx: ArrayLike<number> = CHIN_IDX): JawState {
  const m = new Float32Array(16);
  const cost = (open: number): number => {
    jawMatrix(rig, { open, swing: 0, glide: glideFor(open) }, m);
    let e = 0;
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k];
      const p = applyMat(m, [neutral[3 * i], neutral[3 * i + 1], neutral[3 * i + 2]]);
      // side swing is solved afterwards, so only match in the sagittal plane here
      const dy = p[1] - local[3 * i + 1], dz = p[2] - local[3 * i + 2];
      e += dy * dy + dz * dz;
    }
    return e;
  };
  // Golden-section search on a unimodal cost.
  let a = -0.06, b = JAW_MAX_OPEN;
  const gr = 0.6180339887;
  let x1 = b - gr * (b - a), x2 = a + gr * (b - a);
  let f1 = cost(x1), f2 = cost(x2);
  for (let it = 0; it < 22; it++) {
    if (f1 < f2) { b = x2; x2 = x1; f2 = f1; x1 = b - gr * (b - a); f1 = cost(x1); }
    else { a = x1; x1 = x2; f1 = f2; x2 = a + gr * (b - a); f2 = cost(x2); }
  }
  const open = Math.max(0, 0.5 * (a + b));
  const glide = glideFor(open);

  // Side swing: mean lateral offset of the chin points over their lever arm from the hinge.
  jawMatrix(rig, { open, swing: 0, glide }, m);
  const c = v3.lerp(rig.hingeR, rig.hingeL, 0.5);
  let dx = 0, lever = 0;
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    const p = applyMat(m, [neutral[3 * i], neutral[3 * i + 1], neutral[3 * i + 2]]);
    dx += local[3 * i] - p[0];
    lever += p[2] - c[2] - GLIDE_DIR[2] * glide; // the swing pivots about the un-glided hinge
  }
  dx /= idx.length; lever /= idx.length;
  const swing = lever > 1 ? Math.max(-0.14, Math.min(0.14, Math.atan2(dx, lever))) : 0;
  return { open, swing, glide };
}
