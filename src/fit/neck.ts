/**
 * Cervical spine: C1 follows the skull, C7 follows the torso, and the rotation
 * between head and torso is shared out down the chain (most of the turn happens
 * at C1–C2, nodding at the skull base, the rest spread over C3–C7).
 */
import { mat3Mul, mat3MulVec, mat3Transpose, quatToMat3, v3, type Vec3 } from "../math/linalg";
import { mat3ToQuat } from "../math/similarity";

/** Fraction of the head→torso rotation each vertebra has taken up (C1 … C7). */
const SHARE = [0.1, 0.42, 0.52, 0.62, 0.73, 0.86, 1.0];

function slerpFromIdentity(q: ArrayLike<number>, t: number): [number, number, number, number] {
  let [w, x, y, z] = [q[0], q[1], q[2], q[3]];
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const half = Math.acos(Math.min(1, w));
  const s = Math.sin(half);
  if (s < 1e-6) return [1, 0, 0, 0];
  const k = Math.sin(half * t) / s;
  return [Math.cos(half * t), x * k, y * k, z * k];
}

/**
 * @param joints     8 joint centres in the head frame, from the skull base (above C1) down to below C7
 * @param headR      row-major head rotation (head frame → camera)
 * @param torsoR     row-major torso rotation (torso frame → camera); identity = upright, facing the camera
 * @param out        7 column-major 4×4 matrices in the head frame (one per vertebra)
 * @param target     where the lowest joint should end up (head frame), e.g. the top of the tracked
 *                   chest; the gap is shared out down the chain so the neck meets it
 * @returns the head-frame position of the lowest joint
 */
export function solveNeck(
  joints: ArrayLike<number>, headR: ArrayLike<number>, torsoR: ArrayLike<number>, out: Float32Array[],
  target?: ArrayLike<number> | null,
): Vec3 {
  // torso orientation seen from the head frame
  const rel = mat3ToQuat(mat3Mul(mat3Transpose(headR), torsoR));
  let prevJointNow: Vec3 = v3.at(joints, 0); // the skull-base joint moves with the skull
  for (let k = 0; k < 7; k++) {
    const R = quatToMat3(slerpFromIdentity(rel, SHARE[k]));
    const j0 = v3.at(joints, k);
    // x' = jointNow + R · (x − j0)
    const Rj = mat3MulVec(R, j0);
    const m = out[k];
    m[0] = R[0]; m[1] = R[3]; m[2] = R[6]; m[3] = 0;
    m[4] = R[1]; m[5] = R[4]; m[6] = R[7]; m[7] = 0;
    m[8] = R[2]; m[9] = R[5]; m[10] = R[8]; m[11] = 0;
    m[12] = prevJointNow[0] - Rj[0]; m[13] = prevJointNow[1] - Rj[1]; m[14] = prevJointNow[2] - Rj[2]; m[15] = 1;
    // where this vertebra's lower joint ends up
    const j1 = v3.at(joints, k + 1);
    const d = mat3MulVec(R, v3.sub(j1, j0));
    prevJointNow = v3.add(prevJointNow, d);
  }
  if (target) {
    // never stretch the neck by more than a few centimetres to reach a doubtful chest position
    let gap = v3.sub(target, prevJointNow);
    const gl = v3.len(gap);
    if (gl > 4) gap = v3.scale(gap, 4 / gl);
    for (let k = 0; k < 7; k++) {
      const w = (k + 1) / 7;
      out[k][12] += gap[0] * w; out[k][13] += gap[1] * w; out[k][14] += gap[2] * w;
    }
    prevJointNow = v3.add(prevJointNow, gap);
  }
  return prevJointNow;
}

/** Torso orientation: upright and facing the camera, following a little of the head's turn. */
export function torsoFromHead(headR: ArrayLike<number>, follow = 0.2): Float64Array {
  const yaw = Math.atan2(headR[2], headR[8]) * follow;
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return new Float64Array([c, 0, s, 0, 1, 0, -s, 0, c]);
}
