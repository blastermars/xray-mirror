/**
 * Learns the user's own face shape.
 *
 * Every frame the tracker's landmarks are brought into the head frame. Frames
 * with a neutral expression and a roughly frontal view are averaged into a
 * "neutral shape"; the template anatomy is then warped (thin-plate spline) so
 * its skin landmarks land exactly on that shape. Points the tracker cannot see
 * (vault, back of head, neck) follow the measured face proportions.
 */
import { ThinPlateSpline, fitAxisScale, type AxisScale } from "./tps";
import { NUM_FACE } from "../tracking/faceGeometry";
import type { VaultScale } from "./vault";
import { fitSimilarity, invertSimilarity, applySimilarity } from "../math/similarity";

/**
 * Blendshapes that deform the face away from neutral, with how far above the
 * user's own resting level a score may rise before the frame is rejected.
 * Resting levels differ a lot between faces (some read as a 0.5 "smile" at
 * rest), so each score is judged against a running baseline, not against zero.
 */
const EXPRESSION_LIMITS: Record<string, number> = {
  jawOpen: 0.12, jawLeft: 0.15, jawRight: 0.15, jawForward: 0.25,
  mouthSmileLeft: 0.25, mouthSmileRight: 0.25, mouthPucker: 0.35, mouthFunnel: 0.25,
  mouthLeft: 0.25, mouthRight: 0.25, cheekPuff: 0.3,
  mouthFrownLeft: 0.35, mouthFrownRight: 0.35, mouthStretchLeft: 0.35, mouthStretchRight: 0.35,
  browInnerUp: 0.4, browOuterUpLeft: 0.4, browOuterUpRight: 0.4,
  browDownLeft: 0.5, browDownRight: 0.5,
};
/** Mean landmark distance (cm) that, when it persists, means a different face has appeared. */
const NEW_FACE_CM = 0.4;
/** How fast a resting level may creep upward per frame (it drops instantly). */
const BASELINE_RISE = 0.0004;

export interface TemplateFit {
  /** 468×3 template skin landmarks in the head frame */
  landmarks: ArrayLike<number>;
  /** landmark indices used as spline controls */
  controlIdx: ArrayLike<number>;
  /** extra control points on the template away from the face (flat xyz, head frame) */
  anchors: ArrayLike<number>;
}

export class PersonalFit {
  readonly tps: ThinPlateSpline;
  /** the user's neutral face, 468×3, head frame */
  readonly shape = new Float32Array(NUM_FACE * 3);
  /** effective number of neutral frames averaged so far */
  samples = 0;
  /** 0..1, how settled the shape is */
  confidence = 0;
  axis: AxisScale = { s: [1, 1, 1], t: [0, 0, 0] };
  /** bumps whenever the spline changes (renderer re-uploads on change) */
  version = 0;
  /** correction of the cranial vault from the person's outline (see vault.ts) */
  vault: VaultScale = { side: 1, up: 1 };
  /** template point the vault correction scales about: mid-head at eye level */
  static readonly VAULT_PIVOT = [0, 2.5, -3];

  private nCtrl: number;
  private nAnch: number;
  private targets: Float32Array;
  private starve = 0;
  private initialised = false;
  private allIdx: Int32Array;
  private unrotated = new Float32Array(NUM_FACE * 3);

  constructor(private tpl: TemplateFit, lambdaFace = 0.12, lambdaAnchor = 0.6) {
    this.nCtrl = tpl.controlIdx.length;
    this.nAnch = Math.floor(tpl.anchors.length / 3);
    const n = this.nCtrl + this.nAnch;
    const ctrl = new Float32Array(n * 3);
    const lambda = new Float64Array(n);
    for (let k = 0; k < this.nCtrl; k++) {
      const i = tpl.controlIdx[k];
      ctrl[3 * k] = tpl.landmarks[3 * i]; ctrl[3 * k + 1] = tpl.landmarks[3 * i + 1]; ctrl[3 * k + 2] = tpl.landmarks[3 * i + 2];
      lambda[k] = lambdaFace;
    }
    for (let k = 0; k < this.nAnch * 3; k++) ctrl[this.nCtrl * 3 + k] = tpl.anchors[k];
    for (let k = 0; k < this.nAnch; k++) lambda[this.nCtrl + k] = lambdaAnchor;
    this.tps = new ThinPlateSpline(ctrl, lambda);
    this.targets = new Float32Array(n * 3);
    this.allIdx = Int32Array.from(tpl.controlIdx);
    for (let i = 0; i < NUM_FACE * 3; i++) this.shape[i] = tpl.landmarks[i];
  }

  reset() {
    this.samples = 0; this.confidence = 0; this.starve = 0; this.initialised = false;
    this.baseline = {};
    this.vault = { side: 1, up: 1 };
    for (let i = 0; i < NUM_FACE * 3; i++) this.shape[i] = this.tpl.landmarks[i];
  }

  private baseline: Record<string, number> = {};
  /** running mean distance (cm) between incoming neutral frames and the learned shape */
  private mismatch = 0;

  /**
   * How neutral and frontal this frame is, 0..1.
   * @param baseline resting level per blendshape; updated in place when given
   */
  static frameWeight(
    blend: Record<string, number> | null, yawDeg: number, pitchDeg: number, relax = 1,
    baseline?: Record<string, number>,
  ): number {
    // Judge the frame by its single most deviating expression. (Multiplying a penalty for each of
    // twenty noisy scores would reject almost every frame of a real face.) Scores within a third of
    // their limit above the resting level are treated as noise.
    let worst = 0;
    if (blend) {
      for (const k in EXPRESSION_LIMITS) {
        const v = blend[k];
        if (v === undefined) continue;
        let rest = 0;
        if (baseline) {
          const b = baseline[k];
          rest = baseline[k] = b === undefined ? v : Math.min(v, b + BASELINE_RISE);
        }
        const limit = EXPRESSION_LIMITS[k] * relax, dead = 0.35 * limit;
        const r = Math.max(0, v - rest - dead) / (limit - dead);
        if (r > worst) worst = r;
      }
    }
    let w = Math.max(0, 1 - worst);
    // A laptop camera usually looks up at the face, so a steady 10–15° of pitch is normal.
    const ay = Math.abs(yawDeg), ap = Math.abs(pitchDeg);
    w *= Math.max(0, 1 - Math.max(0, ay - 8) / (22 * relax));
    w *= Math.max(0, 1 - Math.max(0, ap - 15) / (20 * relax));
    return w;
  }

  /**
   * Feed one frame.
   * @param local 478×3 landmarks in the head frame
   * @returns the weight this frame was given
   */
  update(local: ArrayLike<number>, blend: Record<string, number> | null, yawDeg: number, pitchDeg: number): number {
    // If the user never relaxes their face, gradually accept less neutral frames.
    const relax = 1 + Math.min(2, this.starve / 90);
    // If the mouth relaxes well below what was taken as "rest" (the user started out talking or
    // smiling), what has been averaged so far is not neutral: let the new frames take over quickly.
    const before = (this.baseline.jawOpen ?? 0) + (this.baseline.mouthSmileLeft ?? 0);
    let w = PersonalFit.frameWeight(blend, yawDeg, pitchDeg, relax, this.baseline);
    const after = (this.baseline.jawOpen ?? 0) + (this.baseline.mouthSmileLeft ?? 0);
    if (this.initialised && before - after > 0.08) this.samples = Math.min(this.samples, 4);

    if (!this.initialised) {
      // Start from whatever we see so the overlay is roughly right immediately.
      for (let i = 0; i < NUM_FACE * 3; i++) this.shape[i] = local[i];
      this.initialised = true;
      this.samples = w > 0.2 ? w : 0.15;
      this.refit();
      return w;
    }
    if (w < 0.05) { this.starve++; return 0; }
    this.starve = 0;

    // Reject frames that disagree wildly with the settled shape (tracking glitches).
    if (this.samples > 20) {
      let err = 0;
      for (let k = 0; k < this.nCtrl; k++) {
        const i = this.tpl.controlIdx[k];
        err += Math.hypot(local[3 * i] - this.shape[3 * i], local[3 * i + 1] - this.shape[3 * i + 1], local[3 * i + 2] - this.shape[3 * i + 2]);
      }
      err /= this.nCtrl;
      // A different face: one frame far from the learned shape is a tracking glitch, but neutral
      // frames that keep disagreeing mean someone else is in front of the camera. Start learning
      // again from them (it takes about a second).
      this.mismatch += 0.12 * (err - this.mismatch);
      if (this.mismatch > NEW_FACE_CM) { this.relearn(); this.mismatch = 0; }
      else {
        if (err > 1.2) return 0; // > 12 mm mean deviation
        w *= Math.max(0.1, 1 - err / 1.2);
      }
    }

    // Running mean: fast at first, then a slow drift-tracking average.
    const cap = 240;
    const rate = w / (Math.min(this.samples, cap) + w);
    for (let i = 0; i < NUM_FACE * 3; i++) this.shape[i] += rate * (local[i] - this.shape[i]);
    this.samples = Math.min(this.samples + w, 1e6);
    this.confidence = Math.min(1, this.samples / 25);
    this.refit();
    return w;
  }

  /**
   * Forget most of what was learned, so the next neutral frames take over quickly. Used when the
   * person in front of the camera has probably changed.
   */
  relearn() {
    this.samples = Math.min(this.samples, 1.5);
    this.confidence = Math.min(1, this.samples / 25);
    this.baseline = {};
    this.vault = { side: 1, up: 1 };
    this.mismatch = 0;
  }

  /** The user's resting level of a blendshape score (0 until seen). */
  restingLevel(key: string): number { return this.baseline[key] ?? 0; }

  /** Set the outline correction of the vault and refit. */
  setVault(v: VaultScale) {
    if (Math.abs(v.side - this.vault.side) < 1e-4 && Math.abs(v.up - this.vault.up) < 1e-4) return;
    this.vault = v;
    this.refit();
  }

  /** Recompute the spline from the current shape. */
  refit() {
    const { nCtrl, nAnch, targets, tpl, shape } = this;
    for (let k = 0; k < nCtrl; k++) {
      const i = tpl.controlIdx[k];
      targets[3 * k] = shape[3 * i]; targets[3 * k + 1] = shape[3 * i + 1]; targets[3 * k + 2] = shape[3 * i + 2];
    }
    // Unseen regions (vault, back of head, neck) follow the face: first the small rotation between
    // the template and the user's shape (the head frame of a differently shaped face comes out
    // slightly tilted), then a per-axis scale for the measured proportions.
    const rigid = fitSimilarity(tpl.landmarks, shape, this.allIdx, null, false);
    const inv = invertSimilarity(rigid);
    const un = this.unrotated;
    for (let k = 0; k < nCtrl; k++) {
      const i = tpl.controlIdx[k];
      const p = applySimilarity(inv, [shape[3 * i], shape[3 * i + 1], shape[3 * i + 2]]);
      un[3 * i] = p[0]; un[3 * i + 1] = p[1]; un[3 * i + 2] = p[2];
    }
    const ax = fitAxisScale(tpl.landmarks, un, this.allIdx);
    // Depth is the tracker's weakest axis: pull it toward the in-plane size.
    const inPlane = 0.5 * (ax.s[0] + ax.s[1]);
    const sz = Math.min(1.12 * inPlane, Math.max(0.88 * inPlane, 0.5 * ax.s[2] + 0.5 * inPlane));
    // keep the depth translation consistent with the changed scale (pivot on the mean control depth)
    let mz = 0;
    for (let k = 0; k < nCtrl; k++) mz += tpl.landmarks[3 * tpl.controlIdx[k] + 2];
    mz /= nCtrl;
    const tz = ax.t[2] + (ax.s[2] - sz) * mz;
    this.axis = { s: [ax.s[0], ax.s[1], sz], t: [ax.t[0], ax.t[1], tz] };
    const P = PersonalFit.VAULT_PIVOT, V = this.vault;
    const vz = 0.5 * (V.side + V.up);
    for (let k = 0; k < nAnch; k++) {
      const o = 3 * (nCtrl + k);
      const ax0 = tpl.anchors[3 * k], ay0 = tpl.anchors[3 * k + 1], az0 = tpl.anchors[3 * k + 2];
      // the outline correction applies to the vault and fades out toward the jaw and neck
      const t = Math.max(0, Math.min(1, (ay0 + 3) / 5));
      const wv = t * t * (3 - 2 * t);
      const x = P[0] + (1 + wv * (V.side - 1)) * (ax0 - P[0]);
      const y = P[1] + (1 + wv * (V.up - 1)) * (ay0 - P[1]);
      const z = P[2] + (1 + wv * (vz - 1)) * (az0 - P[2]);
      const q = applySimilarity(rigid, [
        this.axis.s[0] * x + this.axis.t[0],
        this.axis.s[1] * y + this.axis.t[1],
        this.axis.s[2] * z + this.axis.t[2],
      ]);
      targets[o] = q[0]; targets[o + 1] = q[1]; targets[o + 2] = q[2];
    }
    this.tps.fit(targets);
    this.version++;
  }
}

/** Yaw / pitch / roll in degrees from a row-major head rotation (head frame → camera). */
export function headAngles(R: ArrayLike<number>): { yaw: number; pitch: number; roll: number } {
  // forward axis of the face (+Z of the head frame) in camera space
  const fx = R[2], fy = R[5], fz = R[8];
  const yaw = Math.atan2(fx, fz) * (180 / Math.PI);
  const pitch = Math.asin(Math.max(-1, Math.min(1, -fy))) * (180 / Math.PI);
  const roll = Math.atan2(R[3], R[0]) * (180 / Math.PI);
  return { yaw, pitch, roll };
}
