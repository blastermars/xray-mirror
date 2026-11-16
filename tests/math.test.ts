import { describe, it, expect } from "vitest";
import canonical from "../src/data/canonicalFace.json";
import { axisAngleMat3, mat3Mul, mat3MulVec, invertMatrix, rotationBetween, v3, type Vec3 } from "../src/math/linalg";
import { fitSimilarity, applySimilarity, invertSimilarity, mat3ToQuat, type Similarity } from "../src/math/similarity";
import { FaceGeometry, projectToImage, refinePose, RIGID_IDX, RIGID_W, NUM_LANDMARKS } from "../src/tracking/faceGeometry";
import { ThinPlateSpline } from "../src/fit/tps";
import { PersonalFit, headAngles } from "../src/fit/neutralShape";
import { jawMatrix, solveJaw, glideFor, CHIN_IDX, type JawRig } from "../src/fit/jaw";
import { liftHand, solveHand, type HandTemplate, type HandBoneDef } from "../src/fit/hand";
import { OneEuro, OneEuroQuat } from "../src/tracking/filters";
import { loopSubdivide } from "../src/render/subdivide";
import { solveTorso, torsoRotation, solveForearm, type TorsoTemplate, type BodyObservation, type ForearmTemplate } from "../src/fit/torso";
import { solveNeck } from "../src/fit/neck";
import { traceOutline, modelOutline, updateVault, RAY_ANGLES, FULL_FRAME, SEG_HAIR } from "../src/fit/vault";

const CANON = Float32Array.from(canonical.vertices);

/** deterministic pseudo-random numbers */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function makePose(yaw: number, pitch: number, roll: number, t: Vec3, s = 1): Similarity {
  const R = mat3Mul(axisAngleMat3([0, 0, 1], roll), mat3Mul(axisAngleMat3([0, 1, 0], yaw), axisAngleMat3([1, 0, 0], pitch)));
  return { s, R, t };
}

/** Build tracker-style landmarks (normalised x, y and weak-perspective z) from head-frame points and a true pose. */
function synthLandmarks(shape: ArrayLike<number>, pose: Similarity, aspect: number, vfov: number): Float32Array {
  const n = Math.floor(shape.length / 3);
  const cam: Vec3[] = [];
  let meanDepth = 0;
  for (let i = 0; i < n; i++) {
    const p = applySimilarity(pose, v3.at(shape, i));
    cam.push(p);
    meanDepth += -p[2];
  }
  meanDepth /= n;
  const hNear = 2 * Math.tan((vfov * Math.PI) / 360);
  const wNear = hNear * aspect;
  const lm = new Float32Array(NUM_LANDMARKS * 3);
  for (let i = 0; i < n; i++) {
    const [x, y] = projectToImage(cam[i], aspect, vfov);
    lm[3 * i] = x; lm[3 * i + 1] = y;
    lm[3 * i + 2] = ((-cam[i][2] - meanDepth) * (1 / meanDepth)) / wNear;
  }
  for (let i = n; i < NUM_LANDMARKS; i++) { lm[3 * i] = 0.5; lm[3 * i + 1] = 0.5; }
  return lm;
}

describe("similarity", () => {
  it("recovers a known scaled rigid transform", () => {
    const r = rng(1);
    const src: number[] = [];
    for (let i = 0; i < 60; i++) src.push(r() * 10 - 5, r() * 10 - 5, r() * 10 - 5);
    const T = makePose(0.4, -0.3, 0.2, [3, -2, -50], 1.3);
    const dst: number[] = [];
    for (let i = 0; i < 60; i++) dst.push(...applySimilarity(T, v3.at(src, i)));
    const F = fitSimilarity(src, dst);
    expect(F.s).toBeCloseTo(1.3, 6);
    for (let i = 0; i < 9; i++) expect(F.R[i]).toBeCloseTo(T.R[i], 6);
    for (let i = 0; i < 3; i++) expect(F.t[i]).toBeCloseTo(T.t[i], 5);
    const inv = invertSimilarity(F);
    const back = applySimilarity(inv, v3.at(dst, 3));
    for (let d = 0; d < 3; d++) expect(back[d]).toBeCloseTo(src[9 + d], 5);
  });

  it("honours weights and index subsets", () => {
    const src = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 5, 5, 5];
    const T = makePose(0.1, 0.2, 0.3, [1, 2, 3]);
    const dst: number[] = [];
    for (let i = 0; i < 5; i++) dst.push(...applySimilarity(T, v3.at(src, i)));
    dst[12] += 40; // outlier on point 4
    const F = fitSimilarity(src, dst, [0, 1, 2, 3], [1, 1, 1, 1]);
    for (let i = 0; i < 9; i++) expect(F.R[i]).toBeCloseTo(T.R[i], 6);
  });

  it("quaternion round trip", () => {
    const R = makePose(0.7, 0.2, -0.4, [0, 0, 0]).R;
    const q = mat3ToQuat(R);
    expect(Math.hypot(...q)).toBeCloseTo(1, 9);
  });

  it("rotationBetween maps a onto b", () => {
    const a = v3.norm([1, 2, 3]), b = v3.norm([-2, 0.5, 1]);
    const r = mat3MulVec(rotationBetween(a, b), a);
    for (let d = 0; d < 3; d++) expect(r[d]).toBeCloseTo(b[d], 9);
    const opp = mat3MulVec(rotationBetween(a, v3.scale(a, -1)), a);
    for (let d = 0; d < 3; d++) expect(opp[d]).toBeCloseTo(-a[d], 6);
  });

  it("invertMatrix", () => {
    const A = new Float64Array([4, 7, 2, 3, 6, 1, 2, 5, 3]);
    const I = invertMatrix(A, 3);
    const P = mat3Mul(A, I);
    const id = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    for (let i = 0; i < 9; i++) expect(P[i]).toBeCloseTo(id[i], 9);
  });
});

describe("face geometry", () => {
  const geo = new FaceGeometry(CANON);
  const cases: Array<[number, number, number, Vec3, number, number]> = [
    [0, 0, 0, [0, 0, -55], 16 / 9, 45],
    [0.5, 0.1, 0.05, [6, -3, -48], 16 / 9, 45],
    [-0.6, -0.25, 0.2, [-9, 4, -70], 4 / 3, 50],
    [0.3, 0.3, -0.3, [2, 1, -38], 9 / 16, 62],
  ];
  it.each(cases)("recovers pose yaw=%f pitch=%f roll=%f", (yaw, pitch, roll, t, aspect, vfov) => {
    const truth = makePose(yaw, pitch, roll, t);
    const lm = synthLandmarks(CANON, truth, aspect, vfov);
    const { pose, metric, local } = geo.solve(lm, aspect, vfov);
    // every landmark re-projects exactly onto its 2D position
    for (const i of [1, 33, 152, 234, 454, 10]) {
      const [x, y] = projectToImage(v3.at(metric, i), aspect, vfov);
      expect(x).toBeCloseTo(lm[3 * i], 5);
      expect(y).toBeCloseTo(lm[3 * i + 1], 5);
    }
    // rotation within 1.5°, depth within 3 %
    const dR = mat3Mul(pose.R, [truth.R[0], truth.R[3], truth.R[6], truth.R[1], truth.R[4], truth.R[7], truth.R[2], truth.R[5], truth.R[8]]);
    const ang = Math.acos(Math.min(1, (dR[0] + dR[4] + dR[8] - 1) / 2)) * (180 / Math.PI);
    expect(ang).toBeLessThan(1.5);
    expect(Math.abs(pose.t[2] - t[2]) / Math.abs(t[2])).toBeLessThan(0.03);
    expect(pose.s).toBeGreaterThan(0.95);
    expect(pose.s).toBeLessThan(1.05);
    // head-frame landmarks come back as the canonical shape (within 2 mm)
    let err = 0;
    for (let i = 0; i < 468; i++) err += v3.dist(v3.at(local, i), v3.at(CANON, i));
    expect(err / 468).toBeLessThan(0.2);
  });

  it("pose ignores jaw opening", () => {
    const rig: JawRig = { hingeR: [-5, 0.8, -3.8], hingeL: [5, 0.8, -3.8] };
    const open = new Float32Array(CANON);
    const m = jawMatrix(rig, { open: 0.4, swing: 0, glide: glideFor(0.4) });
    // move everything below the mouth line with the jaw
    for (let i = 0; i < 468; i++) {
      if (CANON[3 * i + 1] < -4.5) {
        const p = v3.at(CANON, i);
        open[3 * i] = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
        open[3 * i + 1] = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
        open[3 * i + 2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
      }
    }
    const truth = makePose(0.2, 0.05, 0, [1, 0, -50]);
    const a = geo.solve(synthLandmarks(CANON, truth, 16 / 9, 45), 16 / 9, 45);
    const b = geo.solve(synthLandmarks(open, truth, 16 / 9, 45), 16 / 9, 45);
    const ha = headAngles(a.pose.R), hb = headAngles(b.pose.R);
    expect(Math.abs(ha.pitch - hb.pitch)).toBeLessThan(0.6);
    expect(Math.abs(ha.yaw - hb.yaw)).toBeLessThan(0.6);
  });
});

describe("pose refinement from 2D landmarks", () => {
  it.each([[0.5, -0.2, 0.1], [-0.45, 0.3, -0.2], [0.05, 0.02, 0]])("recovers yaw=%f pitch=%f roll=%f from a poor start", (yaw, pitch, roll) => {
    const aspect = 16 / 9, vfov = 45;
    const truth = makePose(yaw, pitch, roll, [4, -2, -52]);
    const lm = new Float32Array(NUM_LANDMARKS * 3);
    for (let i = 0; i < 468; i++) {
      const [x, y] = projectToImage(applySimilarity(truth, v3.at(CANON, i)), aspect, vfov);
      lm[3 * i] = x; lm[3 * i + 1] = y; lm[3 * i + 2] = 123; // depth values must be ignored
    }
    // start several degrees and a few centimetres off
    const start = makePose(yaw + 0.14, pitch - 0.12, roll + 0.08, [5.5, -1, -58]);
    const got = refinePose(CANON, RIGID_IDX, RIGID_W, lm, aspect, vfov, start);
    for (let i = 0; i < 9; i++) expect(got.R[i]).toBeCloseTo(truth.R[i], 4);
    for (let i = 0; i < 3; i++) expect(got.t[i]).toBeCloseTo(truth.t[i], 2);
  });

  it("folds the starting scale into depth", () => {
    const aspect = 4 / 3, vfov = 50;
    const truth = makePose(0.2, 0.1, 0, [0, 0, -45]);
    const lm = new Float32Array(NUM_LANDMARKS * 3);
    for (let i = 0; i < 468; i++) {
      const [x, y] = projectToImage(applySimilarity(truth, v3.at(CANON, i)), aspect, vfov);
      lm[3 * i] = x; lm[3 * i + 1] = y;
    }
    const start = { s: 1.1, R: truth.R, t: [0, 0, -49.5] as Vec3 };
    const got = refinePose(CANON, RIGID_IDX, RIGID_W, lm, aspect, vfov, start);
    expect(got.s).toBe(1);
    expect(got.t[2]).toBeCloseTo(-45, 2);
  });
});

describe("thin-plate spline", () => {
  const r = rng(7);
  const ctrl: number[] = [];
  for (let i = 0; i < 40; i++) ctrl.push(r() * 16 - 8, r() * 18 - 9, r() * 10 - 3);

  it("interpolates the control points", () => {
    const tps = new ThinPlateSpline(ctrl, 0);
    const tgt = ctrl.map((v, i) => v * 1.05 + (r() - 0.5) * 0.8 + (i % 3 === 0 ? 0.3 : 0));
    tps.fit(tgt);
    for (let i = 0; i < 40; i++) {
      const p = tps.apply(v3.at(ctrl, i));
      for (let d = 0; d < 3; d++) expect(p[d]).toBeCloseTo(tgt[3 * i + d], 3);
    }
  });

  it("reproduces an affine map everywhere", () => {
    const tps = new ThinPlateSpline(ctrl, 0);
    const aff = (p: ArrayLike<number>): Vec3 => [1.1 * p[0] + 0.5, 0.9 * p[1] - 1, 1.05 * p[2] + 0.1 * p[0] + 2];
    const tgt: number[] = [];
    for (let i = 0; i < 40; i++) tgt.push(...aff(v3.at(ctrl, i)));
    tps.fit(tgt);
    for (const q of [[0, 0, 0], [0, 14, -15], [20, -30, 5]] as Vec3[]) {
      const p = tps.apply(q), e = aff(q);
      for (let d = 0; d < 3; d++) expect(p[d]).toBeCloseTo(e[d], 2);
    }
    let w = 0;
    for (const v of tps.weights) w += Math.abs(v);
    expect(w).toBeLessThan(1e-2);
  });

  it("regularisation smooths noise instead of chasing it", () => {
    const noisy = ctrl.map((v) => v + (r() - 0.5) * 0.6);
    const exact = new ThinPlateSpline(ctrl, 0);
    const smooth = new ThinPlateSpline(ctrl, 0.5);
    exact.fit(noisy); smooth.fit(noisy);
    let wExact = 0, wSmooth = 0, resid = 0;
    for (let i = 0; i < exact.weights.length; i++) { wExact += exact.weights[i] ** 2; wSmooth += smooth.weights[i] ** 2; }
    for (let i = 0; i < 40; i++) resid += v3.dist(smooth.apply(v3.at(ctrl, i)), v3.at(noisy, i));
    expect(wSmooth).toBeLessThan(wExact);
    expect(resid / 40).toBeGreaterThan(0.01);
    expect(resid / 40).toBeLessThan(0.5); // stays closer to the data than the noise amplitude
  });
});

describe("personal fit", () => {
  it("warps the template onto a differently proportioned face", () => {
    const controlIdx: number[] = [];
    for (let i = 0; i < 468; i += 6) controlIdx.push(i);
    const anchors = [0, 14, -6, 0, 4, -16, -8, 5, -7, 8, 5, -7, 0, -10, -8];
    const fit = new PersonalFit({ landmarks: CANON, controlIdx, anchors });
    const user = new Float32Array(NUM_LANDMARKS * 3);
    for (let i = 0; i < 468; i++) {
      user[3 * i] = CANON[3 * i] * 1.1;
      user[3 * i + 1] = CANON[3 * i + 1] * 0.94 + 0.3;
      user[3 * i + 2] = CANON[3 * i + 2] * 1.02;
    }
    for (let f = 0; f < 40; f++) fit.update(user, { jawOpen: 0.01 }, 2, 1);
    expect(fit.confidence).toBe(1);
    // skin landmarks land on the user's (within 1 mm; regularised, so not exact)
    let err = 0;
    for (const i of controlIdx) err += v3.dist(fit.tps.apply(v3.at(CANON, i)), v3.at(user, i));
    expect(err / controlIdx.length).toBeLessThan(0.1);
    // the vault follows the measured proportions
    const top = fit.tps.apply([0, 14, -6]);
    expect(top[0]).toBeCloseTo(0, 0);
    expect(top[1]).toBeGreaterThan(14 * 0.94 + 0.3 - 0.6);
    expect(top[1]).toBeLessThan(14 * 0.94 + 0.3 + 0.6);
    const side = fit.tps.apply([8, 5, -7]);
    expect(side[0]).toBeGreaterThan(8 * 1.1 - 0.5);
    expect(side[0]).toBeLessThan(8 * 1.1 + 0.5);
  });

  it("relearns when a different face appears", () => {
    const controlIdx: number[] = [];
    for (let i = 0; i < 468; i += 6) controlIdx.push(i);
    const fit = new PersonalFit({ landmarks: CANON, controlIdx, anchors: [0, 14, -6, 0, 4, -16] });
    const face = (sx: number, sy: number) => {
      const f = new Float32Array(NUM_LANDMARKS * 3);
      for (let i = 0; i < 468; i++) { f[3 * i] = CANON[3 * i] * sx; f[3 * i + 1] = CANON[3 * i + 1] * sy; f[3 * i + 2] = CANON[3 * i + 2]; }
      return f;
    };
    const first = face(1.0, 1.0), second = face(1.14, 0.9);
    for (let f = 0; f < 300; f++) fit.update(first, { jawOpen: 0 }, 0, 0);
    expect(fit.confidence).toBe(1);
    // the second person: within three seconds of frames the learned shape is theirs
    for (let f = 0; f < 90; f++) fit.update(second, { jawOpen: 0 }, 0, 0);
    let err = 0;
    for (const i of controlIdx) err += v3.dist(v3.at(fit.shape, i), v3.at(second, i));
    expect(err / controlIdx.length).toBeLessThan(0.05);
    expect(fit.confidence).toBe(1);
  });

  it("ignores non-neutral frames", () => {
    expect(PersonalFit.frameWeight({ jawOpen: 0.5 }, 0, 0)).toBe(0);
    expect(PersonalFit.frameWeight({ jawOpen: 0.0 }, 45, 0)).toBe(0);
    expect(PersonalFit.frameWeight({ jawOpen: 0.01 }, 3, 2)).toBeGreaterThan(0.8);
    // a face that reads as "smiling" at rest is still neutral relative to its own baseline
    const base: Record<string, number> = {};
    expect(PersonalFit.frameWeight({ mouthSmileLeft: 0.6 }, 0, 0, 1, base)).toBe(1);
    expect(PersonalFit.frameWeight({ mouthSmileLeft: 0.95 }, 0, 0, 1, base)).toBe(0);
    expect(PersonalFit.frameWeight({ mouthSmileLeft: 0.3 }, 0, 0, 1, base)).toBe(1);
    expect(base.mouthSmileLeft).toBeCloseTo(0.3, 5);
  });
});

describe("jaw", () => {
  const rig: JawRig = { hingeR: [-5.2, 0.8, -3.8], hingeL: [5.2, 0.8, -3.8] };
  it.each([[0, 0], [0.1, 0], [0.3, 0.05], [0.5, -0.08]])("recovers open=%f swing=%f", (open, swing) => {
    const m = jawMatrix(rig, { open, swing, glide: glideFor(open) });
    const local = new Float32Array(CANON);
    for (const i of CHIN_IDX) {
      const p = v3.at(CANON, i);
      local[3 * i] = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
      local[3 * i + 1] = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
      local[3 * i + 2] = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
    }
    const s = solveJaw(rig, CANON, local);
    expect(s.open).toBeCloseTo(open, 2);
    expect(s.swing).toBeCloseTo(swing, 2);
  });

  it("opening moves the chin down and back", () => {
    const m = jawMatrix(rig, { open: 0.3, swing: 0, glide: 0 });
    const chin = v3.at(CANON, 152);
    const y = m[1] * chin[0] + m[5] * chin[1] + m[9] * chin[2] + m[13];
    const z = m[2] * chin[0] + m[6] * chin[1] + m[10] * chin[2] + m[14];
    expect(y).toBeLessThan(chin[1]);
    expect(z).toBeLessThan(chin[2]);
  });
});

describe("hand", () => {
  // crude right-hand template (cm): wrist at origin, fingers along +Y, thumb toward −X
  const J: number[] = [];
  const set = (i: number, p: Vec3) => { J[3 * i] = p[0]; J[3 * i + 1] = p[1]; J[3 * i + 2] = p[2]; };
  set(0, [0, 0, 0]);
  set(1, [-2.5, 1.5, 0.5]); set(2, [-4.5, 4, 1]); set(3, [-5.5, 6.5, 1.2]); set(4, [-6, 8.5, 1.3]);
  const fx = [-2.5, -0.8, 1, 2.6];
  for (let f = 0; f < 4; f++) {
    const b = 5 + 4 * f;
    set(b, [fx[f], 8.5, 0]); set(b + 1, [fx[f], 12.5, 0.2]); set(b + 2, [fx[f], 15, 0.3]); set(b + 3, [fx[f], 17, 0.35]);
  }
  const bones: HandBoneDef[] = [];
  const add = (name: string, a: number, b: number, parent: number, tip = false, base?: Vec3) =>
    bones.push({ name, a: base ? -1 : a, b, parent, aT: base ?? v3.at(J, a), bT: v3.at(J, b), tip }) - 1;
  const mc1 = add("mc1", 1, 2, -1); const pp1 = add("pp1", 2, 3, mc1); add("dp1", 3, 4, pp1, true);
  for (let f = 0; f < 4; f++) {
    const b = 5 + 4 * f;
    const mc = add(`mc${f + 2}`, 0, b, -1, false, [fx[f] * 0.6, 2.5, 0]);
    const pp = add(`pp${f + 2}`, b, b + 1, mc); const mp = add(`mp${f + 2}`, b + 1, b + 2, pp); add(`dp${f + 2}`, b + 2, b + 3, mp, true);
  }
  const tpl: HandTemplate = { joints: J, bones, rigid: [], forearm: [] };

  it("lifts landmarks to camera space with exact reprojection", () => {
    const aspect = 16 / 9, vfov = 45;
    const T = makePose(0.4, -0.2, 0.3, [8, -4, -45]);
    const cam: Vec3[] = [];
    for (let i = 0; i < 21; i++) cam.push(applySimilarity(T, v3.at(J, i)));
    const c = cam.reduce((a, p) => v3.add(a, p), [0, 0, 0] as Vec3).map((v) => v / 21);
    const image: number[] = [], world: number[] = [];
    for (const p of cam) {
      image.push(...projectToImage(p, aspect, vfov));
      world.push((p[0] - c[0]) / 100, -(p[1] - c[1]) / 100, -(p[2] - c[2]) / 100);
    }
    const out = liftHand(image, world, aspect, vfov);
    for (let i = 0; i < 21; i++) for (let d = 0; d < 3; d++) expect(out[3 * i + d]).toBeCloseTo(cam[i][d], 2);
  });

  it.each([[false], [true]])("places bones between tracked joints (left=%s)", (left) => {
    const mx = left ? -1 : 1;
    const T = makePose(-0.5, 0.3, 0.8, [-5, 2, -40], 1.15);
    const joints = new Float32Array(63);
    for (let i = 0; i < 21; i++) {
      const p = v3.at(J, i);
      // curl the fingers a little so bones are not collinear with the template
      const curl = i >= 5 && (i - 5) % 4 >= 1 ? 0.8 * ((i - 5) % 4) : 0;
      joints.set(applySimilarity(T, [mx * p[0], p[1], p[2] + curl]), 3 * i);
    }
    const pose = solveHand(tpl, joints, left);
    expect(pose.scale).toBeGreaterThan(1.0);
    bones.forEach((bone, k) => {
      const m = pose.bones[k];
      const map = (p: Vec3): Vec3 => [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
      ];
      const b = map(bone.bT);
      const jb = v3.at(joints, bone.b);
      if (bone.a >= 0) {
        const a = map(bone.aT), ja = v3.at(joints, bone.a);
        expect(v3.dist(a, ja)).toBeLessThan(1e-3);
        const want = bone.tip ? v3.lerp(ja, jb, 0.84) : jb;
        expect(v3.dist(b, want)).toBeLessThan(1e-3);
      } else {
        expect(v3.dist(b, jb)).toBeLessThan(1e-3);
      }
      // mirrored hands must flip handedness, right hands must not
      const det =
        m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9]) + m[8] * (m[1] * m[6] - m[2] * m[5]);
      expect(Math.sign(det)).toBe(mx);
    });
  });
});

describe("filters", () => {
  it("one-euro converges and follows", () => {
    const f = new OneEuro(1, 1.5, 0.05);
    let y = 0;
    for (let i = 0; i < 120; i++) y = f.filter([10], 1 / 30)[0];
    expect(y).toBeCloseTo(10, 3);
  });
  it("quaternion filter converges to the target rotation", () => {
    const f = new OneEuroQuat();
    const a = mat3ToQuat(makePose(0, 0, 0, [0, 0, 0]).R);
    const b = mat3ToQuat(makePose(0.6, -0.2, 0.3, [0, 0, 0]).R);
    f.filter(a, 1 / 30);
    let q = a;
    for (let i = 0; i < 90; i++) q = f.filter(b, 1 / 30);
    const dot = Math.abs(q[0] * b[0] + q[1] * b[1] + q[2] * b[2] + q[3] * b[3]);
    expect(dot).toBeGreaterThan(0.99999);
  });
});

describe("vault outline fit", () => {
  const W = 256, H = 256, aspect = 1;
  /** an elliptical head of the given class on a background of zeros */
  const headMask = (rx: number, ry: number, cls = 3) => {
    const m = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const u = (x + 0.5) / W - 0.5, v = (y + 0.5) / H - 0.5;
      if ((u / rx) ** 2 + (v / ry) ** 2 < 1) m[y * W + x] = cls;
    }
    return m;
  };
  const ellipsePoints = (rx: number, ry: number) => {
    const pts: number[] = [];
    for (let a = 0; a < 720; a++) { const t = (a * Math.PI) / 360; pts.push(0.5 + rx * Math.cos(t), 0.5 + ry * Math.sin(t)); }
    return pts;
  };

  it("measures the edge of a mask along every ray", () => {
    const o = traceOutline(headMask(0.2, 0.3), W, H, FULL_FRAME, 0.5, 0.5, 0, -1, aspect);
    const up = o.r[RAY_ANGLES.indexOf(0)], side = o.r[RAY_ANGLES.indexOf(90)];
    // within the built-in edge allowance (2 px = 0.008) plus a pixel
    expect(up).toBeGreaterThan(0.3 - 0.005); expect(up).toBeLessThan(0.3 + 0.015);
    expect(side).toBeGreaterThan(0.2 - 0.005); expect(side).toBeLessThan(0.2 + 0.015);
    expect(o.hair.every((h) => h === 0)).toBe(true);
  });

  it("returns nothing when the origin is off the person or the edge leaves the picture", () => {
    expect(traceOutline(headMask(0.2, 0.3), W, H, FULL_FRAME, 0.05, 0.05, 0, -1, aspect).r.every((r) => r === 0)).toBe(true);
    const big = traceOutline(headMask(0.9, 0.9), W, H, FULL_FRAME, 0.5, 0.5, 0, -1, aspect);
    expect(big.r[RAY_ANGLES.indexOf(0)]).toBe(0);
  });

  it("shrinks a model that is larger than the outline, and converges", () => {
    const seen = traceOutline(headMask(0.2, 0.3), W, H, FULL_FRAME, 0.5, 0.5, 0, -1, aspect);
    let v = { side: 1, up: 1 };
    // the model starts 10 % too tall; its outline scales with the current correction
    for (let i = 0; i < 40; i++) {
      const model = modelOutline(ellipsePoints(0.2 * v.side, 0.33 * v.up), 0.5, 0.5, 0, -1, aspect);
      v = updateVault(v, seen, model) ?? v;
    }
    expect(v.up).toBeGreaterThan(0.89); expect(v.up).toBeLessThan(0.96);
  });

  it("under hair the outline is only an upper bound", () => {
    const hairy = traceOutline(headMask(0.2, 0.3, SEG_HAIR), W, H, FULL_FRAME, 0.5, 0.5, 0, -1, aspect);
    expect(hairy.hair[RAY_ANGLES.indexOf(0)]).toBe(1);
    // model smaller than the hair outline: leave it alone
    const small = modelOutline(ellipsePoints(0.17, 0.25), 0.5, 0.5, 0, -1, aspect);
    const a = updateVault({ side: 1, up: 1 }, hairy, small)!;
    expect(a.up).toBe(1); expect(a.side).toBe(1);
    // model larger than even the hair: shrink
    const large = modelOutline(ellipsePoints(0.24, 0.36), 0.5, 0.5, 0, -1, aspect);
    const b = updateVault({ side: 1, up: 1 }, hairy, large)!;
    expect(b.up).toBeLessThan(1); expect(b.side).toBeLessThan(1);
  });
});

describe("loop subdivision", () => {
  // a cube as 12 triangles
  const P = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1];
  const I = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  it("quadruples the triangles and stays closed", () => {
    const s = loopSubdivide(P, I);
    expect(s.indices.length).toBe(I.length * 4);
    expect(s.positions.length / 3).toBe(8 + 18); // 8 corners + one per edge
    const count = new Map<string, number>();
    for (let t = 0; t < s.indices.length; t += 3)
      for (const [a, b] of [[0, 1], [1, 2], [2, 0]]) {
        const x = s.indices[t + a], y = s.indices[t + b];
        const k = x < y ? `${x}_${y}` : `${y}_${x}`;
        count.set(k, (count.get(k) ?? 0) + 1);
      }
    expect([...count.values()].every((n) => n === 2)).toBe(true);
  });
  it("rounds the shape without moving it", () => {
    const s = loopSubdivide(P, I);
    let cx = 0, cy = 0, cz = 0;
    const n = s.positions.length / 3;
    for (let i = 0; i < n; i++) { cx += s.positions[3 * i]; cy += s.positions[3 * i + 1]; cz += s.positions[3 * i + 2]; }
    expect(cx / n).toBeCloseTo(0.5, 1); expect(cy / n).toBeCloseTo(0.5, 1); expect(cz / n).toBeCloseTo(0.5, 1);
    // corners are pulled in, nothing leaves the original box
    for (let i = 0; i < s.positions.length; i++) { expect(s.positions[i]).toBeGreaterThan(-1e-6); expect(s.positions[i]).toBeLessThan(1 + 1e-6); }
    expect(s.positions[0]).toBeGreaterThan(0.05);
  });
  it("leaves open edges where they are", () => {
    const s = loopSubdivide([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2]);
    expect(Array.from(s.positions.slice(0, 9))).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    expect(s.positions[9]).toBeCloseTo(0.5, 6);
  });
});

describe("torso", () => {
  const tpl: TorsoTemplate = {
    shoulderL: [19, -22, -6], shoulderR: [-19, -22, -6],
    elbowL: [22, -52, -7], elbowR: [-22, -52, -7],
    neckBase: [0, -16, -7],
  };
  const aspect = 16 / 9, vfov = 45;
  const observe = (truth: Similarity, elbowL: Vec3, elbowR: Vec3): BodyObservation => {
    const cam = (q: Vec3) => applySimilarity(truth, q);
    const sL = cam(tpl.shoulderL), sR = cam(tpl.shoulderR);
    const hip = cam([0, -70, -6]);
    const m = (p: Vec3): Vec3 => [p[0] / 100, p[1] / 100, p[2] / 100]; // tracker gives metres
    return {
      shoulderL: projectToImage(sL, aspect, vfov), shoulderR: projectToImage(sR, aspect, vfov),
      elbowL: projectToImage(elbowL, aspect, vfov), elbowR: projectToImage(elbowR, aspect, vfov),
      worldShoulderL: m(sL), worldShoulderR: m(sR), worldElbowL: m(elbowL), worldElbowR: m(elbowR),
      worldHipMid: m(hip),
    };
  };

  it.each([[0, 1.0], [0.3, 1.12], [-0.4, 0.9]])("recovers the chest at yaw=%f scale=%f", (yaw, scale) => {
    const truth = makePose(yaw, 0, 0, [3, 4, -70], scale);
    const eL = applySimilarity(truth, tpl.elbowL), eR = applySimilarity(truth, tpl.elbowR);
    const obs = observe(truth, eL, eR);
    const neckDepth = -applySimilarity(truth, tpl.neckBase)[2];
    const got = solveTorso(tpl, obs, neckDepth, aspect, vfov);
    expect(got.pose.s).toBeCloseTo(scale, 2);
    for (let d = 0; d < 3; d++) {
      expect(got.shoulderL[d]).toBeCloseTo(applySimilarity(truth, tpl.shoulderL)[d], 1);
      expect(got.shoulderR[d]).toBeCloseTo(applySimilarity(truth, tpl.shoulderR)[d], 1);
    }
    // the upper arm reaches the elbow
    const m = got.humerusL, e = tpl.elbowL;
    const mapped: Vec3 = [m[0] * e[0] + m[4] * e[1] + m[8] * e[2] + m[12], m[1] * e[0] + m[5] * e[1] + m[9] * e[2] + m[13], m[2] * e[0] + m[6] * e[1] + m[10] * e[2] + m[14]];
    expect(v3.dist(mapped, eL)).toBeLessThan(0.3);
  });

  it("ends a raised arm on the tracked elbow in the picture, at bone length", () => {
    const truth = makePose(0, 0, 0, [0, 0, -70]);
    const sL = applySimilarity(truth, tpl.shoulderL);
    // arm raised sideways and a little toward the camera, at its true length
    const lenTrue = v3.dist(tpl.shoulderL, tpl.elbowL);
    const eL: Vec3 = [sL[0] + lenTrue * Math.cos(0.5), sL[1], sL[2] + lenTrue * Math.sin(0.5)];
    const obs = observe(truth, eL, applySimilarity(truth, tpl.elbowR));
    // the tracker's depth for the elbow is only rough: here it is 6 cm off
    obs.worldElbowL = [eL[0] / 100, eL[1] / 100, (eL[2] - 6) / 100];
    const got = solveTorso(tpl, obs, 70 + 7, aspect, vfov);
    const m = got.humerusL, e = tpl.elbowL;
    const mapped: Vec3 = [m[0] * e[0] + m[4] * e[1] + m[8] * e[2] + m[12], m[1] * e[0] + m[5] * e[1] + m[9] * e[2] + m[13], m[2] * e[0] + m[6] * e[1] + m[10] * e[2] + m[14]];
    // the bone keeps its length …
    const lenT = v3.dist(tpl.shoulderL, tpl.elbowL);
    expect(v3.dist(mapped, got.shoulderL)).toBeCloseTo(lenT * got.pose.s, 1);
    // … its far end lands on the elbow's pixel …
    const want = projectToImage(eL, aspect, vfov), have = projectToImage(mapped, aspect, vfov);
    expect(have[0]).toBeCloseTo(want[0], 3);
    expect(have[1]).toBeCloseTo(want[1], 3);
    // … and of the two possible depths it picked the true one (toward the camera)
    expect(v3.dist(mapped, eL)).toBeLessThan(1);
  });

  it("orientation follows the shoulder line", () => {
    const truth = makePose(0.5, 0, 0, [0, 0, -70]);
    const obs = observe(truth, [0, 0, 0], [0, 0, 0]);
    const R = torsoRotation(obs);
    for (let i = 0; i < 9; i++) expect(R[i]).toBeCloseTo(truth.R[i], 2);
  });

  it("the neck chain reaches a chest target", () => {
    const joints: number[] = [];
    for (let k = 0; k < 8; k++) joints.push(0, -2 - 1.7 * k, -6);
    const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const mats = Array.from({ length: 7 }, () => new Float32Array(16));
    const free = solveNeck(joints, I, I, mats);
    expect(free[1]).toBeCloseTo(-13.9, 4);
    const end = solveNeck(joints, I, I, mats, [1.5, -14.5, -5]);
    expect(end[0]).toBeCloseTo(1.5, 5); expect(end[1]).toBeCloseTo(-14.5, 5);
    // the top vertebra moves a seventh of the way, the lowest all the way
    expect(mats[0][12]).toBeCloseTo(1.5 / 7, 5);
    expect(mats[6][12]).toBeCloseTo(1.5, 5);
  });
});

describe("forearm", () => {
  // right forearm hanging down in source axes (x left, y back, z up), true cm
  const tpl: ForearmTemplate = { elbow: [-22, -6, 105], wrist: [-25, -9, 80] };
  const aspect = 16 / 9, vfov = 45;
  const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const apply = (m: Float32Array, p: Vec3): Vec3 => [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
  ];
  it.each([[false], [true]])("spans elbow to wrist (left=%s)", (left) => {
    const elbow: Vec3 = [left ? 20 : -20, -5, -60];
    const lenT = v3.dist(tpl.elbow, tpl.wrist);
    // true wrist: forearm raised and tilted toward the camera
    const dir = v3.norm([left ? 0.3 : -0.3, 0.8, 0.5]);
    const wristTrue = v3.add(elbow, v3.scale(dir, lenT * 1.05));
    const m = new Float32Array(16);
    const got = solveForearm(tpl, elbow, projectToImage(wristTrue, aspect, vfov), -wristTrue[2] - 0.3, I, left, 1.05, aspect, vfov, m);
    expect(v3.dist(got, wristTrue)).toBeLessThan(0.05);
    const mx = left ? -1 : 1;
    expect(v3.dist(apply(m, tpl.elbow), elbow)).toBeLessThan(1e-3);
    expect(v3.dist(apply(m, tpl.wrist), wristTrue)).toBeLessThan(0.05);
    const det = m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9]) + m[8] * (m[1] * m[6] - m[2] * m[5]);
    expect(Math.sign(det)).toBe(mx);
  });
});
