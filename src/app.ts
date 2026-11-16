/**
 * The mirror itself: per video frame, track → solve pose → learn the user's
 * shape → solve jaw / neck / hands → render over that same frame.
 */
import * as THREE from "three";
import type { FaceLandmarker, HandLandmarker, ImageSegmenter, PoseLandmarker } from "@mediapipe/tasks-vision";
import canonical from "./data/canonicalFace.json";
import { FaceGeometry, defaultVfov, projectToImage, refinePose, RIGID_IDX, RIGID_W, type FaceSolve } from "./tracking/faceGeometry";
import { OneEuro, OneEuroQuat } from "./tracking/filters";
import { createFaceLandmarker, createHandLandmarker, createSegmenter, createPoseLandmarker, readBody, readFace, readHands, type FaceObservation, type HandObservation } from "./tracking/trackers";
import type { FrameSource } from "./tracking/camera";
import { PersonalFit, headAngles } from "./fit/neutralShape";
import { jawMatrix, solveJaw, type JawRig, type JawState } from "./fit/jaw";
import { liftHand, solveHand, type HandPose } from "./fit/hand";
import { solveNeck, torsoFromHead } from "./fit/neck";
import { solveTorso, torsoRotation, buildTorsoTemplate, buildForearmTemplate, solveForearm, type BodyObservation, type TorsoTemplate, type TorsoPose, type ForearmTemplate } from "./fit/torso";
import { mat3Mul } from "./math/linalg";
import { TorsoRig } from "./render/torso";
import { NerveRig } from "./render/nerves";
import { traceOutline, modelOutline, updateVault, type Roi } from "./fit/vault";
import { mat3ToQuat, similarityToMat4, applySimilarity, invertSimilarity, type Similarity } from "./math/similarity";
import { quatToMat3, type Vec3 } from "./math/linalg";
import { Pipeline, type FrameSettings, type Mode, type Reveal } from "./render/pipeline";
import { WarpUniforms } from "./render/warp";
import { loadAnatomy, type Anatomy } from "./render/anatomy";
import { HandRig } from "./render/hands";

export interface Settings {
  mode: Mode;
  layers: Record<string, boolean>;
  exposure: number;
  real: number;
  grain: number;
  reveal: Reveal;
  /** vertical field of view in degrees; 0 = automatic from the frame shape */
  fov: number;
  labels: boolean;
  /** learn the user's face shape (off = rigid template, for comparison) */
  personalise: boolean;
  trackHands: boolean;
  /** track the shoulders to place the collarbones, shoulder blades and ribs */
  trackBody: boolean;
  /** zoom the picture by pinching with both hands and moving them apart or together */
  gestureZoom: boolean;
  /** "light" draws fewer pixels and runs the trackers less often; "auto" starts light on phones and switches down if the frame rate stays low */
  quality: "auto" | "full" | "light";
  /** use the person / hair outline to fit the cranial vault and to trim the darkened region */
  useOutline: boolean;
}

export const defaultSettings = (): Settings => ({
  mode: "radiograph",
  layers: { bone: true, teeth: true, spine: true, chest: true, hands: true, nerves: true, brain: false, muscles: false, eyes: false, skin: true },
  exposure: 1.0,
  real: 0.12,
  grain: 0.07,
  reveal: "whole",
  fov: 0,
  labels: false,
  personalise: true,
  trackHands: true,
  trackBody: true,
  gestureZoom: true,
  quality: "auto",
  useOutline: true,
});

export interface Status {
  face: boolean;
  hands: number;
  fps: number;
  /** mean reprojection error of the fitted template on the rigid landmarks, % of face width */
  fitError: number;
  /** same, for the un-personalised template */
  rigidError: number;
  confidence: number;
  yaw: number; pitch: number; roll: number;
  distance: number;
  jawOpenDeg: number;
  fov: number;
  width: number; height: number;
  /** current zoom factor and whether a two-hand pinch is holding it */
  zoom: number;
  zooming: boolean;
  /** true while the lighter quality mode is in use */
  light: boolean;
}

export interface LabelPoint { id: string; text: string; x: number; y: number; visible: boolean }

/**
 * Landmarks used to score the fit: points that stay on the same spot of skin (nose, eye corners,
 * brows, cheeks). Jaw-line and face-edge landmarks are left out: the tracker slides them along the
 * silhouette as the head turns, and the chin moves with the jaw.
 */
const SCORE_IDX = [4, 5, 6, 168, 197, 195, 2, 164, 33, 263, 133, 362, 129, 358, 9, 151, 105, 334, 50, 280, 117, 346, 119, 348];

/** Face points that must lie inside the person outline for it to be trusted. */
const SEG_CHECK = [4, 6, 152, 10, 50, 280, 205, 425];

/** A phone or small tablet: touch-first and a small screen. */
export function isPhone(): boolean {
  if (typeof matchMedia === "undefined" || typeof screen === "undefined") return false;
  return matchMedia("(pointer: coarse)").matches && Math.min(screen.width, screen.height) < 820;
}

export class XRayMirror {
  readonly pipeline: Pipeline;
  readonly settings: Settings = defaultSettings();
  readonly status: Status = {
    face: false, hands: 0, fps: 0, fitError: 0, rigidError: 0, confidence: 0,
    yaw: 0, pitch: 0, roll: 0, distance: 0, jawOpenDeg: 0, fov: 45, width: 0, height: 0, zoom: 1, zooming: false, light: false,
  };
  labels: LabelPoint[] = [];
  onFrame?: () => void;
  /** pointer position in image uv (y up), used by the lens when no fingertip is visible */
  pointer = { x: 0.5, y: 0.5 };

  private warp = new WarpUniforms();
  private anatomy!: Anatomy;
  private geo = new FaceGeometry(Float32Array.from(canonical.vertices));
  private fit!: PersonalFit;
  private face!: FaceLandmarker;
  private hands: HandLandmarker | null = null;
  private segmenter: ImageSegmenter | null = null;
  private scalp: Float32Array = new Float32Array(0);
  private scalpProj: Float32Array = new Float32Array(0);
  private person = new Uint8Array(0);
  private handRigs: HandRig[] = [];
  private body: PoseLandmarker | null = null;
  private torsoRig: TorsoRig | null = null;
  private torsoTpl: TorsoTemplate | null = null;
  private bodyObs: { obs: BodyObservation; time: number } | null = null;
  private torsoPose: TorsoPose | null = null;
  private torsoLive = false;
  private forearmTpl: ForearmTemplate | null = null;
  private headUnit = 1.1;
  private foreM = [new Float32Array(16), new Float32Array(16)];
  private foreLive = [false, false];
  private nerveRig: NerveRig | null = null;
  /** head-frame point → forearm template point (true cm): subject's left (mirrored), right */
  private handFromHead = [new THREE.Matrix4(), new THREE.Matrix4()];
  private foreHead = [new THREE.Matrix4(), new THREE.Matrix4()];
  private fBody = new OneEuro(10, 1.2, 0.3);
  private handFilters = [new OneEuro(63, 2.2, 0.08), new OneEuro(63, 2.2, 0.08)];
  private handPoses: (HandPose | undefined)[] = [undefined, undefined];
  private handSeen = [-1e9, -1e9];
  private handWrist: Array<[number, number]> = [[0, 0], [0, 0]];
  /** smoothed left/right vote per slot: > 0 = the subject's left hand */
  private handSide = [0, 0];

  private source: FrameSource | null = null;
  private stopLoop: (() => void) | null = null;
  private obs: FaceObservation | undefined;
  private solve: FaceSolve | undefined;
  private fT = new OneEuro(3, 2.0, 0.12);
  private fQ = new OneEuroQuat(2.0, 0.9);
  private fJaw = new OneEuro(2, 3.0, 4.0);
  private lastTime = 0;
  private lastFace = -1e9;
  private fitVersion = -1;
  private jawRig: JawRig = { hingeR: [-5, 0.8, -3.8], hingeL: [5, 0.8, -3.8] };
  private neckJoints = new Float32Array(24);
  private trueShape = new Float32Array(468 * 3);
  private neckMats = Array.from({ length: 7 }, () => new Float32Array(16));
  private jawM = new Float32Array(16);
  private poseM = new Float64Array(16);
  private frameCount = 0;
  private fpsT0 = 0;
  private fpsN = 0;
  private lensPos = { x: 0.5, y: 0.5 };
  private autoLight = isPhone();
  private slowSince = 0;
  private lastFingertip: { p: [number, number]; time: number } | null = null;
  private headCentreUv = { x: 0.5, y: 0.5, nx: 1, ny: 0 };
  private smoothedPose: Similarity | null = null;
  private drawnPose: Similarity | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.pipeline = new Pipeline(canvas);
  }

  /** Load anatomy and trackers. `progress` receives short status lines. */
  async init(progress: (msg: string) => void = () => {}) {
    progress("Loading anatomy");
    this.anatomy = await loadAnatomy(this.pipeline, this.warp);
    const t = this.anatomy.template;
    this.fit = new PersonalFit({ landmarks: t.landmarks, controlIdx: t.controlIdx, anchors: t.anchors });
    this.warp.setSpline(this.fit.tps);
    this.applyFitDependents();
    for (let i = 0; i < 2; i++) this.handRigs.push(new HandRig(this.pipeline, this.warp, t.hand, this.anatomy.handGeometries));

    progress("Loading face tracker");
    this.face = await createFaceLandmarker("VIDEO");
    // Hands and outline are optional: if a model is slow to download or fails, start without it.
    const optional = async <T>(what: string, make: () => Promise<T>): Promise<T | null> => {
      progress(`Loading ${what} tracker`);
      try {
        return await Promise.race([
          make(),
          new Promise<never>((_, reject) => setTimeout(() => reject(new Error("timed out")), 20000)),
        ]);
      } catch (e) {
        console.warn(`${what} tracking unavailable`, e);
        return null;
      }
    };
    this.hands = await optional("hand", createHandLandmarker);
    this.segmenter = await optional("outline", createSegmenter);
    if (this.anatomy.torsoGeometries.size) {
      const mm = this.anatomy.manifest.meshes;
      const hl = mm.humerus_l, hr = mm.humerus_r;
      if (hl && hr) {
        const toHead = new THREE.Matrix4().fromArray(t.toHead);
        const map = (p: Vec3): Vec3 => { const v = new THREE.Vector3(p[0], p[1], p[2]).applyMatrix4(toHead); return [v.x, v.y, v.z]; };
        const up = this.anatomy.manifest.frame.up ?? "+z";
        const sign = up.trim()[0] === "-" ? -1 : 1, axis = up.trim().slice(-1);
        const upVec: Vec3 = axis === "x" ? [sign, 0, 0] : axis === "y" ? [0, sign, 0] : [0, 0, sign];
        const nb = t.neckJoints.slice(21, 24) as Vec3;
        this.torsoTpl = buildTorsoTemplate(hl, hr, nb, map, upVec);
        this.torsoRig = new TorsoRig(this.pipeline, this.warp, this.anatomy.torsoGeometries, this.anatomy.handGeometries);
        if (mm.radius && mm.ulna) this.forearmTpl = buildForearmTemplate(mm.radius, mm.ulna, t.hand.joints.slice(0, 3) as Vec3);
        // head-frame units per true centimetre (the registration's scale)
        this.headUnit = Math.hypot(t.toHead[0], t.toHead[1], t.toHead[2]) * 10;
        this.body = await optional("body", createPoseLandmarker);
        // nervous system (Anatomy view): needs the chest and arm joints above
        if (this.anatomy.nerveGeometries.size) {
          const fromHead = toHead.clone().invert();
          const tenth = new THREE.Matrix4().makeScale(0.1, 0.1, 0.1);
          this.handFromHead[1].copy(tenth).multiply(fromHead);
          this.handFromHead[0].copy(tenth).multiply(new THREE.Matrix4().makeScale(-1, 1, 1)).multiply(fromHead);
          // template wrists: the hand template's wrist is the RIGHT one, in source mm × 0.1
          const w = t.hand.joints;
          const wristR = map([w[0] * 10, w[1] * 10, w[2] * 10]), wristL = map([-w[0] * 10, w[1] * 10, w[2] * 10]);
          this.nerveRig = new NerveRig(this.pipeline, this.warp, this.anatomy.nerveGeometries, this.anatomy.headRoot,
            { neckJoints: t.neckJoints, torso: this.torsoTpl, wristL, wristR });
        }
      }
    }
    this.scalp = this.samplePart("skin", 1400, (x, y) => y > -1);
    this.scalpProj = new Float32Array((this.scalp.length / 3) * 2);
    progress("Ready");
  }

  get attribution(): string { return this.anatomy?.manifest.attribution ?? ""; }
  get manifest() { return this.anatomy?.manifest; }
  get labelDefs() { return this.anatomy?.template.labels ?? []; }

  start(source: FrameSource) {
    this.stop();
    this.source = source;
    this.pipeline.setVideo(source.el);
    this.resetTracking();
    this.stopLoop = source.onFrame((t) => this.frame(t));
  }

  stop() {
    this.stopLoop?.();
    this.stopLoop = null;
    this.source?.stop();
    this.source = null;
  }

  resetTracking() {
    this.fit?.reset();
    this.fT.reset(); this.fQ.reset(); this.fJaw.reset();
    this.lastFace = -1e9;
    this.fitVersion = -1;
  }

  /** Things derived from the spline: jaw hinge and neck joints follow the user's proportions. */
  private applyFitDependents() {
    const t = this.anatomy.template, tps = this.fit.tps;
    this.jawRig = {
      hingeR: tps.apply(t.jaw.hingeR, [0, 0, 0]) as Vec3,
      hingeL: tps.apply(t.jaw.hingeL, [0, 0, 0]) as Vec3,
    };
    tps.applyArray(t.neckJoints, this.neckJoints);
    // the user's landmarks at true surface depth: the template's true points carried through the warp
    tps.applyArray(t.landmarksTrue, this.trueShape);
  }

  private frame(now: number) {
    const src = this.source;
    if (!src || !src.width || !src.height) return;
    const W = src.width, H = src.height, aspect = W / H;
    const s = this.settings, st = this.status;
    const fov = s.fov > 0 ? s.fov : defaultVfov(W, H);
    this.pipeline.setSize(W, H);
    this.pipeline.setFov(fov);
    st.fov = fov; st.width = W; st.height = H;
    const dt = this.lastTime ? (now - this.lastTime) / 1000 : 1 / 30;
    this.lastTime = now;
    this.frameCount++;

    // ── face ────────────────────────────────────────────────────────────────
    const res = this.face.detectForVideo(src.el, now);
    this.pipeline.uploadVideo(); // the frame the tracker just saw is the frame we draw
    const obs = readFace(res, this.obs);
    if (obs) {
      this.obs = obs;
      const solve = (this.solve = this.geo.solve(obs.landmarks, aspect, fov, this.solve));
      // The pose that is drawn: the user's rigid face shape, at true surface depth, fitted to the 2D
      // landmarks. (The tracker's depth-based pose above is only used near frontal, to learn the shape.)
      const drawn = (this.drawnPose = refinePose(this.trueShape, RIGID_IDX, RIGID_W, obs.landmarks, aspect, fov, solve.pose));
      const ang = headAngles(drawn.R);
      if (now - this.lastFace > 400) { this.fT.reset(); this.fQ.reset(); this.fJaw.reset(); }
      // the face was gone for a while: it may be someone else now, so learn the shape afresh
      if (now - this.lastFace > 1500 && this.lastFace > 0) this.fit.relearn();
      this.lastFace = now;

      if (s.personalise) this.fit.update(solve.local, obs.blend, ang.yaw, ang.pitch);
      else if (this.fit.samples > 0) { this.fit.reset(); this.fit.refit(); }
      if (this.fit.version !== this.fitVersion) {
        this.fitVersion = this.fit.version;
        this.warp.setSpline(this.fit.tps);
        this.applyFitDependents();
      }

      // jaw, measured against the user's own neutral shape
      const jaw: JawState = solveJaw(this.jawRig, this.fit.shape, solve.local);
      // The chin landmarks jitter by a degree or two even with the mouth shut. The tracker's own
      // mouth-open score is not a measurement but is a reliable closed/open signal: use it as a gate.
      const openScore = (obs.blend.jawOpen ?? 1) - this.fit.restingLevel("jawOpen");
      jaw.open *= Math.max(0, Math.min(1, (openScore - 0.02) / 0.08));
      const jf = this.fJaw.filter([jaw.open, jaw.swing], dt);
      jaw.open = Math.max(0, jf[0]); jaw.swing = jf[1];
      // a closed mouth should be fully closed: ignore sub-degree noise
      if (jaw.open < 0.012) jaw.open = 0;
      this.warp.setJaw(jawMatrix(this.jawRig, jaw, this.jawM));
      st.jawOpenDeg = (jaw.open * 180) / Math.PI;

      // smoothed pose for rendering
      const q = this.fQ.filter(mat3ToQuat(drawn.R), dt);
      const tt = this.fT.filter(drawn.t, dt) as number[];
      const pose: Similarity = { s: 1, R: quatToMat3(q), t: [tt[0], tt[1], tt[2]] };
      this.smoothedPose = pose;
      const head = this.anatomy.headRoot;
      head.matrix.fromArray(similarityToMat4(pose, this.poseM) as unknown as number[]);
      head.updateMatrixWorld(true);
      head.visible = true;
      const refDepth = Math.hypot(pose.t[0], pose.t[1], pose.t[2]);
      for (const p of this.anatomy.headParts.values()) p.tau!.uniforms.uRefDepth.value = refDepth;

      // neck
      // neck and chest: with the shoulders tracked, the chest is placed from them and the neck is made
      // to meet it; otherwise the neck hangs under the head on its own
      const body = this.bodyObs && now - this.bodyObs.time < 700 && s.trackBody && s.layers.chest !== false ? this.bodyObs.obs : null;
      const torsoR = body ? torsoRotation(body) : torsoFromHead(pose.R);
      const base = solveNeck(this.neckJoints, pose.R, torsoR, this.neckMats);
      this.nerveRig?.setNeck(this.neckMats);
      if (body && this.torsoTpl && this.torsoRig) {
        const neckDepth = -applySimilarity(pose, base)[2];
        const tp = (this.torsoPose = solveTorso(this.torsoTpl, body, neckDepth, aspect, fov, this.torsoPose));
        // where the chest puts the base of the neck, seen from the head
        const target = applySimilarity(invertSimilarity(pose), applySimilarity(tp.pose, this.torsoTpl.neckBase));
        solveNeck(this.neckJoints, pose.R, torsoR, this.neckMats, target);
        this.nerveRig?.setNeck(this.neckMats);
        this.torsoRig.update(tp);
        this.torsoLive = true;
        // darken the neck between the head's region and the chest's
        const top = applySimilarity(pose, [this.neckJoints[0], this.neckJoints[1], this.neckJoints[2]]);
        this.torsoRig.setBand(0, top, applySimilarity(tp.pose, this.torsoTpl.neckBase), 5.2 * tp.pose.s);
      } else { this.torsoRig?.hide(); this.torsoRig?.setBand(0, null); this.torsoLive = false; }
      for (let k = 0; k < 7; k++) {
        const part = this.anatomy.headParts.get(`c${k + 1}`);
        if (!part) continue;
        for (const m of [part.tau, part.solid]) m?.uniforms.uPart.value.fromArray(this.neckMats[k]);
      }

      // status
      st.face = true;
      st.yaw = ang.yaw; st.pitch = ang.pitch; st.roll = ang.roll;
      st.distance = -drawn.t[2];
      st.confidence = this.fit.confidence;
      st.fitError = this.reprojectionError(drawn, obs.landmarks, this.trueShape, aspect, fov);
      // baseline: the same pose fit, but with the one-size template instead of the user's shape
      const tplShape = this.anatomy.template.landmarksTrue;
      const rigid = s.personalise ? refinePose(tplShape, RIGID_IDX, RIGID_W, obs.landmarks, aspect, fov, solve.pose) : drawn;
      st.rigidError = this.reprojectionError(rigid, obs.landmarks, tplShape, aspect, fov);

      // split line through the middle of the face
      const c = projectToImage(applySimilarity(pose, [0, 0, 0]), aspect, fov);
      const xr = applySimilarity(pose, [1, 0, 0]);
      const cx = projectToImage(xr, aspect, fov);
      let nx = (cx[0] - c[0]) * aspect, ny = -(cx[1] - c[1]);
      const nl = Math.hypot(nx, ny) || 1;
      nx /= nl; ny /= nl;
      this.headCentreUv = { x: c[0], y: 1 - c[1], nx, ny };
    } else if (now - this.lastFace > 300) {
      this.anatomy.headRoot.visible = false;
      this.torsoRig?.hide();
      this.torsoRig?.setBand(0, null);
      this.torsoLive = false;
      st.face = false;
    }

    // ── quality ─────────────────────────────────────────────────────────────
    // Light mode: 60 % of the pixels per side, no edge smoothing, no outline model, and the hand and
    // body models half as often. Auto starts light on phones and drops to light anywhere the frame
    // rate stays under 17 for four seconds (it does not switch back up by itself).
    if (s.quality === "auto" && !this.autoLight && this.frameCount > 150) {
      if (st.fps > 0 && st.fps < 17) { if (!this.slowSince) this.slowSince = now; else if (now - this.slowSince > 4000) this.autoLight = true; }
      else this.slowSince = 0;
    }
    const light = s.quality === "light" || (s.quality === "auto" && this.autoLight);
    st.light = light;
    this.pipeline.setQuality(light ? 0.6 : 1, light ? 0 : 4);

    // ── scheduling ──────────────────────────────────────────────────────────
    // The face model runs on every frame. The other three models each cost about as much again, so
    // at most one of them runs alongside it on any frame: a twelve-frame cycle with the outline once
    // (it changes slowly) and the body three times, the hands taking the rest.
    const slot = this.frameCount % 12;
    const runSeg = !!this.segmenter && s.useOutline && !light && slot === 5;
    const runBody = !!this.body && s.trackBody && s.layers.chest !== false && !runSeg && (light ? slot === 2 || slot === 8 : slot % 3 === 2);

    // ── hands ───────────────────────────────────────────────────────────────
    let fingertip: [number, number] | null = null;
    // Hands are tracked whenever something needs them: their bones, the zoom gesture, or the lens
    // (which follows a fingertip). Hiding the bones must not switch the gestures off.
    const drawHands = s.layers.hands !== false;
    const needHands = drawHands || s.gestureZoom || s.reveal === "lens";
    if (!drawHands) for (const r of this.handRigs) r.hide();
    if (this.hands && s.trackHands && needHands) {
      // While a hand is in view the hand model runs on every frame that has no body or outline pass
      // (three frames in four); with no hand in view it only looks every third frame.
      const idle = now - Math.max(this.handSeen[0], this.handSeen[1]) > 1500;
      if (idle ? slot % 3 === 0 : !(runSeg || runBody) && (!light || slot % 2 === 0)) {
        const list = this.debugHands ?? readHands(this.hands.detectForVideo(src.el, now));
        const used = [false, false];
        for (const h of list) {
          // Keep each hand in the slot it was in (nearest previous wrist), rather than keying slots on
          // the tracker's left/right label, which flickers and would swap bones and filters mid-motion.
          const wx = h.image[0], wy = h.image[1];
          let slot = -1, best = 0.25;
          for (let k = 0; k < 2; k++) {
            if (used[k] || now - this.handSeen[k] > 500) continue;
            const d = Math.hypot((wx - this.handWrist[k][0]) * aspect, wy - this.handWrist[k][1]);
            if (d < best) { best = d; slot = k; }
          }
          const fresh = slot < 0;
          if (fresh) {
            // a new hand takes a slot nobody has used recently, else any free one
            for (let k = 0; k < 2 && slot < 0; k++) if (!used[k] && now - this.handSeen[k] > 500) slot = k;
            for (let k = 0; k < 2 && slot < 0; k++) if (!used[k]) slot = k;
          }
          if (slot < 0) continue;
          used[slot] = true;
          // The tracker assumes a mirrored image, so on a raw camera frame its labels are swapped:
          // "Right" means the subject's left hand. Smooth the vote so one bad frame cannot flip the hand.
          const vote = (h.label === "Right" ? 1 : -1) * Math.max(0.3, h.score);
          if (fresh) { this.handFilters[slot].reset(); this.handSide[slot] = vote; }
          else this.handSide[slot] += 0.12 * (vote - this.handSide[slot]);
          const left = this.handSide[slot] > 0;
          const hdt = fresh ? 0 : (now - this.handSeen[slot]) / 1000;
          this.handSeen[slot] = now;
          this.handWrist[slot] = [wx, wy];
          const raw = liftHand(h.image, h.world, aspect, fov);
          const joints = this.handFilters[slot].filter(raw, hdt, new Float32Array(63)) as Float32Array;
          const pose = (this.handPoses[slot] = solveHand(this.anatomy.template.hand, joints, left, this.handPoses[slot]));
          if (drawHands) this.handRigs[slot].update(pose, joints);
          this.readPinch(slot, h.image, aspect, now);
          if (!fingertip) { fingertip = [h.image[16], 1 - h.image[17]]; this.lastFingertip = { p: fingertip, time: now }; }
        }
        for (let i = 0; i < 2; i++) if (!used[i] && now - this.handSeen[i] > 250) this.handRigs[i].hide();
        st.hands = (used[0] ? 1 : 0) + (used[1] ? 1 : 0);
      }
    } else {
      for (const r of this.handRigs) r.hide();
      st.hands = 0;
    }

    this.updateArms(now, aspect, fov);
    this.updateZoom(now, aspect);

    // ── body (shoulders and elbows), about eight times a second ─────────────
    if (this.body && runBody) {
      try {
        const b = readBody(this.body.detectForVideo(src.el, now));
        if (b) {
          // smooth the shoulders; elbows come and go, so they are passed through as seen
          const prevT = this.bodyObs?.time ?? 0;
          const raw = [...b.shoulderL, ...b.shoulderR, ...b.worldShoulderL, ...b.worldShoulderR];
          const f = this.fBody.filter(raw, prevT && now - prevT < 700 ? (now - prevT) / 1000 : 0, new Array(10)) as number[];
          b.shoulderL = [f[0], f[1]]; b.shoulderR = [f[2], f[3]];
          b.worldShoulderL = [f[4], f[5], f[6]]; b.worldShoulderR = [f[7], f[8], f[9]];
          this.bodyObs = { obs: b, time: now };
        }
      } catch (e) {
        console.warn("body tracking failed, disabling", e);
        this.body = null;
      }
    }

    // ── outline (person + hair) ─────────────────────────────────────────────
    // The outline changes slowly and its model is the most expensive, so it runs a few times a second.
    if (this.segmenter && runSeg) {
      const roi = this.status.face && obs ? this.cropHead(src.el, obs.landmarks, W, H) : null;
      if (!roi) this.pipeline.setSegmentation(null);
      else {
        try {
          this.segmenter.segmentForVideo(this.cropCanvas, now, (r) => {
            const m = r.categoryMask;
            if (!m) return;
            this.useOutline(m.getAsUint8Array(), m.width, m.height, roi, aspect, fov);
            m.close();
          });
        } catch (e) {
          console.warn("outline tracking failed, disabling", e);
          this.segmenter = null;
          this.pipeline.setSegmentation(null);
        }
      }
    } else if (!s.useOutline || !this.segmenter) {
      this.pipeline.setSegmentation(null);
    }

    // ── lens ────────────────────────────────────────────────────────────────
    // hands are not tracked on every frame: keep following the last fingertip for a moment
    if (!fingertip && this.lastFingertip && now - this.lastFingertip.time < 400) fingertip = this.lastFingertip.p;
    const target = fingertip ? { x: fingertip[0], y: fingertip[1] } : this.pointer;
    this.lensPos.x += (target.x - this.lensPos.x) * 0.5;
    this.lensPos.y += (target.y - this.lensPos.y) * 0.5;

    // ── draw ────────────────────────────────────────────────────────────────
    const fs: FrameSettings = {
      mode: s.mode, exposure: s.exposure, real: s.real, grain: s.grain, reveal: s.reveal,
      lens: { x: this.lensPos.x, y: this.lensPos.y, r: 0.2 },
      split: { x: this.headCentreUv.x, y: this.headCentreUv.y, nx: this.headCentreUv.nx, ny: this.headCentreUv.ny },
      layers: this.effectiveLayers(),
      time: now / 1000,
      view: { x: this.view.x, y: this.view.y, zoom: this.view.zoom },
    };
    this.applyCut();
    this.pipeline.render(fs);
    if (s.labels && st.face) this.updateLabels(aspect, fov); else this.labels = [];

    this.fpsN++;
    if (now - this.fpsT0 > 500) { st.fps = (this.fpsN * 1000) / (now - this.fpsT0); this.fpsT0 = now; this.fpsN = 0; }
    this.onFrame?.();
  }

  /** Person / hair classes for this frame: trims the darkened region and fits the vault to the real outline. */
  /** test hook: hand observations to use instead of the tracker's */
  debugHands: HandObservation[] | null = null;
  /** last outline comparison, for diagnostics */
  lastOutline: { seen: Float32Array; hair: Uint8Array; model: Float32Array } | null = null;
  /** The zoomed view: centre in the picture (uv, y up) and factor. */
  readonly view = { x: 0.5, y: 0.5, zoom: 1 };
  private pinch = [{ on: false, x: 0, y: 0, time: -1e9 }, { on: false, x: 0, y: 0, time: -1e9 }];
  private grab: { dist: number; zoom: number; ax: number; ay: number; sx: number; sy: number } | null = null;
  private zoomTarget = 1;

  /** Is this hand pinching (thumb tip on index tip)? Remember where, in picture uv (y up). */
  private readPinch(slot: number, image: ArrayLike<number>, aspect: number, now: number) {
    const d = (a: number, b: number) => Math.hypot((image[2 * a] - image[2 * b]) * aspect, image[2 * a + 1] - image[2 * b + 1]);
    // thumb–index gap relative to the size of the hand (wrist to middle knuckle)
    const gap = d(4, 8) / Math.max(1e-4, d(0, 9));
    const p = this.pinch[slot];
    p.on = p.on ? gap < 0.42 : gap < 0.26; // hysteresis: easy to hold, deliberate to start
    p.x = 0.5 * (image[8] + image[16]);
    p.y = 1 - 0.5 * (image[9] + image[17]);
    p.time = now;
  }

  /**
   * Two-hand zoom: pinch with both hands, then move them apart to zoom in or together to zoom out,
   * like stretching the picture. Letting go keeps the zoom. The picture point that was between the
   * hands when the pinch started stays where it was on screen.
   */
  private updateZoom(now: number, aspect: number) {
    const [a, b] = this.pinch;
    const both = this.settings.gestureZoom && a.on && b.on && now - a.time < 250 && now - b.time < 250;
    const v = this.view;
    if (both) {
      const dist = Math.hypot((a.x - b.x) * aspect, a.y - b.y);
      const mx = 0.5 * (a.x + b.x), my = 0.5 * (a.y + b.y);
      if (!this.grab && dist > 0.08) {
        // anchor: the picture point between the hands, and where it currently sits on screen
        this.grab = { dist, zoom: v.zoom, ax: mx, ay: my, sx: 0.5 + (mx - v.x) * v.zoom, sy: 0.5 + (my - v.y) * v.zoom };
      }
      if (this.grab) this.zoomTarget = Math.min(4, Math.max(1, this.grab.zoom * (dist / this.grab.dist)));
    } else this.grab = null;
    if (!this.settings.gestureZoom) this.zoomTarget = 1;

    const z = v.zoom + (this.zoomTarget - v.zoom) * 0.35;
    v.zoom = Math.abs(z - this.zoomTarget) < 0.002 ? this.zoomTarget : z;
    if (this.grab) {
      v.x = this.grab.ax - (this.grab.sx - 0.5) / v.zoom;
      v.y = this.grab.ay - (this.grab.sy - 0.5) / v.zoom;
    }
    // never show beyond the edge of the picture; at zoom 1 this recentres the view
    const half = 0.5 / v.zoom;
    v.x = Math.min(1 - half, Math.max(half, v.x));
    v.y = Math.min(1 - half, Math.max(half, v.y));
    this.status.zoom = v.zoom;
    this.status.zooming = !!this.grab;
  }

  /** Back to the whole picture. */
  resetZoom() { this.zoomTarget = 1; this.grab = null; }

  /** Picture position (normalised, y down) → position on screen (normalised, y down), given the zoom. */
  toScreen(x: number, y: number): [number, number] {
    const v = this.view;
    return [0.5 + (x - v.x) * v.zoom, 0.5 + (y - (1 - v.y)) * v.zoom];
  }
  /** Screen position (normalised, y up) → picture uv (y up). */
  fromScreen(x: number, y: number): [number, number] {
    const v = this.view;
    return [v.x + (x - 0.5) / v.zoom, v.y + (y - 0.5) / v.zoom];
  }

  /**
   * Forearms and the darkening along the arms. Each forearm runs from the elbow the body tracker found
   * to the wrist: the tracked hand's wrist when a hand is there (and the forearm then turns with that
   * hand), otherwise the body tracker's wrist.
   */
  private updateArms(now: number, aspect: number, fov: number) {
    const rig = this.torsoRig, tp = this.torsoPose, body = this.bodyObs?.obs;
    for (const h of this.handRigs) h.stub = true;
    this.foreLive[0] = this.foreLive[1] = false;
    if (!rig || !tp || !body || !this.torsoLive) { this.nerveRig?.hideBody(); return; }
    const scale = tp.pose.s * this.headUnit; // template true cm → camera units
    // chest orientation applied to source axes (x left, y back, z up → x, z, −y)
    const chestBase = mat3Mul(tp.pose.R, [1, 0, 0, 0, 0, 1, 0, -1, 0]);
    for (const left of [true, false]) {
      const elbow = left ? tp.elbowL : tp.elbowR, shoulder = left ? tp.shoulderL : tp.shoulderR;
      const k = left ? 0 : 1;
      rig.setBand(1 + k, elbow ? shoulder : null, elbow, 5 * tp.pose.s);
      let wristImg = (left ? body.wristL : body.wristR) ?? null;
      const wW = (left ? body.worldWristL : body.worldWristR) ?? null, wE = left ? body.worldElbowL : body.worldElbowR;
      // a tracked hand at this wrist?
      let slot = -1, best = 0.16;
      for (let i = 0; i < 2; i++) {
        if (now - this.handSeen[i] > 250) continue;
        if (wristImg) {
          const d = Math.hypot((this.handWrist[i][0] - wristImg[0]) * aspect, this.handWrist[i][1] - wristImg[1]);
          if (d < best) { best = d; slot = i; }
        } else if (this.handSide[i] > 0 === left && slot < 0) slot = i;
      }
      if (slot >= 0) wristImg = this.handWrist[slot];
      if (!elbow || !wristImg || !this.forearmTpl) { rig.setForearm(left, null); rig.setBand(3 + k, null); continue; }
      const handPose = slot >= 0 ? this.handPoses[slot] : undefined;
      // turn with the hand only if the hand tracker agrees which hand this is
      const base = handPose && this.handSide[slot] > 0 === left ? handPose.palm.R : chestBase;
      const want = wW && wE ? -elbow[2] - (wW[2] - wE[2]) * 100 : null;
      const wrist = solveForearm(this.forearmTpl, elbow, wristImg, want, base, left, scale, aspect, fov, this.foreM[k]);
      rig.setForearm(left, this.foreM[k], Math.hypot(elbow[0], elbow[1], elbow[2]));
      rig.setBand(3 + k, elbow, wrist, 4 * tp.pose.s);
      this.foreLive[k] = true;
      if (slot >= 0) this.handRigs[slot].stub = false;
    }
    if (this.nerveRig) {
      // forearm transforms re-expressed for head-frame meshes
      for (let k = 0; k < 2; k++) if (this.foreLive[k]) this.foreHead[k].fromArray(this.foreM[k]).multiply(this.handFromHead[k]);
      this.nerveRig.setBody(tp, this.foreLive[0] ? this.foreHead[0] : null, this.foreLive[1] ? this.foreHead[1] : null);
    }
  }

  private cropCanvas = document.createElement("canvas");
  private cropCtx: CanvasRenderingContext2D | null = null;

  /**
   * Copy a square around the head into a small canvas for the segmenter, so the head fills the
   * model's input (a full frame leaves the head only a few dozen mask pixels wide).
   * @returns the part of the picture the crop covers, or null when the face is too small
   */
  private cropHead(el: CanvasImageSource, lm: ArrayLike<number>, W: number, H: number): Roi | null {
    const faceW = Math.hypot((lm[3 * 454] - lm[3 * 234]) * W, (lm[3 * 454 + 1] - lm[3 * 234 + 1]) * H);
    if (faceW < 24) return null;
    const cx = 0.5 * (lm[3 * 234] + lm[3 * 454]) * W;
    // centre a little above the eyes so the whole vault and some background above it are included
    const cy = lm[3 * 168 + 1] * H - 0.25 * faceW;
    const half = 1.35 * faceW;
    const c = this.cropCanvas;
    if (c.width !== 256) { c.width = c.height = 256; this.cropCtx = c.getContext("2d", { willReadFrequently: false }); }
    const ctx = this.cropCtx;
    if (!ctx) return null;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 256, 256);
    // clip the source rectangle to the frame: some browsers draw nothing when it overhangs
    const x0 = cx - half, y0 = cy - half, size = 2 * half;
    const sx = Math.max(0, x0), sy = Math.max(0, y0);
    const ex = Math.min(W, x0 + size), ey = Math.min(H, y0 + size);
    if (ex - sx < 8 || ey - sy < 8) return null;
    const k = 256 / size;
    ctx.drawImage(el, sx, sy, ex - sx, ey - sy, (sx - x0) * k, (sy - y0) * k, (ex - sx) * k, (ey - sy) * k);
    return { x: x0 / W, y: y0 / H, w: size / W, h: size / H };
  }

  private useOutline(cls: Uint8Array, w: number, h: number, roi: Roi, aspect: number, fov: number) {
    const lm = this.obs?.landmarks, solve = this.drawnPose;
    // only trust the mask if it actually covers the tracked face
    let cover = 0;
    if (lm && this.status.face) {
      for (const i of SEG_CHECK) {
        const x = Math.min(w - 1, Math.max(0, Math.round(((lm[3 * i] - roi.x) / roi.w) * w)));
        const y = Math.min(h - 1, Math.max(0, Math.round(((lm[3 * i + 1] - roi.y) / roi.h) * h)));
        if (cls[y * w + x] !== 0) cover++;
      }
    }
    if (cover < SEG_CHECK.length - 1 || !solve) { this.pipeline.setSegmentation(null); return; }
    if (this.person.length !== cls.length) this.person = new Uint8Array(cls.length);
    for (let i = 0; i < cls.length; i++) this.person[i] = cls[i] ? 255 : 0;
    this.pipeline.setSegmentation(this.person, w, h, roi);

    // vault fit: only from roughly frontal views, where "up" and "sideways" mean height and breadth
    const st = this.status;
    if (!this.settings.personalise || Math.abs(st.yaw) > 12 || Math.abs(st.pitch) > 12) return;
    const tps = this.fit.tps, pose = solve;
    const P = PersonalFit.VAULT_PIVOT;
    const c = projectToImage(applySimilarity(pose, tps.apply(P, this.tmp)), aspect, fov);
    const u = projectToImage(applySimilarity(pose, tps.apply([P[0], P[1] + 6, P[2]], this.tmp)), aspect, fov);
    let upx = (u[0] - c[0]) * aspect, upy = u[1] - c[1];
    const ul = Math.hypot(upx, upy);
    if (ul < 1e-5) return;
    upx /= ul; upy /= ul;
    const sc = this.scalp, pr = this.scalpProj;
    for (let i = 0, k = 0; i < sc.length; i += 3, k += 2) {
      const q = projectToImage(applySimilarity(pose, tps.apply([sc[i], sc[i + 1], sc[i + 2]], this.tmp)), aspect, fov);
      pr[k] = q[0] * aspect; pr[k + 1] = q[1];
    }
    const seen = traceOutline(cls, w, h, roi, c[0], c[1], upx, upy, aspect);
    const model = modelOutline(pr, c[0], c[1], upx, upy, aspect);
    this.lastOutline = { seen: seen.r, hair: seen.hair, model };
    const next = updateVault(this.fit.vault, seen, model);
    if (next) this.fit.setVault(next);
  }

  private effectiveLayers(): Record<string, boolean> {
    const l = { ...this.settings.layers };
    // the soft-tissue envelope belongs to the radiograph only
    l.skin = this.settings.mode === "radiograph";
    return l;
  }

  /** Anatomy mode: when the brain is shown, lift the skull cap off like an anatomical preparation. */
  private applyCut() {
    const part = this.anatomy.headParts.get("cranium");
    const cutOn = this.settings.mode === "anatomy" && this.settings.layers.brain;
    const y = this.anatomy.template.landmarks[3 * 9 + 1] + 1.4; // a little above the brow ridge
    part?.solid?.uniforms.uCut.value.set(0, 1, 0, cutOn ? y : 1e6);
  }

  /** Mean 2D distance between where the model puts the landmarks and where the tracker sees them, % of face width. */
  private reprojectionError(pose: Similarity, lm: ArrayLike<number>, shape: ArrayLike<number>, aspect: number, fov: number): number {
    const faceW = Math.hypot((lm[3 * 454] - lm[3 * 234]) * aspect, lm[3 * 454 + 1] - lm[3 * 234 + 1]) || 1;
    let e = 0;
    for (const i of SCORE_IDX) {
      const p = applySimilarity(pose, [shape[3 * i], shape[3 * i + 1], shape[3 * i + 2]]);
      const [x, y] = projectToImage(p, aspect, fov);
      e += Math.hypot((x - lm[3 * i]) * aspect, y - lm[3 * i + 1]);
    }
    return (100 * e) / SCORE_IDX.length / faceW;
  }

  private tmp: number[] = [0, 0, 0];
  private updateLabels(aspect: number, fov: number) {
    const pose = this.smoothedPose;
    if (!pose) { this.labels = []; return; }
    const tps = this.fit.tps, J = this.jawM;
    const out: LabelPoint[] = [];
    for (const def of this.labelDefs) {
      if (this.settings.layers[def.layer] === false) continue;
      let p = tps.apply(def.anchor, this.tmp) as number[];
      if (def.jaw) p = [J[0] * p[0] + J[4] * p[1] + J[8] * p[2] + J[12], J[1] * p[0] + J[5] * p[1] + J[9] * p[2] + J[13], J[2] * p[0] + J[6] * p[1] + J[10] * p[2] + J[14]];
      const c = applySimilarity(pose, p);
      const [x, y] = projectToImage(c, aspect, fov);
      out.push({ id: def.id, text: def.text, x, y, visible: x > 0.02 && x < 0.98 && y > 0.02 && y < 0.98 });
    }
    this.labels = out;
  }

  /** Where the app is currently drawing a template point (normalised image coords, y down). */
  imagePoint(templatePoint: ArrayLike<number>, movesWithJaw = false): [number, number] | null {
    const pose = this.smoothedPose, src = this.source;
    if (!pose || !src) return null;
    let p = this.fit.tps.apply(templatePoint, [0, 0, 0]) as number[];
    const J = this.jawM;
    if (movesWithJaw) p = [J[0] * p[0] + J[4] * p[1] + J[8] * p[2] + J[12], J[1] * p[0] + J[5] * p[1] + J[9] * p[2] + J[13], J[2] * p[0] + J[6] * p[1] + J[10] * p[2] + J[14]];
    return projectToImage(applySimilarity(pose, p), src.width / src.height, this.status.fov);
  }

  /** Evenly spaced vertices of a head part, in the template head frame. */
  samplePart(name: string, count: number, keep?: (x: number, y: number, z: number) => boolean): Float32Array {
    const part = this.anatomy.headParts.get(name);
    if (!part) return new Float32Array(0);
    const a = part.mesh.geometry.getAttribute("position").array as Float32Array;
    const picked: number[] = [];
    for (let i = 0; i < a.length; i += 3) if (!keep || keep(a[i], a[i + 1], a[i + 2])) picked.push(i);
    const step = Math.max(1, Math.floor(picked.length / count));
    const out: number[] = [];
    for (let k = 0; k < picked.length; k += step) { const i = picked[k]; out.push(a[i], a[i + 1], a[i + 2]); }
    return Float32Array.from(out);
  }

  /** Current neutral-shape landmark count etc. for debugging / tests. */
  debugState() {
    return {
      status: { ...this.status },
      samples: this.fit?.samples ?? 0,
      axis: this.fit?.axis,
      vault: this.fit?.vault,
      jawRig: this.jawRig,
    };
  }
}

export const _three = THREE;
