import { FilesetResolver, FaceLandmarker, HandLandmarker, ImageSegmenter, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { BodyObservation } from "../fit/torso";
import { NUM_LANDMARKS } from "./faceGeometry";

/**
 * Third-party downloads, all pinned to a fixed version (never "latest"), so what runs cannot change
 * under the site. The wasm version must match the installed @mediapipe/tasks-vision.
 */
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm";
const MODELS = "https://storage.googleapis.com/mediapipe-models";
const FACE_MODEL = `${MODELS}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`;
const HAND_MODEL = `${MODELS}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`;
/** 256×256, classes: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other */
const SEG_MODEL = `${MODELS}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite`;

type Vision = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;
let visionPromise: Promise<Vision> | null = null;
const vision = () => (visionPromise ??= FilesetResolver.forVisionTasks(WASM_URL));

/** Try the GPU delegate first, fall back to CPU. */
async function withDelegate<T>(make: (delegate: "GPU" | "CPU") => Promise<T>): Promise<T> {
  try {
    return await make("GPU");
  } catch (e) {
    console.warn("GPU delegate unavailable, using CPU", e);
    return make("CPU");
  }
}

export async function createFaceLandmarker(mode: "VIDEO" | "IMAGE" = "VIDEO"): Promise<FaceLandmarker> {
  const v = await vision();
  return withDelegate((delegate) =>
    FaceLandmarker.createFromOptions(v, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate },
      runningMode: mode,
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
    }),
  );
}

export async function createHandLandmarker(): Promise<HandLandmarker> {
  const v = await vision();
  return withDelegate((delegate) =>
    HandLandmarker.createFromOptions(v, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate },
      runningMode: "VIDEO",
      numHands: 2,
      // a little more willing to find and keep a hand than the defaults (0.5)
      minHandDetectionConfidence: 0.4,
      minHandPresenceConfidence: 0.4,
      minTrackingConfidence: 0.4,
    }),
  );
}

export async function createSegmenter(): Promise<ImageSegmenter> {
  const v = await vision();
  return withDelegate((delegate) =>
    ImageSegmenter.createFromOptions(v, {
      baseOptions: { modelAssetPath: SEG_MODEL, delegate },
      runningMode: "VIDEO",
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    }),
  );
}

const POSE_MODEL = `${MODELS}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`;

export async function createPoseLandmarker(): Promise<PoseLandmarker> {
  const v = await vision();
  return withDelegate((delegate) =>
    PoseLandmarker.createFromOptions(v, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate },
      runningMode: "VIDEO",
      numPoses: 1,
      outputSegmentationMasks: false,
    }),
  );
}

type PosePoint = { x: number; y: number; z: number; visibility?: number };

/**
 * Shoulders and elbows from a pose result. Landmark 11/13 are the subject's left shoulder/elbow,
 * 12/14 the right, 23/24 the hips. Returns null unless both shoulders are confidently visible.
 */
export function readBody(res: { landmarks?: PosePoint[][]; worldLandmarks?: PosePoint[][] }): BodyObservation | null {
  const im = res.landmarks?.[0], wl = res.worldLandmarks?.[0];
  if (!im || !wl || im.length < 25) return null;
  const vis = (i: number) => im[i].visibility ?? 1;
  if (vis(11) < 0.6 || vis(12) < 0.6) return null;
  // tracker axes are x right, y down, z away from the camera
  const cam = (i: number): [number, number, number] => [wl[i].x, -wl[i].y, -wl[i].z];
  const img = (i: number): [number, number] => [im[i].x, im[i].y];
  const seen = (i: number) => vis(i) > 0.55;
  const hips = seen(23) && seen(24);
  const hl = cam(23), hr = cam(24);
  return {
    shoulderL: img(11), shoulderR: img(12),
    elbowL: seen(13) ? img(13) : null, elbowR: seen(14) ? img(14) : null,
    worldShoulderL: cam(11), worldShoulderR: cam(12),
    worldElbowL: seen(13) ? cam(13) : null, worldElbowR: seen(14) ? cam(14) : null,
    wristL: seen(15) ? img(15) : null, wristR: seen(16) ? img(16) : null,
    worldWristL: seen(15) ? cam(15) : null, worldWristR: seen(16) ? cam(16) : null,
    // the tracker always reports hips, even out of frame; only a rough trunk direction is taken from them
    worldHipMid: hips || wl.length > 24 ? [(hl[0] + hr[0]) / 2, (hl[1] + hr[1]) / 2, (hl[2] + hr[2]) / 2] : null,
  };
}

export interface FaceObservation {
  /** 478 × [x, y, z], normalised */
  landmarks: Float32Array;
  blend: Record<string, number>;
}

export function readFace(
  res: { faceLandmarks: Array<Array<{ x: number; y: number; z: number }>>; faceBlendshapes?: Array<{ categories: Array<{ categoryName: string; score: number }> }> },
  out?: FaceObservation,
): FaceObservation | null {
  const lm = res.faceLandmarks?.[0];
  if (!lm || lm.length < 468) return null;
  const o = out ?? { landmarks: new Float32Array(NUM_LANDMARKS * 3), blend: {} };
  const n = Math.min(lm.length, NUM_LANDMARKS);
  for (let i = 0; i < n; i++) {
    o.landmarks[3 * i] = lm[i].x; o.landmarks[3 * i + 1] = lm[i].y; o.landmarks[3 * i + 2] = lm[i].z;
  }
  const cats = res.faceBlendshapes?.[0]?.categories;
  if (cats) for (const c of cats) o.blend[c.categoryName] = c.score;
  return o;
}

export interface HandObservation {
  /** 21 × [x, y] normalised image coordinates */
  image: Float32Array;
  /** 21 × [x, y, z] metres */
  world: Float32Array;
  /** tracker's label; it assumes a mirrored image, so on raw camera frames it is swapped */
  label: string;
  score: number;
}

export function readHands(res: {
  landmarks: Array<Array<{ x: number; y: number; z: number }>>;
  worldLandmarks: Array<Array<{ x: number; y: number; z: number }>>;
  handedness?: Array<Array<{ categoryName: string; score: number }>>;
  handednesses?: Array<Array<{ categoryName: string; score: number }>>;
}): HandObservation[] {
  const out: HandObservation[] = [];
  const hd = res.handedness ?? res.handednesses ?? [];
  for (let h = 0; h < (res.landmarks?.length ?? 0); h++) {
    const lm = res.landmarks[h], wl = res.worldLandmarks?.[h];
    if (!lm || !wl || lm.length < 21) continue;
    const image = new Float32Array(42), world = new Float32Array(63);
    for (let i = 0; i < 21; i++) {
      image[2 * i] = lm[i].x; image[2 * i + 1] = lm[i].y;
      world[3 * i] = wl[i].x; world[3 * i + 1] = wl[i].y; world[3 * i + 2] = wl[i].z;
    }
    out.push({ image, world, label: hd[h]?.[0]?.categoryName ?? "Right", score: hd[h]?.[0]?.score ?? 0 });
  }
  return out;
}
