/**
 * Test subject: renders the template head, deliberately reshaped into a
 * different "person" and animated (turning, nodding, opening the jaw), into a
 * canvas that the app treats as its camera. Because the reshaping and motion
 * are known exactly, the fitted skull can be scored against ground truth.
 */
import * as THREE from "three";
import { loadGlbGeometries, jawWeight, type TemplateData } from "../render/anatomy";
import { jawMatrix, glideFor, type JawRig } from "../fit/jaw";
import { axisAngleMat3, mat3Mul, type Vec3 } from "../math/linalg";
import { applySimilarity, similarityToMat4, type Similarity } from "../math/similarity";
import { projectToImage } from "../tracking/faceGeometry";

export interface SyntheticOptions {
  width?: number;
  height?: number;
  vfov?: number;
  /** 0 = the template's own proportions, 1 = the full test reshaping */
  reshape?: number;
  /** freeze the motion (frontal, mouth closed) */
  still?: boolean;
  /**
   * Also open the jaw. Off by default: stretching the skin of a render does not part the lips,
   * and the tracker does not read it as an open mouth, so it is not a valid jaw test.
   */
  jaw?: boolean;
}

/** The "other person": wider, shorter face, deeper head, stronger chin. Head frame, cm. */
export function reshapePoint(p: ArrayLike<number>, k: number, out: number[] = [0, 0, 0]): number[] {
  const x = p[0], y = p[1], z = p[2];
  const chin = Math.exp(-((y + 6.5) ** 2 + x * x) / 14);
  const brow = Math.exp(-((y - 4.5) ** 2) / 10);
  out[0] = x * (1 + k * (0.11 - 0.012 * y / 10));
  out[1] = y * (1 - k * 0.06) - k * 0.25;
  out[2] = z * (1 + k * 0.05) + k * (0.45 * chin * Math.max(0, z) / 5 + 0.15 * brow);
  return out;
}

export class SyntheticSubject {
  readonly canvas: HTMLCanvasElement;
  readonly vfov: number;
  truth = { pose: null as Similarity | null, jawOpen: 0, time: 0 };

  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private root = new THREE.Group();
  private skinBase!: Float32Array; // reshaped, jaw closed
  private skinJawW!: Float32Array;
  private skinGeo!: THREE.BufferGeometry;
  private rig!: JawRig;
  private k: number;
  private still: boolean;
  private moveJaw: boolean;
  private poseM = new Float64Array(16);

  constructor(opt: SyntheticOptions = {}) {
    const w = opt.width ?? 1280, h = opt.height ?? 720;
    this.vfov = opt.vfov ?? 45;
    this.k = opt.reshape ?? 1;
    this.still = !!opt.still;
    this.moveJaw = !!opt.jaw;
    this.canvas = document.createElement("canvas");
    this.canvas.width = w; this.canvas.height = h;
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(w, h, false);
    this.camera = new THREE.PerspectiveCamera(this.vfov, w / h, 1, 1000);
    this.scene.background = new THREE.Color(0x4f5a66);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x554433, 1.15));
    const key = new THREE.DirectionalLight(0xfff2e0, 2.1);
    key.position.set(-30, 40, 80);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xdde8ff, 0.8);
    fill.position.set(50, 0, 60);
    this.scene.add(fill);
    this.root.matrixAutoUpdate = false;
    this.scene.add(this.root);
  }

  async load(base = "assets/") {
    const [tpl, geos] = await Promise.all([
      fetch(base + "template.json").then((r) => r.json() as Promise<TemplateData>),
      loadGlbGeometries(base + "head.glb"),
    ]);
    const toHead = new THREE.Matrix4().fromArray(tpl.toHead);
    const tmp = [0, 0, 0];
    const reshape = (g: THREE.BufferGeometry) => {
      g.applyMatrix4(toHead);
      const a = g.getAttribute("position").array as Float32Array;
      for (let i = 0; i < a.length; i += 3) {
        reshapePoint([a[i], a[i + 1], a[i + 2]], this.k, tmp);
        a[i] = tmp[0]; a[i + 1] = tmp[1]; a[i + 2] = tmp[2];
      }
    };
    const skin = geos.get("skin")!;
    // jaw weights are defined on the template shape, before reshaping
    skin.applyMatrix4(toHead);
    const sp = skin.getAttribute("position").array as Float32Array;
    this.skinJawW = new Float32Array(sp.length / 3);
    for (let i = 0; i < this.skinJawW.length; i++) this.skinJawW[i] = jawWeight([sp[3 * i], sp[3 * i + 1], sp[3 * i + 2]], tpl.jawRef);
    for (let i = 0; i < sp.length; i += 3) {
      reshapePoint([sp[i], sp[i + 1], sp[i + 2]], this.k, tmp);
      sp[i] = tmp[0]; sp[i + 1] = tmp[1]; sp[i + 2] = tmp[2];
    }
    this.skinBase = Float32Array.from(sp);
    this.skinGeo = skin;
    skin.computeVertexNormals();
    this.root.add(new THREE.Mesh(skin, new THREE.MeshStandardMaterial({ color: 0xd9a688, roughness: 0.62 })));
    for (const n of ["eye_l", "eye_r"]) {
      const g = geos.get(n);
      if (!g) continue;
      reshape(g);
      g.computeVertexNormals();
      this.root.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0xf2f0ea, roughness: 0.3 })));
    }
    this.rig = {
      hingeR: reshapePoint(tpl.jaw.hingeR, this.k) as Vec3,
      hingeL: reshapePoint(tpl.jaw.hingeL, this.k) as Vec3,
    };
  }

  /** Draw the subject at time t (ms). */
  draw(t: number) {
    const s = t / 1000;
    const still = this.still;
    const yaw = still ? 0 : 0.42 * Math.sin(s * 0.55);
    const pitch = still ? 0 : 0.12 * Math.sin(s * 0.37 + 1);
    const roll = still ? 0 : 0.08 * Math.sin(s * 0.23 + 2);
    // open the jaw for a couple of seconds out of every eight
    const phase = (s % 8) / 8;
    const jawOpen = still || !this.moveJaw ? 0 : phase > 0.55 && phase < 0.85 ? 0.42 * Math.sin(((phase - 0.55) / 0.3) * Math.PI) ** 2 : 0;
    const R = mat3Mul(axisAngleMat3([0, 0, 1], roll), mat3Mul(axisAngleMat3([0, 1, 0], yaw), axisAngleMat3([1, 0, 0], pitch)));
    const pose: Similarity = {
      s: 1, R,
      t: [still ? 0 : 5 * Math.sin(s * 0.31), still ? 1 : 1 + 2 * Math.sin(s * 0.19), still ? -55 : -55 + 8 * Math.sin(s * 0.13)],
    };
    this.truth = { pose, jawOpen, time: t };

    // jaw: blend the lower face toward the rotated mandible
    const m = jawMatrix(this.rig, { open: jawOpen, swing: 0, glide: glideFor(jawOpen) });
    const a = this.skinGeo.getAttribute("position").array as Float32Array, b = this.skinBase, w = this.skinJawW;
    for (let i = 0, k = 0; i < a.length; i += 3, k++) {
      const x = b[i], y = b[i + 1], z = b[i + 2], wk = w[k];
      if (wk <= 0 || jawOpen === 0) { a[i] = x; a[i + 1] = y; a[i + 2] = z; continue; }
      a[i] = x + wk * (m[0] * x + m[4] * y + m[8] * z + m[12] - x);
      a[i + 1] = y + wk * (m[1] * x + m[5] * y + m[9] * z + m[13] - y);
      a[i + 2] = z + wk * (m[2] * x + m[6] * y + m[10] * z + m[14] - z);
    }
    this.skinGeo.getAttribute("position").needsUpdate = true;
    this.skinGeo.computeVertexNormals();

    this.root.matrix.fromArray(similarityToMat4(pose, this.poseM) as unknown as number[]);
    this.root.updateMatrixWorld(true);
    this.renderer.render(this.scene, this.camera);
  }

  /** Where a template point truly is in the image (normalised, y down) for the current frame. */
  truthImagePoint(templatePoint: ArrayLike<number>, movesWithJaw = false): [number, number] {
    let p = reshapePoint(templatePoint, this.k);
    if (movesWithJaw && this.truth.jawOpen > 0) {
      const m = jawMatrix(this.rig, { open: this.truth.jawOpen, swing: 0, glide: glideFor(this.truth.jawOpen) });
      p = [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]];
    }
    const c = applySimilarity(this.truth.pose!, p);
    return projectToImage(c, this.canvas.width / this.canvas.height, this.vfov);
  }
}
