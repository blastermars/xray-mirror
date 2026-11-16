/**
 * Loads the BodyParts3D meshes, moves them into the head frame (or the hand
 * template frame) using the baked registration, and creates the render parts.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import type { Pipeline, Part } from "./pipeline";
import { makeMaskMaterial, makeSolidMaterial, makeTauMaterial, type Channel, type Tissue } from "./materials";
import { jacobiEigen } from "../math/linalg";
import type { WarpUniforms } from "./warp";
import type { HandBoneDef } from "../fit/hand";
import type { Vec3 } from "../math/linalg";
import { loopSubdivide } from "./subdivide";

export interface TemplateData {
  version: number;
  /** column-major 4×4: BodyParts3D mm → head frame cm */
  toHead: number[];
  /** 468 template skin landmarks in the head frame as the tracker perceives them, flat xyz */
  landmarks: number[];
  /** the same landmarks at their true depth: the point of the skin surface under each landmark's pixel */
  landmarksTrue: number[];
  controlIdx: number[];
  anchors: number[];
  jaw: { hingeR: Vec3; hingeL: Vec3 };
  /** stomion / chin heights used to weight soft tissue to the jaw */
  jawRef: { mouthY: number; chinY: number; hingeY: number; hingeZ: number };
  /** neck joints from the skull base down: 8 points (above C1 … below C7), head frame */
  neckJoints: number[];
  hand: { joints: number[]; bones: HandBoneDef[]; rigid: string[]; forearm: string[]; forearmAxis: Vec3 };
  labels: Array<{ id: string; text: string; layer: string; anchor: Vec3; jaw: boolean }>;
}

export interface Manifest {
  attribution: string;
  license: string;
  licenseUrl?: string;
  units: string;
  frame: Record<string, string>;
  files: Record<string, string>;
  /** the facial-muscle file comes from an older release with its own licence */
  muscles?: { source: string; license: string; licenseUrl?: string; attribution: string };
  meshes: Record<string, {
    file: string; triangles: number; closed: boolean; centroid: Vec3;
    bbox: { min: Vec3; max: Vec3 };
    pca: { axes: Vec3[]; min: Vec3; max: Vec3 };
    label: string;
  }>;
}

interface PartStyle {
  layer: string;
  channel: Channel;
  shell: number;
  vol: number;
  color: number;
  gloss?: number;
  /** 1 = moves fully with the mandible, "blend" = per-vertex weight, 0 = fixed to the skull */
  jaw: 0 | 1 | "blend";
  solid?: boolean;
}

/** Radiographic densities (relative): enamel > cortical bone > trabecular ≫ soft tissue. */
function styleFor(name: string): PartStyle | null {
  if (name === "cranium") return { layer: "bone", channel: "bone", shell: 0.03, vol: 0.14, color: 0xe6dcc3, jaw: 0 };
  if (name === "mandible") return { layer: "bone", channel: "bone", shell: 0.035, vol: 0.15, color: 0xe6dcc3, jaw: 1 };
  if (name === "teeth_upper") return { layer: "teeth", channel: "bone", shell: 0.05, vol: 0.5, color: 0xf6f3e8, gloss: 0.5, jaw: 0 };
  if (name === "teeth_lower") return { layer: "teeth", channel: "bone", shell: 0.05, vol: 0.5, color: 0xf6f3e8, gloss: 0.5, jaw: 1 };
  if (name === "hyoid") return { layer: "spine", channel: "bone", shell: 0.03, vol: 0.12, color: 0xe0d5b8, jaw: 0 };
  if (/^c[1-7]$/.test(name)) return { layer: "spine", channel: "bone", shell: 0.03, vol: 0.12, color: 0xe0d5b8, jaw: 0 };
  if (name === "cerebrum" || name === "cerebellum" || name === "brainstem")
    return { layer: "brain", channel: "brain", shell: 0.06, vol: 0.006, color: name === "cerebrum" ? 0xe2c2b8 : 0xd3aca2, jaw: 0 };
  if (name.startsWith("eye")) return { layer: "eyes", channel: "brain", shell: 0.10, vol: 0.04, color: 0xeeeae2, gloss: 0.7, jaw: 0 };
  if (name.startsWith("muscle_")) return { layer: "muscles", channel: "muscle", shell: 0.022, vol: 0.03, color: 0xa3423a, gloss: 0.25, jaw: "blend" };
  if (name === "skin") return { layer: "skin", channel: "soft", shell: 0.012, vol: 0.011, color: 0xd9b8a0, jaw: "blend", solid: false };
  return null;
}

function tissueFor(name: string): Tissue {
  if (name.startsWith("teeth")) return "tooth";
  if (name.startsWith("muscle_")) return "muscle";
  if (name === "cerebrum" || name === "cerebellum" || name === "brainstem") return "brain";
  if (name.startsWith("eye")) return "eye";
  return "bone";
}

/**
 * Per-mesh inputs of the tissue shader: a muscle's fibre direction, centre and half length (taken
 * from the principal axis of its left half, since left and right share a mesh), an eye's centre.
 */
function describeTissue(name: string, g: THREE.BufferGeometry, mat: THREE.ShaderMaterial) {
  const p = g.getAttribute("position").array as Float32Array;
  if (name.startsWith("eye")) {
    g.computeBoundingBox();
    g.boundingBox!.getCenter(mat.uniforms.uCentre.value);
    return;
  }
  if (!name.startsWith("muscle_")) return;
  // use the subject's left half; midline muscles with nothing clearly on one side use everything
  let n = 0;
  for (let i = 0; i < p.length; i += 3) if (p[i] > 0.3) n++;
  const half = n > p.length / 3 / 5;
  const c = [0, 0, 0];
  let k = 0;
  for (let i = 0; i < p.length; i += 3) {
    if (half && p[i] <= 0.3) continue;
    c[0] += p[i]; c[1] += p[i + 1]; c[2] += p[i + 2]; k++;
  }
  if (!k) return;
  c[0] /= k; c[1] /= k; c[2] /= k;
  const cov = new Float64Array(9);
  for (let i = 0; i < p.length; i += 3) {
    if (half && p[i] <= 0.3) continue;
    const d = [p[i] - c[0], p[i + 1] - c[1], p[i + 2] - c[2]];
    for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) cov[r * 3 + q] += d[r] * d[q];
  }
  const { values, vectors } = jacobiEigen(cov, 3);
  let best = 0;
  for (let i = 1; i < 3; i++) if (values[i] > values[best]) best = i;
  const axis = new THREE.Vector3(vectors[best], vectors[3 + best], vectors[6 + best]).normalize();
  let reach = 0;
  for (let i = 0; i < p.length; i += 3) {
    if (half && p[i] <= 0.3) continue;
    reach = Math.max(reach, Math.abs((p[i] - c[0]) * axis.x + (p[i + 1] - c[1]) * axis.y + (p[i + 2] - c[2]) * axis.z));
  }
  mat.uniforms.uFiber.value.copy(axis);
  mat.uniforms.uCentre.value.set(c[0], c[1], c[2]);
  mat.uniforms.uHalfLen.value = reach || 1;
}

/** World-space float positions + indices of a glTF mesh (handles quantised attributes and node transforms). */
function extractGeometry(mesh: THREE.Mesh): THREE.BufferGeometry {
  mesh.updateWorldMatrix(true, false);
  const src = mesh.geometry;
  const pos = src.getAttribute("position");
  const out = new Float32Array(pos.count * 3);
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
    out[3 * i] = v.x; out[3 * i + 1] = v.y; out[3 * i + 2] = v.z;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(out, 3));
  if (src.index) g.setIndex(new THREE.BufferAttribute(Uint32Array.from(src.index.array as ArrayLike<number>), 1));
  return g;
}

export async function loadGlbGeometries(url: string): Promise<Map<string, THREE.BufferGeometry>> {
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  const gltf = await loader.loadAsync(url);
  const byName = new Map<string, THREE.BufferGeometry[]>();
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    let name = m.name;
    if ((!name || /_\d+$/.test(name)) && m.parent && m.parent !== gltf.scene && m.parent.name) name = m.parent.name;
    const list = byName.get(name) ?? [];
    list.push(extractGeometry(m));
    byName.set(name, list);
  });
  const out = new Map<string, THREE.BufferGeometry>();
  for (const [name, list] of byName) out.set(name, list.length === 1 ? list[0] : mergeGeometries(list));
  return out;
}

function mergeGeometries(list: THREE.BufferGeometry[]): THREE.BufferGeometry {
  let nv = 0, ni = 0;
  for (const g of list) { nv += g.getAttribute("position").count; ni += g.index ? g.index.count : g.getAttribute("position").count; }
  const pos = new Float32Array(nv * 3), idx = new Uint32Array(ni);
  let vo = 0, io = 0;
  for (const g of list) {
    const p = g.getAttribute("position");
    pos.set(p.array as Float32Array, vo * 3);
    if (g.index) for (let i = 0; i < g.index.count; i++) idx[io++] = g.index.getX(i) + vo;
    else for (let i = 0; i < p.count; i++) idx[io++] = i + vo;
    vo += p.count;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

/** Signed volume of an indexed mesh; negative means the triangles wind inward. */
export function signedVolume(g: THREE.BufferGeometry): number {
  const p = g.getAttribute("position").array as Float32Array;
  const ix = g.index!.array;
  let vol = 0;
  for (let i = 0; i < ix.length; i += 3) {
    const a = ix[i] * 3, b = ix[i + 1] * 3, c = ix[i + 2] * 3;
    vol += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1])
         - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c])
         + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return vol / 6;
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/**
 * How much a soft-tissue point (template head frame) follows the mandible:
 * 1 on the chin and lower lip, 0 above the mouth and behind the ramus, with the
 * cheeks stretching between the cheekbone and the lower border.
 */
export function jawWeight(p: ArrayLike<number>, ref: TemplateData["jawRef"]): number {
  const [x, y, z] = [p[0], p[1], p[2]];
  const front = 1 - smooth(ref.mouthY - 0.7, ref.mouthY + 0.5, y); // 1 below the mouth line
  const cheekTop = ref.hingeY - 1.2, cheekBot = ref.chinY + 3.0;
  const cheek = 1 - smooth(cheekBot, cheekTop, y); // 0 at the cheekbone → 1 near the lower border
  const lateral = smooth(2.2, 4.8, Math.abs(x));
  let w = front * (1 - lateral) + cheek * lateral;
  w *= smooth(ref.hingeZ - 1.5, ref.hingeZ + 1.0, z); // nothing behind the ramus
  w *= smooth(ref.chinY - 4.5, ref.chinY - 1.0, y); // the neck stays with the neck
  return w;
}

/**
 * The reference body is lean: its cheekbones lie about 3.7 mm under the skin at their widest point
 * (zygion), against a published adult mean of 7.5 mm (Hona & Stephan 2024). Because the skull is
 * carried along with the viewer's skin, that thin layer would make every viewer's cheekbones nearly
 * as wide as their face. Move the zygomatic arches inward by most of the difference, fading out
 * smoothly over the neighbouring bone.
 */
function narrowCheekbones(g: THREE.BufferGeometry, template: TemplateData) {
  const INWARD_CM = 0.3; // ≈ 2.7 mm of true size (head-frame units are ~1.1× true centimetres)
  const SIGMA = 2.3;
  const p = g.getAttribute("position").array as Float32Array;
  for (const lm of [234, 454]) {
    const sx = template.landmarks[3 * lm], sy = template.landmarks[3 * lm + 1], sz = template.landmarks[3 * lm + 2];
    // the point of the arch nearest the face-edge landmark
    let best = 0, bd = Infinity;
    for (let i = 0; i < p.length; i += 3) {
      const d = (p[i] - sx) ** 2 + (p[i + 1] - sy) ** 2 + (p[i + 2] - sz) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    const cx = p[best], cy = p[best + 1], cz = p[best + 2];
    const side = Math.sign(cx) || 1;
    for (let i = 0; i < p.length; i += 3) {
      if (p[i] * side <= 0) continue; // this side of the head only
      const d2 = (p[i] - cx) ** 2 + (p[i + 1] - cy) ** 2 + (p[i + 2] - cz) ** 2;
      p[i] -= side * INWARD_CM * Math.exp(-d2 / (2 * SIGMA * SIGMA));
    }
  }
}

export interface Anatomy {
  template: TemplateData;
  manifest: Manifest;
  headRoot: THREE.Group;
  headParts: Map<string, Part>;
  handGeometries: Map<string, THREE.BufferGeometry>;
  /** shoulder girdle and chest meshes in head-frame units (empty when the file is absent) */
  torsoGeometries: Map<string, THREE.BufferGeometry>;
  /** nervous system meshes in head-frame units (empty when the file is absent) */
  nerveGeometries: Map<string, THREE.BufferGeometry>;
}

export async function loadAnatomy(pipeline: Pipeline, warp: WarpUniforms, base = "assets/"): Promise<Anatomy> {
  const [template, manifest, headGeo, handGeo] = await Promise.all([
    fetch(base + "template.json").then((r) => r.json() as Promise<TemplateData>),
    fetch(base + "manifest.json").then((r) => r.json() as Promise<Manifest>),
    loadGlbGeometries(base + "head.glb"),
    loadGlbGeometries(base + "hand.glb"),
  ]);

  // facial muscles live in their own file (different source release and licence)
  if (manifest.files?.muscles) {
    try {
      const extra = await loadGlbGeometries(base + manifest.files.muscles);
      for (const [name, g] of extra) if (!headGeo.has(name)) headGeo.set(name, g);
    } catch (e) {
      console.warn("muscle meshes unavailable", e);
    }
  }

  const toHead = new THREE.Matrix4().fromArray(template.toHead);
  const headRoot = new THREE.Group();
  headRoot.matrixAutoUpdate = false;
  headRoot.visible = false;
  pipeline.scene.add(headRoot);
  const headParts = new Map<string, Part>();

  for (const [name, g] of headGeo) {
    const st = styleFor(name);
    if (!st) continue;
    g.applyMatrix4(toHead);
    if (name === "cranium") narrowCheekbones(g, template);
    g.computeVertexNormals();
    const n = g.getAttribute("position").count;
    const jw = new Float32Array(n);
    if (st.jaw === 1) jw.fill(1);
    else if (st.jaw === "blend") {
      const p = g.getAttribute("position").array as Float32Array;
      for (let i = 0; i < n; i++) jw[i] = jawWeight([p[3 * i], p[3 * i + 1], p[3 * i + 2]], template.jawRef);
    }
    g.setAttribute("jawW", new THREE.BufferAttribute(jw, 1));
    const closed = manifest.meshes[name]?.closed !== false;
    const volSign = signedVolume(g) < 0 ? -1 : 1;
    const mesh = new THREE.Mesh(g);
    mesh.name = name;
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    headRoot.add(mesh);
    const part: Part = {
      mesh,
      layer: st.layer,
      tau: makeTauMaterial(warp, true, { channel: st.channel, shell: st.shell, vol: closed ? st.vol : 0, volSign }),
      solid: st.solid === false ? undefined : makeSolidMaterial(warp, true, tissueFor(name)),
      mask: name === "skin" ? makeMaskMaterial(warp, true, "head") : undefined,
    };
    if (name === "skin") {
      // the envelope fades out down the neck (the neck skin does not follow the vertebrae)
      const y0 = template.jawRef.chinY - 6.5;
      part.tau!.uniforms.uFadePlane.value.set(0, 1, 0, y0);
      part.tau!.uniforms.uFadeLen.value = 4.5;
      // The darkened region stops just under the jaw. The skin mesh is rigid with the head, so any of
      // it left over the lower neck and shoulders would swing across them as the head moves.
      part.mask!.uniforms.uFadePlane.value.set(0, 1, 0, template.jawRef.chinY - 5);
      part.mask!.uniforms.uFadeLen.value = 3.5;
    }
    if (part.solid) describeTissue(name, g, part.solid);
    pipeline.addPart(part);
    headParts.set(name, part);
  }

  // hand template frame: BodyParts3D mm → cm, nothing else. The source bones are very coarse, so
  // they are subdivided once to stay round when a hand is held close to the camera.
  for (const [name, g] of handGeo) {
    const sub = loopSubdivide(g.getAttribute("position").array as Float32Array, g.index!.array);
    const smooth = new THREE.BufferGeometry();
    smooth.setAttribute("position", new THREE.BufferAttribute(sub.positions, 3));
    smooth.setIndex(new THREE.BufferAttribute(sub.indices, 1));
    smooth.scale(0.1, 0.1, 0.1);
    smooth.computeVertexNormals();
    handGeo.set(name, smooth);
  }
  // shoulder girdle and upper chest: same frame as the head, placed by the body tracker
  let torsoGeo = new Map<string, THREE.BufferGeometry>();
  if (manifest.files?.torso) {
    try {
      torsoGeo = await loadGlbGeometries(base + manifest.files.torso);
      for (const g of torsoGeo.values()) { g.applyMatrix4(toHead); g.computeVertexNormals(); }
    } catch (e) {
      console.warn("chest meshes unavailable", e);
    }
  }
  let nerveGeo = new Map<string, THREE.BufferGeometry>();
  if (manifest.files?.nerves) {
    try {
      nerveGeo = await loadGlbGeometries(base + manifest.files.nerves);
      for (const g of nerveGeo.values()) g.applyMatrix4(toHead);
    } catch (e) {
      console.warn("nerve meshes unavailable", e);
    }
  }
  return { template, manifest, headRoot, headParts, handGeometries: handGeo, torsoGeometries: torsoGeo, nerveGeometries: nerveGeo };
}
