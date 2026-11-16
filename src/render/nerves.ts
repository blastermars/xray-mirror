/**
 * Nervous system for the Anatomy view.
 *
 *   head nerves        ride with the head (same warp as the skull)
 *   spinal cord, neck  cut by vertebral level; each slice follows its neck vertebra
 *   chest nerves       rigid with the chest
 *   arm nerves         skinned: chest ↔ upper arm across the shoulder, upper arm ↔ forearm
 *                      across the elbow, so they bend with the arm instead of tearing
 *
 * All meshes arrive in head-frame units (template pose).
 */
import * as THREE from "three";
import type { Pipeline, Part } from "./pipeline";
import { makeSolidMaterial } from "./materials";
import type { WarpUniforms } from "./warp";
import type { TorsoPose, TorsoTemplate } from "../fit/torso";
import { similarityToMat4 } from "../math/similarity";

const smooth = (a: number, b: number, x: number) => {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Keep the triangles whose centroid passes `keep`; vertices are re-indexed. */
function subset(g: THREE.BufferGeometry, keep: (x: number, y: number, z: number) => boolean): THREE.BufferGeometry | null {
  const p = g.getAttribute("position").array as Float32Array;
  const ix = g.index!.array;
  const remap = new Int32Array(p.length / 3).fill(-1);
  const pos: number[] = [], idx: number[] = [];
  for (let t = 0; t < ix.length; t += 3) {
    const a = ix[t] * 3, b = ix[t + 1] * 3, c = ix[t + 2] * 3;
    if (!keep((p[a] + p[b] + p[c]) / 3, (p[a + 1] + p[b + 1] + p[c + 1]) / 3, (p[a + 2] + p[b + 2] + p[c + 2]) / 3)) continue;
    for (const v of [ix[t], ix[t + 1], ix[t + 2]]) {
      if (remap[v] < 0) { remap[v] = pos.length / 3; pos.push(p[3 * v], p[3 * v + 1], p[3 * v + 2]); }
      idx.push(remap[v]);
    }
  }
  if (!idx.length) return null;
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(pos), 3));
  out.setIndex(new THREE.BufferAttribute(Uint32Array.from(idx), 1));
  out.computeVertexNormals();
  return out;
}

export interface NerveInputs {
  /** head-frame neck joints, 8 points from the skull base down (template) */
  neckJoints: ArrayLike<number>;
  torso: TorsoTemplate;
  /** template wrists in head-frame units (subject's left, right) */
  wristL: [number, number, number];
  wristR: [number, number, number];
}

interface ArmNerve { upper: Part | null; lower: Part | null; ownU: THREE.Matrix4; ownL: THREE.Matrix4 }

export class NerveRig {
  /** head-frame pieces that follow neck vertebra k (0-based), drawn under the head's root */
  readonly neckPieces: Array<{ part: Part; level: number }> = [];
  private chest: Part[] = [];
  private arms: { l: ArmNerve; r: ArmNerve };
  private chestRoot = new THREE.Group();
  private tmpC = new THREE.Matrix4();
  private tmpH = new THREE.Matrix4();
  private m = new Float64Array(16);

  constructor(
    pipeline: Pipeline, private warp: WarpUniforms, geos: Map<string, THREE.BufferGeometry>,
    headRoot: THREE.Group, inp: NerveInputs,
  ) {
    this.chestRoot.matrixAutoUpdate = false;
    pipeline.scene.add(this.chestRoot);
    const add = (g: THREE.BufferGeometry, parent: THREE.Object3D, warped: boolean, name: string): Part => {
      const mesh = new THREE.Mesh(g);
      mesh.name = `nerve:${name}`;
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      parent.add(mesh);
      const part: Part = { mesh, layer: "nerves", solid: makeSolidMaterial(warp, warped, "nerve") };
      pipeline.addPart(part);
      return part;
    };

    // head nerves: warped with the head
    const head = geos.get("nerve_head");
    if (head) { head.computeVertexNormals(); add(head, headRoot, true, "head"); }

    // spinal cord and neck nerves: sliced by vertebral level (joint heights along the neck)
    const jy: number[] = [];
    for (let k = 0; k < 8; k++) jy.push(inp.neckJoints[3 * k + 1]);
    for (const name of ["nerve_cord", "nerve_neck"]) {
      const g = geos.get(name);
      if (!g) continue;
      const above = subset(g, (_x, y) => y >= jy[0]);
      if (above) add(above, headRoot, true, `${name}:head`);
      for (let k = 0; k < 7; k++) {
        const piece = subset(g, (_x, y) => y < jy[k] && y >= jy[k + 1]);
        if (piece) this.neckPieces.push({ part: add(piece, headRoot, true, `${name}:c${k + 1}`), level: k });
      }
      const below = subset(g, (_x, y) => y < jy[7]);
      if (below) this.chest.push(this.tracked(add(below, this.chestRoot, false, `${name}:chest`)));
    }
    const chest = geos.get("nerve_chest");
    if (chest) { chest.computeVertexNormals(); this.chest.push(this.tracked(add(chest, this.chestRoot, false, "chest"))); }

    this.arms = {
      l: this.buildArm(pipeline, geos.get("nerve_arm_l"), inp.torso.shoulderL, inp.torso.elbowL, inp.wristL, "l"),
      r: this.buildArm(pipeline, geos.get("nerve_arm_r"), inp.torso.shoulderR, inp.torso.elbowR, inp.wristR, "r"),
    };
  }

  private tracked(p: Part): Part { p.mesh.userData.tracked = false; return p; }

  /**
   * Split an arm's nerves at mid upper arm into two skinned meshes. In each, `jawW` is the weight
   * of the more distal segment and the material gets its own second transform (not the jaw's).
   */
  private buildArm(pipeline: Pipeline, g: THREE.BufferGeometry | undefined, shoulder: ArrayLike<number>, elbow: ArrayLike<number>, wrist: ArrayLike<number>, tag: string): ArmNerve {
    const out: ArmNerve = { upper: null, lower: null, ownU: new THREE.Matrix4(), ownL: new THREE.Matrix4() };
    if (!g) return out;
    const S = new THREE.Vector3(shoulder[0], shoulder[1], shoulder[2]);
    const E = new THREE.Vector3(elbow[0], elbow[1], elbow[2]);
    const W = new THREE.Vector3(wrist[0], wrist[1], wrist[2]);
    const uH = E.clone().sub(S); const lenH = uH.length(); uH.normalize();
    const uF = W.clone().sub(E).normalize();
    const alongH = (x: number, y: number, z: number) => (x - S.x) * uH.x + (y - S.y) * uH.y + (z - S.z) * uH.z;
    const alongF = (x: number, y: number, z: number) => (x - E.x) * uF.x + (y - E.y) * uF.y + (z - E.z) * uF.z;
    const make = (geo: THREE.BufferGeometry | null, weight: (x: number, y: number, z: number) => number, own: THREE.Matrix4, name: string): Part | null => {
      if (!geo) return null;
      const p = geo.getAttribute("position").array as Float32Array;
      const w = new Float32Array(p.length / 3);
      for (let i = 0; i < w.length; i++) w[i] = weight(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
      geo.setAttribute("jawW", new THREE.BufferAttribute(w, 1));
      const mesh = new THREE.Mesh(geo);
      mesh.name = `nerve:arm_${tag}:${name}`;
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      mesh.userData.tracked = false;
      pipeline.scene.add(mesh);
      const solid = makeSolidMaterial(this.warp, false, "nerve");
      solid.uniforms.uJaw = { value: own }; // this mesh's own second transform
      const part: Part = { mesh, layer: "nerves", solid };
      pipeline.addPart(part);
      return part;
    };
    // upper: from the spine across the shoulder to mid upper arm (chest → upper arm)
    out.upper = make(subset(g, (x, y, z) => alongH(x, y, z) < 0.5 * lenH), (x, y, z) => smooth(-3, 5, alongH(x, y, z)), out.ownU, "upper");
    // lower: mid upper arm across the elbow to the wrist (upper arm → forearm)
    out.lower = make(subset(g, (x, y, z) => alongH(x, y, z) >= 0.5 * lenH), (x, y, z) => smooth(-3, 4, alongF(x, y, z)), out.ownL, "lower");
    return out;
  }

  hideBody() {
    for (const p of this.chest) p.mesh.userData.tracked = false;
    for (const a of [this.arms.l, this.arms.r]) for (const p of [a.upper, a.lower]) if (p) p.mesh.userData.tracked = false;
  }

  /** Slices of the cord and neck nerves follow their vertebra (matrices in the head frame). */
  setNeck(neckMats: Float32Array[]) {
    for (const { part, level } of this.neckPieces) part.solid!.uniforms.uPart.value.fromArray(neckMats[level]);
  }

  /**
   * @param foreL,foreR forearm transforms expressed for head-frame meshes (template → camera), or
   *                    null when that forearm is not tracked (it then stays in line with the upper arm)
   */
  setBody(t: TorsoPose, foreL: THREE.Matrix4 | null, foreR: THREE.Matrix4 | null) {
    const C = this.tmpC.fromArray(similarityToMat4(t.pose, this.m) as unknown as number[]);
    this.chestRoot.matrix.copy(C);
    this.chestRoot.updateMatrixWorld(true);
    for (const p of this.chest) p.mesh.userData.tracked = true;
    const Cinv = C.clone().invert();
    const arm = (a: ArmNerve, humerus: Float32Array, fore: THREE.Matrix4 | null) => {
      const H = this.tmpH.fromArray(humerus);
      if (a.upper) {
        a.upper.mesh.matrix.copy(C); a.upper.mesh.matrixWorld.copy(C);
        a.ownU.multiplyMatrices(Cinv, H);          // chest space → where the upper arm puts the point
        a.upper.mesh.userData.tracked = true;
      }
      if (a.lower) {
        a.lower.mesh.matrix.copy(H); a.lower.mesh.matrixWorld.copy(H);
        if (fore) a.ownL.multiplyMatrices(H.clone().invert(), fore);
        else a.ownL.identity();
        a.lower.mesh.userData.tracked = true;
      }
    };
    arm(this.arms.l, t.humerusL, foreL);
    arm(this.arms.r, t.humerusR, foreR);
  }
}
