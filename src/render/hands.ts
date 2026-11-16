/** One tracked hand: bone meshes placed from the solved pose, plus a soft mask over the real hand. */
import * as THREE from "three";
import type { Pipeline, Part } from "./pipeline";
import { makeMaskMaterial, makeSolidMaterial, makeTauMaterial } from "./materials";
import type { WarpUniforms } from "./warp";
import type { HandPose, HandTemplate } from "../fit/hand";
import type { Vec3 } from "../math/linalg";

const BONE_TAU = { channel: "bone" as const, shell: 0.045, vol: 0.2 };

/** Segments of the hand silhouette: [joint a, joint b, radius in cm at hand scale 1]. */
const MASK_SEGMENTS: Array<[number, number, number]> = [
  [0, 1, 1.5], [1, 2, 1.35], [2, 3, 1.1], [3, 4, 0.95],
  [0, 5, 1.4], [5, 6, 1.0], [6, 7, 0.9], [7, 8, 0.8],
  [0, 9, 1.4], [9, 10, 1.0], [10, 11, 0.9], [11, 12, 0.8],
  [0, 13, 1.4], [13, 14, 0.95], [14, 15, 0.85], [15, 16, 0.78],
  [0, 17, 1.4], [17, 18, 0.9], [18, 19, 0.8], [19, 20, 0.72],
  [5, 9, 1.2], [9, 13, 1.2], [13, 17, 1.2], [1, 5, 1.3],
];
const MASK_FANS: Array<[number, number, number]> = [[0, 1, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17], [1, 2, 5]];
const DISC = 10;

export class HandRig {
  private bones: Part[] = [];
  private rigid: Part[] = [];
  private forearm: Part[] = [];
  private maskPart: Part;
  private maskPos: Float32Array;
  private all: Part[] = [];

  constructor(pipeline: Pipeline, warp: WarpUniforms, private tpl: HandTemplate & { forearmAxis: Vec3 }, geos: Map<string, THREE.BufferGeometry>) {
    const make = (name: string): Part | null => {
      const g = geos.get(name);
      if (!g) { console.warn(`hand mesh missing: ${name}`); return null; }
      const mesh = new THREE.Mesh(g);
      mesh.name = `hand:${name}`;
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      mesh.userData.tracked = false;
      pipeline.scene.add(mesh);
      const part: Part = {
        mesh, layer: "hands",
        tau: makeTauMaterial(warp, false, BONE_TAU),
        solid: makeSolidMaterial(warp, false, "bone"),
      };
      pipeline.addPart(part);
      this.all.push(part);
      return part;
    };
    for (const b of tpl.bones) { const p = make(b.name); if (p) this.bones.push(p); else this.bones.push(null as unknown as Part); }
    for (const n of tpl.rigid) { const p = make(n); if (p) this.rigid.push(p); }
    for (const n of tpl.forearm) {
      const p = make(n);
      if (!p) continue;
      // only the distal forearm is shown; it fades out toward the elbow
      const ax = tpl.forearmAxis;
      const wrist: Vec3 = [tpl.joints[0], tpl.joints[1], tpl.joints[2]];
      const nrm = new THREE.Vector3(-ax[0], -ax[1], -ax[2]);
      const w = nrm.x * wrist[0] + nrm.y * wrist[1] + nrm.z * wrist[2] - 11;
      for (const m of [p.tau!, p.solid!]) { m.uniforms.uFadePlane.value.set(nrm.x, nrm.y, nrm.z, w); m.uniforms.uFadeLen.value = 6; }
      p.solid!.transparent = true;
      this.forearm.push(p);
    }

    // soft mask over the real hand (camera space, rebuilt every frame)
    const nVerts = MASK_SEGMENTS.length * 6 + 21 * DISC * 3 + MASK_FANS.length * 3 + 6;
    this.maskPos = new Float32Array(nVerts * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(this.maskPos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(nVerts * 3), 3));
    const mesh = new THREE.Mesh(g);
    mesh.frustumCulled = false;
    mesh.matrixAutoUpdate = false;
    mesh.userData.tracked = false;
    pipeline.scene.add(mesh);
    this.maskPart = { mesh, layer: "handmask", mask: makeMaskMaterial(warp, false, "hand") };
    pipeline.addPart(this.maskPart);
  }

  /** The short forearm ends attached to the hand; switched off when a full forearm is drawn from the elbow. */
  stub = true;

  hide() {
    for (const p of this.all) p.mesh.userData.tracked = false;
    this.maskPart.mesh.userData.tracked = false;
  }

  /** @param joints 21 joints in camera space (cm) */
  update(pose: HandPose, joints: ArrayLike<number>) {
    const depthOf = (i: number) => Math.hypot(joints[3 * i], joints[3 * i + 1], joints[3 * i + 2]);
    this.tpl.bones.forEach((def, k) => {
      const part = this.bones[k];
      if (!part) return;
      part.mesh.matrix.fromArray(pose.bones[k]);
      part.mesh.matrixWorld.copy(part.mesh.matrix);
      part.mesh.userData.tracked = true;
      part.tau!.uniforms.uRefDepth.value = 0.5 * (depthOf(def.a >= 0 ? def.a : 0) + depthOf(def.b));
    });
    const wristDepth = depthOf(0);
    for (const part of this.rigid) {
      part.mesh.matrix.fromArray(pose.palmMatrix);
      part.mesh.matrixWorld.copy(part.mesh.matrix);
      part.mesh.userData.tracked = true;
      part.tau!.uniforms.uRefDepth.value = wristDepth;
    }
    for (const part of this.forearm) {
      part.mesh.matrix.fromArray(pose.palmMatrix);
      part.mesh.matrixWorld.copy(part.mesh.matrix);
      part.mesh.userData.tracked = this.stub;
      part.tau!.uniforms.uRefDepth.value = wristDepth;
    }
    this.buildMask(joints, pose);
  }

  private buildMask(j: ArrayLike<number>, pose: HandPose) {
    const out = this.maskPos;
    let o = 0;
    const s = pose.scale;
    const put = (x: number, y: number, z: number) => { out[o++] = x; out[o++] = y; out[o++] = z; };
    // Work in the image plane: the camera looks down −Z from the origin, so an offset in XY
    // scaled by depth keeps a constant on-screen width.
    const quad = (ax: number, ay: number, az: number, bx: number, by: number, bz: number, r: number) => {
      // perpendicular to the projected direction
      const pax = ax / -az, pay = ay / -az, pbx = bx / -bz, pby = by / -bz;
      let dx = pbx - pax, dy = pby - pay;
      const l = Math.hypot(dx, dy) || 1;
      dx /= l; dy /= l;
      const nx = -dy * r, ny = dx * r;
      put(ax + nx, ay + ny, az); put(ax - nx, ay - ny, az); put(bx + nx, by + ny, bz);
      put(bx + nx, by + ny, bz); put(ax - nx, ay - ny, az); put(bx - nx, by - ny, bz);
    };
    const radiusAt = new Float32Array(21);
    for (const [a, b, r] of MASK_SEGMENTS) {
      const rr = r * s;
      quad(j[3 * a], j[3 * a + 1], j[3 * a + 2], j[3 * b], j[3 * b + 1], j[3 * b + 2], rr);
      radiusAt[a] = Math.max(radiusAt[a], rr); radiusAt[b] = Math.max(radiusAt[b], rr);
    }
    for (let i = 0; i < 21; i++) {
      const cx = j[3 * i], cy = j[3 * i + 1], cz = j[3 * i + 2], r = radiusAt[i];
      for (let k = 0; k < DISC; k++) {
        const a0 = (k / DISC) * Math.PI * 2, a1 = ((k + 1) / DISC) * Math.PI * 2;
        put(cx, cy, cz); put(cx + Math.cos(a0) * r, cy + Math.sin(a0) * r, cz); put(cx + Math.cos(a1) * r, cy + Math.sin(a1) * r, cz);
      }
    }
    for (const [a, b, c] of MASK_FANS) {
      put(j[3 * a], j[3 * a + 1], j[3 * a + 2]); put(j[3 * b], j[3 * b + 1], j[3 * b + 2]); put(j[3 * c], j[3 * c + 1], j[3 * c + 2]);
    }
    // wrist / distal forearm: continue from the wrist away from the middle knuckle
    {
      const wx = j[0], wy = j[1], wz = j[2];
      let dx = wx - j[27], dy = wy - j[28], dz = wz - j[29];
      const l = Math.hypot(dx, dy, dz) || 1;
      const len = 7 * s;
      quad(wx, wy, wz, wx + (dx / l) * len, wy + (dy / l) * len, wz + (dz / l) * len, 2.6 * s);
    }
    const attr = this.maskPart.mesh.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.needsUpdate = true;
    this.maskPart.mesh.userData.tracked = true;
  }
}
