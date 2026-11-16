/** Shoulder girdle and upper chest: a rigid block placed by the torso solver, plus the two upper arm bones. */
import * as THREE from "three";
import type { Pipeline, Part } from "./pipeline";
import { makeMaskMaterial, makeSolidMaterial, makeTauMaterial, type Channel } from "./materials";
import type { WarpUniforms } from "./warp";
import type { TorsoPose } from "../fit/torso";
import { similarityToMat4 } from "../math/similarity";

interface Style { channel: Channel; shell: number; vol: number; color: number; solid?: boolean; maskOnly?: boolean }

function styleFor(name: string): Style | null {
  if (name === "skin_torso") return { channel: "soft", shell: 0.012, vol: 0, color: 0, solid: false, maskOnly: true };
  // cartilage barely shows on a radiograph
  if (name === "costal_cartilage") return { channel: "soft", shell: 0.012, vol: 0.03, color: 0xcfd6d2 };
  if (name.startsWith("ribs")) return { channel: "bone", shell: 0.03, vol: 0.13, color: 0xe6dcc3 };
  if (/^(clavicle|scapula|humerus|sternum|thoracic_spine)/.test(name)) return { channel: "bone", shell: 0.03, vol: 0.12, color: 0xe6dcc3 };
  return null;
}

export class TorsoRig {
  private root = new THREE.Group();
  private chest: Part[] = [];
  private arms: { l: Part | null; r: Part | null } = { l: null, r: null };
  private all: Part[] = [];
  private m = new Float64Array(16);
  private bands!: Part;
  private bandPos = new Float32Array(5 * 18);
  private foreL: Part[] = [];
  private foreR: Part[] = [];

  /** @param geos torso meshes already in head-frame units (template pose) */
  constructor(pipeline: Pipeline, warp: WarpUniforms, geos: Map<string, THREE.BufferGeometry>, forearmGeos?: Map<string, THREE.BufferGeometry>) {
    this.root.matrixAutoUpdate = false;
    pipeline.scene.add(this.root);
    {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(this.bandPos, 3).setUsage(THREE.DynamicDrawUsage));
      g.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(this.bandPos.length), 3));
      const mesh = new THREE.Mesh(g);
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      mesh.userData.tracked = false;
      pipeline.scene.add(mesh);
      this.bands = { mesh, layer: "spine", mask: makeMaskMaterial(warp, false, "hand") };
      pipeline.addPart(this.bands);
    }
    // forearms: the hand file's radius and ulna (a right arm, true cm), one pair per side
    for (const left of [true, false]) {
      for (const name of ["radius", "ulna"]) {
        const g = forearmGeos?.get(name);
        if (!g) continue;
        const mesh = new THREE.Mesh(g);
        mesh.name = `forearm:${name}:${left ? "l" : "r"}`;
        mesh.frustumCulled = false;
        mesh.matrixAutoUpdate = false;
        mesh.userData.tracked = false;
        pipeline.scene.add(mesh);
        const part: Part = {
          mesh, layer: "chest",
          tau: makeTauMaterial(warp, false, { channel: "bone", shell: 0.04, vol: 0.16 }),
          solid: makeSolidMaterial(warp, false, "bone"),
        };
        pipeline.addPart(part);
        (left ? this.foreL : this.foreR).push(part);
      }
    }
    for (const [name, g] of geos) {
      const st = styleFor(name);
      if (!st) continue;
      const mesh = new THREE.Mesh(g);
      mesh.name = `torso:${name}`;
      mesh.frustumCulled = false;
      mesh.matrixAutoUpdate = false;
      mesh.userData.tracked = false;
      const isArm = name.startsWith("humerus");
      if (isArm) pipeline.scene.add(mesh); else this.root.add(mesh);
      const part: Part = {
        mesh, layer: "chest",
        // the chest envelope only darkens; drawn as film it would streak across the shoulders
        tau: st.maskOnly ? undefined : makeTauMaterial(warp, false, { channel: st.channel, shell: st.shell, vol: st.vol }),
        solid: st.solid === false ? undefined : makeSolidMaterial(warp, false, name === "costal_cartilage" ? "cartilage" : "bone"),
        // the chest envelope darkens the picture behind the ribs; it is not trimmed by the head's outline mask
        mask: st.maskOnly ? makeMaskMaterial(warp, false, "hand") : undefined,
      };
      pipeline.addPart(part);
      this.all.push(part);
      if (name === "humerus_l") this.arms.l = part;
      else if (name === "humerus_r") this.arms.r = part;
      else this.chest.push(part);
    }
  }

  get available(): boolean { return this.all.length > 0; }

  /**
   * Soft dark bands along the neck and arms (the regions between the head, chest and hands).
   * @param index  0 neck, 1/2 left/right upper arm, 3/4 left/right forearm
   * @param a,b    camera-space ends, or null to remove the band
   * @param radius half-width in cm
   */
  setBand(index: number, a: ArrayLike<number> | null, b?: ArrayLike<number> | null, radius = 5) {
    const o = this.bandPos, base = index * 18;
    if (!a || !b) { o.fill(0, base, base + 18); }
    else {
      // perpendicular to the segment in the image plane (the camera looks down −Z from the origin)
      const ax = a[0] / -a[2], ay = a[1] / -a[2], bx = b[0] / -b[2], by = b[1] / -b[2];
      let dx = bx - ax, dy = by - ay;
      const l = Math.hypot(dx, dy) || 1;
      dx /= l; dy /= l;
      const nx = -dy * radius, ny = dx * radius;
      const put = (i: number, x: number, y: number, z: number) => { o[base + 3 * i] = x; o[base + 3 * i + 1] = y; o[base + 3 * i + 2] = z; };
      put(0, a[0] + nx, a[1] + ny, a[2]); put(1, a[0] - nx, a[1] - ny, a[2]); put(2, b[0] + nx, b[1] + ny, b[2]);
      put(3, b[0] + nx, b[1] + ny, b[2]); put(4, a[0] - nx, a[1] - ny, a[2]); put(5, b[0] - nx, b[1] - ny, b[2]);
    }
    (this.bands.mesh.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    this.bands.mesh.userData.tracked = true;
  }

  /** Place (or hide, with null) one forearm: radius and ulna share the matrix. */
  setForearm(left: boolean, m: ArrayLike<number> | null, depth = 50) {
    for (const p of left ? this.foreL : this.foreR) {
      if (!m) { p.mesh.userData.tracked = false; continue; }
      p.mesh.matrix.fromArray(m as number[]);
      p.mesh.matrixWorld.copy(p.mesh.matrix);
      p.mesh.userData.tracked = true;
      p.tau!.uniforms.uRefDepth.value = depth;
    }
  }

  hide() {
    for (const p of this.all) p.mesh.userData.tracked = false;
    this.setForearm(true, null); this.setForearm(false, null);
    for (let i = 1; i < 5; i++) this.setBand(i, null);
  }

  update(t: TorsoPose) {
    this.root.matrix.fromArray(similarityToMat4(t.pose, this.m) as unknown as number[]);
    this.root.updateMatrixWorld(true);
    const mid = [(t.shoulderL[0] + t.shoulderR[0]) / 2, (t.shoulderL[1] + t.shoulderR[1]) / 2, (t.shoulderL[2] + t.shoulderR[2]) / 2];
    const depth = Math.hypot(mid[0], mid[1], mid[2]);
    for (const p of this.chest) { p.mesh.userData.tracked = true; if (p.tau) p.tau.uniforms.uRefDepth.value = depth; }
    const arm = (p: Part | null, m: Float32Array, s: ArrayLike<number>) => {
      if (!p) return;
      p.mesh.matrix.fromArray(m);
      p.mesh.matrixWorld.copy(p.mesh.matrix);
      p.mesh.userData.tracked = true;
      p.tau!.uniforms.uRefDepth.value = Math.hypot(s[0], s[1], s[2]);
    };
    arm(this.arms.l, t.humerusL, t.shoulderL);
    arm(this.arms.r, t.humerusR, t.shoulderR);
  }
}
