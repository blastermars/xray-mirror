/**
 * GPU side of the per-person warp. One small float texture holds the spline's
 * control points and weights; every anatomy material includes WARP_VERTEX and
 * shares the uniforms below, so skull, jaw, teeth, brain, muscles and skin all
 * deform together in the vertex shader.
 */
import * as THREE from "three";
import type { ThinPlateSpline } from "../fit/tps";

export const MAX_CTRL = 160;

export class WarpUniforms {
  private data = new Float32Array(MAX_CTRL * 2 * 4);
  readonly texture: THREE.DataTexture;
  /** Shared uniform objects: spread into each material's `uniforms`. */
  readonly shared: {
    uTps: { value: THREE.DataTexture };
    uTpsN: { value: number };
    uTpsAffine: { value: THREE.Matrix4 };
    uJaw: { value: THREE.Matrix4 };
  };

  constructor() {
    this.texture = new THREE.DataTexture(this.data, MAX_CTRL, 2, THREE.RGBAFormat, THREE.FloatType);
    this.texture.minFilter = this.texture.magFilter = THREE.NearestFilter;
    this.texture.generateMipmaps = false;
    this.texture.needsUpdate = true;
    this.shared = {
      uTps: { value: this.texture },
      uTpsN: { value: 0 },
      uTpsAffine: { value: new THREE.Matrix4() },
      uJaw: { value: new THREE.Matrix4() },
    };
  }

  setSpline(tps: ThinPlateSpline) {
    if (tps.n > MAX_CTRL) throw new Error(`too many spline controls: ${tps.n} > ${MAX_CTRL}`);
    const d = this.data;
    for (let i = 0; i < tps.n; i++) {
      d[4 * i] = tps.controls[3 * i]; d[4 * i + 1] = tps.controls[3 * i + 1]; d[4 * i + 2] = tps.controls[3 * i + 2];
      const o = 4 * (MAX_CTRL + i);
      d[o] = tps.weights[3 * i]; d[o + 1] = tps.weights[3 * i + 1]; d[o + 2] = tps.weights[3 * i + 2];
    }
    this.texture.needsUpdate = true;
    this.shared.uTpsN.value = tps.n;
    const A = tps.affine; // rows: translation, x-coef, y-coef, z-coef
    this.shared.uTpsAffine.value.set(
      A[3], A[6], A[9], A[0],
      A[4], A[7], A[10], A[1],
      A[5], A[8], A[11], A[2],
      0, 0, 0, 1,
    );
  }

  setJaw(m: ArrayLike<number>) {
    this.shared.uJaw.value.fromArray(m as number[]);
  }
}

/**
 * Vertex-shader chunk. Call `warpVertex(position, normal, P, N)`.
 *   uWarp  1 = apply the spline (head anatomy), 0 = leave as is (hand, chest and arm bones)
 *   jawW   per-vertex weight of the second transform `uJaw` (the jaw; for limb nerves, the next segment)
 *   uPart  rigid per-part transform in the head frame (neck vertebrae)
 */
export const WARP_VERTEX = /* glsl */ `
uniform highp sampler2D uTps;
uniform int uTpsN;
uniform mat4 uTpsAffine;
uniform mat4 uJaw;
uniform mat4 uPart;
uniform float uWarp;
attribute float jawW;

void warpVertex(in vec3 p, in vec3 n, out vec3 P, out vec3 N) {
  vec3 q = p;
  vec3 nn = n;
  if (uWarp > 0.5) {
    q = (uTpsAffine * vec4(p, 1.0)).xyz;
    mat3 J = mat3(uTpsAffine);
    for (int i = 0; i < ${MAX_CTRL}; i++) {
      if (i >= uTpsN) break;
      vec3 c = texelFetch(uTps, ivec2(i, 0), 0).xyz;
      vec3 w = texelFetch(uTps, ivec2(i, 1), 0).xyz;
      vec3 d = p - c;
      float r = length(d);
      q += w * r;
      J += outerProduct(w, d / max(r, 1e-4));
    }
    nn = normalize(transpose(inverse(J)) * n);
  }
  // second transform blended in per vertex: the jaw on the head, the next limb segment on nerves
  if (jawW > 0.0) {
    vec3 qj = (uJaw * vec4(q, 1.0)).xyz;
    vec3 nj = mat3(uJaw) * nn;
    q = mix(q, qj, jawW);
    nn = normalize(mix(nn, nj, jawW));
  }
  P = (uPart * vec4(q, 1.0)).xyz;
  N = mat3(uPart) * nn;
}
`;
