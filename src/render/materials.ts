/**
 * Anatomy materials.
 *
 * Radiograph ("tau") material: writes optical depth, not colour. Fragments are
 * blended additively with no depth test, so everything along a ray adds up the
 * way it does on film. Two terms per surface crossing:
 *   shell  — cortical wall: μ·t / |cos θ|  (a thin shell seen edge-on is a long
 *            path, which is physically why bone edges are bright),
 *   volume — signed distance along the ray, − on entry and + on exit, which
 *            sums to the thickness of a closed mesh.
 * Channels of the accumulation target: R bone + teeth, G soft tissue,
 * B brain / eyes, A muscles.
 */
import * as THREE from "three";
import { WARP_VERTEX, type WarpUniforms } from "./warp";

export type Channel = "bone" | "soft" | "brain" | "muscle";
const CHANNEL_VEC: Record<Channel, THREE.Vector4> = {
  bone: new THREE.Vector4(1, 0, 0, 0),
  soft: new THREE.Vector4(0, 1, 0, 0),
  brain: new THREE.Vector4(0, 0, 1, 0),
  muscle: new THREE.Vector4(0, 0, 0, 1),
};

const VERT = /* glsl */ `
${WARP_VERTEX}
varying vec3 vView;
varying vec3 vN;
varying vec3 vLocal;
varying vec3 vRest;
void main() {
  vec3 P; vec3 N;
  warpVertex(position, normal, P, N);
  vLocal = P;
  vRest = position; // un-warped: surface patterns stay glued to the anatomy
  vec4 mv = modelViewMatrix * vec4(P, 1.0);
  vView = mv.xyz;
  vN = normalMatrix * N;
  gl_Position = projectionMatrix * mv;
}
`;

const FADE = /* glsl */ `
uniform vec4 uFadePlane;  // xyz normal, w offset, in the mesh's local (warped) space
uniform float uFadeLen;   // 0 = no fade
float fadeFactor() {
  if (uFadeLen <= 0.0) return 1.0;
  return smoothstep(0.0, uFadeLen, dot(uFadePlane.xyz, vLocal) - uFadePlane.w);
}
`;

const TAU_FRAG = /* glsl */ `
uniform float uShell;
uniform float uVol;
uniform float uVolSign;
uniform float uRefDepth;
uniform float uGain;
uniform vec4 uChan;
varying vec3 vView;
varying vec3 vN;
varying vec3 vLocal;
${FADE}
void main() {
  float c = abs(dot(normalize(vN), normalize(vView)));
  float shell = uShell / max(c, 0.14);
  float d = length(vView) - uRefDepth;
  float vol = uVol * uVolSign * (gl_FrontFacing ? -d : d);
  gl_FragColor = uChan * ((shell + vol) * uGain * fadeFactor());
}
`;

const MASK_FRAG = /* glsl */ `
uniform vec4 uChan;   // red = head, green = hands
varying vec3 vLocal;
${FADE}
void main() {
  gl_FragColor = uChan * fadeFactor();
}
`;

const SOLID_FRAG = /* glsl */ `
uniform vec3 uColor;     // base tint (sRGB)
uniform vec4 uCut;       // discard where dot(xyz, local) > w   (w > 1e5 disables)
uniform float uGloss;
uniform float uKind;     // 0 bone, 1 tooth, 2 muscle, 3 brain, 4 eye, 5 cartilage, 6 nerve
uniform vec3 uFiber;     // muscle: fibre direction of the left half (rest space)
uniform vec3 uCentre;    // muscle: centre of the left half; eye: centre of the globe
uniform float uHalfLen;  // muscle: half length along the fibres
uniform float uPatternScale; // rest-space units per centimetre (meshes come in different units)
varying vec3 vView;
varying vec3 vN;
varying vec3 vLocal;
varying vec3 vRest;
${FADE}

float hash3(vec3 p) {
  p = fract(p * 0.3183099 + 0.1);
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash3(i), hash3(i + vec3(1, 0, 0)), f.x), mix(hash3(i + vec3(0, 1, 0)), hash3(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash3(i + vec3(0, 0, 1)), hash3(i + vec3(1, 0, 1)), f.x), mix(hash3(i + vec3(0, 1, 1)), hash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
  float a = 0.5, s = 0.0;
  for (int i = 0; i < 3; i++) { s += a * vnoise(p); p = p * 2.03 + 11.7; a *= 0.5; }
  return s;
}
vec3 toLinear(vec3 c) { return pow(c, vec3(2.2)); }

// surface bump from a height value, without tangents (Mikkelsen)
vec3 bump(vec3 n, float h, float k) {
  vec3 dp1 = dFdx(vView), dp2 = dFdy(vView);
  float dh1 = dFdx(h), dh2 = dFdy(h);
  vec3 r1 = cross(dp2, n), r2 = cross(n, dp1);
  float det = dot(dp1, r1);
  vec3 grad = sign(det) * (dh1 * r1 + dh2 * r2);
  return normalize(abs(det) * n - k * grad);
}

void main() {
  if (dot(uCut.xyz, vLocal) > uCut.w) discard;
  vec3 n = normalize(vN);
  vec3 v = normalize(-vView);
  bool inside = dot(n, v) < 0.0;   // looking at the inner side (a cut-open skull)
  if (inside) n = -n;
  vec3 p = vRest / uPatternScale;  // centimetres

  vec3 albedo;                      // linear
  float rough = 0.6, spec = 0.04, sss = 0.0, height = 0.0, bumpK = 0.0;
  vec3 sssTint = vec3(1.0, 0.3, 0.2);

  if (uKind < 0.5) {
    // bone: ivory with warmer and greyer patches, fine pores, slightly darker in the grain
    float tone = fbm(p * 0.33);
    float stain = fbm(p * 0.11 + 5.0);
    float pores = fbm(p * 4.2);
    albedo = toLinear(mix(vec3(0.80, 0.72, 0.56), vec3(0.93, 0.89, 0.79), tone));
    albedo = mix(albedo, toLinear(vec3(0.66, 0.55, 0.40)), smoothstep(0.55, 0.8, stain) * 0.35);
    albedo *= 0.91 + 0.12 * pores;
    height = pores * 0.5 + tone * 0.5; bumpK = 0.035;
    rough = 0.62; spec = 0.045;
    sss = 0.08; sssTint = vec3(1.0, 0.75, 0.5);
    // the spongy inner side, only where the skull has actually been opened: some source bones are
    // wound inside-out, and their outer faces must not be mistaken for the inside
    if (inside && uCut.w < 1.0e5) { albedo *= vec3(0.62, 0.48, 0.40); rough = 0.85; }
  } else if (uKind < 1.5) {
    // tooth: enamel, a little warmer toward the body of the tooth
    float t = fbm(p * 1.3);
    albedo = toLinear(mix(vec3(0.90, 0.86, 0.74), vec3(0.97, 0.96, 0.92), t));
    rough = 0.22; spec = 0.07;
    sss = 0.15; sssTint = vec3(0.8, 0.9, 1.0);
  } else if (uKind < 2.5) {
    // muscle: fibres along the muscle, wet surface, pale tendon toward the ends.
    // Left and right muscles share a mesh: mirror the fibre frame for the right side.
    vec3 f = uFiber, c = uCentre;
    if (vRest.x < 0.0) { f.x = -f.x; c.x = -c.x; }
    float along = dot(vRest - c, f);
    vec3 across = (vRest - f * dot(vRest, f)) / uPatternScale;
    float strand = fbm(across * 7.5 + f * (along / uPatternScale) * 0.35);
    float fine = vnoise(across * 26.0);
    float fibre = 0.65 * strand + 0.35 * fine;
    albedo = toLinear(mix(vec3(0.30, 0.06, 0.05), vec3(0.60, 0.19, 0.15), fibre));
    float tendon = smoothstep(0.74, 1.02, abs(along) / max(uHalfLen, 1e-3));
    albedo = mix(albedo, toLinear(vec3(0.84, 0.78, 0.68)), tendon * 0.8);
    height = fibre; bumpK = 0.05;
    rough = mix(0.36, 0.5, tendon); spec = 0.06;
    sss = 0.22; sssTint = vec3(1.0, 0.25, 0.15);
  } else if (uKind < 3.5) {
    // brain: grey-pink cortex with fine surface vessels, glistening
    float mott = fbm(p * 1.1);
    albedo = toLinear(mix(vec3(0.74, 0.58, 0.55), vec3(0.88, 0.74, 0.70), mott));
    float ridge = 1.0 - abs(2.0 * vnoise(p * 1.6 + 3.0) - 1.0);
    float ridge2 = 1.0 - abs(2.0 * vnoise(p * 3.7 + 9.0) - 1.0);
    float vessel = max(smoothstep(0.93, 0.985, ridge), 0.6 * smoothstep(0.95, 0.99, ridge2));
    albedo = mix(albedo, toLinear(vec3(0.42, 0.09, 0.10)), vessel * 0.7);
    height = mott; bumpK = 0.02;
    rough = 0.3; spec = 0.06;
    sss = 0.3; sssTint = vec3(1.0, 0.45, 0.4);
  } else if (uKind < 4.5) {
    // eye: white with fine vessels, iris and pupil looking straight ahead (+Z of the head)
    vec3 d = normalize(vRest - uCentre);
    float ang = acos(clamp(d.z, -1.0, 1.0));
    float az = atan(d.y, d.x);
    float veins = 1.0 - abs(2.0 * vnoise(d * 9.0) - 1.0);
    albedo = toLinear(vec3(0.93, 0.91, 0.88));
    albedo = mix(albedo, toLinear(vec3(0.70, 0.22, 0.20)), smoothstep(0.9, 0.99, veins) * smoothstep(0.55, 1.3, ang) * 0.5);
    float streak = vnoise(vec3(az * 14.0, ang * 30.0, 1.0));
    vec3 iris = toLinear(mix(vec3(0.20, 0.11, 0.05), vec3(0.50, 0.31, 0.14), streak));
    iris *= 0.55 + 0.45 * smoothstep(0.16, 0.42, ang);           // darker toward the pupil
    albedo = mix(albedo, iris, 1.0 - smoothstep(0.44, 0.50, ang));
    albedo = mix(albedo, vec3(0.012), smoothstep(0.50, 0.44, ang) * smoothstep(0.40, 0.47, ang) * 0.6); // limbal ring
    albedo = mix(albedo, vec3(0.004), 1.0 - smoothstep(0.15, 0.19, ang));  // pupil
    rough = 0.08; spec = 0.08;
  } else if (uKind > 5.5) {
    // nerve: pale yellow cords with a faint lengthwise grain
    float grain = fbm(p * vec3(5.0, 1.2, 5.0));
    albedo = toLinear(mix(vec3(0.86, 0.74, 0.34), vec3(0.97, 0.90, 0.56), grain));
    rough = 0.4; spec = 0.05;
    sss = 0.3; sssTint = vec3(1.0, 0.85, 0.4);
  } else {
    // cartilage: pale, slightly blue, smooth
    albedo = toLinear(mix(vec3(0.78, 0.83, 0.84), vec3(0.90, 0.93, 0.92), fbm(p * 0.8)));
    rough = 0.35; spec = 0.05;
    sss = 0.3; sssTint = vec3(0.7, 0.9, 1.0);
  }
  albedo *= toLinear(uColor);
  rough = clamp(rough * (1.0 - 0.5 * uGloss), 0.05, 1.0);
  if (bumpK > 0.0) n = bump(n, height, bumpK);

  // studio lighting in camera space: warm key from the upper left, cool fill, rim from behind
  vec3 L1 = normalize(vec3(-0.45, 0.62, 0.64));
  vec3 L2 = normalize(vec3(0.70, 0.10, 0.55));
  vec3 L3 = normalize(vec3(0.25, 0.55, -0.80));
  vec3 C1 = vec3(1.00, 0.95, 0.88) * 2.6, C2 = vec3(0.78, 0.86, 1.00) * 0.75, C3 = vec3(0.9, 0.95, 1.0) * 1.1;
  float ndv = max(dot(n, v), 1e-3);
  float wrap = 0.25 + sss * 0.5;                                  // light bleeds round soft tissue
  vec3 diff = C1 * max((dot(n, L1) + wrap) / (1.0 + wrap), 0.0)
            + C2 * max((dot(n, L2) + wrap) / (1.0 + wrap), 0.0)
            + C3 * max(dot(n, L3), 0.0) * 0.6;
  vec3 ambient = mix(vec3(0.17, 0.14, 0.12), vec3(0.46, 0.46, 0.47), n.y * 0.5 + 0.5);
  vec3 col = albedo * (diff + ambient) / 3.14159 * 1.9;

  // specular: Blinn-Phong shaped by roughness, plus a soft-box reflection for wet surfaces
  float shin = 2.0 / (rough * rough * rough * rough + 1e-3) - 2.0;
  float fres = spec + (1.0 - spec) * pow(1.0 - ndv, 5.0);
  vec3 h1 = normalize(L1 + v), h2 = normalize(L2 + v);
  float norm = (shin + 8.0) / 25.13;
  vec3 sp = C1 * pow(max(dot(n, h1), 0.0), shin) * max(dot(n, L1), 0.0)
          + C2 * pow(max(dot(n, h2), 0.0), shin) * max(dot(n, L2), 0.0);
  vec3 r = reflect(-v, n);
  float box = smoothstep(0.80, 0.97, dot(r, L1)) + 0.35 * smoothstep(0.1, 0.9, r.y);
  col += fres * (sp * norm * 0.25 + vec3(1.0, 0.97, 0.92) * box * (1.0 - rough) * 0.9);

  // light scattered back out of translucent tissue at grazing angles
  col += sssTint * albedo * sss * pow(1.0 - ndv, 2.0) * 1.4;

  // filmic tone curve, then display encoding
  col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
  col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
  float a = fadeFactor();
  gl_FragColor = vec4(col * a, a);
}
`;

export type Tissue = "bone" | "tooth" | "muscle" | "brain" | "eye" | "cartilage" | "nerve";
const TISSUE_ID: Record<Tissue, number> = { bone: 0, tooth: 1, muscle: 2, brain: 3, eye: 4, cartilage: 5, nerve: 6 };

export interface TauParams {
  channel: Channel;
  /** cortical-wall term per surface crossing (already μ × wall thickness) */
  shell: number;
  /** attenuation per cm of interior */
  vol: number;
  /** −1 when the mesh is wound inside-out */
  volSign?: number;
}

function baseUniforms(warp: WarpUniforms, warped: boolean) {
  return {
    ...warp.shared,
    uWarp: { value: warped ? 1 : 0 },
    uPart: { value: new THREE.Matrix4() },
    uFadePlane: { value: new THREE.Vector4(0, 1, 0, 0) },
    uFadeLen: { value: 0 },
  };
}

export function makeTauMaterial(warp: WarpUniforms, warped: boolean, p: TauParams): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    name: "tau",
    vertexShader: VERT,
    fragmentShader: TAU_FRAG,
    uniforms: {
      ...baseUniforms(warp, warped),
      uShell: { value: p.shell },
      uVol: { value: p.vol },
      uVolSign: { value: p.volSign ?? 1 },
      uRefDepth: { value: 50 },
      uGain: { value: 1 },
      uChan: { value: CHANNEL_VEC[p.channel] },
    },
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
    blendSrcAlpha: THREE.OneFactor,
    blendDstAlpha: THREE.OneFactor,
  });
}

export function makeMaskMaterial(warp: WarpUniforms, warped: boolean, region: "head" | "hand"): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    name: "mask",
    vertexShader: VERT,
    fragmentShader: MASK_FRAG,
    uniforms: {
      ...baseUniforms(warp, warped),
      uChan: { value: region === "head" ? new THREE.Vector4(1, 0, 0, 1) : new THREE.Vector4(0, 1, 0, 1) },
    },
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    blending: THREE.CustomBlending,
    blendEquation: THREE.MaxEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
    blendEquationAlpha: THREE.MaxEquation,
    blendSrcAlpha: THREE.OneFactor,
    blendDstAlpha: THREE.OneFactor,
  });
}

/**
 * Shaded tissue for the Anatomy view.
 * @param tint         multiplies the tissue's own colouring (0xffffff = as is)
 * @param patternScale rest-space units per centimetre of the mesh (1 for head-frame meshes)
 */
export function makeSolidMaterial(
  warp: WarpUniforms, warped: boolean, tissue: Tissue = "bone", tint = 0xffffff, gloss = 0, patternScale = 1,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    name: "solid",
    vertexShader: VERT,
    fragmentShader: SOLID_FRAG,
    uniforms: {
      ...baseUniforms(warp, warped),
      uColor: { value: new THREE.Color(tint) },
      uCut: { value: new THREE.Vector4(0, 1, 0, 1e6) },
      uGloss: { value: gloss },
      uKind: { value: TISSUE_ID[tissue] },
      uFiber: { value: new THREE.Vector3(0, 1, 0) },
      uCentre: { value: new THREE.Vector3() },
      uHalfLen: { value: 1 },
      uPatternScale: { value: patternScale },
    },
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
    transparent: false,
  });
}
