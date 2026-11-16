/**
 * Frame pipeline:
 *   1. tau    — anatomy accumulated as optical depth (half-float, additive, MSAA)
 *   2. mask   — the user's own head / hand region (warped skin envelope + hand capsules), blurred
 *   3. solid  — shaded anatomy with depth, for the Anatomy mode
 *   4. glow   — blurred radiograph intensity (unsharp mask + bloom)
 *   5. film   — everything composited over the exact video frame that was analysed
 */
import * as THREE from "three";

export type Mode = "radiograph" | "anatomy";
export type Reveal = "whole" | "split" | "lens";

export interface Part {
  mesh: THREE.Mesh;
  layer: string;
  tau?: THREE.ShaderMaterial;
  mask?: THREE.ShaderMaterial;
  solid?: THREE.ShaderMaterial;
}

export interface FrameSettings {
  mode: Mode;
  exposure: number;
  /** 0 = film black inside the head, 1 = untouched video */
  real: number;
  grain: number;
  reveal: Reveal;
  /** lens centre in image uv (0..1, y up) and radius as a fraction of image height */
  lens: { x: number; y: number; r: number };
  /** split line: a point (uv) and a direction normal (aspect-corrected space) */
  split: { x: number; y: number; nx: number; ny: number };
  layers: Record<string, boolean>;
  time: number;
  /** zoomed view: centre in the picture (uv, y up) and factor (1 = whole picture) */
  view?: { x: number; y: number; zoom: number };
}

const FS_VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const BLUR_FRAG = /* glsl */ `
uniform sampler2D uTex;
uniform vec2 uStep;
varying vec2 vUv;
void main() {
  vec4 s = texture2D(uTex, vUv) * 0.227027;
  s += (texture2D(uTex, vUv + uStep * 1.3846) + texture2D(uTex, vUv - uStep * 1.3846)) * 0.316216;
  s += (texture2D(uTex, vUv + uStep * 3.2308) + texture2D(uTex, vUv - uStep * 3.2308)) * 0.070270;
  gl_FragColor = s;
}
`;

const INTENSITY_FRAG = /* glsl */ `
uniform sampler2D uTau;
uniform float uExposure;
varying vec2 vUv;
void main() {
  vec4 t = max(texture2D(uTau, vUv), 0.0);
  float I = 1.0 - exp(-(t.r + t.g) * uExposure);
  gl_FragColor = vec4(I, I, I, 1.0);
}
`;

const FILM_FRAG = /* glsl */ `
uniform sampler2D uVideo;
uniform sampler2D uTau;
uniform sampler2D uGlow;
uniform sampler2D uMask;
uniform sampler2D uSolid;
uniform sampler2D uSolidDepth;
uniform vec3 uProj;       // near, far, 1 / (2·tan(fov/2))
uniform vec3 uView;       // centre of the view in the picture (uv, y up) and zoom factor
uniform sampler2D uSeg;
uniform float uHasSeg;
uniform vec4 uSegRect;   // x, y, w, h of the mask in the picture (normalised, y down)
uniform float uHasVideo;
uniform float uMode;      // 0 radiograph, 1 anatomy
uniform float uExposure;
uniform float uReal;
uniform float uGrain;
uniform float uTime;
uniform float uAspect;
uniform float uReveal;    // 0 whole, 1 split, 2 lens
uniform vec3 uLens;
uniform vec4 uSplit;
varying vec2 vUv;

const vec3 FILM_BLACK = vec3(0.012, 0.018, 0.030);
const vec3 FILM_WHITE = vec3(0.86, 0.93, 1.00);

float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

/** Distance in front of the camera (cm) of the shaded anatomy at uv; far away where there is none. */
float viewDepth(vec2 uv) {
  float d = texture2D(uSolidDepth, uv).r * 2.0 - 1.0;
  return 2.0 * uProj.x * uProj.y / (uProj.y + uProj.x - d * (uProj.y - uProj.x));
}

float revealMask(vec2 uv) {
  if (uReveal < 0.5) return 1.0;
  if (uReveal < 1.5) {
    float d = dot((uv - uSplit.xy) * vec2(uAspect, 1.0), uSplit.zw);
    return smoothstep(-0.004, 0.004, d);
  }
  float r = length((uv - uLens.xy) * vec2(uAspect, 1.0));
  return 1.0 - smoothstep(uLens.z * 0.86, uLens.z, r);
}

void main() {
  // the view can be zoomed: screen position → position in the camera picture
  vec2 uv = uView.xy + (vUv - 0.5) / uView.z;
  vec3 vid = uHasVideo > 0.5 ? texture2D(uVideo, uv).rgb : vec3(0.05, 0.06, 0.08);
  float reveal = revealMask(uv);

  vec4 mk = texture2D(uMask, uv);
  float head = mk.r, hand = mk.g;
  if (uHasSeg > 0.5) {
    // The person mask only covers a square around the head. It trims the head region (never the
    // hands), and its influence fades out toward the square's edges so no seam shows.
    vec2 suv = (vec2(uv.x, 1.0 - uv.y) - uSegRect.xy) / uSegRect.zw;
    if (suv.x > 0.0 && suv.y > 0.0 && suv.x < 1.0 && suv.y < 1.0) {
      float edge = smoothstep(0.0, 0.12, min(min(suv.x, 1.0 - suv.x), min(suv.y, 1.0 - suv.y)));
      head *= mix(1.0, smoothstep(0.2, 0.55, texture2D(uSeg, suv).r), edge);
    }
  }
  // In the Anatomy view the bones are solid, so the hands, arms and chest are only shaded lightly
  // (your own skin stays visible around the bones); the face is still hidden behind the skull.
  if (uMode > 0.5) hand *= 0.45;
  float m = max(head, hand) * reveal;

  // the person's own region turns into unexposed film
  float lum = dot(vid, vec3(0.299, 0.587, 0.114));
  vec3 base = mix(vid, FILM_BLACK + vec3(0.07, 0.08, 0.10) * lum, m * (1.0 - uReal));

  vec3 col;
  if (uMode < 0.5) {
    vec4 t = max(texture2D(uTau, uv), 0.0);
    // soft tissue: a faint veil over the whole head, plus the skin line from the envelope
    float I = 1.0 - exp(-(t.r + t.g) * uExposure - 0.07 * max(head, hand));
    float Ig = texture2D(uGlow, uv).r;
    I = clamp(I + 0.55 * (I - Ig), 0.0, 1.0);   // unsharp mask: the crisp edges of a radiograph
    I = pow(I, 1.12);                           // film response: keep the mid-greys open
    I = clamp(I + 0.10 * Ig, 0.0, 1.0);         // slight bloom
    vec3 xr = mix(FILM_BLACK, FILM_WHITE, I) - FILM_BLACK;
    // brain / eyes and muscles are not visible on a plain film: shown as tinted layers that
    // stay below the bone's brightness so they never wash it out
    vec3 over = vec3(0.30, 0.74, 1.00) * min(0.6, 1.0 - exp(-t.b * uExposure))
              + vec3(1.00, 0.36, 0.28) * min(0.55, 1.0 - exp(-t.a * uExposure));
    col = base + (xr + over * (1.0 - 0.55 * I)) * reveal;
    float g = (hash(gl_FragCoord.xy + fract(uTime) * 311.7) - 0.5) * uGrain;
    col += g * (0.25 + 0.75 * I) * max(m, min(1.0, I * 3.0));
  } else {
    vec4 s = texture2D(uSolid, uv);
    // contact shading: how much nearer geometry crowds this point, from the depth buffer
    float ao = 1.0;
    if (s.a > 0.01) {
      float z = viewDepth(uv);
      float radius = 0.9 * uProj.z / z;                 // ~0.9 cm on the anatomy, in units of image height
      float spin = hash(gl_FragCoord.xy) * 6.2832;
      float occ = 0.0;
      for (int i = 0; i < 10; i++) {
        float fi = float(i) + 0.5;
        float r = sqrt(fi / 10.0) * radius;
        float a = fi * 2.39996 + spin;                  // golden-angle spiral
        vec2 o = vec2(cos(a) / uAspect, sin(a)) * r;
        float d = z - viewDepth(uv + o);                // > 0: the neighbour is nearer the camera
        occ += smoothstep(0.08, 0.5, d) * (1.0 - smoothstep(1.5, 4.0, d));
      }
      ao = 1.0 - 0.85 * occ / 10.0;
      ao *= ao;
    }
    col = base * (1.0 - s.a * reveal) + s.rgb * ao * reveal;
  }

  // lens rim
  if (uReveal > 1.5) {
    float r = length((uv - uLens.xy) * vec2(uAspect, 1.0));
    col += vec3(0.5, 0.8, 1.0) * 0.35 * (1.0 - smoothstep(0.0, 0.006, abs(r - uLens.z)));
  }
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;

class FullScreen {
  private scene = new THREE.Scene();
  private cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private quad: THREE.Mesh;
  constructor() {
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quad.frustumCulled = false;
    this.scene.add(this.quad);
  }
  render(r: THREE.WebGLRenderer, mat: THREE.Material, target: THREE.WebGLRenderTarget | null) {
    this.quad.material = mat;
    r.setRenderTarget(target);
    r.render(this.scene, this.cam);
  }
}

function fsMaterial(frag: string, uniforms: Record<string, THREE.IUniform>) {
  return new THREE.ShaderMaterial({ vertexShader: FS_VERT, fragmentShader: frag, uniforms, depthTest: false, depthWrite: false });
}

export class Pipeline {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera = new THREE.PerspectiveCamera(45, 16 / 9, 1, 2000);
  readonly scene = new THREE.Scene();
  readonly parts: Part[] = [];
  width = 2;
  height = 2;

  private fs = new FullScreen();
  private tauRT!: THREE.WebGLRenderTarget;
  private solidRT!: THREE.WebGLRenderTarget;
  private maskRT!: THREE.WebGLRenderTarget;
  private maskTmp!: THREE.WebGLRenderTarget;
  private glowRT!: THREE.WebGLRenderTarget;
  private glowTmp!: THREE.WebGLRenderTarget;
  private blurMat = fsMaterial(BLUR_FRAG, { uTex: { value: null }, uStep: { value: new THREE.Vector2() } });
  private intensityMat = fsMaterial(INTENSITY_FRAG, { uTau: { value: null }, uExposure: { value: 1 } });
  private filmMat: THREE.ShaderMaterial;
  private videoTex: THREE.Texture | null = null;
  private segTex: THREE.DataTexture | null = null;
  private black = new THREE.Color(0, 0, 0);

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: "high-performance", preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(1);
    this.renderer.autoClear = false;
    // display-referred pipeline: video pixels and film colours pass through untouched
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.scene.matrixAutoUpdate = false;
    this.filmMat = fsMaterial(FILM_FRAG, {
      uVideo: { value: null }, uTau: { value: null }, uGlow: { value: null }, uMask: { value: null },
      uSolid: { value: null }, uSolidDepth: { value: null }, uProj: { value: new THREE.Vector3(1, 2000, 1) }, uView: { value: new THREE.Vector3(0.5, 0.5, 1) }, uSeg: { value: null }, uHasSeg: { value: 0 }, uSegRect: { value: new THREE.Vector4(0, 0, 1, 1) }, uHasVideo: { value: 0 },
      uMode: { value: 0 }, uExposure: { value: 1 }, uReal: { value: 0.1 }, uGrain: { value: 0.06 },
      uTime: { value: 0 }, uAspect: { value: 1 }, uReveal: { value: 0 },
      uLens: { value: new THREE.Vector3(0.5, 0.5, 0.2) }, uSplit: { value: new THREE.Vector4(0.5, 0.5, 1, 0) },
    });
    this.setSize(1280, 720);
  }

  private srcW = 0;
  private srcH = 0;
  private renderScale = 1;
  private msaa = 4;

  /**
   * Trade sharpness for speed: `scale` is the fraction of the camera resolution that is drawn,
   * `msaa` the edge smoothing samples (0 = off). Takes effect at the next frame.
   */
  setQuality(scale: number, msaa: number) {
    if (scale === this.renderScale && msaa === this.msaa) return;
    this.renderScale = scale; this.msaa = msaa;
    this.srcW = 0; // force the buffers to be rebuilt
  }

  /** @param w,h size of the camera picture */
  setSize(sw: number, sh: number) {
    if (sw === this.srcW && sh === this.srcH) return;
    this.srcW = sw; this.srcH = sh;
    const w = Math.max(2, Math.round(sw * this.renderScale)), h = Math.max(2, Math.round(sh * this.renderScale));
    this.width = w; this.height = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    for (const rt of [this.tauRT, this.solidRT, this.maskRT, this.maskTmp, this.glowRT, this.glowTmp]) rt?.dispose();
    const lin = { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false };
    this.tauRT = new THREE.WebGLRenderTarget(w, h, { ...lin, type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false, samples: this.msaa });
    // the depth of the shaded anatomy is kept as a texture: the film pass darkens crevices from it
    const depthTexture = new THREE.DepthTexture(w, h);
    depthTexture.minFilter = depthTexture.magFilter = THREE.NearestFilter;
    this.solidRT = new THREE.WebGLRenderTarget(w, h, { ...lin, depthBuffer: true, samples: this.msaa, depthTexture });
    const qw = Math.max(2, w >> 2), qh = Math.max(2, h >> 2);
    this.maskRT = new THREE.WebGLRenderTarget(qw, qh, { ...lin, depthBuffer: false });
    this.maskTmp = new THREE.WebGLRenderTarget(qw, qh, { ...lin, depthBuffer: false });
    const hw = Math.max(2, w >> 1), hh = Math.max(2, h >> 1);
    this.glowRT = new THREE.WebGLRenderTarget(hw, hh, { ...lin, depthBuffer: false });
    this.glowTmp = new THREE.WebGLRenderTarget(hw, hh, { ...lin, depthBuffer: false });
  }

  setFov(vfovDeg: number) {
    if (this.camera.fov !== vfovDeg) { this.camera.fov = vfovDeg; this.camera.updateProjectionMatrix(); }
  }

  /** The frame source. Call `uploadVideo()` right after analysing a frame so overlay and image stay in sync. */
  setVideo(el: TexImageSource | null) {
    this.videoTex?.dispose();
    if (!el) { this.videoTex = null; return; }
    // A <video> must go through VideoTexture: a plain Texture sizes itself from the element's
    // width/height attributes, which are 0 for a video, and uploads nothing.
    const isVideo = typeof HTMLVideoElement !== "undefined" && el instanceof HTMLVideoElement;
    const t = isVideo ? new THREE.VideoTexture(el) : new THREE.Texture(el as HTMLCanvasElement);
    t.minFilter = t.magFilter = THREE.LinearFilter;
    t.generateMipmaps = false;
    t.colorSpace = THREE.NoColorSpace;
    this.videoTex = t;
  }
  uploadVideo() { if (this.videoTex) this.videoTex.needsUpdate = true; }

  /**
   * Person mask (row 0 = top), 0..255, covering the rectangle `roi` of the picture
   * (normalised, y down). Pass null to stop trimming.
   */
  setSegmentation(data: Uint8Array | null, w = 0, h = 0, roi = { x: 0, y: 0, w: 1, h: 1 }) {
    if (!data) { this.filmMat.uniforms.uHasSeg.value = 0; return; }
    if (!this.segTex || this.segTex.image.width !== w || this.segTex.image.height !== h) {
      this.segTex?.dispose();
      this.segTex = new THREE.DataTexture(new Uint8Array(w * h), w, h, THREE.RedFormat, THREE.UnsignedByteType);
      this.segTex.minFilter = this.segTex.magFilter = THREE.LinearFilter;
    }
    (this.segTex.image.data as Uint8Array).set(data);
    this.segTex.needsUpdate = true;
    this.filmMat.uniforms.uSeg.value = this.segTex;
    this.filmMat.uniforms.uSegRect.value.set(roi.x, roi.y, roi.w, roi.h);
    this.filmMat.uniforms.uHasSeg.value = 1;
  }

  addPart(p: Part) { this.parts.push(p); }

  private usePass(kind: "tau" | "mask" | "solid", layers: Record<string, boolean>) {
    for (const p of this.parts) {
      const mat = p[kind];
      const on = !!mat && layers[p.layer] !== false && p.mesh.userData.tracked !== false;
      p.mesh.visible = on;
      if (on) p.mesh.material = mat!;
    }
  }

  render(s: FrameSettings) {
    const r = this.renderer;
    const u = this.filmMat.uniforms;
    r.setClearColor(this.black, 0);

    // mask: always the skin envelope and hand capsules, regardless of layer toggles
    this.usePass("mask", {});
    r.setRenderTarget(this.maskRT); r.clear(); r.render(this.scene, this.camera);
    this.blurMat.uniforms.uTex.value = this.maskRT.texture;
    this.blurMat.uniforms.uStep.value.set(1.6 / this.maskRT.width, 0);
    this.fs.render(r, this.blurMat, this.maskTmp);
    this.blurMat.uniforms.uTex.value = this.maskTmp.texture;
    this.blurMat.uniforms.uStep.value.set(0, 1.6 / this.maskRT.height);
    this.fs.render(r, this.blurMat, this.maskRT);

    if (s.mode === "radiograph") {
      this.usePass("tau", s.layers);
      r.setRenderTarget(this.tauRT); r.clear(); r.render(this.scene, this.camera);
      this.intensityMat.uniforms.uTau.value = this.tauRT.texture;
      this.intensityMat.uniforms.uExposure.value = s.exposure;
      this.fs.render(r, this.intensityMat, this.glowTmp);
      this.blurMat.uniforms.uTex.value = this.glowTmp.texture;
      this.blurMat.uniforms.uStep.value.set(1.2 / this.glowRT.width, 0);
      this.fs.render(r, this.blurMat, this.glowRT);
      this.blurMat.uniforms.uTex.value = this.glowRT.texture;
      this.blurMat.uniforms.uStep.value.set(0, 1.2 / this.glowRT.height);
      this.fs.render(r, this.blurMat, this.glowTmp);
      u.uGlow.value = this.glowTmp.texture;
    } else {
      this.usePass("solid", s.layers);
      r.setRenderTarget(this.solidRT); r.clear(); r.render(this.scene, this.camera);
    }

    u.uVideo.value = this.videoTex;
    u.uHasVideo.value = this.videoTex ? 1 : 0;
    u.uTau.value = this.tauRT.texture;
    u.uMask.value = this.maskRT.texture;
    u.uSolid.value = this.solidRT.texture;
    u.uSolidDepth.value = this.solidRT.depthTexture;
    u.uProj.value.set(this.camera.near, this.camera.far, 1 / (2 * Math.tan((this.camera.fov * Math.PI) / 360)));
    u.uMode.value = s.mode === "radiograph" ? 0 : 1;
    u.uExposure.value = s.exposure;
    u.uReal.value = s.real;
    u.uGrain.value = s.grain;
    u.uTime.value = s.time;
    u.uAspect.value = this.width / this.height;
    u.uReveal.value = s.reveal === "whole" ? 0 : s.reveal === "split" ? 1 : 2;
    u.uLens.value.set(s.lens.x, s.lens.y, s.lens.r);
    u.uSplit.value.set(s.split.x, s.split.y, s.split.nx, s.split.ny);
    u.uView.value.set(s.view?.x ?? 0.5, s.view?.y ?? 0.5, s.view?.zoom ?? 1);
    this.fs.render(r, this.filmMat, null);
  }
}
