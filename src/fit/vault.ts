/**
 * Fits the cranial vault to the person's real outline.
 *
 * Face landmarks stop at the forehead and in front of the ears, so the top and
 * sides of the skull are otherwise an extrapolation. A person/hair segmentation
 * mask shows where the head really ends: rays are cast from the middle of the
 * head upward and sideways, and the distance to the mask's edge is compared
 * with the distance to the edge of the fitted model's own skin outline.
 *
 * Where the edge is bare skin the two must agree. Where it is hair the skull is
 * somewhere underneath, so the outline only acts as an upper bound.
 */

export const SEG_BACKGROUND = 0;
export const SEG_HAIR = 1;

/** Ray directions in degrees from "up" (positive = toward image right). */
export const RAY_ANGLES = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
const UP_LIMIT = 35; // |angle| ≤ this → counts toward the height of the vault
const SIDE_MIN = 50; // side rays lie between these: above the ears, which differ too much between
const SIDE_MAX = 75; // people (and hide under hair) to say anything about the skull
/**
 * The segmentation's class boundary sits a little inside the true silhouette. Measured on a
 * rendered head of known shape with a 256-pixel crop: about two mask pixels.
 */
const EDGE_BIAS_PX = 2.0;

export interface VaultScale {
  /** multiplies the vault's breadth */
  side: number;
  /** multiplies the vault's height */
  up: number;
}

export interface Outline {
  /** distance to the edge along each ray, in units of image height; 0 = not found */
  r: Float32Array;
  /** 1 where the edge of the head along that ray is hair */
  hair: Uint8Array;
}

/** Part of the picture a mask covers, in normalised image coordinates (y down). */
export interface Roi { x: number; y: number; w: number; h: number }
export const FULL_FRAME: Roi = { x: 0, y: 0, w: 1, h: 1 };

/**
 * @param mask   class per pixel (row 0 = top), 0 = background, 1 = hair
 * @param roi    the part of the picture the mask covers
 * @param cx,cy  ray origin, normalised image coordinates (y down)
 * @param upx,upy "up" of the head in aspect-corrected image space (y down), unit length
 */
export function traceOutline(
  mask: Uint8Array, w: number, h: number, roi: Roi,
  cx: number, cy: number, upx: number, upy: number, aspect: number,
  maxR = 0.6,
): Outline {
  const n = RAY_ANGLES.length;
  const r = new Float32Array(n), hair = new Uint8Array(n);
  // half a mask pixel, in units of image height
  const step = (0.5 * roi.h) / h;
  const edgeBias = (EDGE_BIAS_PX * roi.h) / h;
  const at = (u: number, v: number): number => {
    const x = u / aspect; // u is aspect-corrected
    if (x < 0 || v < 0 || x >= 1 || v >= 1) return -1; // outside the picture
    const px = Math.floor(((x - roi.x) / roi.w) * w), py = Math.floor(((v - roi.y) / roi.h) * h);
    if (px < 0 || py < 0 || px >= w || py >= h) return -1; // outside the mask
    return mask[py * w + px];
  };
  const u0 = cx * aspect, v0 = cy;
  if (at(u0, v0) <= 0) return { r, hair }; // the origin must be on the person
  for (let j = 0; j < n; j++) {
    const a = (RAY_ANGLES[j] * Math.PI) / 180;
    // rotate "up" by a (clockwise on screen, where y points down)
    const dx = upx * Math.cos(a) - upy * Math.sin(a);
    const dy = upx * Math.sin(a) + upy * Math.cos(a);
    let last = at(u0, v0), found = 0, lastClass = last;
    for (let d = step; d < maxR; d += step) {
      const c = at(u0 + dx * d, v0 + dy * d);
      if (c < 0) { found = 0; break; } // ran off the picture: the edge is not visible
      if (c === SEG_BACKGROUND) {
        // require the background to persist, so a pinhole in the mask is not taken for the edge
        const c2 = at(u0 + dx * (d + 3 * step), v0 + dy * (d + 3 * step));
        if (c2 === SEG_BACKGROUND || c2 < 0) { found = d - step * 0.5 + edgeBias; lastClass = last; break; }
      } else last = c;
    }
    r[j] = found;
    hair[j] = found > 0 && lastClass === SEG_HAIR ? 1 : 0;
  }
  return { r, hair };
}

/**
 * Edge of the model along the same rays: the farthest projected skin point in each direction.
 * @param pts flat [u, v] projected skin points in aspect-corrected image space (y down)
 */
export function modelOutline(pts: ArrayLike<number>, cx: number, cy: number, upx: number, upy: number, aspect: number): Float32Array {
  const n = RAY_ANGLES.length;
  const out = new Float32Array(n);
  const u0 = cx * aspect, v0 = cy;
  const half = (7 * Math.PI) / 180;
  const dirs: number[] = [];
  for (let j = 0; j < n; j++) {
    const a = (RAY_ANGLES[j] * Math.PI) / 180;
    dirs.push(upx * Math.cos(a) - upy * Math.sin(a), upx * Math.sin(a) + upy * Math.cos(a));
  }
  for (let i = 0; i < pts.length; i += 2) {
    const x = pts[i] - u0, y = pts[i + 1] - v0;
    const len = Math.hypot(x, y);
    if (len < 1e-6) continue;
    for (let j = 0; j < n; j++) {
      const along = x * dirs[2 * j] + y * dirs[2 * j + 1];
      if (along <= out[j]) continue;
      // only points close to this ray's direction define its edge
      const cross = Math.abs(x * dirs[2 * j + 1] - y * dirs[2 * j]);
      if (Math.atan2(cross, along) < half) out[j] = along;
    }
  }
  return out;
}

const median = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  return s.length % 2 ? s[(s.length - 1) / 2] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2]);
};

/**
 * One correction step. Returns the updated scale, or null when the outline was not usable.
 * @param gain how much of the measured mismatch to take up this step (0..1)
 */
export function updateVault(current: VaultScale, seen: Outline, model: Float32Array, gain = 0.3): VaultScale | null {
  const up: number[] = [], side: number[] = [];
  let hairUp = 0, hairSide = 0;
  for (let j = 0; j < RAY_ANGLES.length; j++) {
    if (!(seen.r[j] > 0) || !(model[j] > 0.01)) continue;
    const q = seen.r[j] / model[j];
    if (q < 0.6 || q > 1.6) continue; // not the head's edge (a raised hand, a hat brim, a mask glitch)
    const a = Math.abs(RAY_ANGLES[j]);
    if (a <= UP_LIMIT) { up.push(q); hairUp += seen.hair[j]; }
    else if (a >= SIDE_MIN && a <= SIDE_MAX) { side.push(q); hairSide += seen.hair[j]; }
  }
  if (up.length < 3 && side.length < 3) return null;
  const next = { ...current };
  const apply = (key: "up" | "side", ratios: number[], hairCount: number, minCount: number) => {
    if (ratios.length < minCount) return;
    let q = median(ratios);
    // under hair the skull can only be smaller than the outline, never pushed out to it
    if (hairCount * 2 >= ratios.length) q = Math.min(1, q);
    next[key] = Math.min(1.2, Math.max(0.8, current[key] * Math.pow(q, gain)));
  };
  apply("up", up, hairUp, 3);
  apply("side", side, hairSide, 3);
  return next;
}
