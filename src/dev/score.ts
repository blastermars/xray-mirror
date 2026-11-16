/** Scores the app's overlay against the synthetic subject's ground truth. */
import type { XRayMirror } from "../app";
import type { SyntheticSubject } from "./synthetic";

export interface Score {
  /** mean 2D miss of the skull on screen, % of true face width */
  craniumPct: number;
  mandiblePct: number;
  /** same in millimetres at the subject's distance (face width taken as the template's, reshaped) */
  craniumMm: number;
  mandibleMm: number;
  status: XRayMirror["status"];
  truthJawDeg: number;
}

/** Mean miss per region of the skull (% of face width) with the mean offset, to see where and which way it is off. */
export function makeRegionScorer(app: XRayMirror, subject: SyntheticSubject) {
  let pts: Float32Array | null = null;
  return (): Record<string, string> | null => {
    pts ??= app.samplePart("cranium", 1500);
    if (!subject.truth.pose || !app.status.face) return null;
    const aspect = subject.canvas.width / subject.canvas.height;
    const l = subject.truthImagePoint([7, 1, -2]), r = subject.truthImagePoint([-7, 1, -2]);
    const fw = Math.hypot((l[0] - r[0]) * aspect, l[1] - r[1]);
    const reg: Record<string, { n: number; e: number; dx: number; dy: number }> = {};
    const add = (k: string, dx: number, dy: number) => {
      const g = (reg[k] ??= { n: 0, e: 0, dx: 0, dy: 0 });
      g.n++; g.e += Math.hypot(dx, dy); g.dx += dx; g.dy += dy;
    };
    for (let i = 0; i < pts.length; i += 3) {
      const p = [pts[i], pts[i + 1], pts[i + 2]];
      const a = app.imagePoint(p), t = subject.truthImagePoint(p);
      if (!a) continue;
      const dx = ((a[0] - t[0]) * aspect * 100) / fw, dy = ((a[1] - t[1]) * 100) / fw;
      const k = p[1] > 9 ? "top" : p[2] < -9 ? "back" : Math.abs(p[0]) > 5.5 ? (p[0] > 0 ? "sideL" : "sideR") : p[2] > 2 ? (p[1] > 2 ? "faceUpper" : "faceLower") : "inner";
      add(k, dx, dy); add("all", dx, dy);
    }
    const out: Record<string, string> = {};
    for (const k in reg) {
      const g = reg[k];
      out[k] = `n=${g.n} miss=${(g.e / g.n).toFixed(2)}% dx=${(g.dx / g.n).toFixed(2)} dy=${(g.dy / g.n).toFixed(2)}`;
    }
    return out;
  };
}

export function makeScorer(app: XRayMirror, subject: SyntheticSubject) {
  let cran: Float32Array | null = null, mand: Float32Array | null = null;
  return (): Score | null => {
    cran ??= app.samplePart("cranium", 400);
    mand ??= app.samplePart("mandible", 200);
    if (!subject.truth.pose || !app.status.face) return null;
    const aspect = subject.canvas.width / subject.canvas.height;
    // true face width on screen: cheekbone-level points of the template skin are not available here,
    // so use two fixed template points at the widest part of the cranium's face region
    const l = subject.truthImagePoint([7, 1, -2]), r = subject.truthImagePoint([-7, 1, -2]);
    const faceW = Math.hypot((l[0] - r[0]) * aspect, l[1] - r[1]);
    const faceWCm = 14 * 1.1; // those two points after the test reshaping
    const miss = (pts: Float32Array, jaw: boolean) => {
      let e = 0, n = 0;
      for (let i = 0; i < pts.length; i += 3) {
        const p = [pts[i], pts[i + 1], pts[i + 2]];
        const a = app.imagePoint(p, jaw), t = subject.truthImagePoint(p, jaw);
        if (!a) continue;
        e += Math.hypot((a[0] - t[0]) * aspect, a[1] - t[1]); n++;
      }
      return n ? e / n : NaN;
    };
    const c = miss(cran, false), m = miss(mand, true);
    return {
      craniumPct: (100 * c) / faceW, mandiblePct: (100 * m) / faceW,
      craniumMm: (c / faceW) * faceWCm * 10, mandibleMm: (m / faceW) * faceWCm * 10,
      status: { ...app.status },
      truthJawDeg: (subject.truth.jawOpen * 180) / Math.PI,
    };
  };
}
