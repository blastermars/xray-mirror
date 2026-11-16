import type { XRayMirror } from "../app";
import type { Mode, Reveal } from "../render/pipeline";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const REVEAL_NOTES: Record<Reveal, string> = {
  whole: "",
  split: "One half of your face stays real.",
  lens: "Point with a finger, or move the pointer over the picture.",
};

export function bindUi(app: XRayMirror) {
  const s = app.settings;
  const stage = $("stage");

  // ── segmented controls ─────────────────────────────────────────────────────
  const seg = (id: string, onPick: (v: string) => void) => {
    const root = $(id);
    const buttons = [...root.querySelectorAll<HTMLButtonElement>("button")];
    const pick = (b: HTMLButtonElement) => {
      for (const o of buttons) o.setAttribute("aria-checked", String(o === b));
      onPick(b.dataset.v!);
    };
    for (const b of buttons) b.addEventListener("click", () => pick(b));
    root.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const i = buttons.findIndex((b) => b.getAttribute("aria-checked") === "true");
      const n = buttons[(i + (e.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length];
      n.focus(); pick(n);
    });
  };
  seg("mode", (v) => { s.mode = v as Mode; syncAnno(); });
  seg("reveal", (v) => { s.reveal = v as Reveal; $("revealNote").textContent = REVEAL_NOTES[s.reveal]; });

  // ── layer chips ────────────────────────────────────────────────────────────
  for (const b of $("layers").querySelectorAll<HTMLButtonElement>("button")) {
    const layer = b.dataset.layer!;
    b.setAttribute("aria-pressed", String(s.layers[layer] !== false));
    b.addEventListener("click", () => {
      s.layers[layer] = !s.layers[layer];
      b.setAttribute("aria-pressed", String(s.layers[layer]));
    });
  }

  // ── sliders and switches ───────────────────────────────────────────────────
  const exposure = $<HTMLInputElement>("exposure"), real = $<HTMLInputElement>("real");
  exposure.addEventListener("input", () => { s.exposure = +exposure.value; $("exposureOut").textContent = s.exposure.toFixed(2); syncAnno(); });
  real.addEventListener("input", () => { s.real = +real.value; $("realOut").textContent = Math.round(s.real * 100) + "%"; });
  $<HTMLInputElement>("labelsOn").addEventListener("change", (e) => { s.labels = (e.target as HTMLInputElement).checked; });
  const fov = $<HTMLInputElement>("fov"), fovAuto = $<HTMLInputElement>("fovAuto");
  const syncFov = () => {
    s.fov = fovAuto.checked ? 0 : +fov.value;
    fov.disabled = fovAuto.checked;
    $("fovOut").textContent = fovAuto.checked ? "auto" : fov.value + "°";
  };
  fov.addEventListener("input", syncFov);
  fovAuto.addEventListener("change", syncFov);
  $<HTMLInputElement>("personalise").addEventListener("change", (e) => { s.personalise = (e.target as HTMLInputElement).checked; });
  $<HTMLInputElement>("useOutline").addEventListener("change", (e) => { s.useOutline = (e.target as HTMLInputElement).checked; });
  $("refit").addEventListener("click", () => app.resetTracking());
  $<HTMLInputElement>("gestureZoom").addEventListener("change", (e) => { s.gestureZoom = (e.target as HTMLInputElement).checked; });
  $("resetZoom").addEventListener("click", () => app.resetZoom());
  $<HTMLSelectElement>("quality").addEventListener("change", (e) => { s.quality = (e.target as HTMLSelectElement).value as typeof s.quality; });
  $("snapshot").addEventListener("click", () => {
    // the canvas is shown mirrored; save it the way it is seen
    const src = app.pipeline.renderer.domElement;
    const c = document.createElement("canvas");
    c.width = src.width; c.height = src.height;
    const ctx = c.getContext("2d")!;
    ctx.translate(c.width, 0); ctx.scale(-1, 1);
    ctx.drawImage(src, 0, 0);
    const a = document.createElement("a");
    a.download = "xray-mirror.png";
    a.href = c.toDataURL("image/png");
    a.click();
  });

  // ── lens follows the pointer when no fingertip is in view ───────────────────
  stage.addEventListener("pointermove", (e) => {
    const r = stage.getBoundingClientRect();
    // the picture is mirrored: flip x back into image space; image uv has y up
    const [px, py] = app.fromScreen(1 - (e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height);
    app.pointer.x = px;
    app.pointer.y = py;
  });

  // ── per-frame readouts (throttled) ──────────────────────────────────────────
  const fitCard = $("fitCard"), fitBar = $("fitBar"), fitState = $("fitState");
  const labelsEl = $("labels"), leaders = document.getElementById("leaders") as unknown as SVGSVGElement;
  const labelNodes = new Map<string, HTMLDivElement>();
  let lastText = 0;

  function syncAnno() {
    $("annoTitle").textContent = s.mode === "radiograph" ? "SKULL · PA · LIVE" : "ANATOMY · LIVE";
    $("annoTech").textContent = s.mode === "radiograph" ? `EXP ${s.exposure.toFixed(2)}\nFOV ${app.status.fov.toFixed(0)}°` : `FOV ${app.status.fov.toFixed(0)}°`;
  }

  function drawLabels() {
    const pts = app.labels;
    const seen = new Set<string>();
    let lines = "";
    const W = stage.clientWidth, H = stage.clientHeight;
    // head centre on screen decides which side each label hangs on
    let cx = 0, n = 0;
    // labels are placed in the camera picture; the view may be zoomed
    const placed = pts.map((p) => { const [sx, sy] = app.toScreen(p.x, p.y); return { ...p, x: sx, y: sy, visible: p.visible && sx > 0.02 && sx < 0.98 && sy > 0.02 && sy < 0.98 }; });
    for (const p of placed) if (p.visible) { cx += 1 - p.x; n++; }
    cx = n ? cx / n : 0.5;
    const used: number[] = [];
    for (const p of [...placed].sort((a, b) => a.y - b.y)) {
      if (!p.visible) continue;
      const x = (1 - p.x) * W, y = p.y * H; // mirrored like the picture
      const right = 1 - p.x < cx;
      // keep labels on one side from overlapping
      let ly = y;
      const key = right ? -1 : 1;
      for (let k = 0; k < used.length; k += 2) if (used[k] === key && Math.abs(used[k + 1] - ly) < 18) ly = used[k + 1] + 18;
      used.push(key, ly);
      const lx = right ? x - 46 : x + 46;
      let node = labelNodes.get(p.id);
      if (!node) {
        node = document.createElement("div");
        node.className = "label";
        node.textContent = p.text;
        labelsEl.appendChild(node);
        labelNodes.set(p.id, node);
      }
      node.classList.toggle("right", right);
      node.style.left = lx + "px";
      node.style.top = ly + "px";
      node.hidden = false;
      seen.add(p.id);
      lines += `<line x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${lx.toFixed(1)}" y2="${ly.toFixed(1)}"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2"/>`;
    }
    for (const [id, node] of labelNodes) if (!seen.has(id)) node.hidden = true;
    leaders.innerHTML = lines;
  }

  app.onFrame = () => {
    drawLabels();
    const now = performance.now();
    if (now - lastText < 200) return;
    lastText = now;
    const st = app.status;
    // the marker shows which side of the picture is the subject's right (mirror view: your right is on the right)
    $("annoPose").textContent = st.face
      ? `DIST ${st.distance.toFixed(0)} CM\nYAW ${st.yaw.toFixed(0)}°  PITCH ${st.pitch.toFixed(0)}°\nJAW ${st.jawOpenDeg.toFixed(0)}°`
      : "NO FACE IN VIEW";
    const zoomText = st.zooming ? `ZOOM ${st.zoom.toFixed(1)}×  PINCHING` : st.zoom > 1.01 ? `ZOOM ${st.zoom.toFixed(1)}×` : "";
    $("annoFit").textContent = `${zoomText ? zoomText + "\n" : ""}${st.width}×${st.height}  ${st.fps.toFixed(0)} FPS${st.light ? "  LIGHT" : ""}\nHANDS ${st.hands}`;
    syncAnno();

    const done = st.confidence >= 1;
    fitCard.classList.toggle("done", done && st.face);
    fitBar.style.width = Math.round(st.confidence * 100) + "%";
    fitState.textContent = !st.face ? "Waiting for a face" : !s.personalise ? "Fitting is off" : done ? "Fitted to you" : "Learning your face";
    $("fitErr").textContent = st.face ? st.fitError.toFixed(1) + "%" : "–";
    $("rigidErr").textContent = st.face ? st.rigidError.toFixed(1) + "%" : "–";
  };

  syncFov();
  syncAnno();
  return {
    setState(text: string) {
      $("stateText").textContent = text;
      $("state").className = "state " + (text === "Live" ? "ok" : "");
    },
  };
}
