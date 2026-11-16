import { XRayMirror, isPhone } from "./app";
import { openCamera, openVideoFile, canvasSource, streamSource, type FrameSource } from "./tracking/camera";
import { bindUi } from "./ui/ui";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);

async function boot() {
  const canvas = $<HTMLCanvasElement>("view");
  const stage = $("stage");
  const start = $<HTMLButtonElement>("start");
  const hint = $("introHint");
  const app = new XRayMirror(canvas);
  const ui = bindUi(app);

  const fail = (msg: string) => { hint.textContent = msg; hint.classList.add("error"); };

  try {
    await app.init((m) => { start.textContent = m === "Ready" ? "Start camera" : m + "…"; ui.setState(m); });
  } catch (e) {
    console.error(e);
    fail("Could not load the anatomy or the trackers. Check the connection and reload.");
    start.textContent = "Unavailable";
    return;
  }
  const mf = app.manifest;
  const link = (url: string | undefined, text: string) => (url ? `<a href="${url}" target="_blank" rel="noreferrer">${text}</a>` : text);
  $("credits").innerHTML =
    `Anatomy: ${app.attribution} (${link(mf?.licenseUrl, mf?.license ?? "")})` +
    (mf?.muscles ? `; facial muscles: ${mf.muscles.attribution} (${link(mf.muscles.licenseUrl, mf.muscles.license)})` : "") +
    `. Meshes were simplified and are reshaped live to fit the viewer. Tracking: MediaPipe. ` +
    `This is an illustration, not an image of your body, and not a medical device.`;

  const go = (source: FrameSource) => {
    stage.style.setProperty("--ar", String(source.width / source.height));
    stage.classList.add("live");
    app.start(source);
    ui.setState("Live");
  };

  // ── test sources ──────────────────────────────────────────────────────────
  if (params.has("synthetic")) {
    const { SyntheticSubject } = await import("./dev/synthetic");
    const subject = new SyntheticSubject({
      reshape: params.has("reshape") ? Number(params.get("reshape")) : 1,
      still: params.has("still"),
      jaw: params.has("jaw"),
      vfov: params.has("truefov") ? Number(params.get("truefov")) : 45,
    });
    await subject.load();
    if (params.has("viastream")) {
      // exercise the real camera code path: the test render goes through a MediaStream and a <video>
      let t = 0;
      setInterval(() => subject.draw((t += 1000 / 30)), 1000 / 30);
      subject.draw(0);
      go(await streamSource(subject.canvas.captureStream(30)));
    } else go(canvasSource(subject.canvas, (t) => subject.draw(t)));
    const { makeScorer, makeRegionScorer } = await import("./dev/score");
    (window as unknown as Record<string, unknown>).__xray = { app, subject, score: makeScorer(app, subject), regions: makeRegionScorer(app, subject) };
    return;
  }
  if (params.has("src")) {
    start.disabled = false;
    start.textContent = "Play test video";
    start.onclick = async () => go(await openVideoFile(params.get("src")!));
    (window as unknown as Record<string, unknown>).__xray = { app };
    return;
  }

  // ── camera ────────────────────────────────────────────────────────────────
  start.disabled = false;
  start.onclick = async () => {
    start.disabled = true;
    start.textContent = "Opening camera…";
    try {
      go(await openCamera(app.settings.quality === "light" || (app.settings.quality === "auto" && isPhone())));
    } catch (e) {
      console.error(e);
      const name = (e as DOMException)?.name;
      fail(
        name === "NotAllowedError" ? "Camera access was blocked. Allow the camera for this page and try again."
        : name === "NotFoundError" ? "No camera was found on this device."
        : "The camera could not be opened: " + ((e as Error)?.message ?? e),
      );
      start.disabled = false;
      start.textContent = "Start camera";
    }
  };
  (window as unknown as Record<string, unknown>).__xray = { app };
}

boot();
