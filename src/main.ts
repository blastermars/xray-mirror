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
  // Credits are built as text nodes and links, never as HTML: the strings come from a data file.
  const mf = app.manifest;
  const credits = $("credits");
  credits.textContent = "";
  const say = (text: string) => credits.append(text);
  const cite = (label: string, url?: string) => {
    if (!url || !/^https?:\/\//.test(url)) { say(label); return; }
    const a = document.createElement("a");
    a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.textContent = label;
    credits.append(a);
  };
  say(`Anatomy: ${app.attribution} (`); cite(mf?.license ?? "", mf?.licenseUrl); say(")");
  if (mf?.muscles) { say(`; facial muscles: ${mf.muscles.attribution} (`); cite(mf.muscles.license, mf.muscles.licenseUrl); say(")"); }
  say(". Meshes were simplified and are reshaped live to fit the viewer. Tracking: MediaPipe. This is an illustration, not an image of your body, and not a medical device.");

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
    // test hook: play a video file instead of the camera. Only files served by this same site, so a
    // crafted link cannot make the page load media from somewhere else.
    const url = new URL(params.get("src")!, location.href);
    if (url.origin !== location.origin) { fail("Test videos must come from this site."); start.textContent = "Unavailable"; return; }
    start.disabled = false;
    start.textContent = "Play test video";
    start.onclick = async () => go(await openVideoFile(url.href));
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
  // debugging handle, development builds only
  if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__xray = { app };
}

boot();
