/** Where frames come from: the webcam, a video file, or any canvas (test renders). */

export interface FrameSource {
  el: HTMLVideoElement | HTMLCanvasElement;
  width: number;
  height: number;
  /** Calls back once for every new frame. Returns a function that stops the loop. */
  onFrame(cb: (timeMs: number) => void): () => void;
  stop(): void;
}

/** Keep the element in the page (invisible): detached videos are not reliably decoded frame by frame. */
function park(video: HTMLVideoElement) {
  video.setAttribute("aria-hidden", "true");
  Object.assign(video.style, { position: "fixed", left: "0", top: "0", width: "2px", height: "2px", opacity: "0", pointerEvents: "none" });
  document.body.appendChild(video);
}

function videoSource(video: HTMLVideoElement, cleanup: () => void): FrameSource {
  return {
    el: video,
    get width() { return video.videoWidth; },
    get height() { return video.videoHeight; },
    onFrame(cb) {
      let live = true;
      let frames = 0;
      const anyVideo = video as HTMLVideoElement & { requestVideoFrameCallback?: (f: (now: number) => void) => number };
      // fallback: poll for a new frame on every display refresh
      const poll = () => {
        let last = -1;
        const tick = (now: number) => {
          if (!live) return;
          if (video.currentTime !== last && video.readyState >= 2) { last = video.currentTime; cb(now); }
          requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      };
      if (anyVideo.requestVideoFrameCallback) {
        let usePoll = false;
        const tick = (now: number) => {
          if (!live || usePoll) return;
          frames++;
          cb(now);
          anyVideo.requestVideoFrameCallback!(tick);
        };
        anyVideo.requestVideoFrameCallback(tick);
        // some browsers never fire per-frame callbacks for a video that is not being shown
        setTimeout(() => { if (live && frames === 0) { usePoll = true; poll(); } }, 1500);
      } else poll();
      return () => { live = false; };
    },
    stop: cleanup,
  };
}

/** @param light ask for a smaller picture (phones, slower machines) */
export async function openCamera(light = false): Promise<FrameSource> {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser cannot open a camera (needs HTTPS or localhost).");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user", width: { ideal: light ? 960 : 1280 }, height: { ideal: light ? 540 : 720 }, frameRate: { ideal: 30 } },
  });
  return streamSource(stream);
}

/** Frames from any video MediaStream (the webcam, or a captured canvas in tests). */
export async function streamSource(stream: MediaStream): Promise<FrameSource> {
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  park(video);
  await video.play();
  if (!video.videoWidth) await new Promise<void>((r) => video.addEventListener("loadedmetadata", () => r(), { once: true }));
  return videoSource(video, () => { stream.getTracks().forEach((t) => t.stop()); video.remove(); });
}

export async function openVideoFile(url: string): Promise<FrameSource> {
  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.loop = true;
  video.crossOrigin = "anonymous";
  video.src = url;
  park(video);
  await video.play();
  return videoSource(video, () => { video.pause(); video.remove(); });
}

/** A canvas that something else draws into; `tick` is called before each frame so it can redraw. */
export function canvasSource(canvas: HTMLCanvasElement, tick?: (timeMs: number) => void): FrameSource {
  return {
    el: canvas,
    get width() { return canvas.width; },
    get height() { return canvas.height; },
    onFrame(cb) {
      // A fixed 30 fps virtual clock on a timer: test renders stay deterministic and keep
      // running at full speed even where requestAnimationFrame is throttled.
      let live = true;
      let t = 0;
      const loop = () => {
        if (!live) return;
        t += 1000 / 30;
        tick?.(t);
        cb(t);
        setTimeout(loop, 0);
      };
      setTimeout(loop, 0);
      return () => { live = false; };
    },
    stop() {},
  };
}
