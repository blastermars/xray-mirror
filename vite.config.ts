import { defineConfig } from "vitest/config";
import type { Plugin } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname, basename } from "node:path";

/**
 * Dev-only endpoint used by tools/register.html to write the baked template
 * registration into public/assets. Only bare .json file names are accepted.
 */
function saveAssetPlugin(): Plugin {
  return {
    name: "xray-save-asset",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/__save", (req, res) => {
        const url = new URL(req.url ?? "", "http://localhost");
        const name = basename(url.searchParams.get("name") ?? "");
        // Only this machine may write assets, and only the registration page itself: the custom
        // header cannot be sent by another website without a preflight this server never answers,
        // and the Origin must be this dev server's own.
        const from = req.socket.remoteAddress ?? "";
        const local = from === "127.0.0.1" || from === "::1" || from === "::ffff:127.0.0.1";
        const origin = req.headers.origin ?? "";
        const sameOrigin = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(origin) && origin.endsWith("//" + (req.headers.host ?? ""));
        const marked = req.headers["x-xray-save"] === "1";
        if (!local || !sameOrigin || !marked || req.method !== "POST" || !/^[\w.-]+\.json$/.test(name)) {
          res.statusCode = 400;
          res.end("bad request");
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", async () => {
          const file = resolve(server.config.root, "public/assets", name);
          await mkdir(dirname(file), { recursive: true });
          await writeFile(file, Buffer.concat(chunks));
          res.end("ok");
        });
      });
    },
  };
}

/**
 * PHONE=1 serves over HTTPS on the local network, so a phone on the same Wi-Fi can open the page
 * and use its camera (browsers only allow the camera on HTTPS or localhost). The certificate is
 * self-signed: the phone shows a warning once that has to be accepted.
 */
const phone = process.env.PHONE === "1";

/**
 * Content Security Policy for the built site (as a meta tag: static hosts cannot set headers).
 * Scripts only from this site and the pinned MediaPipe runtime on jsDelivr, no inline script, no
 * eval; network requests only to this site and the two hosts the trackers download from.
 * Not applied in development, where Vite injects styles and uses a websocket.
 */
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval' https://cdn.jsdelivr.net",
  "connect-src 'self' https://cdn.jsdelivr.net https://storage.googleapis.com",
  "img-src 'self' data: blob:",
  "media-src 'self' blob: mediastream:",
  "style-src 'self'",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
].join("; ");

function cspPlugin(): Plugin {
  return {
    name: "xray-csp",
    apply: "build",
    transformIndexHtml(html) {
      return html.replace("<meta charset=\"utf-8\">", `<meta charset="utf-8">\n<meta http-equiv="Content-Security-Policy" content="${CSP}">\n<meta name="referrer" content="no-referrer">`);
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [saveAssetPlugin(), cspPlugin(), ...(phone ? [basicSsl()] : [])],
  server: phone ? { host: true } : undefined,
  preview: phone ? { host: true } : undefined,
  build: {
    target: "es2022",
    rollupOptions: {
      input: { main: resolve(import.meta.dirname, "index.html") },
    },
  },
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
