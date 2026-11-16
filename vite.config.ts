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
        // only this machine may write assets, even when the server is reachable from the network
        const from = req.socket.remoteAddress ?? "";
        const local = from === "127.0.0.1" || from === "::1" || from === "::ffff:127.0.0.1";
        if (!local || req.method !== "POST" || !/^[\w.-]+\.json$/.test(name)) {
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

export default defineConfig({
  base: "./",
  plugins: [saveAssetPlugin(), ...(phone ? [basicSsl()] : [])],
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
