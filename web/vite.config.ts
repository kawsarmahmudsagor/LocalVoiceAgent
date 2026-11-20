import { defineConfig, loadEnv, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { apiPort, callPageRewrite, devProxy } from "./src/devProxy";

/** Serves `/call` from `call.html` in dev, as the FastAPI app does in production. */
function callPage(): Plugin {
  return {
    name: "call-page",
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        const r = req as { url?: string }; // no @types/node here: only the URL is touched
        const to = r.url ? callPageRewrite(r.url) : null;
        if (to) r.url = to;
        next();
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  // The API reads WEB_PORT from the repo-root .env; the dev proxy follows it. A shell variable wins.
  const env = loadEnv(mode, "..", "WEB_"); // shell variables override .env (Vite loadEnv)
  return {
    // Tailwind only touches files that import it: src/call/call.css, loaded by call.html alone.
    plugins: [react(), tailwindcss(), callPage()],
    server: {
      port: 5173,
      proxy: devProxy(apiPort(env)),
    },
    build: {
      rollupOptions: { input: { main: "index.html", call: "call.html" } },
    },
  };
});
