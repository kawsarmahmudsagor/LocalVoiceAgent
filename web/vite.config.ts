import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { apiPort, devProxy } from "./src/devProxy";

export default defineConfig(({ mode }) => {
  // The API reads WEB_PORT from the repo-root .env; the dev proxy follows it. A shell variable wins.
  const env = loadEnv(mode, "..", "WEB_"); // shell variables override .env (Vite loadEnv)
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: devProxy(apiPort(env)),
    },
  };
});
