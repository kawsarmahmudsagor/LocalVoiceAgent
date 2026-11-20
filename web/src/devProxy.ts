/** Vite dev-server proxy to the FastAPI app. The port follows WEB_PORT in the repo's .env (default 8000). */
export function apiPort(env: Record<string, string | undefined>): number {
  const raw = (env.WEB_PORT ?? "").trim();
  if (!raw) return 8000;
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error(`WEB_PORT=${raw} is not a valid port`);
  return port;
}

export function devProxy(port: number) {
  return {
    "/api": `http://127.0.0.1:${port}`,
    "/ws": { target: `ws://127.0.0.1:${port}`, ws: true },
  };
}

/** Dev server: `/call` is the call page, served from `call.html` (the built server does the same). */
export function callPageRewrite(url: string): string | null {
  const m = /^\/call\/?(\?.*)?$/.exec(url);
  return m ? `/call.html${m[1] ?? ""}` : null;
}
