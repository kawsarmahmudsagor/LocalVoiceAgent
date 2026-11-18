import { describe, it, expect } from "vitest";
import { apiPort, devProxy } from "./devProxy";
describe("dev proxy", () => {
  it("follows WEB_PORT from .env, default 8000", () => {
    expect(apiPort({})).toBe(8000);
    expect(apiPort({ WEB_PORT: "8001" })).toBe(8001);
    expect(apiPort({ WEB_PORT: " " })).toBe(8000);
    expect(() => apiPort({ WEB_PORT: "abc" })).toThrow(/WEB_PORT/);
  });
  it("points /api and /ws at the API port", () => {
    const p = devProxy(8001);
    expect(p["/api"]).toBe("http://127.0.0.1:8001");
    expect(p["/ws"]).toEqual({ target: "ws://127.0.0.1:8001", ws: true });
  });
});
