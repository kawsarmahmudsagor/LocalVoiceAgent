import { describe, it, expect } from "vitest";
import { vramUsed } from "./vram";
describe("vramUsed", () => {
  it("sums selections and flags over budget", () => {
    const cat = [{ slug: "a", vram_mb: 1000 }, { slug: "b-rt", vram_mb: 500 }, { slug: "t", vram_mb: 2000 }] as any;
    const cfg = { stt_final: "a", stt_realtime: "off", tts: "t", gpu_total_mb: 10000, llm_vram_mb: 7000 } as any;
    expect(vramUsed(cfg, cat)).toEqual({ used: 10000, total: 10000, over: false });
    expect(vramUsed({ ...cfg, stt_realtime: "b-rt" }, cat).over).toBe(true);
  });
});
