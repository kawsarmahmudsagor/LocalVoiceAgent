import { describe, it, expect } from "vitest";
import { floatTo16BitPCM, resample, pcm16ToFloat } from "./pcm";
describe("pcm", () => {
  it("converts and clamps", () => {
    expect(Array.from(floatTo16BitPCM(new Float32Array([0, 1, -1, 2, -2, 0.5])))).toEqual([0, 32767, -32768, 32767, -32768, 16384]);
  });
  it("resamples 48k to 16k", () => {
    const out = resample(new Float32Array(480).fill(0.25), 48000, 16000);
    expect(out.length).toBe(160); expect(out.every(v => Math.abs(v - 0.25) < 1e-6)).toBe(true);
  });
  it("same rate copies", () => { const a = new Float32Array([0.1, 0.2]); const b = resample(a, 16000, 16000); expect(b).toEqual(a); expect(b).not.toBe(a); });
  it("pcm16 to float", () => { expect(Array.from(pcm16ToFloat(new Int16Array([0, 16384, -32768]).buffer))).toEqual([0, 0.5, -1]); });
});
