import { describe, it, expect } from "vitest";
import { rmsLevel } from "./level";

describe("rmsLevel", () => {
  it("is 0 for silence", () => expect(rmsLevel(new Float32Array(128))).toBe(0));
  it("is the RMS of the samples", () => expect(rmsLevel(new Float32Array(128).fill(0.5))).toBeCloseTo(0.5));
  it("clamps to 1", () => expect(rmsLevel(new Float32Array(128).fill(2))).toBe(1));
  it("is 0 for no samples", () => expect(rmsLevel(new Float32Array(0))).toBe(0));
});
