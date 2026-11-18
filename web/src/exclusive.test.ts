import { it, expect } from "vitest";
import { createExclusive } from "./exclusive";

it("ignores calls while one is still running", async () => {
  const run = createExclusive();
  let starts = 0;
  let release!: () => void;
  const slow = () => { starts++; return new Promise<void>(r => { release = r; }); };
  const first = run(slow);
  const second = run(slow);                     // double-click while the permission prompt is open
  expect(starts).toBe(1);
  release(); await first; await second;
  await run(async () => { starts++; });         // free again afterwards
  expect(starts).toBe(2);
});

it("frees the guard when the action throws", async () => {
  const run = createExclusive();
  await expect(run(async () => { throw new Error("denied"); })).rejects.toThrow("denied");
  let ran = false; await run(async () => { ran = true; }); expect(ran).toBe(true);
});
