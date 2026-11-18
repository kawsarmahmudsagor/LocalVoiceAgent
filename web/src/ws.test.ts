import { it, expect } from "vitest";
import { nextDelay } from "./ws";
it("backs off exponentially with cap", () => {
  expect([0, 1, 2, 3, 4, 5].map(nextDelay)).toEqual([500, 1000, 2000, 4000, 8000, 8000]);
});
