import { describe, it, expect } from "vitest";
import { callActivity, callStatusLabel } from "./status";

describe("callStatusLabel", () => {
  it("names each state", () => {
    expect(callStatusLabel("idle", true)).toBe("Connected");
    expect(callStatusLabel("listening", true)).toBe("Listening…");
    expect(callStatusLabel("thinking", true)).toBe("Thinking…");
    expect(callStatusLabel("speaking", true)).toBe("Speaking…");
  });
  it("says Connecting… while the socket is down, whatever the status", () => {
    expect(callStatusLabel("speaking", false)).toBe("Connecting…");
    expect(callStatusLabel("idle", false)).toBe("Connecting…");
  });
});

describe("callActivity", () => {
  it("follows whose voice is live", () => {
    expect(callActivity("speaking")).toBe("agent");
    expect(callActivity("listening")).toBe("user");
    expect(callActivity("thinking")).toBeNull();
    expect(callActivity("idle")).toBeNull();
  });
});
