import { describe, it, expect } from "vitest";
import { callReducer, initialCall } from "./flow";

describe("callReducer", () => {
  it("starts closed and unmuted, out of a call", () => {
    expect(initialCall).toEqual({ inCall: false, paneOpen: false, speakerMuted: false });
  });
  it("start enters the call; the mic outcome is not part of the view", () => {
    expect(callReducer(initialCall, { kind: "start" })).toEqual({ ...initialCall, inCall: true });
  });
  it("opens and closes the pane", () => {
    const open = callReducer({ ...initialCall, inCall: true }, { kind: "pane", open: true });
    expect(open.paneOpen).toBe(true);
    expect(callReducer(open, { kind: "pane", open: false }).paneOpen).toBe(false);
  });
  it("mutes the speaker", () => {
    expect(callReducer(initialCall, { kind: "speaker", muted: true }).speakerMuted).toBe(true);
  });
  it("end resets the pane and the speaker", () => {
    expect(callReducer({ inCall: true, paneOpen: true, speakerMuted: true }, { kind: "end" })).toEqual(initialCall);
  });
});
