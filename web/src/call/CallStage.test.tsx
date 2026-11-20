import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { CallStage } from "./CallStage";
import type { Status } from "../protocol";

const zero = () => 0;
const render = (status: Status, speakerMuted = false, error: string | null = null) => renderToStaticMarkup(
  <CallStage status={status} connected speakerMuted={speakerMuted} error={error} agentLevel={zero} userLevel={zero} />);

describe("CallStage", () => {
  it("announces the status politely", () => {
    const html = render("listening");
    expect(html).toContain('role="status"');
    expect(html).toContain('aria-live="polite"');
    expect(html).toContain("Listening…");
  });
  it("says Speaker off only when the speaker is muted", () => {
    expect(render("idle", true)).toContain("Speaker off");
    expect(render("idle", false)).not.toContain("Speaker off");
  });
  it("shows an error under the status", () => {
    expect(render("idle", false, "Microphone permission denied — typing still works.")).toContain("Microphone permission denied");
  });
  it("draws 23 waveform bars", () => {
    expect(render("idle").match(/data-bar=/g)).toHaveLength(23);
  });
});
