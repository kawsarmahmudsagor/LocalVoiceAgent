import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { CallControls } from "./CallControls";
import { CallComposer } from "./CallComposer";
import { CallTranscript } from "./CallTranscript";
import { ConversationButton } from "./ConversationButton";
import { ConversationPane } from "./ConversationPane";
import type { ChatMessage } from "../state";

const noop = () => {};

describe("CallControls", () => {
  it("labels the mic by what a click does, and marks it pressed when off", () => {
    const html = renderToStaticMarkup(<CallControls micOn={false} speakerMuted={false} disabled={false}
      onToggleMic={noop} onEndCall={noop} onToggleSpeaker={noop} />);
    expect(html).toMatch(/aria-label="Unmute microphone"[^>]*aria-pressed="true"|aria-pressed="true"[^>]*aria-label="Unmute microphone"/);
    expect(html).toContain('aria-label="End call"');
    expect(html).toContain('aria-label="Mute speaker"');
  });
  it("offers to turn the speaker back on when muted", () => {
    const html = renderToStaticMarkup(<CallControls micOn speakerMuted disabled={false}
      onToggleMic={noop} onEndCall={noop} onToggleSpeaker={noop} />);
    expect(html).toContain('aria-label="Turn speaker on"');
    expect(html).toContain('aria-label="Mute microphone"');
  });
});

describe("ConversationButton", () => {
  it("controls the pane", () => {
    const html = renderToStaticMarkup(<ConversationButton open={false} controls="pane" onToggle={noop} />);
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain('aria-controls="pane"');
    expect(html).toContain('aria-label="Open conversation"');
  });
});

describe("ConversationPane", () => {
  const render = (open: boolean) => renderToStaticMarkup(<ConversationPane id="pane" open={open} onClose={noop}
    messages={[]} partial={null} connected onSend={noop} />);
  it("is inert while closed", () => {
    const html = render(false);
    expect(html).toContain('aria-label="Conversation"');
    expect(html).toContain("inert");
    expect(html).not.toContain("data-open");
  });
  it("is marked open", () => {
    const html = render(true);
    expect(html).toContain("data-open");
    expect(html).not.toMatch(/\sinert/);
  });
});

describe("CallTranscript", () => {
  it("shows the heard and unheard parts of an interrupted reply, and the live partial", () => {
    const messages: ChatMessage[] = [
      { id: 1, role: "user", text: "Hi", streaming: false, interrupted: false },
      { id: 2, role: "assistant", text: "Hello", streaming: false, interrupted: true, unheard: "there" },
    ];
    const html = renderToStaticMarkup(<CallTranscript messages={messages} partial="how are" />);
    expect(html).toContain("Hello");
    expect(html).toContain("there");
    expect(html).toContain("how are");
  });
  it("invites typing when empty", () => {
    expect(renderToStaticMarkup(<CallTranscript messages={[]} partial={null} />)).toContain("Say something");
  });
});

describe("CallComposer", () => {
  it("disables sending while disconnected", () => {
    const html = renderToStaticMarkup(<CallComposer disabled onSend={noop} />);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Send message"|<button[^>]*aria-label="Send message"[^>]*disabled=""/);
  });
});
