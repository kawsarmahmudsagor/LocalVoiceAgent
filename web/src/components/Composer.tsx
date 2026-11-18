import { useState, type KeyboardEvent } from "react";
import type { Status } from "../protocol";

interface Props {
  status: Status;
  listening: boolean;
  connected: boolean;
  onToggleMic(): void;
  onSend(text: string): void;
  onStop(): void;
  onClear(): void;
}

export function Composer({ status, listening, connected, onToggleMic, onSend, onStop, onClear }: Props) {
  const [text, setText] = useState("");
  const busy = status === "thinking" || status === "speaking";

  const send = () => {
    const t = text.trim();
    if (!t || !connected) return;
    onSend(t);
    setText("");
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="composer">
      <button type="button" className={`mic ${listening ? "mic-on" : ""}`} aria-pressed={listening}
        disabled={!connected} onClick={onToggleMic} title={listening ? "Stop listening" : "Start listening"}>
        <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
          <path fill="currentColor" d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z" />
        </svg>
        <span>{listening ? "Listening" : "Mic"}</span>
      </button>
      <textarea className="input" rows={1} placeholder={connected ? "Type a message…" : "Disconnected"}
        value={text} disabled={!connected} onChange={e => setText(e.target.value)} onKeyDown={onKey} />
      <button type="button" className="primary" disabled={!connected || !text.trim()} onClick={send}>Send</button>
      <button type="button" disabled={!busy} onClick={onStop}>Stop</button>
      <button type="button" className="ghost" onClick={onClear}>Clear</button>
    </div>
  );
}
