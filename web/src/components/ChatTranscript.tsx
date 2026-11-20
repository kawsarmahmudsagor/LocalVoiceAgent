import { useEffect, useRef } from "react";
import type { ChatMessage } from "../state";

export function ChatTranscript({ messages, partial }: { messages: ChatMessage[]; partial: string | null }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages, partial]);

  return (
    <div className="transcript" role="log" aria-live="polite">
      {messages.length === 0 && !partial && (
        <div className="empty">
          <p>Press the mic and start talking, or type a message below.</p>
        </div>
      )}
      {messages.map(m => (
        <div key={m.id} className={`msg msg-${m.role}${m.ignored ? " msg-ignored" : ""}`}>
          <div className="msg-role">
            {m.ignored ? `Speaker ${m.speaker === "unknown" || m.speaker === undefined ? "?" : m.speaker} (ignored)`
              : m.role === "user" ? "You" : "Assistant"}
            {!m.ignored && m.role === "user" && m.speaker !== undefined &&
              <span className="speaker-tag">S{m.speaker === "unknown" ? "?" : m.speaker}</span>}
          </div>
          <div className="msg-text">
            {m.text}
            {m.unheard && <span className="unheard">{m.unheard}</span>}
            {m.streaming && <span className="cursor" aria-hidden="true" />}
            {m.interrupted && <span className="tag">interrupted</span>}
          </div>
          {m.sources && m.sources.length > 0 && (
            <details className="sources">
              <summary>Sources ({m.sources.length})</summary>
              <ul>
                {m.sources.map((s, i) => (
                  <li key={`${s.name}-${s.chunk}-${i}`}>
                    <span className="source-name">{s.name} #{s.chunk}</span>
                    <span className="source-score"> {s.score.toFixed(2)}</span>
                    <div className="source-preview">{s.preview}</div>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      ))}
      {partial && (
        <div className="msg msg-user msg-partial" aria-live="off">
          <div className="msg-role">You (live)</div>
          <div className="msg-text">{partial}</div>
        </div>
      )}
      <div ref={end} />
    </div>
  );
}
