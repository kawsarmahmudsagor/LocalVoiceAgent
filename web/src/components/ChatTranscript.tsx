import { useEffect, useRef } from "react";
import type { ChatMessage } from "../state";

export function ChatTranscript({ messages }: { messages: ChatMessage[] }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  return (
    <div className="transcript" role="log" aria-live="polite">
      {messages.length === 0 && (
        <div className="empty">
          <p>Press the mic and start talking, or type a message below.</p>
        </div>
      )}
      {messages.map(m => (
        <div key={m.id} className={`msg msg-${m.role}`}>
          <div className="msg-role">{m.role === "user" ? "You" : "Assistant"}</div>
          <div className="msg-text">
            {m.text}
            {m.streaming && <span className="cursor" aria-hidden="true" />}
            {m.interrupted && <span className="tag">interrupted</span>}
          </div>
        </div>
      ))}
      <div ref={end} />
    </div>
  );
}
