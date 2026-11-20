import { useEffect, useRef } from "react";
import type { ChatMessage } from "../state";

const BUBBLE = "max-w-[85%] rounded-2xl px-3.5 py-2.5 text-[15px] leading-relaxed";

/** The call's transcript: what you said, what the agent said (as you heard it), and your live words. */
export function CallTranscript({ messages, partial }: { messages: ChatMessage[]; partial: string | null }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages, partial]);

  return (
    <div role="log" aria-live="polite" className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto overscroll-contain p-4">
      {messages.length === 0 && !partial && (
        <p className="m-auto max-w-[30ch] text-center text-sm text-ink-muted">
          Say something, or type a message below.
        </p>
      )}
      {messages.map(m => {
        const mine = m.role === "user";
        return (
          <div key={m.id} className={`flex flex-col ${mine ? "items-end" : "items-start"} ${m.ignored ? "opacity-50" : ""}`}>
            {m.ignored && <span className="mb-1 text-xs text-ink-muted">Another speaker (ignored)</span>}
            <div className={`${BUBBLE} ${mine ? "rounded-br-md bg-user/15 text-ink" : "rounded-bl-md bg-panel text-ink"}`}>
              {m.text}
              {m.unheard && <span className="text-ink-muted/70 line-through decoration-ink-muted/40"> {m.unheard}</span>}
              {m.streaming && <span aria-hidden="true" className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-agent align-middle motion-reduce:animate-none" />}
            </div>
            {m.interrupted && <span className="mt-1 text-xs text-ink-muted">Interrupted</span>}
          </div>
        );
      })}
      {partial && (
        <div className="flex flex-col items-end" aria-live="off">
          <div className={`${BUBBLE} rounded-br-md border border-dashed border-user/40 text-ink-muted`}>{partial}</div>
        </div>
      )}
      <div ref={end} />
    </div>
  );
}
