import { useEffect, useRef, useState } from "react";
import { SendHorizontalIcon } from "lucide-react";

interface CallComposerProps {
  disabled: boolean;
  onSend(text: string): void;
  /** Focuses the input whenever this turns true (the pane opening). */
  focusWhen?: boolean;
}

/** The conversation's message box. */
export function CallComposer({ disabled, onSend, focusWhen }: CallComposerProps) {
  const [text, setText] = useState("");
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (focusWhen) input.current?.focus();
  }, [focusWhen]);

  const canSend = !disabled && text.trim().length > 0;
  return (
    <form onSubmit={e => {
      e.preventDefault();
      if (!canSend) return;
      onSend(text.trim());
      setText("");
    }}
      className="flex h-12 items-center gap-2 rounded-full border border-line bg-white/[0.06] pl-4 pr-1.5 focus-within:border-user/50">
      <label htmlFor="call-message" className="sr-only">Message the agent</label>
      <input id="call-message" ref={input} value={text} onChange={e => setText(e.target.value)} autoComplete="off"
        placeholder="Type a message…"
        className="min-w-0 flex-1 bg-transparent text-[15px] text-ink outline-none placeholder:text-ink-muted" />
      <button type="submit" disabled={!canSend} aria-label="Send message"
        className="grid size-10 shrink-0 place-items-center rounded-full bg-user text-stage outline-none transition-[opacity,filter] hover:brightness-110 focus-visible:ring-[3px] focus-visible:ring-ink/60 disabled:opacity-40">
        <SendHorizontalIcon aria-hidden="true" className="size-[18px]" />
      </button>
    </form>
  );
}
