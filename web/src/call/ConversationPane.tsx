import { XIcon } from "lucide-react";
import type { ChatMessage } from "../state";
import { CallComposer } from "./CallComposer";
import { CallTranscript } from "./CallTranscript";

interface ConversationPaneProps {
  id: string;
  open: boolean;
  onClose(): void;
  messages: ChatMessage[];
  partial: string | null;
  connected: boolean;
  onSend(text: string): void;
}

/**
 * The transcript and message box, in a pane on the right of the call (like a meeting's chat).
 * On a narrow screen it covers the call. Always mounted, so the transcript keeps its scroll;
 * closed, it is collapsed and inert.
 */
export function ConversationPane({ id, open, onClose, messages, partial, connected, onSend }: ConversationPaneProps) {
  return (
    <aside id={id} aria-label="Conversation" data-open={open || undefined}
      // React 18 has no boolean `inert` prop; an empty string sets the attribute.
      {...(open ? {} : { inert: "" })}
      onKeyDown={e => {
        if (e.key === "Escape") onClose();
      }}
      className={[
        "z-40 overflow-hidden bg-stage",
        // Narrow: covers the call, sliding in from the right.
        "invisible absolute inset-0 translate-x-full transition-[translate,visibility] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]",
        "data-[open]:visible data-[open]:translate-x-0",
        // Wide: a column beside the call that opens to 400px.
        "@4xl/call-root:relative @4xl/call-root:inset-auto @4xl/call-root:w-0 @4xl/call-root:translate-x-0 @4xl/call-root:border-l @4xl/call-root:border-line @4xl/call-root:transition-[width,visibility] @4xl/call-root:data-[open]:w-[400px]",
        "motion-reduce:transition-none",
      ].join(" ")}>
      <div className="flex h-full w-full flex-col @4xl/call-root:w-[400px]">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-line pl-5 pr-2">
          <h2 className="font-display text-base font-semibold">Conversation</h2>
          <button type="button" onClick={onClose} aria-label="Close conversation"
            className="grid size-11 place-items-center rounded-full text-ink-muted outline-none transition-colors hover:bg-white/[0.08] hover:text-ink focus-visible:ring-[3px] focus-visible:ring-ring/60">
            <XIcon aria-hidden="true" className="size-5" />
          </button>
        </header>
        <CallTranscript messages={messages} partial={partial} />
        <div className="shrink-0 border-t border-line p-3">
          <CallComposer disabled={!connected} onSend={onSend} focusWhen={open} />
        </div>
      </div>
    </aside>
  );
}
