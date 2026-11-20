import type { Ref } from "react";
import { MessageCircleIcon } from "lucide-react";

interface ConversationButtonProps {
  open: boolean;
  /** The pane's id, for aria-controls. */
  controls: string;
  onToggle(): void;
  buttonRef?: Ref<HTMLButtonElement>;
}

/** The chat button in the dock: opens and closes the conversation pane. */
export function ConversationButton({ open, controls, onToggle, buttonRef }: ConversationButtonProps) {
  const label = open ? "Close conversation" : "Open conversation";
  return (
    <button ref={buttonRef} type="button" onClick={onToggle} aria-label={label} title={label}
      aria-expanded={open} aria-controls={controls}
      className={`grid size-12 shrink-0 place-items-center justify-self-end rounded-full shadow-lg outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ink/60 @5xl:size-14 ${
        open ? "bg-user text-stage hover:brightness-110" : "border border-line bg-white/[0.08] text-ink backdrop-blur-md hover:bg-white/[0.14]"
      }`}>
      <MessageCircleIcon aria-hidden="true" className="size-6" />
    </button>
  );
}
