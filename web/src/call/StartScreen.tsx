import type { CSSProperties, Ref } from "react";
import { Loader2Icon, PhoneIcon } from "lucide-react";

const ARC_RADII = [13, 17.5, 22];

/** Brass sound arcs over a rising sun; they pulse faster while connecting. */
function CallMark({ connecting }: { connecting: boolean }) {
  return (
    <svg viewBox="-6 -2 48 28" aria-hidden="true" className="mb-7 h-auto w-40 text-ink">
      {ARC_RADII.map((r, i) => (
        <path key={r} d={`M${18 - r} 22A${r} ${r} 0 0 1 ${18 + r} 22`} fill="none" strokeWidth="1.1"
          strokeLinecap="round"
          className="stroke-agent opacity-40 motion-safe:animate-[call-wave_var(--wave-duration)_ease-in-out_infinite]"
          style={{ "--wave-duration": connecting ? "1.2s" : "2.6s", animationDelay: `${i * (connecting ? 0.2 : 0.45)}s` } as CSSProperties} />
      ))}
      <path d="M9 22a9 9 0 0 1 18 0Z" className="fill-agent" />
      <rect x="-2" y="22" width="40" height="2" rx="1" fill="currentColor" />
    </svg>
  );
}

interface StartScreenProps {
  connected: boolean;
  error: string | null;
  onStart(): void;
  buttonRef?: Ref<HTMLButtonElement>;
}

/** Before the call: one button. Browsers allow the mic and audio only after a click. */
export function StartScreen({ connected, error, onStart, buttonRef }: StartScreenProps) {
  return (
    <main className="flex h-full flex-col items-center justify-center bg-stage px-4 text-center">
      <CallMark connecting={!connected} />
      <h1 className="text-balance font-display text-[1.75rem] font-semibold leading-tight tracking-[-0.02em]">
        Voice agent
      </h1>
      <p className="mt-2 max-w-[30ch] text-pretty text-[0.9375rem] leading-relaxed text-ink-muted">
        Talk to the local voice agent. Your microphone turns on when the call starts.
      </p>
      <button ref={buttonRef} type="button" onClick={onStart} disabled={!connected} aria-busy={!connected}
        className="mt-8 flex h-12 min-w-56 items-center justify-center gap-2.5 rounded-full bg-user px-7 text-[0.9375rem] font-semibold text-on-user shadow-md outline-none transition-[transform,box-shadow,opacity] hover:shadow-lg focus-visible:ring-[3px] focus-visible:ring-ring/60 active:scale-[0.98] disabled:opacity-50 motion-reduce:transition-none">
        {connected
          ? <PhoneIcon aria-hidden="true" className="size-[18px]" />
          : <Loader2Icon aria-hidden="true" className="size-[18px] animate-spin motion-reduce:animate-none" />}
        {connected ? "Start call" : "Connecting…"}
      </button>
      {error && <p role="alert" className="mt-4 max-w-sm text-sm text-danger">{error}</p>}
    </main>
  );
}
