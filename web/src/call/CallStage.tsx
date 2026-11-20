import type { Status } from "../protocol";
import { CallAvatar } from "./CallAvatar";
import { CallWaveform } from "./CallWaveform";
import { callStatusLabel } from "./status";

interface CallStageProps {
  status: Status;
  connected: boolean;
  speakerMuted: boolean;
  error: string | null;
  agentLevel: () => number;
  userLevel: () => number;
}

/** The middle of the call: the agent, its status, and the voice line. */
export function CallStage({ status, connected, speakerMuted, error, agentLevel, userLevel }: CallStageProps) {
  return (
    <div className="relative isolate flex h-full flex-col items-center justify-center px-4">
      {/* A low evergreen glow behind the agent. */}
      <div aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(ellipse_70%_55%_at_50%_36%,color-mix(in_oklch,var(--color-user)_12%,transparent),transparent_70%)]" />
      <div className="p-10 @5xl:p-16 [@media(max-height:500px)]:p-4">
        <CallAvatar status={status} speakerMuted={speakerMuted} agentLevel={agentLevel} userLevel={userLevel}
          className="w-28 @5xl:w-48 [@media(max-height:500px)]:w-16" />
      </div>
      <div className="text-center">
        <h1 className="font-display text-xl font-semibold tracking-tight @5xl:text-[1.75rem]">Voice agent</h1>
        <p role="status" aria-live="polite" className="mt-1 text-sm text-ink-muted">
          {callStatusLabel(status, connected)}
          {speakerMuted && <span className="block text-xs text-agent">Speaker off</span>}
        </p>
        {error && <p className="mx-auto mt-2 max-w-sm text-sm text-danger">{error}</p>}
      </div>
      <CallWaveform status={status} speakerMuted={speakerMuted} agentLevel={agentLevel} userLevel={userLevel}
        className="mt-4 @5xl:mt-8" />
    </div>
  );
}
