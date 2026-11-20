import type { Status } from "../protocol";
import { callActivity } from "./status";
import { useLevel } from "./useLevel";

const BARS = 23;
// A fixed contour, tallest in the middle, so the line reads as a voice and not a meter.
const SHAPE = Array.from({ length: BARS }, (_, i) => {
  const x = (i - (BARS - 1) / 2) / ((BARS - 1) / 2);
  return 0.35 + 0.65 * Math.cos((x * Math.PI) / 2) * (0.75 + 0.25 * Math.sin(i * 1.7));
});
const MAX_PX = 44;
const MIN_PX = 5;

interface CallWaveformProps {
  status: Status;
  speakerMuted: boolean;
  agentLevel: () => number;
  userLevel: () => number;
  className?: string;
}

/**
 * The voice line under the agent's name: brass for the agent, mint for the caller. With the
 * speaker off there is no reply audio to follow, so the agent's line pulses instead.
 */
export function CallWaveform({ status, speakerMuted, agentLevel, userLevel, className = "" }: CallWaveformProps) {
  const activity = callActivity(status);
  const read = activity === "user" ? userLevel : agentLevel;
  const live = activity === "user" || (activity === "agent" && !speakerMuted);
  const v = Math.min(1, useLevel(read, live) * 4);
  const sweep = (activity === "agent" && speakerMuted) || status === "thinking";
  const tone = activity === "agent" ? "text-agent"
    : activity === "user" ? "text-user"
    : status === "thinking" ? "text-agent/80" : "text-ink/35";

  return (
    <div aria-hidden="true" className={`flex h-12 items-center gap-[5px] transition-colors duration-300 ${tone} ${className}`}>
      {SHAPE.map((s, i) => (
        <span key={i} data-bar={i}
          className={`w-[3px] rounded-full bg-current transition-[height] duration-100 ease-out ${sweep ? "animate-pulse motion-reduce:animate-none" : ""}`}
          style={{
            height: `${Math.max(MIN_PX, sweep ? s * 0.4 * MAX_PX : s * v * MAX_PX)}px`,
            animationDelay: sweep ? `${i * 60}ms` : undefined,
          }} />
      ))}
    </div>
  );
}
