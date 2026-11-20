import { motion, useReducedMotion } from "motion/react";
import type { Status } from "../protocol";
import { callActivity } from "./status";
import { useLevel } from "./useLevel";

// Resting scale of each ripple ring, relative to the avatar.
const RING_STEPS = [1.14, 1.3, 1.46];

/** Brass ripples sized by the agent's voice; with the speaker off they pulse on the state instead. */
function AgentRings({ level, active, muted }: { level: () => number; active: boolean; muted: boolean }) {
  const reduceMotion = useReducedMotion();
  const v = Math.min(1, useLevel(level, active && !muted) * 3);
  return RING_STEPS.map((step, i) => {
    const rest = 0.14 - i * 0.04;
    let animate: { scale: number | number[]; opacity: number | number[] };
    let transition: object = { duration: 0.14, ease: "easeOut" };
    if (!active) {
      animate = { scale: step, opacity: rest };
      transition = { duration: 0.6, ease: "easeOut" };
    } else if (muted) {
      animate = reduceMotion
        ? { scale: step, opacity: [rest, 0.5 - i * 0.12, rest] }
        : { scale: [step, step + 0.08, step], opacity: [rest, 0.5 - i * 0.12, rest] };
      transition = { duration: 1.4, repeat: Infinity, ease: "easeInOut", delay: i * 0.18 };
    } else {
      animate = { scale: reduceMotion ? step : step + v * 0.06 * (i + 1), opacity: rest + v * (0.42 - i * 0.12) };
    }
    return (
      <motion.span key={step} aria-hidden="true" initial={false} animate={animate} transition={transition}
        className="absolute inset-0 rounded-full border-2 border-agent" />
    );
  });
}

/** A mint halo that swells with the caller's voice. */
function UserHalo({ level, active }: { level: () => number; active: boolean }) {
  const reduceMotion = useReducedMotion();
  const v = Math.min(1, useLevel(level, active) * 4);
  return (
    <motion.span aria-hidden="true" initial={false}
      animate={{ opacity: active ? 0.45 + v * 0.55 : 0, scale: reduceMotion ? 1.06 : 1.04 + v * 0.08 }}
      transition={{ duration: active ? 0.12 : 0.4, ease: "easeOut" }}
      className="absolute inset-0 rounded-full border-[3px] border-user shadow-[0_0_32px_-4px_var(--color-user)]" />
  );
}

interface CallAvatarProps {
  status: Status;
  speakerMuted: boolean;
  agentLevel: () => number;
  userLevel: () => number;
  className?: string;
}

export function CallAvatar({ status, speakerMuted, agentLevel, userLevel, className = "" }: CallAvatarProps) {
  const activity = callActivity(status);
  return (
    <div className={`relative aspect-square ${className}`}>
      <AgentRings level={agentLevel} active={activity === "agent"} muted={speakerMuted} />
      <UserHalo level={userLevel} active={activity === "user"} />
      {/* Thinking: a slow brass arc around the portrait. */}
      <svg aria-hidden="true" viewBox="0 0 100 100"
        className={`absolute -inset-[5%] size-[110%] text-agent transition-opacity duration-300 animate-spin [animation-duration:2.4s] motion-reduce:animate-none ${status === "thinking" ? "opacity-100" : "opacity-0"}`}>
        <circle cx="50" cy="50" r="48" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"
          strokeDasharray="70 232" />
      </svg>
      <div className="relative size-full overflow-hidden rounded-full bg-panel ring-2 ring-agent/50 ring-offset-4 ring-offset-stage">
        <img src="/agent-avatar.svg" alt="Voice agent" draggable={false} className="size-full select-none object-cover" />
      </div>
    </div>
  );
}
