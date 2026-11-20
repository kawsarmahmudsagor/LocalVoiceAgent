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

/** The agent: an ink orb with three voice lines, not a face. */
function Orb() {
  return (
    <svg viewBox="0 0 100 100" aria-hidden="true" className="size-full">
      <defs>
        <radialGradient id="call-orb" cx="38%" cy="32%" r="75%">
          <stop offset="0" stopColor="#2a3550" />
          <stop offset="1" stopColor="#141b2a" />
        </radialGradient>
      </defs>
      <circle cx="50" cy="50" r="50" fill="url(#call-orb)" />
      <g stroke="var(--color-agent)" strokeWidth="5" strokeLinecap="round">
        <line x1="36" y1="42" x2="36" y2="58" />
        <line x1="50" y1="32" x2="50" y2="68" />
        <line x1="64" y1="40" x2="64" y2="60" />
      </g>
    </svg>
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
      {/* Thinking: a slow brass arc around the orb. */}
      <svg aria-hidden="true" viewBox="0 0 100 100"
        className={`absolute -inset-[5%] size-[110%] text-agent transition-opacity duration-300 animate-spin [animation-duration:2.4s] motion-reduce:animate-none ${status === "thinking" ? "opacity-100" : "opacity-0"}`}>
        <circle cx="50" cy="50" r="48" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"
          strokeDasharray="70 232" />
      </svg>
      <div className="relative size-full overflow-hidden rounded-full ring-2 ring-agent/40 ring-offset-4 ring-offset-stage">
        <Orb />
      </div>
    </div>
  );
}
