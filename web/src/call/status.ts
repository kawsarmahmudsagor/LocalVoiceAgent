import type { Status } from "../protocol";

/** Whose voice the call screen animates: the agent's, the caller's, or nobody's. */
export type CallActivity = "agent" | "user" | null;

export function callActivity(status: Status): CallActivity {
  if (status === "speaking") return "agent";
  if (status === "listening") return "user";
  return null;
}

/** The line under the agent's name. */
export function callStatusLabel(status: Status, connected: boolean): string {
  if (!connected) return "Connecting…";
  switch (status) {
    case "listening": return "Listening…";
    case "thinking": return "Thinking…";
    case "speaking": return "Speaking…";
    default: return "Connected";
  }
}
