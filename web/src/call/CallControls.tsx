import { MicIcon, MicOffIcon, PhoneOffIcon, Volume2Icon, VolumeXIcon } from "lucide-react";

const ROUND =
  "grid shrink-0 place-items-center rounded-full outline-none transition-colors focus-visible:ring-[3px] focus-visible:ring-ink/60 disabled:opacity-50";
const QUIET = "size-12 @5xl:size-14 border border-line bg-white/[0.06] text-ink backdrop-blur-md hover:bg-white/[0.12]";
// A switched-off mic or speaker reads as a filled light button.
const QUIET_OFF = "border-transparent bg-ink text-stage hover:bg-ink/85";

interface CallControlsProps {
  micOn: boolean;
  speakerMuted: boolean;
  /** While the socket is down. */
  disabled: boolean;
  onToggleMic(): void;
  onEndCall(): void;
  onToggleSpeaker(): void;
}

/** Mic, End call and Speaker: the call's three buttons. */
export function CallControls({ micOn, speakerMuted, disabled, onToggleMic, onEndCall, onToggleSpeaker }: CallControlsProps) {
  const micLabel = micOn ? "Mute microphone" : "Unmute microphone";
  const speakerLabel = speakerMuted ? "Turn speaker on" : "Mute speaker";
  return (
    <div className="flex items-center gap-4 @5xl:gap-8">
      <button type="button" onClick={onToggleMic} disabled={disabled} aria-label={micLabel} aria-pressed={!micOn}
        title={micLabel} className={`${ROUND} ${QUIET} ${micOn ? "" : QUIET_OFF}`}>
        {micOn ? <MicIcon aria-hidden="true" className="size-6" /> : <MicOffIcon aria-hidden="true" className="size-6" />}
      </button>
      <button type="button" onClick={onEndCall} aria-label="End call" title="End call"
        className={`${ROUND} size-14 bg-danger text-white shadow-[0_8px_28px_-6px_var(--color-danger)] hover:brightness-110 @5xl:size-16`}>
        <PhoneOffIcon aria-hidden="true" className="size-7" />
      </button>
      <button type="button" onClick={onToggleSpeaker} aria-label={speakerLabel} aria-pressed={speakerMuted}
        title={speakerLabel} className={`${ROUND} ${QUIET} ${speakerMuted ? QUIET_OFF : ""}`}>
        {speakerMuted ? <VolumeXIcon aria-hidden="true" className="size-6" /> : <Volume2Icon aria-hidden="true" className="size-6" />}
      </button>
    </div>
  );
}
