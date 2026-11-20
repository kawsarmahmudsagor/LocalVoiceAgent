import { useEffect, useReducer, useRef } from "react";
import { useVoiceSession } from "../session/useVoiceSession";
import { CallControls } from "./CallControls";
import { CallStage } from "./CallStage";
import { ConversationButton } from "./ConversationButton";
import { ConversationPane } from "./ConversationPane";
import { callReducer, initialCall } from "./flow";
import { StartScreen } from "./StartScreen";

const PANE_ID = "call-conversation";

/** The call page: a start screen, then the call with its collapsible conversation pane. */
export default function CallApp() {
  const session = useVoiceSession();
  const { state } = session;
  const [view, send] = useReducer(callReducer, initialCall);
  const startButton = useRef<HTMLButtonElement>(null);
  const paneButton = useRef<HTMLButtonElement>(null);
  const wasInCall = useRef(false);

  // After End call, focus goes back to Start.
  useEffect(() => {
    if (wasInCall.current && !view.inCall) startButton.current?.focus();
    wasInCall.current = view.inCall;
  }, [view.inCall]);

  const start = () => {
    session.resumeAudio();
    session.dispatch({ kind: "dismiss_error" });
    send({ kind: "start" });
    // A denied mic lands in state.error and shows on the stage; the call stays open for typing.
    if (!state.listening) void session.toggleMic();
  };

  const end = () => {
    session.stop(); // cut playback first, so nothing more is heard
    if (state.listening) void session.toggleMic();
    session.clear();
    session.setSpeakerMuted(false);
    send({ kind: "end" });
  };

  const toggleSpeaker = () => {
    session.setSpeakerMuted(!view.speakerMuted);
    send({ kind: "speaker", muted: !view.speakerMuted });
  };

  const closePane = () => {
    send({ kind: "pane", open: false });
    paneButton.current?.focus();
  };

  if (!view.inCall) {
    return <StartScreen connected={state.connected} error={state.error} onStart={start} buttonRef={startButton} />;
  }

  return (
    <section aria-label="Call" className="@container/call-root relative isolate flex h-full w-full overflow-hidden bg-stage text-ink">
      <div className="@container relative flex min-w-0 flex-1 flex-col">
        <div className="relative min-h-0 flex-1">
          <CallStage status={state.status} connected={state.connected} speakerMuted={view.speakerMuted}
            error={state.error} agentLevel={session.agentLevel} userLevel={session.userLevel} />
        </div>
        <div className="grid shrink-0 grid-cols-[1fr_auto_1fr] items-center gap-2 px-3 pb-5 @5xl:px-6 @5xl:pb-8">
          <span />
          <CallControls micOn={state.listening} speakerMuted={view.speakerMuted} disabled={!state.connected}
            onToggleMic={() => void session.toggleMic()} onEndCall={end} onToggleSpeaker={toggleSpeaker} />
          <ConversationButton open={view.paneOpen} controls={PANE_ID} buttonRef={paneButton}
            onToggle={() => send({ kind: "pane", open: !view.paneOpen })} />
        </div>
      </div>
      <ConversationPane id={PANE_ID} open={view.paneOpen} onClose={closePane} messages={state.messages}
        partial={state.partial} connected={state.connected} onSend={session.send} />
    </section>
  );
}
