import { useCallback, useEffect, useReducer, useRef } from "react";
import { startMic } from "./audio/mic";
import { createPlayer, type Player } from "./audio/player";
import { ChatTranscript } from "./components/ChatTranscript";
import { Composer } from "./components/Composer";
import { SettingsPanel } from "./components/SettingsPanel";
import { StatusBar } from "./components/StatusBar";
import type { Defaults, Role } from "./protocol";
import { createExclusive } from "./exclusive";
import { initialState, reducer } from "./state";
import { VoiceSocket, socketUrl } from "./ws";

const SETTINGS_DEBOUNCE_MS = 300;

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const socket = useRef<VoiceSocket | null>(null);
  const player = useRef<Player | null>(null);
  const playerRate = useRef<number | null>(null);
  const mic = useRef<{ stop(): void } | null>(null);
  const turnId = useRef<number | null>(null);
  const ackedTurn = useRef<number | null>(null);
  const pendingPatch = useRef<Partial<Defaults>>({});
  const debounce = useRef<number | undefined>(undefined);

  const stopMic = useCallback(() => {
    mic.current?.stop();
    mic.current = null;
  }, []);

  useEffect(() => {
    // Tell the server when the browser actually starts playing a turn (stage timing).
    const ackPlayback = () => {
      const id = turnId.current;
      if (id == null || ackedTurn.current === id) return;
      ackedTurn.current = id;
      socket.current?.send({ type: "playback_started", turn_id: id });
    };

    const s = new VoiceSocket(socketUrl(), {
      onOpen: () => dispatch({ kind: "connected", value: true }),
      onClose: () => {
        stopMic();
        player.current?.flush();
        dispatch({ kind: "connected", value: false });
      },
      onEvent: e => {
        if (e.type === "config" && e.output_format && playerRate.current !== e.output_format.sample_rate) {
          // First config, or a TTS switch that changed the sample rate: never play at the wrong rate.
          player.current?.close();
          player.current = createPlayer(e.output_format, ackPlayback);
          playerRate.current = e.output_format.sample_rate;
        }
        if (e.type === "user_transcript") {
          turnId.current = e.turn_id;
          player.current?.resetClock(); // the previous reply has finished playing before a new turn starts
        }
        if (e.type === "assistant_segment") {
          const spoken = () => dispatch({ kind: "spoken", turn_id: e.turn_id, text: e.text });
          if (e.at_s != null && player.current) player.current.schedule(e.at_s, spoken);
          else spoken();
          return;
        }
        dispatch({ kind: "server", event: e });
      },
      onAudio: buf => player.current?.push(buf),
    });
    socket.current = s;
    return () => {
      s.close();
      stopMic();
      player.current?.close();
      player.current = null;
      playerRate.current = null;
      window.clearTimeout(debounce.current);
    };
  }, [stopMic]);

  const onSettingsChange = (patch: Partial<Defaults>) => {
    dispatch({ kind: "settings", patch });
    pendingPatch.current = { ...pendingPatch.current, ...patch };
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(() => {
      socket.current?.send({ type: "update_settings", ...pendingPatch.current });
      pendingPatch.current = {};
    }, SETTINGS_DEBOUNCE_MS);
  };

  // A second click while the permission prompt is open would start a second capture.
  const micToggle = useRef(createExclusive()).current;

  const onToggleMic = () => micToggle(toggleMic);

  const toggleMic = async () => {
    player.current?.resume();
    const s = socket.current;
    if (!s || !state.config) return;
    if (state.listening) {
      stopMic();
      s.send({ type: "stop_listening" });
      dispatch({ kind: "listening", value: false });
      return;
    }
    if (state.config.audio_io === "browser") {
      try {
        mic.current = await startMic(pcm => s.sendAudio(pcm));
      } catch (err) {
        const denied = err instanceof DOMException && (err.name === "NotAllowedError" || err.name === "SecurityError");
        dispatch({
          kind: "error",
          message: denied
            ? "Microphone permission denied — typing still works."
            : `Microphone unavailable (${err instanceof Error ? err.message : String(err)}) — typing still works.`,
        });
        return;
      }
    }
    s.send({ type: "start_listening" });
    dispatch({ kind: "listening", value: true });
  };

  const onSend = (text: string) => {
    player.current?.resume();
    socket.current?.send({ type: "send_text", text });
  };

  const onStop = () => {
    socket.current?.send({ type: "stop_speaking" });
    player.current?.flush();
  };

  const onSelectModel = (role: Role, slug: string) => {
    socket.current?.send({ type: "select_model", role, slug });
  };

  const onClear = () => {
    socket.current?.send({ type: "clear" });
    dispatch({ kind: "clear" });
    player.current?.flush();
  };

  return (
    <div className="app">
      <SettingsPanel config={state.config} settings={state.settings} onChange={onSettingsChange}
        catalog={state.catalog} switching={state.switching} status={state.status} onSelectModel={onSelectModel} />
      <main className="main">
        <ChatTranscript messages={state.messages} partial={state.partial} />
        <div className="dock">
          <StatusBar config={state.config} switchingSlug={state.switchingSlug} connected={state.connected} status={state.status} metrics={state.metrics} error={state.error}
            onDismissError={() => dispatch({ kind: "dismiss_error" })} />
          <Composer status={state.status} listening={state.listening} connected={state.connected}
            onToggleMic={() => void onToggleMic()} onSend={onSend} onStop={onStop} onClear={onClear} />
        </div>
      </main>
    </div>
  );
}
