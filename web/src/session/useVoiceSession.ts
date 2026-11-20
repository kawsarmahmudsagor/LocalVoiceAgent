import { useCallback, useEffect, useReducer, useRef, type Dispatch } from "react";
import { startMic } from "../audio/mic";
import { createPlayer, type Player } from "../audio/player";
import { createExclusive } from "../exclusive";
import type { Defaults, RagSettings, Role } from "../protocol";
import { initialState, reducer, type Action, type AppState } from "../state";
import { VoiceSocket, socketUrl } from "../ws";
import { HeardText, SentenceWords, type TurnWord } from "../words";

const SETTINGS_DEBOUNCE_MS = 300;

/** One voice-bot session: the socket, mic, player and the reducer state. Both pages use it. */
export interface VoiceSession {
  state: AppState;
  dispatch: Dispatch<Action>;
  /** Exclusive-guarded; resumes the player first (user gesture). */
  toggleMic(): Promise<void>;
  send(text: string): void;
  /** stop_speaking + cut playback. */
  stop(): void;
  /** clear + flush. */
  clear(): void;
  selectModel(role: Role, slug: string): void;
  /** Dispatches "settings" and sends a debounced update_settings. */
  updateSettings(patch: Partial<Defaults>): void;
  updateRag(patch: Partial<RagSettings>): void;
  resumeAudio(): void;
  /** Remembered and re-applied when the player is recreated on a sample-rate change. */
  setSpeakerMuted(muted: boolean): void;
  agentLevel(): number;
  userLevel(): number;
}

export function useVoiceSession(init: (s: AppState) => AppState = s => s): VoiceSession {
  const [state, dispatch] = useReducer(reducer, initialState, init);
  const stateRef = useRef(state);
  stateRef.current = state;
  const socket = useRef<VoiceSocket | null>(null);
  const player = useRef<Player | null>(null);
  const playerRate = useRef<number | null>(null);
  const speakerMuted = useRef(false);
  const mic = useRef<{ stop(): void; level(): number } | null>(null);
  const turnId = useRef<number | null>(null);
  const sentenceWords = useRef(new SentenceWords());
  const heard = useRef(new HeardText());
  const ackedTurn = useRef<number | null>(null);
  const pendingPatch = useRef<Partial<Defaults>>({});
  const debounce = useRef<number | undefined>(undefined);
  // A second click while the permission prompt is open would start a second capture.
  const micExclusive = useRef(createExclusive()).current;

  const stopMic = useCallback(() => {
    mic.current?.stop();
    mic.current = null;
  }, []);

  // Silence the player and tell the server which words the user actually heard.
  const cutPlayback = useCallback((id: number) => {
    player.current?.flush();
    sentenceWords.current.reset();
    socket.current?.send({ type: "playback_cut", turn_id: id, heard: heard.current.text(id) });
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
        sentenceWords.current.reset();
        dispatch({ kind: "connected", value: false });
      },
      onEvent: e => {
        if (e.type === "config" && e.output_format && playerRate.current !== e.output_format.sample_rate) {
          // First config, or a TTS switch that changed the sample rate: never play at the wrong rate.
          player.current?.close();
          player.current = createPlayer(e.output_format, ackPlayback);
          player.current.setMuted(speakerMuted.current);
          playerRate.current = e.output_format.sample_rate;
        }
        if (e.type === "user_transcript" && !e.ignored && e.turn_id != null) {
          turnId.current = e.turn_id;
          player.current?.resetClock(); // the previous reply has finished playing before a new turn starts
          sentenceWords.current.reset();
          heard.current.reset(e.turn_id);
        }
        if (e.type === "barge_in" && e.turn_id === turnId.current) cutPlayback(e.turn_id);
        // The reply appears word by word as its audio plays: each word is shown when playback reaches it.
        const show = (words: TurnWord[]) => {
          for (const w of words) {
            const spoken = () => {
              heard.current.add(w.turn_id, w.word);
              dispatch({ kind: "spoken", turn_id: w.turn_id, text: w.word });
            };
            if (player.current) player.current.schedule(w.at, spoken);
            else spoken();
          }
        };
        if (e.type === "assistant_segment") {
          if (e.at_s == null) { // PC speakers: at once
            heard.current.add(e.turn_id, e.text);
            dispatch({ kind: "spoken", turn_id: e.turn_id, text: e.text });
          } else show(sentenceWords.current.segment({ turn_id: e.turn_id, text: e.text, at_s: e.at_s }));
          return;
        }
        if (e.type === "assistant_words") {
          show(sentenceWords.current.words(e));
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
  }, [stopMic, cutPlayback]);

  const toggleMicNow = async () => {
    player.current?.resume();
    const s = socket.current;
    const { config, listening } = stateRef.current;
    if (!s || !config) return;
    if (listening) {
      stopMic();
      s.send({ type: "stop_listening" });
      dispatch({ kind: "listening", value: false });
      return;
    }
    if (config.audio_io === "browser") {
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

  return {
    state,
    dispatch,
    toggleMic: () => micExclusive(toggleMicNow),
    send(text) {
      player.current?.resume();
      socket.current?.send({ type: "send_text", text });
    },
    stop() {
      socket.current?.send({ type: "stop_speaking" });
      if (turnId.current != null) cutPlayback(turnId.current);
      else {
        player.current?.flush();
        sentenceWords.current.reset();
      }
    },
    clear() {
      socket.current?.send({ type: "clear" });
      dispatch({ kind: "clear" });
      player.current?.flush();
      sentenceWords.current.reset();
    },
    selectModel(role, slug) {
      socket.current?.send({ type: "select_model", role, slug });
    },
    updateSettings(patch) {
      dispatch({ kind: "settings", patch });
      pendingPatch.current = { ...pendingPatch.current, ...patch };
      window.clearTimeout(debounce.current);
      debounce.current = window.setTimeout(() => {
        socket.current?.send({ type: "update_settings", ...pendingPatch.current });
        pendingPatch.current = {};
      }, SETTINGS_DEBOUNCE_MS);
    },
    updateRag(patch) {
      socket.current?.send({ type: "update_rag", ...patch });
    },
    resumeAudio() {
      player.current?.resume();
    },
    setSpeakerMuted(muted) {
      speakerMuted.current = muted;
      player.current?.setMuted(muted);
    },
    agentLevel: () => player.current?.level() ?? 0,
    userLevel: () => mic.current?.level() ?? 0,
  };
}
