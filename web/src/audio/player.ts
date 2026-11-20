import type { OutputFormat } from "../protocol";
import { analyserLevel } from "./level";
import { pcm16ToFloat, resample } from "./pcm";
import workletUrl from "./player-worklet.ts?worker&url";

export interface Player {
  push(buf: ArrayBuffer): void;
  flush(): void;
  close(): void;
  /** Resume the AudioContext; call from a user gesture (browsers block autoplay). */
  resume(): void;
  /** A new reply starts: schedule() offsets count from its first sample. */
  resetClock(): void;
  /** Call cb once playback of the current reply passes atSeconds. flush() drops pending callbacks. */
  schedule(atSeconds: number, cb: () => void): void;
  /** Speaker off: silence the output only, so word timing and playback_started keep running. */
  setMuted(muted: boolean): void;
  /** The reply's current level, 0–1 (measured before the mute, so it moves while muted). */
  level(): number;
}

/** Streams server TTS PCM (s16le at format.sample_rate) to the speakers. */
export function createPlayer(format: OutputFormat, onStarted: () => void): Player {
  const ctx = new AudioContext();
  let node: AudioWorkletNode | null = null;
  const pending: (Float32Array | object)[] = [];
  const callbacks = new Map<number, () => void>();
  let nextMark = 1;
  // worklet → analyser (level), worklet → gain (speaker mute) → speakers.
  const gain = ctx.createGain();
  gain.connect(ctx.destination);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  const levelBuf = new Float32Array(analyser.fftSize);

  const post = (m: Float32Array | object) =>
    m instanceof Float32Array ? node!.port.postMessage(m, [m.buffer]) : node!.port.postMessage(m);
  const send = (m: Float32Array | object) => (node ? post(m) : pending.push(m));

  void ctx.audioWorklet.addModule(workletUrl).then(() => {
    node = new AudioWorkletNode(ctx, "player-processor", { numberOfInputs: 0, outputChannelCount: [1] });
    node.port.onmessage = (e: MessageEvent) => {
      if (e.data === "started") onStarted();
      else if (e.data?.type === "reached") {
        for (const id of e.data.ids as number[]) {
          const cb = callbacks.get(id);
          callbacks.delete(id);
          cb?.();
        }
      }
    };
    node.connect(gain);
    node.connect(analyser);
    pending.splice(0).forEach(post);
  });

  return {
    push(buf) {
      if (ctx.state === "suspended") void ctx.resume();
      send(resample(pcm16ToFloat(buf), format.sample_rate, ctx.sampleRate));
    },
    flush() {
      pending.length = 0;
      callbacks.clear();
      node?.port.postMessage("flush");
    },
    resetClock() {
      send({ type: "reset" });
    },
    schedule(atSeconds, cb) {
      const id = nextMark++;
      callbacks.set(id, cb);
      send({ type: "mark", id, at: Math.round(atSeconds * ctx.sampleRate) });
    },
    close() {
      void ctx.close();
    },
    resume() {
      if (ctx.state === "suspended") void ctx.resume();
    },
    setMuted(muted) {
      gain.gain.value = muted ? 0 : 1;
    },
    level() {
      return node ? analyserLevel(analyser, levelBuf) : 0;
    },
  };
}
