import type { OutputFormat } from "../protocol";
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
}

/** Streams server TTS PCM (s16le at format.sample_rate) to the speakers. */
export function createPlayer(format: OutputFormat, onStarted: () => void): Player {
  const ctx = new AudioContext();
  let node: AudioWorkletNode | null = null;
  const pending: (Float32Array | object)[] = [];
  const callbacks = new Map<number, () => void>();
  let nextMark = 1;

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
    node.connect(ctx.destination);
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
  };
}
