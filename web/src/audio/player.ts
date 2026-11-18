import type { OutputFormat } from "../protocol";
import { pcm16ToFloat, resample } from "./pcm";
import workletUrl from "./player-worklet.ts?worker&url";

export interface Player {
  push(buf: ArrayBuffer): void;
  flush(): void;
  close(): void;
  /** Resume the AudioContext; call from a user gesture (browsers block autoplay). */
  resume(): void;
}

/** Streams server TTS PCM (s16le at format.sample_rate) to the speakers. */
export function createPlayer(format: OutputFormat, onStarted: () => void): Player {
  const ctx = new AudioContext();
  let node: AudioWorkletNode | null = null;
  const pending: Float32Array[] = [];

  const post = (f: Float32Array) => node!.port.postMessage(f, [f.buffer]);

  void ctx.audioWorklet.addModule(workletUrl).then(() => {
    node = new AudioWorkletNode(ctx, "player-processor", { numberOfInputs: 0, outputChannelCount: [1] });
    node.port.onmessage = (e: MessageEvent) => {
      if (e.data === "started") onStarted();
    };
    node.connect(ctx.destination);
    pending.splice(0).forEach(post);
  });

  return {
    push(buf) {
      if (ctx.state === "suspended") void ctx.resume();
      const f = resample(pcm16ToFloat(buf), format.sample_rate, ctx.sampleRate);
      if (node) post(f);
      else pending.push(f);
    },
    flush() {
      pending.length = 0;
      node?.port.postMessage("flush");
    },
    close() {
      void ctx.close();
    },
    resume() {
      if (ctx.state === "suspended") void ctx.resume();
    },
  };
}
