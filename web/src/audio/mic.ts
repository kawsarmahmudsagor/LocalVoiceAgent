import { analyserLevel } from "./level";
import workletUrl from "./mic-worklet.ts?worker&url";

/** Capture the mic and deliver 16 kHz mono Int16 frames of 40 ms. */
export async function startMic(onFrame: (pcm: Int16Array) => void): Promise<{ stop(): void; level(): number }> {
  const media = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });
  const ctx = new AudioContext();
  try {
    await ctx.audioWorklet.addModule(workletUrl);
  } catch (e) {
    media.getTracks().forEach(t => t.stop());
    void ctx.close();
    throw e;
  }
  const source = ctx.createMediaStreamSource(media);
  // No outputs: the node is still processed, and nothing is played back.
  const node = new AudioWorkletNode(ctx, "mic-processor", { numberOfInputs: 1, numberOfOutputs: 0 });
  node.port.onmessage = (e: MessageEvent) => onFrame(e.data as Int16Array);
  source.connect(node);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);
  const levelBuf = new Float32Array(analyser.fftSize);
  return {
    /** The caller's current voice level, 0–1. */
    level: () => analyserLevel(analyser, levelBuf),
    stop() {
      source.disconnect();
      node.port.onmessage = null;
      media.getTracks().forEach(t => t.stop());
      void ctx.close();
    },
  };
}
