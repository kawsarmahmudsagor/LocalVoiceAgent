// AudioWorklet: mic Float32 at the context rate -> 16 kHz Int16 frames of 40 ms (640 samples).
import { floatTo16BitPCM, resample } from "./pcm";

const TARGET_RATE = 16000;
const FRAME_SECONDS = 0.04;

class MicProcessor extends AudioWorkletProcessor {
  // Collect 40 ms at the input rate, then resample the whole block, so every
  // frame is exactly 640 samples.
  private readonly inFrame = Math.round(sampleRate * FRAME_SECONDS);
  private readonly outFrame = Math.round(TARGET_RATE * FRAME_SECONDS);
  private buf = new Float32Array(this.inFrame);
  private fill = 0;

  process(inputs: Float32Array[][]): boolean {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    let i = 0;
    while (i < ch.length) {
      const n = Math.min(ch.length - i, this.inFrame - this.fill);
      this.buf.set(ch.subarray(i, i + n), this.fill);
      this.fill += n;
      i += n;
      if (this.fill === this.inFrame) {
        const pcm = floatTo16BitPCM(resample(this.buf, sampleRate, TARGET_RATE).subarray(0, this.outFrame));
        this.port.postMessage(pcm, [pcm.buffer as ArrayBuffer]);
        this.fill = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
