// AudioWorklet: plays queued Float32 chunks; silence when empty; "flush" clears the queue.
// Posts "started" when it begins outputting audio after the queue was empty.

class PlayerProcessor extends AudioWorkletProcessor {
  private queue: Float32Array[] = [];
  private offset = 0;
  private wasEmpty = true;

  constructor() {
    super();
    this.port.onmessage = (e: MessageEvent) => {
      if (e.data === "flush") {
        this.queue = [];
        this.offset = 0;
        this.wasEmpty = true;
      } else if (e.data instanceof Float32Array) {
        this.queue.push(e.data);
      }
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const channels = outputs[0];
    const out = channels[0];
    let i = 0;
    while (i < out.length && this.queue.length) {
      const cur = this.queue[0];
      const n = Math.min(out.length - i, cur.length - this.offset);
      out.set(cur.subarray(this.offset, this.offset + n), i);
      i += n;
      this.offset += n;
      if (this.offset >= cur.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    out.fill(0, i);
    for (let c = 1; c < channels.length; c++) channels[c].set(out);
    const wrote = i > 0;
    if (wrote && this.wasEmpty) this.port.postMessage("started");
    this.wasEmpty = !wrote;
    return true;
  }
}

registerProcessor("player-processor", PlayerProcessor);
