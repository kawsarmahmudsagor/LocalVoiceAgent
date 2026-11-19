// AudioWorklet: plays queued Float32 chunks; silence when empty; "flush" clears the queue.
// Posts "started" when it begins outputting audio after the queue was empty.
// {type: "mark", id, at}: posts {type: "reached", ids} once playback passes sample `at` of the current reply.
// {type: "reset"}: a new reply starts; its marks count from the next sample played.
import { PlaybackMarks } from "./marks";

class PlayerProcessor extends AudioWorkletProcessor {
  private queue: Float32Array[] = [];
  private offset = 0;
  private wasEmpty = true;
  private marks = new PlaybackMarks();

  constructor() {
    super();
    this.port.onmessage = (e: MessageEvent) => {
      if (e.data === "flush") {
        this.queue = [];
        this.offset = 0;
        this.wasEmpty = true;
        this.marks.reset();
      } else if (e.data instanceof Float32Array) {
        this.queue.push(e.data);
      } else if (e.data?.type === "mark") {
        this.marks.add(e.data.id, e.data.at);
      } else if (e.data?.type === "reset") {
        this.marks.reset();
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
    const ids = this.marks.advance(i);
    if (ids.length) this.port.postMessage({ type: "reached", ids });
    if (wrote && this.wasEmpty) this.port.postMessage("started");
    this.wasEmpty = !wrote;
    return true;
  }
}

registerProcessor("player-processor", PlayerProcessor);
