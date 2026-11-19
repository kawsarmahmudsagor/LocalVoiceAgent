/** Sample-accurate playback marks for the player worklet: each reply sentence is shown when playback
 * passes the sample where its audio starts. Only samples actually played advance the clock, so gaps
 * while the next sentence is still being synthesized do not shift the marks. */
export class PlaybackMarks {
  private played = 0;
  private marks: { id: number; at: number }[] = [];

  reset(): void {
    this.played = 0;
    this.marks = [];
  }

  add(id: number, atSample: number): void {
    this.marks.push({ id, at: atSample });
  }

  /** Count n more played samples; returns the ids whose start sample has now sounded, in order. */
  advance(n: number): number[] {
    this.played += n;
    const reached = this.marks.filter(m => m.at < this.played).map(m => m.id);
    if (reached.length) this.marks = this.marks.filter(m => m.at >= this.played);
    return reached;
  }
}
