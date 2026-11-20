// Word-by-word reveal of the reply, timed to its audio.
// The server sends each sentence (assistant_segment: text and where its audio starts) and, once it is
// synthesized, when its words start (assistant_words). Engines without word times send word_s = null;
// the words are then spread over the sentence's audio in proportion to their length.

export interface TimedWord { at: number; word: string }

/** When each whitespace word of `text` starts, in seconds on the reply clock. */
export function wordSchedule(text: string, atS: number, wordS: number[] | null, endS: number): TimedWord[] {
  const words = text.split(/\s+/).filter(Boolean);
  if (wordS && wordS.length === words.length) return words.map((word, i) => ({ at: wordS[i], word }));
  const total = words.reduce((n, w) => n + w.length, 0);
  const span = Math.max(0, endS - atS);
  let before = 0;
  return words.map(word => {
    const at = atS + (total ? (span * before) / total : 0);
    before += word.length;
    return { at, word };
  });
}

interface Segment { turn_id: number; text: string; at_s: number }
export interface TurnWord extends TimedWord { turn_id: number }

/** Pairs each assistant_words event with the sentence sent before it (sentences are synthesized in order). */
export class SentenceWords {
  private waiting: Segment | null = null;

  /** A new sentence. Returns the previous one whole if its words never arrived (e.g. synthesis failed). */
  segment(s: Segment): TurnWord[] {
    const stale = this.waiting;
    this.waiting = s;
    return stale ? [{ turn_id: stale.turn_id, at: stale.at_s, word: stale.text.trim() }] : [];
  }

  /** The waiting sentence's word times: the words to show and when. */
  words(w: { turn_id: number; word_s: number[] | null; end_s: number }): TurnWord[] {
    const s = this.waiting;
    if (!s || s.turn_id !== w.turn_id) return [];
    this.waiting = null;
    return wordSchedule(s.text, s.at_s, w.word_s, w.end_s).map(x => ({ ...x, turn_id: s.turn_id }));
  }

  reset(): void {
    this.waiting = null;
  }
}

/** The words of the current turn's reply that have been revealed (heard) so far. */
export class HeardText {
  private turn: number | null = null;
  private words: string[] = [];

  reset(turnId: number): void {
    this.turn = turnId;
    this.words = [];
  }

  add(turnId: number, word: string): void {
    if (turnId !== this.turn) return;
    this.words.push(word);
  }

  text(turnId: number): string {
    return turnId === this.turn ? this.words.join(" ").trim() : "";
  }
}
