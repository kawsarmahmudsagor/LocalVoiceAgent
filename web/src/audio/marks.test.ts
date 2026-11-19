import { describe, expect, it } from "vitest";
import { PlaybackMarks } from "./marks";

describe("PlaybackMarks", () => {
  it("reports a mark once playback has passed its sample offset", () => {
    const m = new PlaybackMarks();
    m.add(1, 0); m.add(2, 100);
    expect(m.advance(0)).toEqual([]);          // silence: nothing played yet, even a mark at 0 waits
    expect(m.advance(50)).toEqual([1]);
    expect(m.advance(50)).toEqual([]);         // exactly at 100: sentence 2 has not started sounding
    expect(m.advance(1)).toEqual([2]);
    expect(m.advance(500)).toEqual([]);        // each mark fires once
  });
  it("reset restarts the clock and drops pending marks", () => {
    const m = new PlaybackMarks();
    m.add(1, 10); m.advance(5); m.reset();
    expect(m.advance(100)).toEqual([]);
    m.add(2, 0); expect(m.advance(1)).toEqual([2]);
  });
});
