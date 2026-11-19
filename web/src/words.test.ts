import { describe, expect, it } from "vitest";
import { SentenceWords, wordSchedule } from "./words";

describe("wordSchedule", () => {
  it("uses the engine's word times when there is one per word", () => {
    expect(wordSchedule("Hi there, friend.", 1, [1, 1.2, 1.5], 2)).toEqual([
      { at: 1, word: "Hi" }, { at: 1.2, word: "there," }, { at: 1.5, word: "friend." }]);
  });

  it("spreads the words over the sentence's audio by length when there are no word times", () => {
    // 2 + 5 + 3 letters over 1 s: words start at 0, 0.2 and 0.7 of the way.
    expect(wordSchedule("ab abcde abc", 2, null, 3)).toEqual([
      { at: 2, word: "ab" }, { at: 2.2, word: "abcde" }, { at: 2.7, word: "abc" }]);
  });

  it("spreads the words when the word times don't match the words", () => {
    expect(wordSchedule("one two", 0, [0], 1).map(w => w.at)).toEqual([0, 0.5]);
  });

  it("returns nothing for an empty sentence", () => {
    expect(wordSchedule("  ", 0, null, 1)).toEqual([]);
  });
});

describe("SentenceWords", () => {
  it("pairs each word list with the sentence before it", () => {
    const f = new SentenceWords();
    expect(f.segment({ turn_id: 1, text: "Hi there.", at_s: 0 })).toEqual([]);
    expect(f.words({ turn_id: 1, word_s: [0, 0.3], end_s: 0.5 })).toEqual([
      { turn_id: 1, at: 0, word: "Hi" }, { turn_id: 1, at: 0.3, word: "there." }]);
    expect(f.words({ turn_id: 1, word_s: null, end_s: 1 })).toEqual([]);   // no sentence waiting
  });

  it("shows a sentence whole if its words never came", () => {
    const f = new SentenceWords();
    f.segment({ turn_id: 1, text: "Lost words.", at_s: 0.4 });
    expect(f.segment({ turn_id: 1, text: "Next.", at_s: 1 })).toEqual([{ turn_id: 1, at: 0.4, word: "Lost words." }]);
    expect(f.words({ turn_id: 1, word_s: null, end_s: 1.5 })).toEqual([{ turn_id: 1, at: 1, word: "Next." }]);
  });

  it("forgets a waiting sentence on reset", () => {
    const f = new SentenceWords();
    f.segment({ turn_id: 1, text: "Old.", at_s: 0 }); f.reset();
    expect(f.words({ turn_id: 1, word_s: null, end_s: 1 })).toEqual([]);
  });
});
