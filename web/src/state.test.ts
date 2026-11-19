import { describe, it, expect } from "vitest";
import { reducer, initialState } from "./state";
const cfg = { type: "config", llm_backend: "llamacpp", audio_io: "browser", model_label: "m", voices: ["af_heart"], output_format: null,
  defaults: { system_prompt: "S", temperature: 0.7, top_p: 0.8, max_tokens: 256, voice: "af_heart" } } as const;
const srv = (s: any, event: any) => reducer(s, { kind: "server", event });
describe("reducer", () => {
  it("config sets config and settings once", () => {
    const s = srv(initialState, cfg); expect(s.settings?.temperature).toBe(0.7);
    const s2 = srv(reducer(s, { kind: "settings", patch: { temperature: 0.1 } }), cfg); expect(s2.settings?.temperature).toBe(0.1);
  });
  it("shows sentences as they are spoken, not as the LLM writes them", () => {
    let s = srv(initialState, { type: "user_transcript", text: "hi", turn_id: 3 }); expect(s.turnId).toBe(3);
    s = srv(s, { type: "assistant_delta", text: "Hel" }); s = srv(s, { type: "assistant_delta", text: "lo. Bye." });
    expect(s.messages.map(m => [m.role, m.text, m.streaming])).toEqual([["user", "hi", false], ["assistant", "", true]]);
    s = srv(s, { type: "assistant_segment", turn_id: 3, text: "Hello.", at_s: 0 });
    expect(s.messages[1].text).toBe("");                      // segments wait for playback
    s = reducer(s, { kind: "spoken", turn_id: 3, text: "Hello." });
    s = reducer(s, { kind: "spoken", turn_id: 2, text: "Stale." });   // an older turn's sentence
    expect(s.messages[1].text).toBe("Hello.");
    s = reducer(s, { kind: "spoken", turn_id: 3, text: "Bye." }); expect(s.messages[1].text).toBe("Hello. Bye.");
    s = srv(s, { type: "assistant_done", text: "Hello. Bye.", interrupted: false });
    expect(s.messages[1]).toMatchObject({ text: "Hello. Bye.", streaming: false, interrupted: false });
    expect(reducer(s, { kind: "spoken", turn_id: 3, text: "Late." }).messages[1].text).toBe("Hello. Bye.");
  });
  it("an interrupted reply keeps only what was heard", () => {
    let s = srv(initialState, { type: "user_transcript", text: "hi", turn_id: 1 });
    s = srv(s, { type: "assistant_delta", text: "One. Two." }); s = reducer(s, { kind: "spoken", turn_id: 1, text: "One." });
    s = srv(s, { type: "assistant_done", text: "One. Two.", interrupted: true });
    expect(s.messages[1]).toMatchObject({ text: "One.", streaming: false, interrupted: true });
  });
  it("assistant_done without deltas adds a message if text non-empty", () => {
    expect(srv(initialState, { type: "assistant_done", text: "Sorry.", interrupted: false }).messages).toHaveLength(1);
    expect(srv(initialState, { type: "assistant_done", text: "", interrupted: false }).messages).toHaveLength(0);
  });
  it("status, metrics, error, clear, dismiss", () => {
    let s = srv(initialState, { type: "status", state: "thinking" }); expect(s.status).toBe("thinking");
    s = srv(s, { type: "metrics", turn_id: 1, source: "text", interrupted: false, marks: {},
      spans: { silence_wait_ms: null, stt_ms: null, llm_ttft_ms: 300, filter_ms: 0, tts_first_audio_ms: 200,
               playback_ms: 30, perceived_ms: 530, llm_total_ms: 900, total_ms: 2000 } });
    expect(s.metrics?.spans.perceived_ms).toBe(530);
    s = srv(s, { type: "error", code: "busy", message: "Busy" }); expect(s.error).toBe("Busy");
    s = reducer(reducer(s, { kind: "dismiss_error" }), { kind: "clear" }); expect(s.error).toBeNull(); expect(s.messages).toEqual([]);
  });
  it("model_status, partial and catalog", () => {
    let s = srv(initialState, { type: "catalog", entries: [{ slug: "kokoro" }] }); expect(s.catalog).toHaveLength(1);
    s = srv(s, { type: "model_status", role: "tts", slug: "piper", state: "loading" }); expect(s.switching).toBe("tts");
    s = srv(s, { type: "model_status", role: "tts", slug: "piper", state: "failed", message: "no" });
    expect(s.switching).toBeNull(); expect(s.error).toBe("no");
    s = srv(s, { type: "user_partial", text: "hel" }); expect(s.partial).toBe("hel");
    s = srv(s, { type: "user_transcript", text: "hello", turn_id: 1 }); expect(s.partial).toBeNull();
  });
});
