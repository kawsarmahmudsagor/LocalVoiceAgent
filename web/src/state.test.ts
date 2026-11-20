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
  it("config overrides the local barge-in settings (the server may fall back to onset)", () => {
    let s = srv(initialState, { ...cfg, defaults: { ...cfg.defaults, barge_in_mode: "words", barge_in_min_ms: 200 } });
    s = reducer(s, { kind: "settings", patch: { temperature: 0.1 } });
    s = srv(s, { ...cfg, defaults: { ...cfg.defaults, barge_in_mode: "onset", barge_in_min_ms: 300 } });
    expect(s.settings?.barge_in_mode).toBe("onset"); expect(s.settings?.barge_in_min_ms).toBe(300);
    expect(s.settings?.temperature).toBe(0.1);            // other local edits stay
    s = srv(s, cfg);                                      // a config without them keeps them
    expect(s.settings?.barge_in_mode).toBe("onset");
  });
  it("tracks forced word alignment from config and word_align events", () => {
    let s = srv(initialState, cfg); expect(s.wordAlign).toEqual({ state: "off", message: null });
    s = srv(initialState, { ...cfg, defaults: { ...cfg.defaults, word_align: true } });
    expect(s.wordAlign.state).toBe("ready");
    s = srv(s, { type: "word_align", state: "loading" }); expect(s.wordAlign.state).toBe("loading");
    s = srv(s, { type: "word_align", state: "error", message: "no weights" });
    expect(s.wordAlign).toEqual({ state: "error", message: "no weights" });
    s = srv(s, cfg);                                    // a reconnect's config doesn't hide the error
    expect(s.wordAlign.state).toBe("error");
  });
  it("tags speakers, keeps ignored transcripts out of the turn, tracks diarization", () => {
    let s = srv(initialState, { ...cfg, defaults: { ...cfg.defaults, diarization: true } });
    expect(s.diarization.state).toBe("ready");
    s = srv(s, { type: "user_transcript", text: "hi", turn_id: 4, speaker: 0 });
    expect(s.messages[0]).toMatchObject({ text: "hi", speaker: 0 }); expect(s.turnId).toBe(4);
    s = srv(s, { type: "user_partial", text: "not" });
    s = srv(s, { type: "user_transcript", text: "not me", turn_id: null, speaker: 1, ignored: true });
    expect(s.messages[1]).toMatchObject({ text: "not me", speaker: 1, ignored: true });
    expect(s.turnId).toBe(4); expect(s.partial).toBeNull();
    s = srv(s, { type: "diarization", state: "error", message: "oom" });
    expect(s.diarization).toEqual({ state: "error", message: "oom" });
    s = srv(s, cfg); expect(s.diarization.state).toBe("error");          // a reconnect keeps the error
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
    s = srv(s, { type: "assistant_done", text: "Hello. Bye.", interrupted: false, heard: null });
    expect(s.messages[1]).toMatchObject({ text: "Hello. Bye.", streaming: false, interrupted: false });
    expect(reducer(s, { kind: "spoken", turn_id: 3, text: "Late." }).messages[1].text).toBe("Hello. Bye.");
  });
  it("an interrupted reply keeps only what was heard", () => {
    let s = srv(initialState, { type: "user_transcript", text: "hi", turn_id: 1 });
    s = srv(s, { type: "assistant_delta", text: "One. Two." }); s = reducer(s, { kind: "spoken", turn_id: 1, text: "One." });
    s = srv(s, { type: "assistant_done", text: "One. Two.", interrupted: true, heard: null });
    expect(s.messages[1]).toMatchObject({ text: "One.", streaming: false, interrupted: true });
  });
  it("assistant_done without deltas adds a message if text non-empty", () => {
    expect(srv(initialState, { type: "assistant_done", text: "Sorry.", interrupted: false, heard: null }).messages).toHaveLength(1);
    expect(srv(initialState, { type: "assistant_done", text: "", interrupted: false, heard: null }).messages).toHaveLength(0);
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
  it("rag_sources attach to the current turn's reply", () => {
    let s = srv(initialState, { type: "user_transcript", text: "q", turn_id: 2 });
    s = srv(s, { type: "assistant_delta", text: "x" });
    s = srv(s, { type: "rag_sources", turn_id: 2, sources: [{ name: "a.txt", chunk: 0, score: 0.8, preview: "p" }] });
    expect(s.messages[1].sources?.[0].name).toBe("a.txt");
    expect(srv(s, { type: "rag_sources", turn_id: 1, sources: [] }).messages[1].sources).toHaveLength(1);
  });
  it("rag_sources arriving first open the bubble", () => {
    let s = srv(initialState, { type: "user_transcript", text: "q", turn_id: 2 });
    s = srv(s, { type: "rag_sources", turn_id: 2, sources: [{ name: "a.txt", chunk: 0, score: 0.8, preview: "p" }] });
    expect(s.messages[1]).toMatchObject({ role: "assistant", streaming: true, text: "" });
    s = srv(s, { type: "assistant_delta", text: "x" });
    expect(s.messages).toHaveLength(2);
  });
  it("rag_config and rag_status fill rag state", () => {
    const RAG_DEFAULTS = { enabled: true, top_k: 4, min_score: 0.3, max_context_chars: 3000, chunk_size: 800, chunk_overlap: 100, embedder: "e" };
    const DOC = { doc_id: "d", name: "a.txt", ext: ".txt", size: 1, state: "indexed", chunks: 1 };
    let s = srv(initialState, { type: "rag_config", settings: RAG_DEFAULTS, embedders: [], collection: null });
    s = srv(s, { type: "rag_status", documents: [DOC], indexing: { done: 1, total: 2 } });
    expect(s.rag?.documents).toEqual([DOC]); expect(s.rag?.indexing).toEqual({ done: 1, total: 2 });
    s = srv(s, { type: "rag_config", settings: RAG_DEFAULTS, embedders: [], collection: null, error: "boom" });
    expect(s.rag?.documents).toEqual([DOC]); expect(s.rag?.error).toBe("boom");
  });
  it("tab action switches tabs", () => {
    expect(initialState.tab).toBe("playground");
    expect(reducer(initialState, { kind: "tab", tab: "rag" }).tab).toBe("rag");
  });
  it("an interrupted reply keeps the unheard tail separately", () => {
    let s = srv(initialState, { type: "user_transcript", text: "hi", turn_id: 1 });
    s = srv(s, { type: "assistant_delta", text: "One." });
    s = reducer(s, { kind: "spoken", turn_id: 1, text: "One." });
    s = srv(s, { type: "assistant_done", text: "One. Two three.", interrupted: true, heard: "One." });
    expect(s.messages[1]).toMatchObject({ text: "One.", unheard: "Two three.", interrupted: true });
  });
  it("barge_in leaves state unchanged", () => {
    expect(srv(initialState, { type: "barge_in", turn_id: 1 })).toBe(initialState);
  });
});
