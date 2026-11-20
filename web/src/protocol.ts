export type WordAlignState = "off" | "loading" | "ready" | "error";
export type Status = "idle" | "listening" | "thinking" | "speaking";
export interface OutputFormat { sample_rate: number; channels: number; encoding: "pcm_s16le" }
export type BargeInMode = "off" | "onset" | "words";
export interface Defaults { system_prompt: string; temperature: number; top_p: number; max_tokens: number; voice: string; rag_template: string;
  /** Forced word alignment for TTS engines without word timings (absent: an older server). */
  word_align?: boolean;
  barge_in_mode?: BargeInMode; barge_in_min_ms?: number }
export type Role = "stt_final" | "stt_realtime" | "tts";
export interface CatalogEntry { slug: string; kind: "stt" | "tts"; role: "final" | "realtime" | null; device: "cpu" | "cuda";
  vram_mb: number; runtime: "inprocess" | "worker"; streaming: boolean }
/** v2 fields (stt_final … llm_vram_mb) are absent when the server runs without a model catalog. */
export interface ServerConfig { llm_backend: "transformers" | "llamacpp"; audio_io: "browser" | "server"; model_label: string;
  voices: string[]; output_format: OutputFormat | null; defaults: Defaults;
  stt_final?: string; stt_realtime?: string; tts?: string; gpu_total_mb?: number; llm_vram_mb?: number; barge_words_available?: boolean }
export interface RagSettings { enabled: boolean; top_k: number; min_score: number; max_context_chars: number;
  chunk_size: number; chunk_overlap: number; embedder: string }
export interface RagEmbedder { slug: string; available: boolean; reason?: string }
export interface RagCollection { embedder: string; dim: number | null; state: "indexed" | "needs re-index" | "indexing";
  chunk_size: number; chunk_overlap: number }
export interface RagDocument { doc_id: string; name: string; ext: string; size: number; state: string; chunks: number }
export interface RagSource { name: string; chunk: number; score: number; preview: string }
export const SPANS = ["silence_wait_ms", "stt_ms", "retrieval_ms", "llm_ttft_ms", "filter_ms", "tts_first_audio_ms",
  "playback_ms", "perceived_ms", "llm_total_ms", "total_ms"] as const;
export type SpanName = (typeof SPANS)[number];
export interface Metrics { turn_id: number; source: "voice" | "text"; interrupted: boolean;
  marks: Record<string, number | null>; spans: Record<SpanName, number | null> }
export type ServerEvent = ({ type: "config" } & ServerConfig) | { type: "status"; state: Status }
  | { type: "user_transcript"; text: string; turn_id: number }
  | { type: "assistant_delta"; text: string }
  /** One sentence of the reply; at_s is where its audio starts in the reply (null: no browser playback). */
  | { type: "assistant_segment"; turn_id: number; text: string; at_s: number | null }
  /** After a sentence is synthesized: when each of its words starts (null: spread them up to end_s). */
  | { type: "assistant_words"; turn_id: number; word_s: number[] | null; end_s: number }
  | { type: "word_align"; state: WordAlignState; message?: string } | { type: "assistant_done"; text: string; interrupted: boolean; heard: string | null }
  | { type: "barge_in"; turn_id: number }
  | ({ type: "metrics" } & Metrics) | { type: "error"; code: string; message: string }
  | { type: "catalog"; entries: CatalogEntry[] }
  | { type: "model_status"; role: Role; slug: string; state: "loading" | "ready" | "failed"; message?: string; load_ms?: number }
  | { type: "user_partial"; text: string }
  | { type: "rag_config"; settings: RagSettings; embedders: RagEmbedder[]; collection: RagCollection | null; error?: string }
  | { type: "rag_status"; documents: RagDocument[]; indexing: { done: number; total: number } | null }
  | { type: "rag_sources"; turn_id: number; sources: RagSource[] };
export type ClientCommand = { type: "start_listening" } | { type: "stop_listening" } | { type: "stop_speaking" }
  | { type: "send_text"; text: string } | ({ type: "update_settings" } & Partial<Defaults>) | { type: "clear" }
  | { type: "playback_started"; turn_id: number }
  | { type: "playback_cut"; turn_id: number; heard: string } | { type: "select_model"; role: Role; slug: string }
  | ({ type: "update_rag" } & Partial<RagSettings>);
