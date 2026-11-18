export type Status = "idle" | "listening" | "thinking" | "speaking";
export interface OutputFormat { sample_rate: number; channels: number; encoding: "pcm_s16le" }
export interface Defaults { system_prompt: string; temperature: number; top_p: number; max_tokens: number; voice: string }
export type Role = "stt_final" | "stt_realtime" | "tts";
export interface CatalogEntry { slug: string; kind: "stt" | "tts"; role: "final" | "realtime" | null; device: "cpu" | "cuda";
  vram_mb: number; runtime: "inprocess" | "worker"; streaming: boolean }
/** v2 fields (stt_final … llm_vram_mb) are absent when the server runs without a model catalog. */
export interface ServerConfig { llm_backend: "transformers" | "llamacpp"; audio_io: "browser" | "server"; model_label: string;
  voices: string[]; output_format: OutputFormat | null; defaults: Defaults;
  stt_final?: string; stt_realtime?: string; tts?: string; gpu_total_mb?: number; llm_vram_mb?: number }
export const SPANS = ["silence_wait_ms", "stt_ms", "llm_ttft_ms", "filter_ms", "tts_first_audio_ms",
  "playback_ms", "perceived_ms", "llm_total_ms", "total_ms"] as const;
export type SpanName = (typeof SPANS)[number];
export interface Metrics { turn_id: number; source: "voice" | "text"; interrupted: boolean;
  marks: Record<string, number | null>; spans: Record<SpanName, number | null> }
export type ServerEvent = ({ type: "config" } & ServerConfig) | { type: "status"; state: Status }
  | { type: "user_transcript"; text: string; turn_id: number }
  | { type: "assistant_delta"; text: string } | { type: "assistant_done"; text: string; interrupted: boolean }
  | ({ type: "metrics" } & Metrics) | { type: "error"; code: string; message: string }
  | { type: "catalog"; entries: CatalogEntry[] }
  | { type: "model_status"; role: Role; slug: string; state: "loading" | "ready" | "failed"; message?: string; load_ms?: number }
  | { type: "user_partial"; text: string };
export type ClientCommand = { type: "start_listening" } | { type: "stop_listening" } | { type: "stop_speaking" }
  | { type: "send_text"; text: string } | ({ type: "update_settings" } & Partial<Defaults>) | { type: "clear" }
  | { type: "playback_started"; turn_id: number } | { type: "select_model"; role: Role; slug: string };
