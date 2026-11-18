export type Status = "idle" | "listening" | "thinking" | "speaking";
export interface OutputFormat { sample_rate: number; channels: number; encoding: "pcm_s16le" }
export interface Defaults { system_prompt: string; temperature: number; top_p: number; max_tokens: number; voice: string }
export interface ServerConfig { llm_backend: "transformers" | "llamacpp"; audio_io: "browser" | "server"; model_label: string;
  voices: string[]; output_format: OutputFormat | null; defaults: Defaults }
export const SPANS = ["silence_wait_ms", "stt_ms", "llm_ttft_ms", "filter_ms", "tts_first_audio_ms",
  "playback_ms", "perceived_ms", "llm_total_ms", "total_ms"] as const;
export type SpanName = (typeof SPANS)[number];
export interface Metrics { turn_id: number; source: "voice" | "text"; interrupted: boolean;
  marks: Record<string, number | null>; spans: Record<SpanName, number | null> }
export type ServerEvent = ({ type: "config" } & ServerConfig) | { type: "status"; state: Status }
  | { type: "user_transcript"; text: string; turn_id: number }
  | { type: "assistant_delta"; text: string } | { type: "assistant_done"; text: string; interrupted: boolean }
  | ({ type: "metrics" } & Metrics) | { type: "error"; code: string; message: string };
export type ClientCommand = { type: "start_listening" } | { type: "stop_listening" } | { type: "stop_speaking" }
  | { type: "send_text"; text: string } | ({ type: "update_settings" } & Partial<Defaults>) | { type: "clear" }
  | { type: "playback_started"; turn_id: number };
