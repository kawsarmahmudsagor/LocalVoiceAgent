import type { CatalogEntry, ServerConfig } from "./protocol";

/** GPU memory of the LLM plus the selected final, realtime (unless off) and TTS models (spec §4.8). */
export function vramUsed(cfg: ServerConfig, catalog: CatalogEntry[]): { used: number; total: number; over: boolean } {
  const mb = (slug: string | undefined) => (slug && slug !== "off" ? catalog.find(e => e.slug === slug)?.vram_mb ?? 0 : 0);
  const used = (cfg.llm_vram_mb ?? 0) + mb(cfg.stt_final) + mb(cfg.stt_realtime) + mb(cfg.tts);
  const total = cfg.gpu_total_mb ?? 0;
  return { used, total, over: used > total };
}
