import type { Metrics, ServerConfig, SpanName, Status } from "../protocol";

interface Props {
  config?: ServerConfig | null;
  switchingSlug?: string | null;
  connected: boolean;
  status: Status;
  metrics: Metrics | null;
  error: string | null;
  onDismissError(): void;
}

const BREAKDOWN: [SpanName, string][] = [
  ["silence_wait_ms", "silence"],
  ["stt_ms", "stt"],
  ["diar_wait_ms", "diar"],
  ["retrieval_ms", "rag"],
  ["llm_ttft_ms", "llm"],
  ["filter_ms", "filter"],
  ["tts_first_audio_ms", "tts"],
  ["playback_ms", "play"],
];

function tooltip(m: Metrics, config?: ServerConfig | null): string {
  const fmt = (o: Record<string, number | null>) =>
    Object.entries(o).map(([k, v]) => `  ${k}: ${v ?? "-"}`).join("\n");
  const models = config?.stt_final !== undefined
    ? `\nmodels: stt_final=${config?.stt_final} stt_realtime=${config?.stt_realtime} tts=${config?.tts}`
    : "";
  return `turn ${m.turn_id} (${m.source}${m.interrupted ? ", interrupted" : ""})${models}\nspans (ms):\n${fmt(m.spans)}\nmarks (ms since start):\n${fmt(m.marks)}`;
}

export function StatusBar({ config, switchingSlug, connected, status, metrics, error, onDismissError }: Props) {
  const state = connected ? status : "disconnected";
  const pillText = connected && switchingSlug ? `loading ${switchingSlug}` : state;
  const parts = metrics
    ? BREAKDOWN.filter(([k]) => metrics.spans[k] != null).map(([k, label]) => `${label} ${metrics.spans[k]}`)
    : [];
  return (
    <div className="statusbar">
      {error && (
        <div className="error" role="alert">
          <span>{error}</span>
          <button type="button" className="ghost" onClick={onDismissError} aria-label="Dismiss error">×</button>
        </div>
      )}
      <div className="status-row">
        <span className={`pill pill-${switchingSlug && connected ? "loading" : state}`}>{pillText}</span>
        {metrics && (
          <div className="metrics" title={tooltip(metrics, config)}>
            <span className="perceived">
              Response: {metrics.spans.perceived_ms != null ? `${metrics.spans.perceived_ms} ms` : "–"}
            </span>
            {parts.length > 0 && <span className="breakdown">{parts.join(" · ")}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
