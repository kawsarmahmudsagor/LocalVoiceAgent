import type { Metrics, SpanName, Status } from "../protocol";

interface Props {
  connected: boolean;
  status: Status;
  metrics: Metrics | null;
  error: string | null;
  onDismissError(): void;
}

const BREAKDOWN: [SpanName, string][] = [
  ["silence_wait_ms", "silence"],
  ["stt_ms", "stt"],
  ["llm_ttft_ms", "llm"],
  ["filter_ms", "filter"],
  ["tts_first_audio_ms", "tts"],
  ["playback_ms", "play"],
];

function tooltip(m: Metrics): string {
  const fmt = (o: Record<string, number | null>) =>
    Object.entries(o).map(([k, v]) => `  ${k}: ${v ?? "-"}`).join("\n");
  return `turn ${m.turn_id} (${m.source}${m.interrupted ? ", interrupted" : ""})\nspans (ms):\n${fmt(m.spans)}\nmarks (ms since start):\n${fmt(m.marks)}`;
}

export function StatusBar({ connected, status, metrics, error, onDismissError }: Props) {
  const state = connected ? status : "disconnected";
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
        <span className={`pill pill-${state}`}>{state}</span>
        {metrics && (
          <div className="metrics" title={tooltip(metrics)}>
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
