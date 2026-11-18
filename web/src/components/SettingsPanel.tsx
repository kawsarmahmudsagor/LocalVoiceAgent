import type { Defaults, ServerConfig } from "../protocol";

interface Props {
  config: ServerConfig | null;
  settings: Defaults | null;
  onChange(patch: Partial<Defaults>): void;
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange(v: number): void;
}

function Slider({ label, value, min, max, step, disabled, onChange }: SliderProps) {
  const id = `s-${label.replace(/\s+/g, "-").toLowerCase()}`;
  return (
    <div className="field">
      <div className="field-head">
        <label htmlFor={id}>{label}</label>
        <span className="field-value">{value}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={e => onChange(Number(e.target.value))} />
    </div>
  );
}

export function SettingsPanel({ config, settings, onChange }: Props) {
  const disabled = !settings;
  return (
    <aside className="sidebar">
      <h1 className="brand">Voice Playground</h1>

      <section className="badges" aria-label="Server configuration">
        <span className="badge" title="LLM backend (LLM_BACKEND)">{config?.llm_backend ?? "…"}</span>
        <span className="badge" title="Audio mode (AUDIO_IO)">{config ? `${config.audio_io} audio` : "…"}</span>
        <span className="badge badge-model" title="Model">{config?.model_label ?? "connecting…"}</span>
      </section>

      <div className="field">
        <label htmlFor="system-prompt">System prompt</label>
        <textarea id="system-prompt" rows={7} value={settings?.system_prompt ?? ""} disabled={disabled}
          onChange={e => onChange({ system_prompt: e.target.value })} />
      </div>

      <Slider label="Temperature" value={settings?.temperature ?? 0} min={0} max={2} step={0.05} disabled={disabled}
        onChange={v => onChange({ temperature: v })} />
      <Slider label="Top P" value={settings?.top_p ?? 0} min={0} max={1} step={0.05} disabled={disabled}
        onChange={v => onChange({ top_p: v })} />
      <Slider label="Max tokens" value={settings?.max_tokens ?? 32} min={32} max={1024} step={16} disabled={disabled}
        onChange={v => onChange({ max_tokens: v })} />

      <div className="field">
        <label htmlFor="voice">Voice</label>
        <select id="voice" value={settings?.voice ?? ""} disabled={disabled}
          onChange={e => onChange({ voice: e.target.value })}>
          {(config?.voices ?? []).map(v => <option key={v} value={v}>{v}</option>)}
        </select>
      </div>

      <p className="hint">Backend and audio mode are set in <code>.env</code> and apply on restart.</p>
    </aside>
  );
}
