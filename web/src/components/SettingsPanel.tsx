import type { BargeInMode, DiarMode, DiarState, CatalogEntry, Defaults, Role, ServerConfig, Status, WordAlignState } from "../protocol";
import type { AppState } from "../state";
import { vramUsed } from "../vram";

interface Props {
  config: ServerConfig | null;
  settings: Defaults | null;
  onChange(patch: Partial<Defaults>): void;
  catalog: CatalogEntry[];
  switching: Role | null;
  status: Status;
  onSelectModel(role: Role, slug: string): void;
  wordAlign: AppState["wordAlign"];
  diarization: AppState["diarization"];
}

const DIAR_NOTE: Record<DiarState, string> = {
  off: "Answers everyone.",
  loading: "Loading the diarization model…",
  ready: "Answers Speaker 0 only. Clear makes the next voice Speaker 0.",
  error: "",
};

const ALIGN_NOTE: Record<WordAlignState, string> = {
  off: "Text follows Kokoro's own word timings; other TTS models spread words evenly.",
  loading: "Loading the alignment model (about 1.2 GB on first use)…",
  ready: "Words of every TTS model appear as they are spoken. Uses the GPU after each sentence.",
  error: "",
};

const label = (e: CatalogEntry) => `${e.slug} · ${e.device} · ${e.vram_mb} MB`;

interface PickerProps {
  id: string;
  title: string;
  value: string;
  options: CatalogEntry[];
  withOff?: boolean;
  disabled: boolean;
  onPick(slug: string): void;
}

function Picker({ id, title, value, options, withOff, disabled, onPick }: PickerProps) {
  return (
    <div className="field">
      <label htmlFor={id}>{title}</label>
      <select id={id} value={value} disabled={disabled} onChange={e => onPick(e.target.value)}>
        {withOff && <option value="off">off</option>}
        {options.map(e => <option key={e.slug} value={e.slug}>{label(e)}</option>)}
      </select>
    </div>
  );
}

function ModelPickers({ config, catalog, switching, status, onSelectModel }:
  Pick<Props, "catalog" | "switching" | "status" | "onSelectModel"> & { config: ServerConfig }) {
  const locked = switching !== null || !(status === "idle" || status === "listening");
  const finals = catalog.filter(e => e.kind === "stt" && e.role === "final");
  const realtimes = catalog.filter(e => e.kind === "stt" && e.role === "realtime");
  const ttss = catalog.filter(e => e.kind === "tts");
  const vram = vramUsed(config, catalog);
  const pct = vram.total ? Math.min(100, Math.round((vram.used / vram.total) * 100)) : 0;
  return (
    <section className="models" aria-label="Models">
      <Picker id="stt-final" title="Final STT" value={config.stt_final ?? ""} options={finals} disabled={locked}
        onPick={slug => onSelectModel("stt_final", slug)} />
      <Picker id="stt-realtime" title="Realtime STT" value={config.stt_realtime ?? "off"} options={realtimes} withOff
        disabled={locked} onPick={slug => onSelectModel("stt_realtime", slug)} />
      <Picker id="tts-model" title="TTS" value={config.tts ?? ""} options={ttss} disabled={locked}
        onPick={slug => onSelectModel("tts", slug)} />
      <div className="field">
        <div className="field-head">
          <span className="field-label">VRAM (estimate)</span>
          <span className="field-value">{vram.used} / {vram.total} MB</span>
        </div>
        <div className={`meter${vram.over ? " meter-over" : ""}`} role="meter" aria-valuemin={0}
          aria-valuemax={vram.total} aria-valuenow={vram.used} aria-label="Estimated GPU memory">
          <div className="meter-fill" style={{ width: `${pct}%` }} />
        </div>
      </div>
    </section>
  );
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

export function SettingsPanel({ config, settings, onChange, catalog, switching, status, onSelectModel, wordAlign, diarization }: Props) {
  const disabled = !settings;
  return (
    <aside className="sidebar">
      <h1 className="brand">Voice Playground</h1>

      <section className="badges" aria-label="Server configuration">
        <span className="badge" title="LLM backend (LLM_BACKEND)">{config?.llm_backend ?? "…"}</span>
        <span className="badge" title="Audio mode (AUDIO_IO)">{config ? `${config.audio_io} audio` : "…"}</span>
        <span className="badge badge-model" title="Model">{config?.model_label ?? "connecting…"}</span>
      </section>

      {config && config.stt_final !== undefined && catalog.length > 0 && (
        <ModelPickers config={config} catalog={catalog} switching={switching} status={status}
          onSelectModel={onSelectModel} />
      )}

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

      {config?.defaults.word_align !== undefined && (
        <div className="field">
          <label className="check" title="Forced word alignment for TTS models without word timings (Piper, Qwen, Inflect, Coqui)">
            <input type="checkbox" checked={wordAlign.state === "ready" || wordAlign.state === "loading"}
              disabled={disabled || wordAlign.state === "loading"}
              onChange={e => onChange({ word_align: e.target.checked })} />
            <span>Exact word timing</span>
          </label>
          {wordAlign.state === "error"
            ? <p className="field-error" role="alert">{wordAlign.message}</p>
            : <p className="hint">{ALIGN_NOTE[wordAlign.state]}</p>}
        </div>
      )}

      {config?.defaults.barge_in_mode !== undefined && (
        <>
          <div className="field">
            <label htmlFor="barge-in">Barge-in</label>
            <select id="barge-in" value={settings?.barge_in_mode ?? config.defaults.barge_in_mode} disabled={disabled}
              onChange={e => onChange({ barge_in_mode: e.target.value as BargeInMode })}>
              <option value="off">Off</option>
              <option value="onset">Voice onset</option>
              <option value="words" disabled={config.barge_words_available !== true}>First words</option>
            </select>
          </div>
          <Slider label="Min speech (ms)" value={settings?.barge_in_min_ms ?? config.defaults.barge_in_min_ms ?? 0}
            min={0} max={1000} step={50}
            disabled={disabled || (settings?.barge_in_mode ?? config.defaults.barge_in_mode) === "off"}
            onChange={v => onChange({ barge_in_min_ms: v })} />
          <p className="hint">Speak over the bot to stop it. Use headphones for clean measurements.
            A short noise can cancel "Voice onset" only when Min speech is above about 650 ms; below that,
            the latency report counts such false fires afterwards.</p>
        </>
      )}

      {config?.defaults.diarization !== undefined && (
        <>
          <div className="field">
            <label className="check" title="Answer only the first speaker heard (Speaker 0)">
              <input type="checkbox" checked={diarization.state === "ready" || diarization.state === "loading"}
                disabled={disabled || !config.diarization_available || diarization.state === "loading"}
                onChange={e => onChange({ diarization: e.target.checked })} />
              <span>Speaker diarization</span>
            </label>
            {diarization.state === "error"
              ? <p className="field-error" role="alert">{diarization.message}</p>
              : <p className="hint">{DIAR_NOTE[diarization.state]}</p>}
          </div>
          <div className="field">
            <label htmlFor="diar-mode">Diarization latency</label>
            <select id="diar-mode" value={settings?.diar_mode ?? config.defaults.diar_mode ?? "ultra_low_latency"}
              disabled={disabled || !config.diarization_available}
              onChange={e => onChange({ diar_mode: e.target.value as DiarMode })}>
              <option value="ultra_low_latency">0.32 s</option>
              <option value="very_low_latency">0.64 s</option>
              <option value="low_latency">1.04 s</option>
            </select>
          </div>
          <Slider label="Max wait (ms)" value={settings?.diar_wait_ms ?? config.defaults.diar_wait_ms ?? 0}
            min={0} max={2000} step={50} disabled={disabled || !config.diarization_available}
            onChange={v => onChange({ diar_wait_ms: v })} />
        </>
      )}

      <p className="hint">Backend and audio mode are set in <code>.env</code> and apply on restart. Models switch while idle.</p>
    </aside>
  );
}
