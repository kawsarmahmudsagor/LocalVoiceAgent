import { useEffect, useRef, useState } from "react";
import type { RagSettings } from "../protocol";
import { chunkPatch, type UploadResult } from "../rag";
import type { AppState } from "../state";

interface Props {
  rag: AppState["rag"];
  disabled: boolean;
  onUpdate(patch: Partial<RagSettings>): void;
  onUpload(files: File[]): Promise<UploadResult[]>;
  onDelete(id: string): void;
  onReindex(): void;
}

const CONFIRM = "Re-index all documents?";
const ACCEPT = ".txt,.md,.pdf,.docx";

interface NumProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled: boolean;
  /** Return false when nothing was sent (e.g. the confirm was cancelled): the box then shows the real value again. */
  onCommit(v: number): boolean | void;
  /** Changes when the server sends fresh settings, so a rejected edit does not linger. */
  resetKey?: unknown;
}

/** A number box that applies on blur or Enter, so typing "800" does not fire three updates. */
function NumField({ id, label, value, min, max, step = 1, disabled, onCommit, resetKey }: NumProps) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value, resetKey]);
  const commit = () => {
    const n = Number(draft);
    if (draft.trim() === "" || Number.isNaN(n)) { setDraft(String(value)); return; }
    const v = Math.min(max, Math.max(min, n));
    setDraft(String(v));
    if (v !== value && onCommit(v) === false) setDraft(String(value));
  };
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span className="range">({min}–{max})</span></label>
      <input id={id} type="number" min={min} max={max} step={step} value={draft} disabled={disabled}
        onChange={e => setDraft(e.target.value)} onBlur={commit}
        onKeyDown={e => { if (e.key === "Enter") commit(); }} />
    </div>
  );
}

export function RagPanel({ rag, disabled, onUpdate, onUpload, onDelete, onReindex }: Props) {
  const [errors, setErrors] = useState<{ name: string; error: string }[]>([]);
  const [uploading, setUploading] = useState(false);
  const [over, setOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  if (!rag) return <div className="page"><p className="hint">Waiting for the server…</p></div>;

  const { settings, embedders, collection, documents, indexing } = rag;
  const busy = disabled || indexing !== null;
  const embedder = embedders.find(e => e.slug === settings.embedder);
  const unavailable = embedders.filter(e => !e.available);

  // Index settings change what the stored vectors mean: confirm, then the server re-indexes.
  const indexChange = (patch: Partial<RagSettings>): boolean => {
    if (!window.confirm(CONFIRM)) return false;
    onUpdate(patch);
    return true;
  };
  const rk = rag.settings;

  const upload = async (list: FileList | File[] | null) => {
    const files = list ? Array.from(list) : [];
    if (files.length === 0 || disabled) return;
    setUploading(true);
    try {
      const results = await onUpload(files);
      setErrors(results.flatMap(r => ("error" in r ? [{ name: r.name, error: r.error }] : [])));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrors(files.map(f => ({ name: f.name, error: msg })));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  return (
    <div className="page rag-page">
      {rag.error && <div className="error" role="alert"><span>{rag.error}</span></div>}

      <label className="check">
        <input type="checkbox" checked={settings.enabled} disabled={busy}
          onChange={e => onUpdate({ enabled: e.target.checked })} />
        <span>Use documents when answering</span>
      </label>

      <fieldset className="rag-section" disabled={busy}>
        <legend>Embedding model</legend>
        <div className="field">
          <label htmlFor="rag-embedder">Embedder</label>
          <select id="rag-embedder" value={settings.embedder} aria-describedby="rag-embedder-note"
            onChange={e => indexChange({ embedder: e.target.value })}>
            {!embedder && <option value={settings.embedder}>{settings.embedder}</option>}
            {embedders.map(e => (
              <option key={e.slug} value={e.slug} disabled={!e.available}>{e.slug}{e.available ? "" : " (unavailable)"}</option>
            ))}
          </select>
          <div id="rag-embedder-note">
            {unavailable.map(e => (
              <p key={e.slug} className="hint"><code>{e.slug}</code>: {e.reason ?? "unavailable"}</p>
            ))}
          </div>
        </div>
        {collection && (
          <p className="hint">Index: {collection.embedder}{collection.dim ? ` · ${collection.dim} dims` : ""} · {collection.state}</p>
        )}
      </fieldset>

      <fieldset className="rag-section" disabled={busy}>
        <legend>Retrieval</legend>
        <NumField id="rag-top-k" label="Top K" value={settings.top_k} min={1} max={10} disabled={busy}
          resetKey={rk} onCommit={v => onUpdate({ top_k: v })} />
        <NumField id="rag-min-score" label="Min similarity" value={settings.min_score} min={0} max={1} step={0.05}
          disabled={busy} resetKey={rk} onCommit={v => onUpdate({ min_score: v })} />
        <NumField id="rag-max-chars" label="Max context chars" value={settings.max_context_chars} min={200} max={8000}
          step={100} disabled={busy} resetKey={rk} onCommit={v => onUpdate({ max_context_chars: v })} />
      </fieldset>

      <fieldset className="rag-section" disabled={busy}>
        <legend>Indexing</legend>
        <NumField id="rag-chunk-size" label="Chunk size" value={settings.chunk_size} min={200} max={4000} step={50}
          disabled={busy} resetKey={rk}
          onCommit={v => indexChange(chunkPatch(v, settings.chunk_overlap))} />
        <NumField id="rag-overlap" label="Overlap" value={settings.chunk_overlap} min={0}
          max={Math.floor(settings.chunk_size / 2)} step={10} disabled={busy}
          resetKey={rk} onCommit={v => indexChange({ chunk_overlap: v })} />
        <div className="row">
          <button type="button" disabled={busy || documents.length === 0}
            onClick={() => { if (window.confirm(CONFIRM)) onReindex(); }}>Re-index all</button>
        </div>
      </fieldset>

      <section className="rag-section" aria-label="Documents">
        <h2 className="section-title">Documents</h2>
        <div className={`dropzone${over ? " dropzone-over" : ""}`}
          onDragOver={e => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={e => { e.preventDefault(); setOver(false); void upload(e.dataTransfer.files); }}>
          <p>Drop files here, or choose them ({ACCEPT.split(",").join(" ")}).</p>
          <input ref={fileInput} type="file" multiple accept={ACCEPT} disabled={uploading || disabled}
            aria-label="Upload documents" onChange={e => void upload(e.target.files)} />
          {uploading && <p className="hint">Uploading…</p>}
        </div>

        {errors.length > 0 && (
          <ul className="upload-errors" role="alert">
            {errors.map((e, i) => <li key={`${e.name}-${i}`}><strong>{e.name}</strong>: {e.error}</li>)}
          </ul>
        )}

        {indexing && (
          <p className="hint" role="status">Indexing {indexing.done} / {indexing.total}…</p>
        )}

        {documents.length === 0
          ? <p className="hint">No documents yet.</p>
          : (
            <ul className="docs">
              {documents.map(d => (
                <li key={d.doc_id} className="doc">
                  <span className="doc-name" title={d.name}>{d.name}</span>
                  <span className="doc-meta">{d.state} · {d.chunks} chunks · {Math.max(1, Math.round(d.size / 1024))} KB</span>
                  <button type="button" className="ghost small" disabled={busy} aria-label={`Delete ${d.name}`}
                    onClick={() => onDelete(d.doc_id)}>Delete</button>
                </li>
              ))}
            </ul>
          )}
      </section>
    </div>
  );
}
