import { useEffect, useState } from "react";
import type { Defaults } from "../protocol";
import { validTemplate } from "../rag";

interface Props {
  settings: Defaults | null;
  defaults: Defaults | null;
  disabled: boolean;
  onChange(patch: Partial<Defaults>): void;
}

export function SystemPromptPanel({ settings, defaults, disabled, onChange }: Props) {
  const off = disabled || !settings;
  // The template is held locally so an invalid edit stays visible but is never sent.
  const [template, setTemplate] = useState(settings?.rag_template ?? "");
  const saved = settings?.rag_template ?? "";
  useEffect(() => {
    if (validTemplate(saved)) setTemplate(saved);
  }, [saved]);
  const templateOk = validTemplate(template);

  const editTemplate = (value: string) => {
    setTemplate(value);
    if (validTemplate(value)) onChange({ rag_template: value });
  };

  return (
    <div className="page prompt-page">
      <div className="field">
        <div className="field-head">
          <label htmlFor="system-prompt">System prompt</label>
          <button type="button" className="ghost small" disabled={off || !defaults}
            onClick={() => defaults && onChange({ system_prompt: defaults.system_prompt })}>Reset</button>
        </div>
        <textarea id="system-prompt" rows={10} value={settings?.system_prompt ?? ""} disabled={off}
          onChange={e => onChange({ system_prompt: e.target.value })} />
      </div>

      <div className="field">
        <div className="field-head">
          <label htmlFor="rag-template">RAG template</label>
          <button type="button" className="ghost small" disabled={off || !defaults}
            onClick={() => defaults && editTemplate(defaults.rag_template)}>Reset</button>
        </div>
        <textarea id="rag-template" rows={8} value={template} disabled={off} aria-invalid={!templateOk}
          aria-describedby="rag-template-hint" onChange={e => editTemplate(e.target.value)} />
        {templateOk
          ? <p id="rag-template-hint" className="hint">Added to the system prompt when notes are found. <code>{"{context}"}</code> is replaced by the retrieved text.</p>
          : <p id="rag-template-hint" className="field-error" role="alert">The template must contain <code>{"{context}"}</code>. It is not sent until it does.</p>}
      </div>
    </div>
  );
}
