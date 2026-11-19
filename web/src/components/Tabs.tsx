import type { AppState } from "../state";

type Tab = AppState["tab"];

const TABS: [Tab, string][] = [
  ["playground", "Playground"],
  ["prompt", "System Prompt"],
  ["rag", "RAG"],
];

export function Tabs({ tab, onChange }: { tab: Tab; onChange(tab: Tab): void }) {
  return (
    <div className="tabs" role="tablist" aria-label="Sections">
      {TABS.map(([id, text]) => (
        <button key={id} type="button" role="tab" id={`tab-${id}`} aria-controls={`panel-${id}`}
          aria-selected={tab === id} className={`tab${tab === id ? " tab-on" : ""}`} onClick={() => onChange(id)}>
          {text}
        </button>
      ))}
    </div>
  );
}
