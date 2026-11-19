import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { RagPanel } from "./RagPanel";
import type { AppState } from "../state";

const rag: NonNullable<AppState["rag"]> = {
  settings: { enabled: true, top_k: 3, min_score: 0.5, max_context_chars: 2000, chunk_size: 800, chunk_overlap: 120,
    embedder: "hf:BAAI/bge-small-en-v1.5" },
  embedders: [{ slug: "hf:BAAI/bge-small-en-v1.5", available: true }],
  collection: null, documents: [], indexing: null, error: null,
};
const noop = () => {};
const render = (error: string | null) => renderToStaticMarkup(
  <RagPanel rag={rag} disabled={false} onUpdate={noop} onUpload={async () => []} onDelete={noop} onReindex={noop}
    error={error} onDismissError={noop} />);

describe("RagPanel", () => {
  it("shows the current error so RAG-tab rejections are visible on this tab", () => {
    const html = render("top_k: must be between 1 and 10");
    expect(html).toContain('role="alert"');
    expect(html).toContain("top_k: must be between 1 and 10");
    expect(html).toContain('aria-label="Dismiss error"');
  });
  it("shows no alert without an error", () => {
    expect(render(null)).not.toContain('role="alert"');
  });
});
