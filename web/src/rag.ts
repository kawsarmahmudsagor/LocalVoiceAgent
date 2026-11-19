export type UploadResult = { name: string; doc_id: string; state: string } | { name: string; error: string };

async function fail(r: Response): Promise<never> {
  let detail = `HTTP ${r.status}`;
  try { const j = await r.json(); if (j && typeof j.detail === "string") detail = j.detail; } catch { /* keep status text */ }
  throw new Error(detail);
}

export async function uploadDocuments(files: File[], f: typeof fetch = fetch): Promise<UploadResult[]> {
  const body = new FormData();
  for (const file of files) body.append("files", file, file.name);
  const r = await f("/api/rag/documents", { method: "POST", body });
  if (!r.ok) return fail(r);
  return (await r.json()) as UploadResult[];
}

export async function deleteDocument(id: string, f: typeof fetch = fetch): Promise<void> {
  const r = await f(`/api/rag/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) await fail(r);
}

export async function reindexAll(f: typeof fetch = fetch): Promise<void> {
  const r = await f("/api/rag/reindex", { method: "POST" });
  if (!r.ok) await fail(r);
}

export function validTemplate(t: string): boolean { return t.includes("{context}"); }
