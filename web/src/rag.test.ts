import { describe, it, expect } from "vitest";
import { uploadDocuments, deleteDocument, reindexAll, validTemplate } from "./rag";
describe("rag api", () => {
  it("uploads every file in one multipart request", async () => {
    const calls: [string, RequestInit][] = [];
    const f = (async (u: string, i: RequestInit) => { calls.push([u, i]); return new Response("[]", { status: 202 }); }) as typeof fetch;
    await uploadDocuments([new File(["a"], "a.txt"), new File(["b"], "b.md")], f);
    expect(calls[0][0]).toBe("/api/rag/documents"); expect((calls[0][1].body as FormData).getAll("files")).toHaveLength(2);
  });
  it("delete and reindex surface the server detail", async () => {
    const ok = (async () => new Response(null, { status: 204 })) as unknown as typeof fetch;
    await expect(deleteDocument("d1", ok)).resolves.toBeUndefined();
    const busy = (async () => new Response(JSON.stringify({ detail: "busy" }), { status: 409 })) as typeof fetch;
    await expect(reindexAll(busy)).rejects.toThrow("busy");
  });
  it("validTemplate requires {context}", () => { expect(validTemplate("x {context}")).toBe(true); expect(validTemplate("x")).toBe(false); });
});
