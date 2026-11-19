"""Chroma-backed chunk store: one collection per embedder."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings


@dataclass(frozen=True)
class Hit:
    doc_id: str
    name: str
    chunk: int
    text: str
    score: float


def collection_name(embedder_slug: str) -> str:
    return "docs-" + re.sub(r"[^a-z0-9]+", "-", embedder_slug.lower()).strip("-")


class ChunkStore:
    def __init__(self, path: Path) -> None:
        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )

    def _get(self, slug: str):
        try:
            return self._client.get_collection(collection_name(slug), embedding_function=None)
        except Exception:
            return None

    def info(self, slug: str) -> dict | None:
        col = self._get(slug)
        if col is None:
            return None
        m = col.metadata or {}
        return {k: m[k] for k in ("embedder", "dim", "chunk_size", "chunk_overlap")}

    def reset(self, slug: str, *, dim: int, chunk_size: int, chunk_overlap: int) -> None:
        name = collection_name(slug)
        if self._get(slug) is not None:
            self._client.delete_collection(name)
        self._client.create_collection(
            name,
            metadata={
                "hnsw:space": "cosine",
                "embedder": slug,
                "dim": dim,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            embedding_function=None,
        )

    def add(self, slug: str, doc_id: str, name: str, chunks: list[str],
            embeddings: list[list[float]], *, start: int = 0) -> None:
        """Add chunks numbered from ``start``; sliced to Chroma's max batch size."""
        if not chunks:
            return
        col = self._get(slug)
        if col is None:
            raise KeyError(f"no collection for embedder {slug!r}; call reset() first")
        step = max(1, int(self._client.get_max_batch_size()))
        for i in range(0, len(chunks), step):
            nums = range(start + i, start + min(i + step, len(chunks)))
            col.add(
                ids=[f"{doc_id}:{n}" for n in nums],
                documents=chunks[i:i + step],
                embeddings=embeddings[i:i + step],
                metadatas=[{"doc_id": doc_id, "name": name, "chunk": n} for n in nums],
            )

    def delete_doc(self, slug: str, doc_id: str) -> None:
        col = self._get(slug)
        if col is not None:
            col.delete(where={"doc_id": doc_id})

    def doc_ids(self, slug: str) -> set[str]:
        col = self._get(slug)
        if col is None:
            return set()
        return {m["doc_id"] for m in col.get(include=["metadatas"])["metadatas"]}

    def count(self, slug: str) -> int:
        col = self._get(slug)
        return 0 if col is None else col.count()

    def query(self, slug: str, embedding: list[float], top_k: int) -> list[Hit]:
        col = self._get(slug)
        if col is None:
            return []
        n = min(top_k, col.count())
        if n <= 0:
            return []
        r = col.query(query_embeddings=[embedding], n_results=n,
                      include=["documents", "metadatas", "distances"])
        return [
            Hit(doc_id=m["doc_id"], name=m["name"], chunk=int(m["chunk"]), text=d, score=1.0 - float(dist))
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
