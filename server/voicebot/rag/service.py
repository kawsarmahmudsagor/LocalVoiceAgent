"""RagService: one indexing worker, retrieval on the turn thread, embedder switching."""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from voicebot.rag.chunker import chunk
from voicebot.rag.embedders import Embedder, EmbedderError
from voicebot.rag.library import INDEX_KEYS, Library, RagSettings
from voicebot.rag.loaders import SUPPORTED, NoText, load_text
from voicebot.rag.store import ChunkStore, Hit

log = logging.getLogger(__name__)

BATCH = 32


@dataclass(frozen=True)
class Retrieval:
    chunks: list[Hit]
    context: str  # "" when nothing qualified


EMPTY = Retrieval([], "")


class RagError(RuntimeError):
    pass


class RagService:
    def __init__(
        self,
        root: Path,
        default: RagSettings,
        *,
        make_embedder: Callable[[str], Embedder],
        on_event: Callable[[dict], None] | None = None,
        list_embedders: Callable[[], list[dict]] = lambda: [],
        max_upload_mb: int = 25,
    ) -> None:
        self._root = Path(root)
        self._default = default
        self._make = make_embedder
        self._on_event = on_event
        self._list_embedders = list_embedders
        self._max_upload = max_upload_mb * 1024 * 1024

        self._lock = threading.RLock()        # embedder, settings, store, generation
        self._update_lock = threading.Lock()  # serialises update() / reindex_all()
        self._lib = Library(self._root)
        self._store: ChunkStore | None = None
        self._settings = default
        self._embedder: Embedder | None = None
        self._error: str | None = None
        self._gen = 0  # bumped whenever the active collection is reset; stale index jobs are skipped

        self._q: queue.Queue = queue.Queue()
        self._cv = threading.Condition()
        self._pending = 0
        self._total = 0
        self._done = 0
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------
    @property
    def settings(self) -> RagSettings:
        return self._settings

    def start(self) -> None:
        self._store = ChunkStore(self._root / "chroma")
        self._settings = self._lib.load_settings(self._default)
        rebuild = False
        try:
            emb = self._make(self._settings.embedder)
            dim = self._dim(emb)
            with self._lock:
                self._embedder = emb
                info = self._store.info(self._settings.embedder)
                if info is None or info["dim"] != dim:
                    self._reset_locked()
                    rebuild = True
        except Exception as exc:  # noqa: BLE001 - the service must come up without an embedder
            log.warning("RAG embedder unavailable: %s", exc)
            self._error = str(exc)
        self._thread = threading.Thread(target=self._run, name="rag-indexer", daemon=True)
        self._thread.start()
        if self._embedder is None:
            return
        if rebuild:
            if self._lib.documents():
                self._enqueue(("reindex",))
            return
        for rec in self._lib.documents():
            if rec.state in ("queued", "indexing"):
                self._lib.set_state(rec.doc_id, "queued")
                self._enqueue(("index", rec.doc_id, self._gen))

    def close(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=10)
            self._thread = None
        with self._lock:
            if self._embedder is not None:
                try:
                    self._embedder.close()
                except Exception:  # noqa: BLE001
                    pass
                self._embedder = None

    # -- public API ---------------------------------------------------
    def upload(self, name: str, data: bytes) -> dict:
        base = name.replace("\\", "/").split("/")[-1] or "file"
        ext = Path(base).suffix.lower()
        if ext not in SUPPORTED:
            return {"name": base, "error": f"unsupported file type: {ext or base}"}
        if len(data) > self._max_upload:
            return {"name": base, "error": f"file is larger than {self._max_upload // (1024 * 1024)} MB"}
        rec, created = self._lib.add_file(base, data)
        if not created:
            return {"name": base, "doc_id": rec.doc_id, "state": "already indexed"}
        with self._lock:
            gen = self._gen
        self._enqueue(("index", rec.doc_id, gen))
        self._emit(self.status_event())
        return {"name": base, "doc_id": rec.doc_id, "state": "queued"}

    def delete(self, doc_id: str) -> bool:
        # Under the lock, so a worker batch in flight either finishes first (and we then
        # delete its chunks) or sees the doc gone and adds nothing.
        with self._lock:
            if self._store is not None:
                try:
                    self._store.delete_doc(self._settings.embedder, doc_id)
                except Exception:  # noqa: BLE001
                    log.exception("could not delete chunks for %s", doc_id)
            removed = self._lib.remove(doc_id)
        self._emit(self.status_event())
        return removed

    def reindex_all(self) -> None:
        with self._update_lock:
            with self._lock:
                if self._embedder is None:
                    raise RagError(self._error or "no embedder loaded")
                self._reset_locked()
            self._enqueue(("reindex",))
        self._emit(self.config_event())

    def update(self, patch: dict) -> None:
        with self._update_lock:
            cur = self._settings
            new = cur.updated(patch)  # ValueError on bad values
            switch = new.embedder != cur.embedder
            index_changed = any(getattr(new, k) != getattr(cur, k) for k in INDEX_KEYS)
            new_emb: Embedder | None = None
            if switch:  # build and probe outside the lock: this can be slow
                try:
                    new_emb = self._make(new.embedder)
                    self._dim(new_emb)
                except Exception as exc:  # noqa: BLE001
                    self._error = str(exc)
                    self._emit(self.config_event())
                    raise RagError(str(exc)) from exc
            old_emb = None
            rebuild = False
            with self._lock:
                if switch:
                    old_emb, self._embedder = self._embedder, new_emb
                self._settings = new
                if self._embedder is not None and self._store is not None:
                    if switch:
                        info = self._store.info(new.embedder)
                        ready = sum(d.chunks for d in self._lib.documents() if d.state == "ready")
                        rebuild = (
                            info is None
                            or info["dim"] != self._embedder.dim
                            or info["chunk_size"] != new.chunk_size
                            or info["chunk_overlap"] != new.chunk_overlap
                            or self._store.count(new.embedder) != ready
                        )
                    else:
                        rebuild = index_changed
                    if rebuild:
                        self._reset_locked()
                self._error = None
                self._lib.save_settings(new)
            if rebuild:
                self._enqueue(("reindex",))
            if old_emb is not None:
                try:
                    old_emb.close()
                except Exception:  # noqa: BLE001
                    pass
        self._emit(self.config_event())

    def retrieve(self, query: str) -> Retrieval:
        with self._lock:
            s = self._settings
            if not s.enabled or self._store is None:
                return EMPTY
            if self._embedder is None:
                raise EmbedderError(self._error or "no embedder loaded")
            if self._store.count(s.embedder) == 0:
                return EMPTY
            vec = self._embedder.embed([query], "query")[0]
            hits = self._store.query(s.embedder, vec, s.top_k)
        kept: list[Hit] = []
        blocks: list[str] = []
        used = 0
        for h in hits:
            if h.score < s.min_score:
                continue
            block = f"[{h.name}, part {h.chunk + 1}]\n{h.text}"
            extra = len(block) + (2 if blocks else 0)
            if used + extra > s.max_context_chars:
                if blocks:
                    continue
                block = block[: s.max_context_chars]
                extra = len(block)
            blocks.append(block)
            kept.append(h)
            used += extra
        if not kept:
            return EMPTY
        return Retrieval(kept, "\n\n".join(blocks))

    def config_event(self) -> dict:
        try:
            embedders = self._list_embedders()
        except Exception:  # noqa: BLE001
            embedders = []
        with self._lock:
            s = self._settings
            info = self._store.info(s.embedder) if self._store is not None else None
            if info is None:
                state = "needs re-index"
                dim = self._embedder.dim if self._embedder is not None else None
            else:
                dim = info["dim"]
                if self._pending > 0:
                    state = "indexing"
                elif info["chunk_size"] != s.chunk_size or info["chunk_overlap"] != s.chunk_overlap:
                    state = "needs re-index"
                else:
                    state = "indexed"
            ev = {
                "type": "rag_config",
                "settings": asdict(s),
                "embedders": embedders,
                "collection": {
                    "embedder": s.embedder,
                    "dim": dim,
                    "state": state,
                    "chunk_size": s.chunk_size,
                    "chunk_overlap": s.chunk_overlap,
                },
            }
            if self._error:
                ev["error"] = self._error
        return ev

    def status_event(self) -> dict:
        with self._lock:
            docs = [asdict(d) for d in self._lib.documents()]
        with self._cv:
            indexing = {"done": self._done, "total": self._total} if self._total > 0 else None
        return {"type": "rag_status", "documents": docs, "indexing": indexing}

    def wait_idle(self, timeout: float = 5.0) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: self._pending == 0, timeout)

    # -- internals ----------------------------------------------------
    def _dim(self, emb: Embedder) -> int:
        if emb.dim is None:
            emb.embed(["probe"], "document")
        if emb.dim is None:
            raise EmbedderError("could not determine the embedding size")
        return int(emb.dim)

    def _reset_locked(self) -> None:
        s = self._settings
        assert self._store is not None and self._embedder is not None
        self._store.reset(s.embedder, dim=int(self._embedder.dim), chunk_size=s.chunk_size,
                          chunk_overlap=s.chunk_overlap)
        self._gen += 1

    def _enqueue(self, job: tuple) -> None:
        with self._cv:
            self._pending += 1
            if job[0] == "index":
                self._total += 1
        self._q.put(job)

    def _emit(self, ev: dict) -> None:
        if self._on_event is not None:
            try:
                self._on_event(ev)
            except Exception:  # noqa: BLE001
                log.exception("rag on_event failed")

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is None:
                return
            try:
                if job[0] == "index":
                    self._do_index(job[1], job[2])
                elif job[0] == "reindex":
                    self._do_reindex()
            except Exception:  # noqa: BLE001 - the worker must never die
                log.exception("rag job failed: %s", job)
            finally:
                with self._cv:
                    if job[0] == "index":
                        self._done += 1
                    self._pending -= 1
                    if self._pending == 0:
                        self._total = self._done = 0
                    self._cv.notify_all()
            self._emit(self.status_event())

    def _do_reindex(self) -> None:
        with self._lock:
            gen = self._gen
            docs = self._lib.documents()
            for d in docs:
                self._lib.set_state(d.doc_id, "queued")
        for d in docs:
            self._enqueue(("index", d.doc_id, gen))

    def _fail(self, doc_id: str, reason: str) -> None:
        self._lib.set_state(doc_id, f"failed: {reason}", 0)

    def _do_index(self, doc_id: str, gen: int) -> None:
        with self._lock:
            rec = self._lib.get(doc_id)
            if rec is None or gen != self._gen:
                return
            emb, s, name = self._embedder, self._settings, rec.name
            if emb is None:
                self._fail(doc_id, self._error or "no embedder loaded")
                return
            path = self._lib.path(doc_id)
            self._lib.set_state(doc_id, "indexing")
        self._emit(self.status_event())
        try:
            parts = chunk(load_text(path), s.chunk_size, s.chunk_overlap)
            if not parts:
                raise NoText("no extractable text")
        except NoText:
            with self._lock:
                if gen == self._gen:
                    self._lib.set_state(doc_id, "no text", 0)
            return
        except Exception as exc:  # noqa: BLE001 - corrupt PDFs raise raw library errors
            with self._lock:
                if gen == self._gen:
                    self._fail(doc_id, str(exc) or type(exc).__name__)
            return
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(parts), BATCH):
                with self._lock:  # one batch per lock hold
                    if self._lib.get(doc_id) is None or gen != self._gen:
                        return
                    vectors += emb.embed(parts[i:i + BATCH], "document")
            with self._lock:
                if self._lib.get(doc_id) is None or gen != self._gen:
                    return
                assert self._store is not None
                self._store.delete_doc(s.embedder, doc_id)
                self._store.add(s.embedder, doc_id, name, parts, vectors)
                self._lib.set_state(doc_id, "ready", len(parts))
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                if self._lib.get(doc_id) is not None and gen == self._gen:
                    self._fail(doc_id, str(exc) or type(exc).__name__)
