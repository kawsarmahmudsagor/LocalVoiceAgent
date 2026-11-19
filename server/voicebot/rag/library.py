"""Document library on disk: uploaded files, a manifest, and saved RAG settings."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path

INDEX_KEYS = ("chunk_size", "chunk_overlap")

_RANGES = {
    "top_k": (int, 1, 10),
    "min_score": (float, 0.0, 1.0),
    "max_context_chars": (int, 200, 8000),
    "chunk_size": (int, 200, 4000),
}


@dataclass(frozen=True)
class RagSettings:
    enabled: bool = False
    top_k: int = 3
    min_score: float = 0.5
    max_context_chars: int = 2000
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedder: str = "hf:BAAI/bge-small-en-v1.5"

    def updated(self, patch: dict) -> "RagSettings":
        known = {f.name for f in fields(self)}
        values = asdict(self)
        errors: list[str] = []
        for key, val in patch.items():
            if key not in known:
                errors.append(f"{key}: unknown setting")
                continue
            values[key] = val
        for key in patch:
            if key not in known:
                continue
            v = values[key]
            if key == "enabled":
                if not isinstance(v, bool):
                    errors.append("enabled: must be true or false")
            elif key in _RANGES:
                kind, lo, hi = _RANGES[key]
                if isinstance(v, bool) or not isinstance(v, (int, float) if kind is float else int):
                    errors.append(f"{key}: must be a number" if kind is float else f"{key}: must be an integer")
                elif not lo <= v <= hi:
                    errors.append(f"{key}: must be between {lo} and {hi}")
                elif kind is float:
                    values[key] = float(v)
            elif key == "chunk_overlap":
                if isinstance(v, bool) or not isinstance(v, int):
                    errors.append("chunk_overlap: must be an integer")
            elif key == "embedder":
                if not isinstance(v, str) or not re.match(r"^(hf|ollama):\S.*$", v.strip()):
                    errors.append("embedder: must be 'hf:<model>' or 'ollama:<model>'")
                else:
                    values[key] = v.strip()
        # overlap depends on the final chunk_size, so check it once everything else is settled
        if not any(e.startswith(("chunk_overlap", "chunk_size")) for e in errors):
            size, ov = values["chunk_size"], values["chunk_overlap"]
            if not 0 <= ov <= size // 2:
                errors.append(f"chunk_overlap: must be between 0 and {size // 2} (half the chunk size)")
        if errors:
            raise ValueError("; ".join(errors))
        return RagSettings(**values)


@dataclass
class DocRecord:
    doc_id: str
    name: str
    ext: str
    size: int
    sha256: str
    state: str
    chunks: int = 0


class Library:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.files = self.root / "files"
        self.files.mkdir(parents=True, exist_ok=True)
        self._manifest = self.root / "manifest.json"
        self._settings = self.root / "settings.json"
        self._lock = threading.RLock()
        self._docs: dict[str, DocRecord] = {}
        self._load()

    # -- manifest -----------------------------------------------------
    def _load(self) -> None:
        try:
            raw = json.loads(self._manifest.read_text(encoding="utf-8"))
            self._docs = {r["doc_id"]: DocRecord(**r) for r in raw}
        except (OSError, ValueError, TypeError, KeyError):
            self._docs = {}

    def _atomic_write(self, target: Path, text: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=target.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _save(self) -> None:
        self._atomic_write(self._manifest, json.dumps([asdict(d) for d in self._docs.values()], indent=2))

    # -- documents ----------------------------------------------------
    def add_file(self, name: str, data: bytes) -> tuple[DocRecord, bool]:
        sha = hashlib.sha256(data).hexdigest()
        doc_id = sha[:12]
        with self._lock:
            existing = self._docs.get(doc_id)
            if existing is not None:
                return existing, False
            base = re.split(r"[\/]", name)[-1] or "file"
            ext = re.sub(r"[^a-z0-9.]", "", Path(base).suffix.lower())
            (self.files / f"{doc_id}{ext}").write_bytes(data)
            rec = DocRecord(doc_id, base, ext, len(data), sha, "queued")
            self._docs[doc_id] = rec
            self._save()
            return rec, True

    def documents(self) -> list[DocRecord]:
        with self._lock:
            return list(self._docs.values())

    def get(self, doc_id: str) -> DocRecord | None:
        return self._docs.get(doc_id)

    def path(self, doc_id: str) -> Path:
        rec = self._docs.get(doc_id)
        return self.files / f"{doc_id}{rec.ext if rec else ''}"

    def set_state(self, doc_id: str, state: str, chunks: int | None = None) -> None:
        with self._lock:
            rec = self._docs.get(doc_id)
            if rec is None:
                return
            rec.state = state
            if chunks is not None:
                rec.chunks = chunks
            self._save()

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            rec = self._docs.get(doc_id)
            if rec is None:
                return False
            p = self.path(doc_id)
            del self._docs[doc_id]
            self._save()
            try:
                p.unlink()
            except OSError:
                pass
            return True

    # -- settings -----------------------------------------------------
    def load_settings(self, default: RagSettings) -> RagSettings:
        try:
            patch = json.loads(self._settings.read_text(encoding="utf-8"))
            return default.updated(patch)
        except (OSError, ValueError, TypeError, AttributeError):
            return default

    def save_settings(self, s: RagSettings) -> None:
        self._atomic_write(self._settings, json.dumps(asdict(s), indent=2))
