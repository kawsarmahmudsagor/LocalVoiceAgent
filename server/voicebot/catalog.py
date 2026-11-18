"""STT/TTS entries for the benchmark pool (bench/pool.toml) and the shortlist (catalog.toml).

See v2 spec §3.1 and §4.1. Stdlib only.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FETCH_RE = re.compile(r"^(auto|hf:[^:\s]+(:[^:\s]+)?|sherpa:(parakeet|nemotron)|kroko:\S+|piper:\S+|manual:.+)$")


class CatalogError(ValueError):
    """Invalid catalog entry. The message starts with the offending slug."""


@dataclass(frozen=True)
class Entry:
    slug: str
    kind: str
    role: str | None
    engine: str
    model: str
    device: str
    family: str
    options: dict = field(default_factory=dict)
    streaming: bool = False
    vram_mb: int = 0
    fetch: str = "auto"
    runtime: str = "inprocess"
    group: str | None = None
    voice: str | None = None


_FIELDS = {f.name for f in fields(Entry)}


def _validate(e: Entry) -> None:
    def bad(msg: str) -> CatalogError:
        return CatalogError(f"{e.slug}: {msg}")

    if not SLUG_RE.match(e.slug):
        raise bad("slug must be kebab-case: lowercase letters, digits and single hyphens")
    if e.kind == "stt":
        if e.role not in ("final", "realtime"):
            raise bad("stt entries need role 'final' or 'realtime'")
    elif e.kind == "tts":
        if e.role is not None:
            raise bad("tts entries have no role")
    else:
        raise bad(f"kind must be 'stt' or 'tts', not {e.kind!r}")
    if e.slug.endswith("-rt") != (e.role == "realtime"):
        raise bad("the slug ends in -rt if and only if role = 'realtime'")
    if e.streaming and e.role != "realtime":
        raise bad("streaming is only allowed for realtime entries")
    if e.device not in ("cpu", "cuda"):
        raise bad(f"device must be 'cpu' or 'cuda', not {e.device!r}")
    if e.runtime not in ("inprocess", "worker"):
        raise bad(f"runtime must be 'inprocess' or 'worker', not {e.runtime!r}")
    if e.runtime == "worker" and not e.group:
        raise bad("runtime = 'worker' needs a group")
    if not FETCH_RE.match(e.fetch):
        raise bad(f"fetch {e.fetch!r} must be auto, hf:<repo>[:<file>], sherpa:<parakeet|nemotron>, "
                  "kroko:<file>, piper:<voice> or manual:<note>")


def load_catalog(path: Path) -> list[Entry]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    entries: list[Entry] = []
    seen: set[str] = set()
    for raw in data.get("entry", []):
        slug = str(raw.get("slug", "?"))
        unknown = set(raw) - _FIELDS
        if unknown:
            raise CatalogError(f"{slug}: unknown field(s) {', '.join(sorted(unknown))}")
        try:
            e = Entry(**{"role": None, **raw})
        except TypeError as exc:
            raise CatalogError(f"{slug}: {exc}") from None
        _validate(e)
        if e.slug in seen:
            raise CatalogError(f"{e.slug}: duplicate slug")
        seen.add(e.slug)
        entries.append(e)
    return entries


def by_slug(entries: list[Entry]) -> dict[str, Entry]:
    return {e.slug: e for e in entries}


def for_role(entries: list[Entry], kind: str, role: str | None = None) -> list[Entry]:
    return [e for e in entries if e.kind == kind and e.role == role]
