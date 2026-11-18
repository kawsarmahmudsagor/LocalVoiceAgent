"""ModelManager: builds and swaps the STT recorder and TTS stream in-process (v2 spec §4.4, plan Task 13)."""
from __future__ import annotations

import gc
import logging
import sys
from typing import Callable

from voicebot.catalog import Entry, by_slug

log = logging.getLogger(__name__)

ROLES = ("stt_final", "stt_realtime", "tts")
_ROLE_KIND = {"stt_final": ("stt", "final"), "stt_realtime": ("stt", "realtime"), "tts": ("tts", None)}


class ModelError(RuntimeError):
    """A bad slug ('bad_model: ...') or a failed load. `restored` is the rebuilt previous object, if any."""

    def __init__(self, message: str, restored=None) -> None:
        super().__init__(message)
        self.restored = restored


def default_unload(obj) -> None:
    try:
        shutdown = getattr(obj, "shutdown", None)
        if callable(shutdown):
            shutdown()
        gc.collect()
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()
    except Exception:
        log.exception("unload failed")


class ModelManager:
    def __init__(self, entries: list[Entry], settings, *, recorder_factory: Callable | None,
                 tts_factory: Callable | None, unload: Callable = default_unload) -> None:
        self.entries = by_slug(entries)
        self.settings = settings
        self.recorder_factory, self.tts_factory, self.unload = recorder_factory, tts_factory, unload
        self.current: dict[str, str] = {}
        self._use_microphone = False
        self._callbacks: dict = {}

    def validate(self, role: str, slug: str) -> Entry | None:
        if role not in ROLES:
            raise ModelError(f"bad_model: unknown role {role!r}")
        if role == "stt_realtime" and slug == "off":
            return None
        kind, sub = _ROLE_KIND[role]
        e = self.entries.get(slug)
        if e is None or e.kind != kind or e.role != sub:
            raise ModelError(f"bad_model: {slug!r} is not a {role} model in the catalog")
        return e

    def entry(self, role: str) -> Entry | None:
        slug = self.current.get(role)
        return None if slug in (None, "off") else self.entries[slug]

    def runtime(self) -> dict[str, str | None]:
        return {role: (e.runtime if (e := self.entry(role)) else None) for role in ROLES}

    def start(self, use_microphone: bool, callbacks: dict):
        s = self.settings
        wanted = {"stt_final": s.stt_final, "stt_realtime": s.stt_realtime, "tts": s.tts_model}
        for role, slug in wanted.items():
            try:
                self.validate(role, slug)
            except ModelError:
                raise ModelError(f"startup slug {slug!r} for {role} is not in the catalog") from None
        self._use_microphone, self._callbacks = use_microphone, dict(callbacks)
        final, realtime = self.validate("stt_final", wanted["stt_final"]), self.validate("stt_realtime", wanted["stt_realtime"])
        recorder = self._build_recorder(final, realtime)
        stream = self.tts_factory(self.validate("tts", wanted["tts"]))
        self.current = dict(wanted)
        return recorder, stream

    def _build_recorder(self, final: Entry, realtime: Entry | None):
        return self.recorder_factory(final, realtime, use_microphone=self._use_microphone, callbacks=self._callbacks)

    def switch_stt(self, role: str, slug: str, old_recorder):
        new = self.validate(role, slug)
        if role not in ("stt_final", "stt_realtime"):
            raise ModelError(f"bad_model: {role} is not an STT role")
        prev_final, prev_rt = self.entry("stt_final"), self.entry("stt_realtime")
        final = new if role == "stt_final" else prev_final
        realtime = new if role == "stt_realtime" else prev_rt
        self.unload(old_recorder)
        try:
            recorder = self._build_recorder(final, realtime)
        except Exception as e:
            log.exception("loading %s=%s failed; restoring the previous recorder", role, slug)
            raise ModelError(str(e), restored=self._build_recorder(prev_final, prev_rt)) from e
        self.current[role] = slug
        return recorder

    def switch_tts(self, slug: str, old_stream):
        new = self.validate("tts", slug)
        prev = self.entry("tts")
        self.unload(old_stream)
        try:
            stream = self.tts_factory(new)
        except Exception as e:
            log.exception("loading tts=%s failed; restoring the previous stream", slug)
            raise ModelError(str(e), restored=self.tts_factory(prev)) from e
        self.current["tts"] = slug
        return stream
