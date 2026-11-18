"""Turn a TTS catalog entry into RealtimeTTS engine kwargs (v2 spec §3.1, §4.1).

Catalog options stay flat and declarative. Engines whose docs construct a voice object
(QwenVoice, PiperVoice, ChatterboxVoice) get it built here. Relative paths resolve against
the repo root. Stdlib only; the RealtimeTTS module is passed in.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from voicebot.catalog import Entry


def _abs(root: Path, value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else root / p)


def default_piper_path() -> str:
    """The piper executable from piper-tts: next to this interpreter, else on PATH."""
    local = Path(sys.executable).parent / "piper"
    return str(local) if local.exists() else (shutil.which("piper") or "piper")


def engine_kwargs(entry: Entry, module, *, root: Path, piper_path: str | None = None) -> dict:
    opts = dict(entry.options)
    if entry.engine == "QwenEngine":
        ref_audio = opts.pop("reference_audio", None)
        ref_text_file = opts.pop("reference_text_file", None)
        if ref_audio:
            ref_text = Path(_abs(root, ref_text_file)).read_text(encoding="utf-8").strip() if ref_text_file else None
            opts["voice"] = module.QwenVoice(name=entry.slug, ref_audio=_abs(root, ref_audio), ref_text=ref_text,
                                             language="english")
    elif entry.engine == "PiperEngine":
        model_dir = Path(_abs(root, opts.pop("model_dir", "models/piper")))
        opts["voice"] = module.PiperVoice(model_file=str(model_dir / f"{entry.model}.onnx"),
                                          config_file=str(model_dir / f"{entry.model}.onnx.json"))
        opts["piper_path"] = piper_path or default_piper_path()
    elif entry.engine == "CoquiEngine":
        if isinstance(opts.get("voice"), str) and opts["voice"].endswith(".wav"):
            opts["voice"] = _abs(root, opts["voice"])  # coqui.md: voice = reference WAV filename
    elif entry.engine == "ChatterboxEngine":
        prompt = opts.pop("audio_prompt_path", None)
        if prompt:
            opts["voice"] = module.ChatterboxVoice(audio_prompt_path=_abs(root, prompt))
    if entry.voice is not None and "voice" not in opts:
        opts["voice"] = entry.voice
    return opts


def build_engine(entry: Entry, *, root: Path, module=None):
    if module is None:
        import RealtimeTTS as module  # noqa: PLC0415  (heavy; only where an engine is built)
    return getattr(module, entry.engine)(**engine_kwargs(entry, module, root=root))
