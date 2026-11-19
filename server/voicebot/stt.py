"""Builds the RealtimeSTT recorder. See spec §4.5 and plan Task 7b (timing)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

from voicebot.config import REPO_ROOT, Settings


def _torch_dir() -> Path | None:
    spec = importlib.util.find_spec("torch")  # locate without importing torch
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def expose_cuda_dlls(*, torch_dir: Path | None = None, add_dll_directory=None) -> None:
    """Let faster-whisper (CTranslate2) find cuBLAS/cuDNN on Windows.

    CTranslate2 needs cublas64_12.dll, which the CUDA build of torch ships in
    torch\\lib but does not put on the DLL search path. PATH is updated as well
    because RealtimeSTT transcribes in a spawned child process, which inherits it.
    """
    torch_dir = torch_dir or _torch_dir()
    if torch_dir is None:
        return
    lib = torch_dir / "lib"
    if not lib.is_dir():
        return
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(lib) not in parts:
        os.environ["PATH"] = os.pathsep.join([str(lib), *parts])
    add = add_dll_directory or getattr(os, "add_dll_directory", None)
    if add is not None:
        add(str(lib))


class STTError(RuntimeError):
    pass


def _load_dll(name: str) -> None:
    import ctypes
    ctypes.WinDLL(name)  # type: ignore[attr-defined]


def preflight(settings: Settings, *, import_module=importlib.import_module, load_dll=_load_dll,
              windows: bool = sys.platform == "win32", engine: str = "faster_whisper") -> None:
    """Check what the STT worker needs before starting it.

    RealtimeSTT loads the model in a child process, and if that fails the parent
    waits forever, so startup would hang with no message. Catch the known causes here.
    """
    if engine != "faster_whisper":
        return  # other engines report their own import errors when the recorder builds them
    try:
        import_module("faster_whisper")
    except ImportError as e:
        raise STTError(f"faster-whisper is not installed ({e}): run scripts/setup_venv.ps1, "
                       "which installs RealtimeSTT[faster-whisper]") from None
    if windows and settings.stt_device == "cuda":
        try:
            load_dll("cublas64_12.dll")
        except OSError as e:
            raise STTError(f"cublas64_12.dll could not be loaded ({e}): faster-whisper needs the CUDA 12 cuBLAS "
                           "that ships with the CUDA build of torch (run scripts/setup_venv.ps1), "
                           "or set STT_DEVICE=cpu") from None


def _repo_path(value: str) -> str:
    """Catalog paths under models/ or assets/ are relative to the repo root."""
    return str(REPO_ROOT / value) if value.startswith(("models/", "assets/")) else value


def engine_options(options: dict) -> dict:
    """Engine options with repo paths made absolute. sherpa-onnx takes `provider="cpu:<session config>"`
    (RealtimeSTT/docs/engines/sherpa-onnx.md, "CPU performance and installation"); the file needs an absolute path."""
    opts = dict(options)
    provider = opts.get("provider")
    if isinstance(provider, str) and provider.startswith("cpu:"):
        opts["provider"] = "cpu:" + _repo_path(provider.removeprefix("cpu:"))
    return opts


def entry_kwargs(final, realtime=None) -> dict:
    """Recorder kwargs for catalog entries (v2 spec §4.3). See plan Task 12."""
    opts = engine_options(final.options)
    kw: dict = dict(transcription_engine=final.engine, model=_repo_path(final.model), device=final.device)
    if "compute_type" in opts:
        kw["compute_type"] = opts["compute_type"]
    if "download_root" in opts:  # a recorder argument, not an engine option (sherpa-onnx.md)
        kw["download_root"] = _repo_path(opts.pop("download_root"))
    kw["transcription_engine_options"] = opts
    if realtime is not None:
        rt_opts = engine_options(realtime.options)
        rt_root = rt_opts.pop("download_root", None)
        if rt_root is not None and "download_root" not in kw:  # one download_root serves both engines
            kw["download_root"] = _repo_path(rt_root)
        kw.update(enable_realtime_transcription=True, realtime_transcription_engine=realtime.engine,
                  realtime_model_type=_repo_path(realtime.model),
                  realtime_transcription_engine_options=rt_opts)
    else:
        kw["enable_realtime_transcription"] = False
    return kw


def make_recorder(settings: Settings, *, use_microphone: bool, callbacks: dict[str, Callable] | None = None,
                  recorder_cls=None, final=None, realtime=None):
    device = final.device if final is not None else settings.stt_device
    if recorder_cls is None:
        if sys.platform == "win32" and device == "cuda":
            expose_cuda_dlls()
        preflight(settings, engine=final.engine if final is not None else "faster_whisper")
        from RealtimeSTT import AudioToTextRecorder
        recorder_cls = AudioToTextRecorder
    if final is None:
        kwargs = dict(model=settings.stt_model, language="en", device=settings.stt_device,
                      compute_type=settings.stt_compute_type, use_microphone=use_microphone,
                      post_speech_silence_duration=settings.stt_silence_sec, spinner=False,
                      enable_realtime_transcription=False)
    else:
        kwargs = dict(entry_kwargs(final, realtime), language="en", use_microphone=use_microphone,
                      post_speech_silence_duration=settings.stt_silence_sec, spinner=False)
    kwargs.update(callbacks or {})
    return recorder_cls(**kwargs)


def timing_callbacks(timer) -> dict[str, Callable]:
    """RealtimeSTT callbacks that mark the STT stages on a TurnTimer.

    on_vad_stop is not used: RealtimeSTT fires it after the silence wait, not at
    the end of speech.
    """
    def on_transcription_start(_audio) -> None:
        timer.mark("stt_start")
        return None  # a truthy return makes RealtimeSTT abort the transcription

    return {
        "on_turn_detection_start": lambda: timer.mark("speech_end"),
        # The user resumed speaking. RealtimeSTT also fires this right after it ends the
        # utterance (after on_recording_stop); keep speech_end once the endpoint is marked.
        "on_turn_detection_stop": lambda: None if timer.has("endpoint") else timer.unmark("speech_end"),
        "on_recording_stop": lambda: timer.mark("endpoint"),
        "on_transcription_start": on_transcription_start,
    }
