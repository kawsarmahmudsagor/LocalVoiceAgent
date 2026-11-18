"""Builds the RealtimeSTT recorder. See spec §4.5 and plan Task 7b (timing)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

from voicebot.config import Settings


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
              windows: bool = sys.platform == "win32") -> None:
    """Check what the STT worker needs before starting it.

    RealtimeSTT loads the model in a child process, and if that fails the parent
    waits forever, so startup would hang with no message. Catch the known causes here.
    """
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


def make_recorder(settings: Settings, *, use_microphone: bool, callbacks: dict[str, Callable] | None = None,
                  recorder_cls=None):
    if recorder_cls is None:
        if sys.platform == "win32" and settings.stt_device == "cuda":
            expose_cuda_dlls()
        preflight(settings)
        from RealtimeSTT import AudioToTextRecorder
        recorder_cls = AudioToTextRecorder
    kwargs = dict(model=settings.stt_model, language="en", device=settings.stt_device,
                  compute_type=settings.stt_compute_type, use_microphone=use_microphone,
                  post_speech_silence_duration=settings.stt_silence_sec, spinner=False,
                  enable_realtime_transcription=False)
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
        "on_turn_detection_stop": lambda: timer.unmark("speech_end"),  # the user resumed speaking
        "on_recording_stop": lambda: timer.mark("endpoint"),
        "on_transcription_start": on_transcription_start,
    }
