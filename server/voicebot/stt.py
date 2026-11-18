"""Builds the RealtimeSTT recorder. See spec §4.5 and plan Task 7b (timing)."""
from __future__ import annotations

from typing import Callable

from voicebot.config import Settings


def make_recorder(settings: Settings, *, use_microphone: bool, callbacks: dict[str, Callable] | None = None,
                  recorder_cls=None):
    if recorder_cls is None:
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
