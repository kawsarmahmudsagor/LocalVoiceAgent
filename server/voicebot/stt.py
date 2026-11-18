"""Builds the RealtimeSTT recorder. See spec §4.5."""
from __future__ import annotations

from typing import Callable

from voicebot.config import Settings


def make_recorder(settings: Settings, *, use_microphone: bool, recorder_cls=None,
                  on_recording_stop: Callable[[], None] | None = None):
    """on_recording_stop fires when VAD ends the utterance (the "silence end" timing mark)."""
    if recorder_cls is None:
        from RealtimeSTT import AudioToTextRecorder
        recorder_cls = AudioToTextRecorder
    kwargs = dict(model=settings.stt_model, language="en", device=settings.stt_device,
                  compute_type=settings.stt_compute_type, use_microphone=use_microphone,
                  post_speech_silence_duration=settings.stt_silence_sec, spinner=False,
                  enable_realtime_transcription=False)
    if on_recording_stop is not None:
        kwargs["on_recording_stop"] = on_recording_stop
    return recorder_cls(**kwargs)
