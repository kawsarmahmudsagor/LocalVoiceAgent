"""Where audio comes from and goes to: the browser or this PC. See spec §4.6."""
from __future__ import annotations

import threading
from typing import Callable, Protocol

from voicebot.config import Settings
from voicebot.stt import make_recorder
from voicebot.tts import output_format as tts_output_format

STT_SAMPLE_RATE = 16000


class AudioIO(Protocol):
    recorder: object
    output_format: dict | None

    def on_client_audio(self, data: bytes) -> None: ...
    def mute_input(self, muted: bool) -> None: ...
    def set_sink(self, send_binary: Callable[[bytes], None] | None) -> None: ...
    def play(self, stream, on_first_audio: Callable[[], None]) -> None: ...


def _once(fn: Callable[[], None]) -> Callable[[], None]:
    done = False

    def call() -> None:
        nonlocal done
        if not done:
            done = True
            fn()
    return call


class BrowserAudio:
    """Mic PCM arrives over the WebSocket; TTS PCM is sent back over it."""

    def __init__(self, recorder, output_format: dict) -> None:
        self.recorder = recorder
        self.output_format = output_format
        self._muted = False
        self._sink: Callable[[bytes], None] | None = None
        self._lock = threading.Lock()

    def on_client_audio(self, data: bytes) -> None:
        if not self._muted:
            self.recorder.feed_audio(data, original_sample_rate=STT_SAMPLE_RATE)

    def mute_input(self, muted: bool) -> None:
        self._muted = muted

    def set_sink(self, send_binary: Callable[[bytes], None] | None) -> None:
        with self._lock:
            self._sink = send_binary

    def play(self, stream, on_first_audio: Callable[[], None]) -> None:
        first = _once(on_first_audio)

        def on_chunk(chunk: bytes) -> None:
            first()
            with self._lock:
                sink = self._sink
            if sink is not None:
                sink(chunk)

        stream.play(on_audio_chunk=on_chunk, muted=True)


class ServerAudio:
    """This PC's microphone and speakers; the browser plays nothing."""

    output_format = None

    def __init__(self, recorder) -> None:
        self.recorder = recorder

    def on_client_audio(self, data: bytes) -> None:
        pass

    def mute_input(self, muted: bool) -> None:
        self.recorder.set_microphone(not muted)

    def set_sink(self, send_binary: Callable[[bytes], None] | None) -> None:
        pass

    def play(self, stream, on_first_audio: Callable[[], None]) -> None:
        first = _once(on_first_audio)
        stream.play(on_audio_chunk=lambda _chunk: first())


def make_audio_io(settings: Settings, stream, *, recorder_factory=make_recorder,
                  on_recording_stop: Callable[[], None] | None = None) -> AudioIO:
    browser = settings.audio_io == "browser"
    extra = {"on_recording_stop": on_recording_stop} if on_recording_stop is not None else {}
    recorder = recorder_factory(settings, use_microphone=not browser, **extra)
    if browser:
        return BrowserAudio(recorder, tts_output_format(stream))
    return ServerAudio(recorder)
