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
    def play(self, stream, on_first_audio: Callable[[], None]) -> float | None:
        """Blocks until synthesis ends. BrowserAudio returns the seconds of audio sent to
        the browser (which plays it later); ServerAudio returns None (already played)."""
    def cut(self) -> None:
        """Stop pressed: drop any audio the current play() still produces."""


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

    def __init__(self, recorder, output_format: dict, *, play_kwargs: dict | None = None) -> None:
        self.recorder = recorder
        self.output_format = output_format
        self.play_kwargs = dict(play_kwargs or {})  # extra TextToAudioStream.play() arguments
        self._muted = False
        self._sink: Callable[[bytes], None] | None = None
        self._lock = threading.Lock()
        self._generation = 0  # bumped by cut(); chunks from an older play() are dropped

    def on_client_audio(self, data: bytes) -> None:
        if not self._muted:
            self.recorder.feed_audio(data, original_sample_rate=STT_SAMPLE_RATE)

    def mute_input(self, muted: bool) -> None:
        self._muted = muted

    def set_recorder(self, recorder) -> None:
        self.recorder = recorder

    def set_output_format(self, output_format: dict) -> None:
        """A TTS switch can change the sample rate (v2 spec §4.4)."""
        self.output_format = output_format

    def set_sink(self, send_binary: Callable[[bytes], None] | None) -> None:
        with self._lock:
            self._sink = send_binary

    def cut(self) -> None:
        with self._lock:
            self._generation += 1

    def play(self, stream, on_first_audio: Callable[[], None]) -> float:
        first = _once(on_first_audio)
        with self._lock:
            generation = self._generation
        sent = 0

        def on_chunk(chunk: bytes) -> None:
            nonlocal sent
            with self._lock:
                sink = self._sink if self._generation == generation else None
            if sink is None:
                return
            first()
            sink(chunk)
            sent += len(chunk)

        stream.play(on_audio_chunk=on_chunk, muted=True, **self.play_kwargs)
        rate = self.output_format.get("sample_rate", 24000)
        channels = self.output_format.get("channels", 1)
        return sent / (2 * channels * rate)  # s16le


class ServerAudio:
    """This PC's microphone and speakers; the browser plays nothing."""

    output_format = None

    def __init__(self, recorder, *, play_kwargs: dict | None = None) -> None:
        self.recorder = recorder
        self.play_kwargs = dict(play_kwargs or {})

    def on_client_audio(self, data: bytes) -> None:
        pass

    def mute_input(self, muted: bool) -> None:
        self.recorder.set_microphone(not muted)

    def set_recorder(self, recorder) -> None:
        self.recorder = recorder

    def set_sink(self, send_binary: Callable[[bytes], None] | None) -> None:
        pass

    def cut(self) -> None:
        pass  # stream.stop() silences the PC speakers

    def play(self, stream, on_first_audio: Callable[[], None]) -> None:
        first = _once(on_first_audio)
        stream.play(on_audio_chunk=lambda _chunk: first(), **self.play_kwargs)
        return None  # play() returns after the speakers have played everything


def play_kwargs(settings: Settings) -> dict:
    """TextToAudioStream.play() pacing. fast_sentence_fragment=True synthesizes the first sentence's opening
    fragment (cut at a comma, or after force_first_fragment_after_words) on its own, which starts speech sooner
    but puts a pause and a pitch reset mid-sentence. RealtimeTTS/docs/llm-streaming.md: "For engines that sound
    better with complete sentences, disable early fragments". sentence_silence_duration adds a pause after each
    sentence."""
    return {"force_first_fragment_after_words": settings.tts_first_fragment_words,
            "fast_sentence_fragment": settings.tts_fast_sentence_fragment,
            "sentence_silence_duration": settings.tts_sentence_silence_ms / 1000}


def make_audio_io(settings: Settings, stream, *, callbacks: dict | None = None,
                  recorder_factory=make_recorder) -> AudioIO:
    browser = settings.audio_io == "browser"
    recorder = recorder_factory(settings, use_microphone=not browser, callbacks=callbacks)
    if browser:
        return BrowserAudio(recorder, tts_output_format(stream), play_kwargs=play_kwargs(settings))
    return ServerAudio(recorder, play_kwargs=play_kwargs(settings))
