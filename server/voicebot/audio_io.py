"""Where audio comes from and goes to: the browser or this PC. See spec §4.6."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Protocol

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
    def play(self, stream, on_first_audio: Callable[[], None],
             on_sentence: Callable[[str, float | None], None] | None = None) -> float | None:
        """Blocks until synthesis ends. BrowserAudio returns the seconds of audio sent to
        the browser (which plays it later); ServerAudio returns None (already played).
        on_sentence(text, at_s) fires before each sentence is synthesized; at_s is where that
        sentence starts in the reply's audio (BrowserAudio), or None (ServerAudio)."""
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


def _nbytes(chunk) -> int:
    n = getattr(chunk, "nbytes", None)  # numpy arrays (Kokoro queues int16 arrays)
    if n is not None:
        return int(n)
    try:
        return len(chunk)
    except TypeError:
        return 0


@contextmanager
def _count_queued(engine) -> Iterator[Callable[[], int]]:
    """Count the audio bytes the engine puts on its queue during one play().

    Muted RealtimeTTS sends chunks to on_audio_chunk as fast as they are queued, and an engine may queue a
    whole sentence as one chunk, so the bytes *sent* when a sentence starts can trail by a sentence. Sentences
    are synthesized in order, so the bytes *queued* before a sentence starts are exactly its start offset.
    The count wraps put() on this queue instance only; the vendored code is unchanged."""
    q = getattr(engine, "queue", None)
    if q is None or not hasattr(q, "put"):
        yield lambda: 0
        return
    count = 0
    had_own, own, real_put = "put" in vars(q), vars(q).get("put"), q.put

    def put(item, *args, **kwargs):
        nonlocal count
        count += _nbytes(item)
        return real_put(item, *args, **kwargs)

    q.put = put
    try:
        yield lambda: count
    finally:
        if had_own:
            q.put = own
        else:
            del q.put


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

    def play(self, stream, on_first_audio: Callable[[], None],
             on_sentence: Callable[[str, float | None], None] | None = None) -> float:
        first = _once(on_first_audio)
        with self._lock:
            generation = self._generation
        sent = 0
        rate = self.output_format.get("sample_rate", 24000)
        channels = self.output_format.get("channels", 1)
        bytes_per_s = 2 * channels * rate  # s16le

        def on_chunk(chunk: bytes) -> None:
            nonlocal sent
            with self._lock:
                sink = self._sink if self._generation == generation else None
            if sink is None:
                return
            first()
            sink(chunk)
            sent += len(chunk)

        with _count_queued(stream.engine) as queued:
            def before_sentence(text: str) -> None:
                with self._lock:
                    current = self._generation == generation
                if current and on_sentence is not None:
                    on_sentence(text, queued() / bytes_per_s)

            stream.play(on_audio_chunk=on_chunk, muted=True, before_sentence_synthesized=before_sentence,
                        **self.play_kwargs)
        return sent / bytes_per_s


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

    def play(self, stream, on_first_audio: Callable[[], None],
             on_sentence: Callable[[str, float | None], None] | None = None) -> None:
        first = _once(on_first_audio)

        def before_sentence(text: str) -> None:
            if on_sentence is not None:
                on_sentence(text, None)  # the PC speakers play it; the UI shows it at once

        stream.play(on_audio_chunk=lambda _chunk: first(), before_sentence_synthesized=before_sentence,
                    **self.play_kwargs)
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
