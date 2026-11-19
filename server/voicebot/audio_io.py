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
             on_sentence: Callable[[str, float | None], None] | None = None,
             on_words: Callable[[str, list[float] | None, float], None] | None = None) -> float | None:
        """Blocks until synthesis ends. BrowserAudio returns the seconds of audio sent to
        the browser (which plays it later); ServerAudio returns None (already played).
        on_sentence(text, at_s) fires before each sentence is synthesized; at_s is where that
        sentence starts in the reply's audio (BrowserAudio), or None (ServerAudio).
        on_words(text, word_s, end_s) fires after it is synthesized (BrowserAudio only): word_s holds
        each whitespace word's start on the reply clock, or None when the engine gave no usable word
        times; end_s is where the sentence's audio ends."""
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
def _count_queued(engine, on_put: Callable[[object], None] | None = None) -> Iterator[Callable[[], int]]:
    """Count the audio bytes the engine puts on its queue during one play().

    Muted RealtimeTTS sends chunks to on_audio_chunk as fast as they are queued, and an engine may queue a
    whole sentence as one chunk, so the bytes *sent* when a sentence starts can trail by a sentence. Sentences
    are synthesized in order, so the bytes *queued* before a sentence starts are exactly its start offset.
    The count wraps put() on this queue instance only; the vendored code is unchanged. on_put, if given,
    sees every queued item too (forced alignment captures each sentence's audio with it)."""
    q = getattr(engine, "queue", None)
    if q is None or not hasattr(q, "put"):
        yield lambda: 0
        return
    count = 0
    had_own, own, real_put = "put" in vars(q), vars(q).get("put"), q.put

    def put(item, *args, **kwargs):
        nonlocal count
        count += _nbytes(item)
        if on_put is not None:
            on_put(item)
        return real_put(item, *args, **kwargs)

    q.put = put
    try:
        yield lambda: count
    finally:
        if had_own:
            q.put = own
        else:
            del q.put


def _pcm_bytes(item) -> bytes | None:
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    tobytes = getattr(item, "tobytes", None)  # numpy arrays
    return tobytes() if callable(tobytes) else None


def _alnum(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _drain(q) -> list:
    out = []
    if q is None:
        return out
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:  # queue.Empty
            return out


def word_times(text: str, timings: list, at_s: float, engine_start: float) -> list[float] | None:
    """Start time, on the reply clock, of each whitespace word of `text`, from an engine's word timings.

    Timings are on the engine's clock; `engine_start` is that clock when the sentence began and `at_s` is
    where the sentence starts in the reply. Tokens are matched to words by their letters and digits, so
    "It's" may be the tokens "It" + "'s" and punctuation tokens are skipped. A word with no letters or
    digits ("—") takes the previous word's time. Returns None when the tokens don't spell the text."""
    toks = [(_alnum(str(t.word)), float(t.start_time)) for t in timings]
    toks = [tk for tk in toks if tk[0]]
    out: list[float] = []
    i, prev = 0, at_s
    for word in text.split():
        target = _alnum(word)
        if not target:
            out.append(prev)
            continue
        acc, start = "", None
        while len(acc) < len(target) and i < len(toks):
            if start is None:
                start = toks[i][1]
            acc += toks[i][0]
            i += 1
        if acc != target or start is None:
            return None
        prev = at_s + (start - engine_start)
        out.append(prev)
    return out if i == len(toks) else None


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
        # voicebot.aligner.WordAligner, or None: word times for engines without native ones (Piper, Qwen, ...).
        self.word_aligner = None

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
             on_sentence: Callable[[str, float | None], None] | None = None,
             on_words: Callable[[str, list[float] | None, float], None] | None = None) -> float:
        first = _once(on_first_audio)
        engine = stream.engine
        timings_q = getattr(engine, "timings", None)
        # Native word times line up with the sent audio only when the engine reports them and doesn't trim it.
        native = bool(getattr(engine, "provides_word_timings", False)) and not getattr(engine, "trim_silence", False)
        _drain(timings_q)  # timings left over from an earlier reply
        sentence: dict = {}
        aligner = self.word_aligner
        align = not native and aligner is not None and getattr(aligner, "ready", False)
        captured: list[bytes] = []  # the current sentence's audio, for forced alignment

        def capture(item) -> None:
            if align and sentence.get("capturing"):
                b = _pcm_bytes(item)
                if b:
                    captured.append(b)
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

        def is_current() -> bool:
            with self._lock:
                return self._generation == generation

        with _count_queued(engine, capture) as queued:
            def before_sentence(text: str) -> None:
                _drain(timings_q)
                captured.clear()
                at_s = queued() / bytes_per_s
                sentence.update(at_s=at_s, engine_start=float(getattr(engine, "audio_duration", 0.0) or 0.0),
                                capturing=True)
                if is_current() and on_sentence is not None:
                    on_sentence(text, at_s)

            def after_sentence(text: str) -> None:
                timings = _drain(timings_q)
                if not sentence or not is_current() or on_words is None:
                    return
                sentence["capturing"] = False
                words = None
                if native:
                    words = word_times(text, timings, sentence["at_s"], sentence["engine_start"])
                elif align:
                    # Runs on the synthesis thread before the next sentence: tens of ms on the GPU, while
                    # the browser is still playing audio queued earlier.
                    rel = aligner.word_starts(text, b"".join(captured), rate, channels)
                    words = [sentence["at_s"] + r for r in rel] if rel is not None else None
                on_words(text, words, queued() / bytes_per_s)

            stream.play(on_audio_chunk=on_chunk, muted=True, before_sentence_synthesized=before_sentence,
                        on_sentence_synthesized=after_sentence, **self.play_kwargs)
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
             on_sentence: Callable[[str, float | None], None] | None = None,
             on_words: Callable[[str, list[float] | None, float], None] | None = None) -> None:
        first = _once(on_first_audio)  # on_words is unused: the UI shows each sentence at once

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
