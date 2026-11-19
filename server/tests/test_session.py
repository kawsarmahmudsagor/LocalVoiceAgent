import logging, threading, time
from types import SimpleNamespace as NS
import pytest
from voicebot.conversation import Conversation
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.session import APOLOGY, VoiceSession, is_noise
from voicebot.timing import TurnTimer
from voicebot.rag.embedders import EmbedderError
from voicebot.rag.library import RagSettings
from voicebot.rag.service import EMPTY, RagError, Retrieval
from voicebot.rag.store import Hit


# ---------- fakes ----------

class FakeRecorder:
    """Like RealtimeSTT: text() returns the next transcript (firing on_stop first, like VAD
    ending the utterance, then 'transcribing' for 20 ms) or waits until abort().
    abort() wakes one waiting text(); with no text() running it blocks forever, like
    RealtimeSTT's abort_recording() when the recorder was left in 'transcribing'."""
    def __init__(self):
        self.texts, self.aborted, self.on_stop = [], False, None
        self._abort = threading.Event()
        self._lock = threading.Lock()
        self.active = self.max_active = 0
        self.hang_on_idle_abort = False
    def text(self):
        with self._lock:
            self.active += 1; self.max_active = max(self.max_active, self.active)
        try:
            while True:
                if self.texts:
                    t = self.texts.pop(0)
                    if self.on_stop: self.on_stop()
                    time.sleep(0.02)
                    return t
                if self._abort.is_set():
                    self._abort.clear(); return ""
                time.sleep(0.005)
        finally:
            with self._lock: self.active -= 1
    @property
    def waiting(self): return self.active > 0
    def abort(self):
        self.aborted = True
        if self.hang_on_idle_abort and self.active == 0:
            threading.Event().wait()          # never returns
        self._abort.set()


class FakeStream:
    def __init__(self):
        self.it, self.fed_text, self.consumed, self.stopped = None, None, [], False
        self.engine = NS(voice=None)
        self.engine.set_voice = lambda n: setattr(self.engine, "voice", n)
    def feed(self, x):
        if isinstance(x, str): self.fed_text = x; x = iter([x])
        self.it, self.consumed, self.stopped = x, [], False
    def text(self): return "".join(self.consumed)
    def stop(self):
        self.stopped = True
        time.sleep(getattr(self, "stop_delay", 0))


class FakeAudio:
    output_format = {"sample_rate": 24000}
    def __init__(self): self.log, self.recorder, self.sink = [], None, None
    def mute_input(self, muted): self.log.append(f"mute:{muted}")
    def cut(self): self.cuts = getattr(self, "cuts", 0) + 1
    def set_sink(self, s): self.sink = s
    def on_client_audio(self, data): pass
    def set_recorder(self, r): self.recorder = r
    def set_output_format(self, fmt): self.output_format = fmt
    play_seconds = None   # BrowserAudio returns the seconds of audio it sent to the browser
    word_times = None     # what on_words reports per sentence (None: the engine gave no word times)
    def play(self, stream, on_first_audio, on_sentence=None, on_words=None):
        self.log.append("play"); first = True
        for i, chunk in enumerate(stream.it):
            stream.consumed.append(chunk)
            if on_sentence: on_sentence(chunk, i * 0.5)   # each chunk stands in for one sentence
            if on_words: on_words(chunk, self.word_times, (i + 1) * 0.5)
            if first: on_first_audio(); first = False
            if stream.stopped: break
        return self.play_seconds


class FakeBackend:
    name, model_label = "fake", "fake-model"
    def __init__(self):
        self.chunks, self.calls, self.gate, self.error = ["Hello", " there."], [], None, None
    def stream(self, messages, params, stop_event):
        self.calls.append((messages, params))
        if self.error: raise self.error
        for i, c in enumerate(self.chunks):
            if stop_event.is_set(): return
            yield c
            if self.gate is not None and i == 0:
                while not self.gate.wait(0.01):
                    if stop_event.is_set(): return
    def start(self): pass
    def close(self): pass


class FakeRag:
    def __init__(self):
        self.settings, self.result, self.error, self.updates = RagSettings(enabled=True), EMPTY, None, []
        self.queries, self.update_error = [], None
    def retrieve(self, q):
        self.queries.append(q)
        if self.error: raise self.error
        return self.result
    def update(self, patch):
        if self.update_error: raise self.update_error
        self.updates.append(patch)
    def config_event(self): return {"type": "rag_config"}
    def status_event(self): return {"type": "rag_status"}


def wait_for(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v: return v
        time.sleep(0.01)
    pytest.fail("condition not met in time")

def types(events): return [e["type"] for e in events]


# ---------- fixtures ----------

@pytest.fixture
def events(): return []
@pytest.fixture
def conv(): return Conversation("SYS", 10)
@pytest.fixture
def stream(): return FakeStream()
@pytest.fixture
def audio(): return FakeAudio()
@pytest.fixture
def recorder(): return FakeRecorder()
@pytest.fixture
def backend(): return FakeBackend()
@pytest.fixture
def timer(): return TurnTimer()
@pytest.fixture
def sink(): return []

INFO = {"llm_backend": "llamacpp", "audio_io": "browser", "model_label": "fake-model",
        "stt_model": "small.en", "stt_device": "cuda", "stt_compute_type": "int8", "stt_silence_sec": 0.6}

@pytest.fixture
def make(backend, audio, stream, conv, recorder, events, timer, sink):
    made = []
    def _make(**extra):
        audio.recorder = recorder
        # The real recorder marks these through stt.timing_callbacks (Task 8).
        recorder.on_stop = lambda: (timer.mark("speech_end"), timer.mark("endpoint"), timer.mark("stt_start"))
        s = VoiceSession(backend=backend, audio=audio, stream=stream, conversation=conv,
                         params=GenerationParams(0.7, 0.8, 256), voice="af_heart", voices=["af_heart", "am_adam"],
                         info=INFO, timer=timer, metrics_sink=sink.append, ack_timeout=0.05, **extra)
        s.attach(events.append, lambda b: None); made.append(s); return s
    yield _make
    if backend.gate is not None: backend.gate.set()
    for s in made: s.close()

@pytest.fixture
def session(make): return make()

@pytest.fixture
def rag(): return FakeRag()

@pytest.fixture
def rag_session(make, rag): return make(rag=rag)

@pytest.fixture
def session_with_texts(request, make, recorder):
    recorder.texts = list(request.node.get_closest_marker("texts").args[0]); return make()

@pytest.fixture
def gated_session(make, backend):
    backend.gate = threading.Event(); return make()

@pytest.fixture
def failing_session(make, backend):
    backend.error = LLMError("down"); return make()


# ---------- tests ----------

def test_is_noise():
    assert all(is_noise(t) for t in ["", "  ", ".", "You.", "you", "a", "I"])
    assert not any(is_noise(t) for t in ["Hi", "What time is it?", "young"])

def test_typed_turn_event_order_and_history(session, events, conv):
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: types(events)[-1:] == ["status"] and events[-1]["state"] == "idle")
    assert types(events)[:3] == ["user_transcript", "status", "assistant_delta"] and events[1]["state"] == "thinking"
    assert types(events)[-3:] == ["assistant_done", "metrics", "status"]
    states = [e["state"] for e in events if e["type"] == "status"]; assert states == ["thinking", "speaking", "idle"]
    done = next(e for e in events if e["type"] == "assistant_done")
    assert done == {"type": "assistant_done", "text": "Hello there.", "interrupted": False}
    assert conv.turns == [("Hi", "Hello there.")]
    m = next(e for e in events if e["type"] == "metrics")
    assert m["source"] == "text" and m["turn_id"] == events[0]["turn_id"] == 1 and m["interrupted"] is False
    assert m["spans"]["total_ms"] >= 0 and m["spans"]["llm_ttft_ms"] is not None and m["spans"]["stt_ms"] is None

def test_spoken_sentences_reach_the_ui_with_audio_offsets(session, events):
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: "assistant_done" in types(events))
    segs = [e for e in events if e["type"] == "assistant_segment"]
    assert segs == [{"type": "assistant_segment", "turn_id": 1, "text": "Hello", "at_s": 0.0},
                    {"type": "assistant_segment", "turn_id": 1, "text": " there.", "at_s": 0.5}]
    assert types(events).index("assistant_segment") < types(events).index("assistant_done")
    words = [e for e in events if e["type"] == "assistant_words"]
    assert words == [{"type": "assistant_words", "turn_id": 1, "word_s": None, "end_s": 0.5},
                     {"type": "assistant_words", "turn_id": 1, "word_s": None, "end_s": 1.0}]
    order = [e["type"] for e in events if e["type"] in ("assistant_segment", "assistant_words")]
    assert order == ["assistant_segment", "assistant_words"] * 2     # each sentence's words follow it

def test_native_word_times_reach_the_ui(session, events):
    session.audio.word_times = [0.0, 0.2]
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: "assistant_done" in types(events))
    assert next(e for e in events if e["type"] == "assistant_words")["word_s"] == [0.0, 0.2]

@pytest.mark.texts(["What time is it?"])
def test_voice_turn_metrics_and_sink(session_with_texts, sink, events):     # sink fixture: list.append
    session_with_texts.handle({"type": "start_listening"}); wait_for(lambda: sink)
    assert sink[0]["source"] == "voice" and sink[0]["marks"]["stt_done"] is not None
    assert sink[0]["stt_model"] == "small.en" and "ts" in sink[0] and "type" not in sink[0]

def test_playback_ack_only_for_current_turn(gated_session, events, timer):   # Review Focus 6
    gated_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: "user_transcript" in types(events))
    gated_session.handle({"type": "playback_started", "turn_id": 99}); assert not timer.has("playback_started")
    gated_session.handle({"type": "playback_started", "turn_id": events[0]["turn_id"]}); assert timer.has("playback_started")

def test_mute_before_play_unmute_after(session, audio):
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: "mute:False" in audio.log)
    assert audio.log == ["mute:True", "play", "mute:False"]

@pytest.mark.texts(["What time is it?"])
def test_voice_turn_and_stop_listening(session_with_texts, recorder, backend):
    session_with_texts.handle({"type": "start_listening"}); wait_for(lambda: backend.calls)
    assert backend.calls[0][0][-1] == {"role": "user", "content": "What time is it?"}
    session_with_texts.handle({"type": "stop_listening"}); wait_for(lambda: recorder.aborted)

@pytest.mark.texts(["", ".", "you", "a"])
def test_noise_skipped(session_with_texts, backend):
    session_with_texts.handle({"type": "start_listening"}); time.sleep(0.2); assert backend.calls == []

def test_stop_speaking_marks_interrupted_and_keeps_partial(gated_session, backend, stream, conv, events):
    gated_session.handle({"type": "send_text", "text": "Tell me"}); wait_for(lambda: "assistant_delta" in types(events))
    gated_session.handle({"type": "stop_speaking"}); backend.gate.set()
    done = wait_for(lambda: next((e for e in events if e["type"] == "assistant_done"), None))
    assert done["interrupted"] is True and stream.stopped and conv.turns[0][1] == done["text"] != ""

def test_stop_during_thinking_unmutes(gated_session, audio, events):          # Review Focus 2
    gated_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: "mute:True" in audio.log)
    gated_session.handle({"type": "stop_speaking"}); wait_for(lambda: audio.log[-1] == "mute:False")

def test_stop_when_idle_is_noop(session, events):                            # Review Focus 2
    session.handle({"type": "stop_speaking"}); assert events == []

def test_busy(gated_session, events):
    gated_session.handle({"type": "send_text", "text": "a"}); gated_session.handle({"type": "send_text", "text": "b"})
    assert any(e["type"] == "error" and e["code"] == "busy" for e in events)

def test_empty_send_text_ignored(session, backend, events):                  # Review Focus 5
    session.handle({"type": "send_text", "text": "   "}); time.sleep(0.1); assert backend.calls == [] and events == []

def test_llm_error_apologizes_and_recovers(failing_session, stream, events, conv):  # Review Focus 4
    failing_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: events and events[-1].get("state") == "idle")
    assert any(e["type"] == "error" and e["code"] == "llm_error" for e in events) and stream.fed_text == APOLOGY
    assert {"type": "assistant_done", "text": APOLOGY, "interrupted": False} in events
    assert "metrics" not in types(events) and conv.turns == []

def test_update_settings_next_turn(session, backend, stream):
    session.handle({"type": "update_settings", "temperature": 0.2, "max_tokens": 100, "system_prompt": "S", "voice": "am_adam"})
    session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: backend.calls)
    msgs, params = backend.calls[0]
    assert params == GenerationParams(0.2, 0.8, 100) and msgs[0]["content"] == "S" and stream.engine.voice == "am_adam"

@pytest.mark.parametrize("patch", [{"temperature": "hot"}, {"max_tokens": 0}, {"voice": "nope"}, {"top_p": 1.5}])
def test_update_settings_rejects_invalid(session, events, patch):            # Review Focus 1
    session.handle({"type": "update_settings", "temperature": 0.1, **patch})
    assert events[-1]["code"] == "bad_settings" and session.config()["defaults"]["temperature"] == 0.7

def test_emit_after_detach_is_safe(session, events):                          # Review Focus 3
    session.detach(); session.handle({"type": "send_text", "text": "Hi"}); time.sleep(0.2); assert events == []

def test_unknown_command_and_clear(session, events, conv):
    conv.add_exchange("u", "a"); session.handle({"type": "clear"}); session.handle({"type": "dance"})
    assert conv.turns == [] and events[-1]["code"] == "bad_command"

# Per-stage timing (user request): silence end, STT done, LLM first token, TTS first audio.

@pytest.mark.texts(["What time is it?"])
def test_voice_turn_stage_spans_and_log_line(session_with_texts, events, caplog):
    caplog.set_level(logging.INFO, logger="voicebot.session")
    session_with_texts.handle({"type": "start_listening"})
    m = wait_for(lambda: next((e for e in events if e["type"] == "metrics"), None))
    sp = m["spans"]
    assert sp["stt_ms"] >= 15                     # FakeRecorder transcribes for 20 ms after endpoint
    assert sp["llm_ttft_ms"] is not None and sp["tts_first_audio_ms"] is not None and sp["perceived_ms"] >= sp["stt_ms"]
    assert m["marks"]["speech_end"] == 0 and sp["playback_ms"] is None        # no browser ack within ack_timeout
    line = wait_for(lambda: next((r.getMessage() for r in caplog.records if "turn timing" in r.getMessage()), None))
    for span in ("silence_wait_ms", "stt_ms", "llm_ttft_ms", "tts_first_audio_ms", "perceived_ms"): assert span in line

def test_timer_reset_after_turn(session, events, timer):
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: events and events[-1].get("state") == "idle")
    assert not timer.has("input") and not timer.has("play_done")

def test_stop_speaking_does_not_block_and_cuts_audio(gated_session, stream, audio, events):
    # RealtimeTTS stream.stop() can block for seconds; handle() runs on the web event loop.
    stream.stop_delay = 0.5
    gated_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: "assistant_delta" in types(events))
    t0 = time.monotonic(); gated_session.handle({"type": "stop_speaking"})
    assert time.monotonic() - t0 < 0.1 and audio.cuts == 1
    wait_for(lambda: stream.stopped)

def test_turn_drops_audio_left_from_a_stopped_turn(session, stream, events):
    # RealtimeTTS can leave synthesized audio of a stopped turn in engine.queue;
    # it must not play at the start of the next turn.
    import queue
    stream.engine.queue = queue.Queue(); stream.engine.queue.put(b"stale")
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: events and events[-1].get("state") == "idle")
    assert stream.engine.queue.empty()

def test_stop_silences_player_immediately_before_stream_stop(gated_session, stream, events):
    # stream.stop() lets play() reach its own cleanup, which drains the speaker buffer
    # (seconds of audio). The immediate player stop must happen first.
    order = []
    stream.player = NS(stop=lambda immediate=False: order.append(("player", immediate)))
    real_stop = stream.stop
    stream.stop = lambda: (order.append(("stream", None)), real_stop())
    gated_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: "assistant_delta" in types(events))
    gated_session.handle({"type": "stop_speaking"}); wait_for(lambda: len(order) == 2)
    assert order == [("player", True), ("stream", None)]

def test_quick_mic_toggle_keeps_one_listener(session, recorder, backend, events):     # review #4
    session.handle({"type": "start_listening"}); wait_for(lambda: recorder.active == 1)
    session.handle({"type": "stop_listening"}); session.handle({"type": "start_listening"})
    time.sleep(0.2)
    recorder.texts.append("What time is it?"); wait_for(lambda: backend.calls)
    time.sleep(0.1)
    assert recorder.max_active == 1 and session._listening is True

def test_close_does_not_hang_on_idle_recorder_abort(session, recorder):                # review #3
    session.handle({"type": "start_listening"}); wait_for(lambda: recorder.active == 1)
    session.handle({"type": "stop_listening"}); wait_for(lambda: recorder.active == 0)
    recorder.hang_on_idle_abort = True
    t0 = time.monotonic(); session.close(); assert time.monotonic() - t0 < 3

def test_browser_turn_stays_open_until_audio_has_played(session, audio, events):       # review #2
    # Kokoro synthesizes faster than real time; the turn (muted mic, Stop enabled) must
    # last until the browser has played what it was sent.
    audio.play_seconds = 0.6
    t0 = time.monotonic(); session.handle({"type": "send_text", "text": "Hi"})
    wait_for(lambda: "assistant_done" in types(events))
    assert time.monotonic() - t0 >= 0.5 and audio.log[-1] == "mute:False"
    assert [e["state"] for e in events if e["type"] == "status"][-2:] == ["speaking", "idle"]

def test_stop_during_browser_playback_ends_turn_at_once(session, audio, events):        # review #2
    audio.play_seconds = 5.0
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: "speaking" in [e.get("state") for e in events])
    time.sleep(0.1); t0 = time.monotonic(); session.handle({"type": "stop_speaking"})
    done = wait_for(lambda: next((e for e in events if e["type"] == "assistant_done"), None))
    assert time.monotonic() - t0 < 0.5 and done["interrupted"] is True


# ---------- v2: model switching (plan Task 13) ----------

from voicebot.catalog import by_slug, load_catalog
from voicebot.config import REPO_ROOT, Settings
from voicebot.models import ModelManager

def fake_tts_stream(entry):
    s = FakeStream(); rate = 22050 if entry.slug == "piper" else 24000
    s.engine.get_stream_info = lambda: (8, 1, rate); s.slug = entry.slug
    return s

@pytest.fixture
def session_with_models(make, recorder, stream):
    pool = by_slug(load_catalog(REPO_ROOT / "bench" / "pool.toml"))
    entries = [pool[s] for s in ("whisper-small-en", "nemotron-rt", "kokoro", "piper")]
    first = [recorder]
    rf = lambda final, realtime, **kw: first.pop() if first else FakeRecorder()
    models = ModelManager(entries, Settings.load(env={}), recorder_factory=rf, tts_factory=fake_tts_stream,
                          unload=lambda o: None)
    models.start(False, {})
    return make(models=models, callbacks={})

def test_select_model_busy_during_turn(gated_session, events):
    gated_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: "user_transcript" in types(events))
    gated_session.handle({"type": "select_model", "role": "tts", "slug": "kokoro"})
    assert events[-1]["code"] == "busy"

def test_select_model_bad_slug(session_with_models, events):
    session_with_models.handle({"type": "select_model", "role": "stt_final", "slug": "nemotron-rt"})
    assert events[-1]["type"] == "error" and events[-1]["code"] == "bad_model"

def test_select_model_while_listening_restarts_loop(session_with_models, recorder, events):   # Review Focus 1
    session_with_models.handle({"type": "start_listening"}); wait_for(lambda: recorder.waiting)
    session_with_models.handle({"type": "select_model", "role": "stt_realtime", "slug": "nemotron-rt"})
    st = wait_for(lambda: next((e for e in events if e["type"] == "model_status" and e["state"] == "ready"), None))
    assert recorder.aborted and st["role"] == "stt_realtime" and session_with_models.audio.recorder is not recorder
    assert wait_for(lambda: session_with_models.audio.recorder.waiting)          # listening again on the new recorder

def test_switch_while_listening_loses_no_turn(session_with_models, recorder, backend, events):  # Review Focus 1
    session_with_models.handle({"type": "start_listening"}); wait_for(lambda: recorder.waiting)
    session_with_models.handle({"type": "select_model", "role": "stt_realtime", "slug": "nemotron-rt"})
    wait_for(lambda: any(e["type"] == "model_status" and e["state"] == "ready" for e in events))
    new = session_with_models.audio.recorder; wait_for(lambda: new.waiting)
    new.texts.append("What time is it?"); wait_for(lambda: "assistant_done" in types(events))
    time.sleep(0.1)
    assert len(backend.calls) == 1 and recorder.max_active == 1 and new.max_active == 1

def test_tts_switch_sends_new_output_format(session_with_models, events):     # Review Focus 4
    session_with_models.handle({"type": "select_model", "role": "tts", "slug": "piper"})   # fake stream reports 22050 Hz
    cfg = wait_for(lambda: next((e for e in events if e["type"] == "config"), None))
    assert cfg["output_format"]["sample_rate"] == 22050 and cfg["tts"] == "piper"
    assert session_with_models.stream.slug == "piper"

def test_partials_and_metrics_slugs(session_with_models, sink, events):
    session_with_models.on_partial("hel"); assert events[-1] == {"type": "user_partial", "text": "hel"}
    session_with_models.handle({"type": "send_text", "text": "x"}); wait_for(lambda: sink)
    assert sink[0]["stt_final"] == "whisper-small-en" and sink[0]["tts_model"] == "kokoro" and "runtime" in sink[0]

def test_config_has_model_fields(session_with_models):
    c = session_with_models.config()
    assert (c["stt_final"], c["stt_realtime"], c["tts"], c["gpu_total_mb"], c["llm_vram_mb"]) == \
        ("whisper-small-en", "off", "kokoro", 32607, 7000)

def test_tts_failure_reports_error_not_an_empty_turn(session, audio, events, sink):   # e2e: NLTK data missing
    # RealtimeTTS fails on its own thread (e.g. stream2sentence cannot load NLTK data) and play()
    # returns without ever reading the LLM stream. The user must see an error, and no metrics are logged.
    audio.play = lambda stream, on_first_audio, on_sentence=None, on_words=None: None
    session.handle({"type": "send_text", "text": "Hi"})
    err = wait_for(lambda: next((e for e in events if e["type"] == "error"), None))
    assert err["code"] == "tts_error" and sink == []

def test_metrics_record_keeps_transcript_and_reply(session_with_models, sink):
    session_with_models.handle({"type": "send_text", "text": "hello there"}); wait_for(lambda: sink)
    assert sink[0]["transcript"] == "hello there"
    assert sink[0]["reply"] and len(sink[0]["reply"]) == sink[0]["reply_chars"]


# ---------- v3: RAG (Task 7) ----------

def test_rag_context_sources_and_timing(rag_session, rag, backend, events, sink):
    rag.result = Retrieval([Hit("d1", "geo.txt", 0, "Canberra is the capital.", 0.82)], "[geo.txt, part 1]\nCanberra is the capital.")
    rag_session.handle({"type": "send_text", "text": "Capital?"}); wait_for(lambda: sink)
    assert "Canberra is the capital." in backend.calls[0][0][0]["content"]
    src = next(e for e in events if e["type"] == "rag_sources")
    assert src["sources"] == [{"name": "geo.txt", "chunk": 0, "score": 0.82, "preview": "Canberra is the capital."}]
    assert sink[0]["spans"]["retrieval_ms"] is not None and sink[0]["rag_chunks_used"] == 1
    assert sink[0]["rag_embedder"] == rag.settings.embedder

def test_rag_sources_before_llm_and_history_has_no_context(rag_session, rag, events, conv, sink):
    rag.result = Retrieval([Hit("d1", "a.txt", 2, "x" * 300, 0.7)], "[a.txt, part 3]\n" + "x" * 300)
    rag_session.handle({"type": "send_text", "text": "Q"}); wait_for(lambda: sink)
    t = types(events)
    assert t.index("rag_sources") < t.index("assistant_delta")
    src = next(e for e in events if e["type"] == "rag_sources")
    assert src["turn_id"] == 1 and len(src["sources"][0]["preview"]) == 160
    assert conv.turns == [("Q", "Hello there.")] and rag.queries == ["Q"]
    r = sink[0]
    assert r["rag_enabled"] is True and r["rag_context_chars"] == len(rag.result.context)
    assert (r["rag_top_k"], r["rag_min_score"], r["rag_max_context_chars"], r["rag_chunk_size"],
            r["rag_chunk_overlap"]) == (3, 0.5, 2000, 800, 120)
    assert "rag_error" not in r

def test_rag_empty_result_sends_no_sources(rag_session, rag, backend, events, sink):
    rag_session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: sink)
    assert "rag_sources" not in types(events) and backend.calls[0][0][0]["content"] == "SYS"
    assert sink[0]["rag_chunks_used"] == 0 and sink[0]["rag_context_chars"] == 0

def test_rag_disabled_skips_retrieval(rag_session, rag, sink):
    rag.settings = RagSettings(enabled=False)
    rag_session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: sink)
    assert rag.queries == [] and sink[0]["rag_enabled"] is False and sink[0]["rag_embedder"] is None
    assert sink[0]["spans"]["retrieval_ms"] is None

@pytest.mark.parametrize("fail", [False, True])
def test_rag_log_names_settings_retrieve_used(rag_session, rag, sink, fail):
    # An embedder switch that lands while retrieve() runs: the log must name the new settings.
    def retrieve(q):
        rag.settings = RagSettings(enabled=True, embedder="ollama:nomic-embed-text", top_k=5)
        if fail: raise EmbedderError("down")
        return EMPTY
    rag.retrieve = retrieve
    rag_session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: sink)
    assert (sink[0]["rag_embedder"], sink[0]["rag_top_k"]) == ("ollama:nomic-embed-text", 5)
    assert sink[0]["rag_enabled"] is True and ("rag_error" in sink[0]) is fail

def test_rag_failure_still_answers(rag_session, rag, events, sink):
    rag.error = EmbedderError("Ollama not reachable at http://h")
    rag_session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: sink)
    assert any(e["type"] == "error" and e["code"] == "rag_error" for e in events)
    assert sink[0]["reply"] and sink[0]["rag_error"].startswith("Ollama not reachable")

def test_update_rag_forwards_and_rejects_while_busy(rag_session, rag, events, backend):
    rag_session.handle({"type": "update_rag", "top_k": 4}); wait_for(lambda: rag.updates)
    assert rag.updates == [{"top_k": 4}]
    backend.gate = threading.Event(); rag_session.handle({"type": "send_text", "text": "x"})
    wait_for(lambda: "assistant_delta" in types(events))
    before = types(events).count("rag_config")
    rag_session.handle({"type": "update_rag", "embedder": "ollama:nomic-embed-text"})
    assert wait_for(lambda: any(e["type"] == "error" and e["code"] == "rag_busy" for e in events))
    # a fresh rag_config resyncs the RAG tab's drafts to the unchanged settings
    assert wait_for(lambda: types(events).count("rag_config") == before + 1)

def test_rag_busy_resync_runs_off_the_calling_thread(rag_session, rag, events, backend):
    threads = []
    rag.config_event = lambda: (threads.append(threading.current_thread()), {"type": "rag_config"})[1]
    backend.gate = threading.Event(); rag_session.handle({"type": "send_text", "text": "x"})
    wait_for(lambda: "assistant_delta" in types(events))
    rag_session.handle({"type": "update_rag", "top_k": 4})
    assert threads == [] or threads[0] is not threading.current_thread()
    assert wait_for(lambda: any(e["type"] == "error" and e["code"] == "rag_busy" for e in events))
    assert wait_for(lambda: len(threads) == 1 and "rag_config" in types(events))
    assert threads[0] is not threading.current_thread()

def test_update_rag_maps_errors_and_emits_config(rag_session, rag, events):
    rag.update_error = ValueError("top_k: must be between 1 and 10")
    rag_session.handle({"type": "update_rag", "top_k": 99})
    wait_for(lambda: any(e["type"] == "error" and e["code"] == "bad_settings" for e in events))
    rag.update_error = RagError("Ollama model not pulled")
    rag_session.handle({"type": "update_rag", "embedder": "ollama:x"})
    wait_for(lambda: any(e["type"] == "error" and e["code"] == "rag_error" for e in events))
    # bad values: the session sends a fresh rag_config; a failed switch: the service does (not the session)
    wait_for(lambda: types(events).count("rag_config") == 1)

def test_update_rag_without_rag_is_bad_command(session, events):
    session.handle({"type": "update_rag", "top_k": 4})
    assert events[-1]["type"] == "error" and events[-1]["code"] == "bad_command"

def test_rag_events(session, rag_session):
    assert session.rag is None and session.rag_events() == []
    assert rag_session.rag_events() == [{"type": "rag_config"}, {"type": "rag_status"}]

def test_rag_template_needs_context_placeholder(session, events):
    session.handle({"type": "update_settings", "rag_template": "no placeholder"})
    assert events[-1]["code"] == "bad_settings" and "{context}" in events[-1]["message"]

def test_rag_template_update_and_default(session, conv):
    assert session.config()["defaults"]["rag_template"] == conv.rag_template
    session.handle({"type": "update_settings", "rag_template": "Use {notes}: {context}"})
    assert conv.rag_template == "Use {notes}: {context}"
    assert session.config()["defaults"]["rag_template"] == "Use {notes}: {context}"

def test_session_without_rag_logs_rag_off(session, sink):
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: sink)
    assert sink[0]["rag_enabled"] is False and sink[0]["spans"]["retrieval_ms"] is None


# ---------- forced word alignment toggle ----------

class FakeAligner:
    def __init__(self, fail=False): self.ready, self.loads, self.fail = False, 0, fail
    def load(self):
        self.loads += 1
        if self.fail: raise RuntimeError("no weights")
        self.ready = True

def align_events(events): return [e for e in events if e["type"] == "word_align"]

def test_word_align_loads_the_model_once_then_switches_on_and_off(make, events, audio):
    fa = FakeAligner(); s = make(word_aligner=fa)
    assert s.config()["defaults"]["word_align"] is False and getattr(audio, "word_aligner", None) is None
    s.handle({"type": "update_settings", "word_align": True})
    wait_for(lambda: any(e["state"] == "ready" for e in align_events(events)))
    assert [e["state"] for e in align_events(events)] == ["loading", "ready"]
    assert audio.word_aligner is fa and s.config()["defaults"]["word_align"] is True
    s.handle({"type": "update_settings", "word_align": False})
    assert audio.word_aligner is None and align_events(events)[-1] == {"type": "word_align", "state": "off"}
    s.handle({"type": "update_settings", "word_align": True})
    wait_for(lambda: align_events(events)[-1]["state"] == "ready")
    assert fa.loads == 1 and audio.word_aligner is fa                 # already loaded: no second load

def test_word_align_load_failure_reports_and_stays_off(make, events, audio):
    s = make(word_aligner=FakeAligner(fail=True))
    s.handle({"type": "update_settings", "word_align": True})
    wait_for(lambda: align_events(events) and align_events(events)[-1]["state"] == "error")
    assert "no weights" in align_events(events)[-1]["message"]
    assert getattr(audio, "word_aligner", None) is None and s.config()["defaults"]["word_align"] is False

def test_word_align_without_an_aligner_or_with_a_bad_value_is_rejected(session, events):
    session.handle({"type": "update_settings", "word_align": "yes"})
    assert events[-1]["code"] == "bad_settings"
    session.handle({"type": "update_settings", "word_align": True})
    assert align_events(events)[-1]["state"] == "error"                # this server has no aligner

def test_turn_log_records_word_align(make, events, sink, audio):
    fa = FakeAligner(); s = make(word_aligner=fa)
    s.handle({"type": "update_settings", "word_align": True})
    wait_for(lambda: audio.__dict__.get("word_aligner") is fa)
    s.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: sink)
    assert sink[-1]["word_align"] is True
