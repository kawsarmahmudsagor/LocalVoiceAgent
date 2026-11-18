import logging, threading, time
from types import SimpleNamespace as NS
import pytest
from voicebot.conversation import Conversation
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.session import APOLOGY, VoiceSession, is_noise


# ---------- fakes ----------

class FakeRecorder:
    """text() pops the next transcript (firing on_stop first, like VAD ending the
    utterance, then 'transcribing' for 20 ms), or blocks until abort()."""
    def __init__(self):
        self.texts, self.aborted, self.on_stop = [], False, None
        self._abort = threading.Event()
    def text(self):
        if self.texts:
            t = self.texts.pop(0)
            if self.on_stop: self.on_stop()
            time.sleep(0.02)
            return t
        self._abort.wait()
        return ""
    def abort(self):
        self.aborted = True; self._abort.set()


class FakeStream:
    def __init__(self):
        self.it, self.fed_text, self.consumed, self.stopped = None, None, [], False
        self.engine = NS(voice=None)
        self.engine.set_voice = lambda n: setattr(self.engine, "voice", n)
    def feed(self, x):
        if isinstance(x, str): self.fed_text = x; x = iter([x])
        self.it, self.consumed, self.stopped = x, [], False
    def text(self): return "".join(self.consumed)
    def stop(self): self.stopped = True


class FakeAudio:
    output_format = None
    def __init__(self): self.log, self.recorder, self.sink = [], None, None
    def mute_input(self, muted): self.log.append(f"mute:{muted}")
    def set_sink(self, s): self.sink = s
    def on_client_audio(self, data): pass
    def play(self, stream, on_first_audio):
        self.log.append("play"); first = True
        for chunk in stream.it:
            stream.consumed.append(chunk)
            if first: on_first_audio(); first = False
            if stream.stopped: break


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

def _make(backend, audio, stream, conv, recorder, events):
    audio.recorder = recorder
    s = VoiceSession(backend=backend, audio=audio, stream=stream, conversation=conv,
                     params=GenerationParams(0.7, 0.8, 256), voice="af_heart", voices=["af_heart", "am_adam"],
                     info={"llm_backend": "llamacpp", "audio_io": "browser", "model_label": "fake-model"})
    recorder.on_stop = s.mark_silence_end
    s.attach(events.append, lambda b: None)
    return s

@pytest.fixture
def session(backend, audio, stream, conv, recorder, events):
    s = _make(backend, audio, stream, conv, recorder, events); yield s; s.close()

@pytest.fixture
def session_with_texts(request, backend, audio, stream, conv, recorder, events):
    recorder.texts = list(request.node.get_closest_marker("texts").args[0])
    s = _make(backend, audio, stream, conv, recorder, events); yield s; s.close()

@pytest.fixture
def gated_session(backend, audio, stream, conv, recorder, events):
    backend.gate = threading.Event()
    s = _make(backend, audio, stream, conv, recorder, events); yield s; backend.gate.set(); s.close()

@pytest.fixture
def failing_session(backend, audio, stream, conv, recorder, events):
    backend.error = LLMError("down")
    s = _make(backend, audio, stream, conv, recorder, events); yield s; s.close()


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
    m = next(e for e in events if e["type"] == "metrics"); assert m["total_ms"] >= 0 and m["ttft_ms"] is not None

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
def test_voice_turn_stage_timings(session_with_texts, events, caplog):
    caplog.set_level(logging.INFO, logger="voicebot.session")
    session_with_texts.handle({"type": "start_listening"})
    m = wait_for(lambda: next((e for e in events if e["type"] == "metrics"), None))
    assert m["stt_ms"] >= 15                                  # FakeRecorder transcribes for 20 ms after silence end
    assert m["ttft_ms"] is not None and m["first_audio_ms"] >= m["ttft_ms"]
    assert m["speech_to_audio_ms"] >= m["stt_ms"] + m["first_audio_ms"] - 1
    line = wait_for(lambda: next((r.getMessage() for r in caplog.records if "turn timing" in r.getMessage()), None))
    for stage in ("silence_end", "stt_done", "first_token", "first_audio"): assert stage in line

def test_typed_turn_has_no_stt_timings(session, events):
    session.handle({"type": "send_text", "text": "Hi"})
    m = wait_for(lambda: next((e for e in events if e["type"] == "metrics"), None))
    assert m["stt_ms"] is None and m["speech_to_audio_ms"] is None and m["first_audio_ms"] is not None
