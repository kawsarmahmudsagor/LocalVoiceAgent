import threading, httpx, openai, pytest
from types import SimpleNamespace as NS
from voicebot.config import Settings
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.llm.llamacpp_backend import LlamaCppBackend

class FakeStream:
    def __init__(self, deltas): self.deltas, self.closed = deltas, False
    def __iter__(self):
        for d in self.deltas: yield NS(choices=[NS(delta=NS(content=d))])
    def close(self): self.closed = True

class FakeClient:
    def __init__(self, deltas=(), error=None):
        self.kwargs, self.stream_obj, self.error = None, FakeStream(list(deltas)), error
        self.chat = NS(completions=NS(create=self.create))
    def create(self, **kw):
        if self.error: raise self.error
        self.kwargs = kw; return self.stream_obj

P = GenerationParams(temperature=0.3, top_p=0.9, max_tokens=64)
MSGS = [{"role": "user", "content": "hi"}]

def backend(client, **env):
    return LlamaCppBackend(Settings.load(env={"LLAMA_AUTOSTART": "false", **env}), client=client,
                           status=lambda u: 200, sleep=lambda s: None)

def test_sends_thinking_off_and_params():
    c = FakeClient(["Hi"]); list(backend(c).stream(MSGS, P, threading.Event()))
    assert c.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert (c.kwargs["stream"], c.kwargs["temperature"], c.kwargs["top_p"], c.kwargs["max_tokens"]) == (True, 0.3, 0.9, 64)
    assert c.kwargs["messages"] == MSGS

def test_skips_empty_deltas():
    assert list(backend(FakeClient(["Hi", None, "", " there"])).stream(MSGS, P, threading.Event())) == ["Hi", " there"]

def test_stop_event_closes_stream():
    c, ev = FakeClient(["Hi", " more", " text"]), threading.Event(); out = []
    for t in backend(c).stream(MSGS, P, ev): out.append(t); ev.set()
    assert out == ["Hi"] and c.stream_obj.closed

def test_connection_error_becomes_llm_error():   # Review Focus 4
    err = openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8081/v1/chat/completions"))
    with pytest.raises(LLMError, match="llama-server"):
        list(backend(FakeClient(error=err)).stream(MSGS, P, threading.Event()))

def test_start_autostart_uses_server_and_close_stops_it():
    calls = []
    class FakeServer:
        def __init__(self, s): calls.append("init")
        def start(self): calls.append("start")
        def close(self): calls.append("close")
    b = LlamaCppBackend(Settings.load(env={}), client=FakeClient(), server_factory=FakeServer)
    b.start(); b.close()
    assert calls == ["init", "start", "close"]

def test_start_external_waits_for_health():
    seq = [503, 200]
    b = LlamaCppBackend(Settings.load(env={"LLAMA_AUTOSTART": "false"}), client=FakeClient(),
                        server_factory=lambda s: pytest.fail("must not start server"),
                        status=lambda u: seq.pop(0), sleep=lambda s: None)
    b.start(); assert seq == []
