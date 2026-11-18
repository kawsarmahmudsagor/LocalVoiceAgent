from voicebot import smoke_llm

class FakeBackend:
    name, model_label = "fake", "fake-model"
    def __init__(self, chunks): self.chunks, self.closed = chunks, False
    def start(self): pass
    def stream(self, messages, params, stop_event): yield from self.chunks
    def close(self): self.closed = True

def test_ok(capsys):
    b = FakeBackend(["A voice ", "assistant."])
    assert smoke_llm.main([], backend_factory=lambda s: b) == 0 and b.closed
    out = capsys.readouterr().out
    assert "fake-model" in out and "A voice assistant." in out and "ttft_ms=" in out

def test_think_leak_fails():
    assert smoke_llm.main([], backend_factory=lambda s: FakeBackend(["<think>x</think>Hi"])) == 1
