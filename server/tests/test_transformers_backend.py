import threading, pytest, torch
from voicebot.config import Settings
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.llm.transformers_backend import TransformersBackend, StopOnEvent, build_inputs

class FakeInputs(dict):
    def to(self, device): return self

class FakeTokenizer:
    def __init__(self): self.kwargs = None
    def apply_chat_template(self, messages, **kw): self.kwargs = kw; return FakeInputs(input_ids=[[1, 2]])

class FakeModel:
    device = "cpu"
    def __init__(self, texts=("Hel", "lo"), error=None): self.texts, self.error, self.kwargs = texts, error, None
    def generate(self, **kw):
        self.kwargs = kw
        if self.error: raise self.error
        for i, t in enumerate(self.texts): kw["streamer"].on_finalized_text(t, stream_end=i == len(self.texts) - 1)

P = GenerationParams(0.4, 0.85, 64)

def test_build_inputs_disables_thinking():
    tok = FakeTokenizer(); build_inputs(tok, [{"role": "user", "content": "x"}])
    assert tok.kwargs["enable_thinking"] is False and tok.kwargs["add_generation_prompt"] is True and tok.kwargs["return_dict"] is True

def test_stop_on_event():
    ev = threading.Event(); s = StopOnEvent(ev); assert s(None, None) is False; ev.set(); assert s(None, None) is True

def test_stream_yields_text_and_passes_params():
    model = FakeModel(); b = TransformersBackend(Settings.load(env={}), loader=lambda: (model, FakeTokenizer())); b.start()
    assert "".join(b.stream([{"role": "user", "content": "x"}], P, threading.Event())) == "Hello"
    assert (model.kwargs["max_new_tokens"], model.kwargs["temperature"], model.kwargs["top_p"], model.kwargs["do_sample"]) == (64, 0.4, 0.85, True)
    assert any(isinstance(c, StopOnEvent) for c in model.kwargs["stopping_criteria"])

def test_generate_error_raises_without_hanging():
    b = TransformersBackend(Settings.load(env={}), loader=lambda: (FakeModel(error=RuntimeError("boom")), FakeTokenizer())); b.start()
    with pytest.raises(LLMError, match="boom"): list(b.stream([], P, threading.Event()))

def test_oom_hint():
    def loader(): raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(LLMError, match="STT_DEVICE=cpu"): TransformersBackend(Settings.load(env={}), loader=loader).start()
