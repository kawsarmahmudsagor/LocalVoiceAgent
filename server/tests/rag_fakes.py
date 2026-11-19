"""Shared fakes for the RAG tests."""
from voicebot.rag.embedders import EmbedderError


class FakeEmbedder:
    """Two-dim vectors: texts containing 'canberra' point at [1, 0], everything else at [0, 1]."""
    def __init__(self, slug="hf:fake", dim=2, fail=False): self.slug, self.dim, self.fail = slug, dim, fail
    def embed(self, texts, kind):
        if self.fail: raise EmbedderError("Ollama not reachable at http://h")
        hit = lambda t: "canberra" in t.lower()
        return [([1.0] if hit(t) else [0.0]) + ([0.0] if hit(t) else [1.0]) + [0.0] * (self.dim - 2) for t in texts]
    def close(self): pass
