import pytest
from fastapi.testclient import TestClient

from rag_fakes import FakeEmbedder
from voicebot.config import Settings
from voicebot.rag.library import RagSettings
from voicebot.rag.service import RagService
from voicebot.web import create_app


class StubSession:
    def __init__(self, rag):
        self.rag, self.emit = rag, None
    def attach(self, emit, send_binary): self.emit = emit
    def detach(self): pass
    def handle(self, cmd): pass
    def on_client_audio(self, data): pass
    def config(self): return {"model_label": "m"}
    def catalog(self): return []
    def close(self): pass
    def rag_events(self):
        return [] if self.rag is None else [self.rag.config_event(), self.rag.status_event()]


def _client(rag):
    return TestClient(create_app(Settings.load(env={}), session_factory=lambda s: StubSession(rag)))


@pytest.fixture
def client(tmp_path):
    rag = RagService(tmp_path, RagSettings(enabled=True), make_embedder=lambda slug: FakeEmbedder(slug),
                     list_embedders=lambda: [{"slug": "hf:fake", "available": True, "reason": ""}])
    rag.start()
    with _client(rag) as c:
        yield c
    rag.close()


@pytest.fixture
def client_without_rag():
    with _client(None) as c:
        yield c


def test_upload_several_with_one_bad(client):
    r = client.post("/api/rag/documents", files=[("files", ("a.txt", b"Canberra.", "text/plain")),
                                                 ("files", ("b.doc", b"x", "application/msword"))])
    assert r.status_code == 202
    assert [("doc_id" in x, "error" in x) for x in r.json()] == [(True, False), (False, True)]


def test_list_delete_reindex(client):
    d = client.post("/api/rag/documents", files=[("files", ("a.txt", b"Canberra.", "text/plain"))]).json()[0]["doc_id"]
    assert [x["doc_id"] for x in client.get("/api/rag/documents").json()] == [d]
    assert client.delete(f"/api/rag/documents/{d}").status_code == 204
    assert client.delete("/api/rag/documents/nope").status_code == 404
    assert client.post("/api/rag/reindex").status_code == 202


def test_embedders_route(client):
    assert client.get("/api/rag/embedders").json()[0]["slug"] == "hf:fake"


def test_ws_sends_rag_config_and_status_on_connect(client):
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        kinds = []
        while "rag_status" not in kinds and len(kinds) < 10:
            kinds.append(ws.receive_json()["type"])
    assert kinds.index("catalog") < kinds.index("rag_config") < kinds.index("rag_status")


def test_routes_503_without_rag(client_without_rag):
    c = client_without_rag
    for r in (c.get("/api/rag/documents"), c.get("/api/rag/embedders"), c.post("/api/rag/reindex"),
              c.delete("/api/rag/documents/x"), c.post("/api/rag/documents", files=[("files", ("a.txt", b"x", "text/plain"))])):
        assert r.status_code == 503 and r.json() == {"detail": "RAG is not available"}
    with c.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "config" and ws.receive_json()["type"] == "catalog"
