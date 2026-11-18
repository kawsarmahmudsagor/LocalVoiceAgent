import threading, pytest
from fastapi.testclient import TestClient
from voicebot.config import Settings
from voicebot.web import create_app

class FakeSession:
    def __init__(self): self.handled, self.audio, self.emit, self.send_binary, self.detached, self.closed = [], [], None, None, False, False
    def attach(self, emit, send_binary): self.emit, self.send_binary, self.detached = emit, send_binary, False
    def detach(self): self.detached = True
    def handle(self, cmd): self.handled.append(cmd)
    def on_client_audio(self, data): self.audio.append(data)
    def config(self): return {"llm_backend": "llamacpp", "audio_io": "browser", "model_label": "m", "voices": ["af_heart"],
                              "output_format": None, "defaults": {}}
    def close(self): self.closed = True

@pytest.fixture
def ctx():
    fake = FakeSession()
    with TestClient(create_app(Settings.load(env={}), session_factory=lambda s: fake)) as client:
        yield client, fake
    assert fake.closed

def test_config_endpoint(ctx):
    client, _ = ctx; assert client.get("/api/config").json()["model_label"] == "m"

def test_ws_config_commands_binary(ctx):
    client, fake = ctx
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "config"
        ws.send_json({"type": "send_text", "text": "hi"}); ws.send_bytes(b"\x00\x01"); ws.send_text("{bad")
        assert ws.receive_json()["code"] == "bad_command"
        assert {"type": "send_text", "text": "hi"} in fake.handled and fake.audio == [b"\x00\x01"]

def test_events_from_worker_thread(ctx):
    client, fake = ctx
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        t = threading.Thread(target=lambda: (fake.emit({"type": "status", "state": "idle"}), fake.send_binary(b"pcm"))); t.start(); t.join()
        assert ws.receive_json() == {"type": "status", "state": "idle"} and ws.receive_bytes() == b"pcm"

def test_second_client_busy(ctx):
    client, _ = ctx
    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json()
        with client.websocket_connect("/ws") as ws2:
            assert ws2.receive_json()["code"] == "session_busy"

def test_disconnect_detaches_and_stops(ctx):                 # Review Focus 3
    client, fake = ctx
    with client.websocket_connect("/ws") as ws: ws.receive_json()
    assert fake.detached and {"type": "stop_listening"} in fake.handled and {"type": "stop_speaking"} in fake.handled

def test_shutdown_always_closes_backend():                    # review #3: never orphan llama-server
    import time
    from voicebot.web import shutdown_resources
    calls = []
    def bad_close(): calls.append("session"); raise RuntimeError("boom")
    class HangingRecorder:
        def shutdown(self): calls.append("recorder"); threading.Event().wait()
    class Backend:
        def close(self): calls.append("backend")
    t0 = time.monotonic()
    shutdown_resources(bad_close, HangingRecorder(), Backend(), recorder_timeout=0.2)
    assert calls == ["session", "recorder", "backend"] and time.monotonic() - t0 < 2

def test_ws_rejects_cross_origin_pages(ctx):                  # review #5: other sites must not drive the bot
    from starlette.websockets import WebSocketDisconnect
    client, fake = ctx
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_json()
    assert fake.emit is None                                   # never attached
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "config"           # same origin is fine

def test_main_passes_ssl(monkeypatch):
    from voicebot import web
    seen, real_load = {}, Settings.load
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr("voicebot.web.Settings.load", classmethod(lambda cls, **k: real_load(
        env={"WEB_SSL_CERTFILE": "c.pem", "WEB_SSL_KEYFILE": "k.pem"}, platform="linux")))
    web.main(); assert seen["ssl_certfile"].endswith("c.pem") and seen["ssl_keyfile"].endswith("k.pem")
