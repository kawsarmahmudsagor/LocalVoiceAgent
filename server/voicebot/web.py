"""FastAPI app: /api/config, the /ws WebSocket and the built UI. See spec §4.8.

    python -m voicebot.web
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from voicebot.config import REPO_ROOT, Settings

log = logging.getLogger(__name__)

WEB_DIST = REPO_ROOT / "web" / "dist"
TURNS_LOG = REPO_ROOT / "logs" / "turns.jsonl"
TURNS_TEXT_LOG = REPO_ROOT / "logs" / "turns.log"  # human-readable copy of TURNS_LOG
CLOSE_TRY_AGAIN_LATER = 1013
CLOSE_POLICY_VIOLATION = 1008


def _same_origin(origin: str | None, host: str | None) -> bool:
    """Allow non-browser clients (no Origin header) and pages served from this host.

    The Vite dev proxy forwards the browser's Host (localhost:5173), so the dev UI passes too.
    """
    if not origin:
        return True
    return bool(host) and urlsplit(origin).netloc.lower() == host.lower()


def build_session(settings: Settings):
    """The real wiring: LLM backend, Kokoro TTS, STT recorder with timing callbacks."""
    from voicebot.audio_io import make_audio_io
    from voicebot.conversation import Conversation
    from voicebot.llm.base import GenerationParams
    from voicebot.llm.factory import make_backend
    from voicebot.session import VoiceSession
    from voicebot.stt import timing_callbacks
    from voicebot.timing import TurnTimer, append_jsonl, append_turn_text
    from voicebot.tts import list_voices, make_tts, warm_up

    s = settings
    log.info("starting LLM backend: %s", s.llm_backend)
    backend = make_backend(s)
    backend.start()
    try:
        log.info("loading Kokoro TTS")
        stream = make_tts(s)
        warm_up(stream)
        timer = TurnTimer()
        log.info("loading STT model %s on %s (audio mode: %s)", s.stt_model, s.stt_device, s.audio_io)
        audio = make_audio_io(s, stream, callbacks=timing_callbacks(timer))
    except BaseException:
        backend.close()
        raise
    info = {"llm_backend": s.llm_backend, "audio_io": s.audio_io, "model_label": backend.model_label,
            "stt_model": s.stt_model, "stt_device": s.stt_device, "stt_compute_type": s.stt_compute_type,
            "stt_silence_sec": s.stt_silence_sec}
    session = VoiceSession(
        backend=backend, audio=audio, stream=stream,
        conversation=Conversation(s.system_prompt, s.history_turns),
        params=GenerationParams(s.llm_temperature, s.llm_top_p, s.llm_max_tokens),
        voice=s.tts_voice, voices=list_voices(stream), info=info, timer=timer,
        metrics_sink=lambda record: (append_jsonl(TURNS_LOG, record), append_turn_text(TURNS_TEXT_LOG, record)))
    _close_session = session.close
    session.close = lambda: shutdown_resources(_close_session, audio.recorder, backend)  # type: ignore[method-assign]
    return session


def shutdown_resources(close_session: Callable[[], None], recorder, backend, *,
                       recorder_timeout: float = 10.0) -> None:
    """Stop everything; the LLM backend (llama-server) is closed whatever else fails or hangs."""
    try:
        close_session()
    except Exception:
        log.exception("session.close failed")
    try:
        t = threading.Thread(target=_safe_call, args=(recorder.shutdown, "recorder.shutdown"),
                             name="stt-shutdown", daemon=True)
        t.start()
        t.join(timeout=recorder_timeout)
        if t.is_alive():
            log.warning("recorder.shutdown did not finish within %.0fs; continuing", recorder_timeout)
    finally:
        backend.close()


def _safe_call(fn: Callable[[], None], what: str) -> None:
    try:
        fn()
    except Exception:
        log.exception("%s failed", what)


def create_app(settings: Settings | None = None, *,
               session_factory: Callable[[Settings], Any] | None = None) -> FastAPI:
    settings = settings or Settings.load()
    factory = session_factory or build_session
    state: dict[str, Any] = {"session": None, "connected": False}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Loading models blocks for a while; keep it off the event loop.
        state["session"] = await asyncio.to_thread(factory, settings)
        log.info("ready")
        try:
            yield
        finally:
            await asyncio.to_thread(state["session"].close)

    app = FastAPI(title="Local voice bot", lifespan=lifespan)

    @app.get("/api/config")
    def get_config() -> dict:
        return state["session"].config()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        # Browsers do not apply CORS to WebSockets: without this check any page the user
        # visits could connect, change settings and, in server audio mode, switch on the
        # PC microphone and read the transcripts.
        if not _same_origin(ws.headers.get("origin"), ws.headers.get("host")):
            log.warning("rejected WebSocket from origin %r", ws.headers.get("origin"))
            await ws.close(code=CLOSE_POLICY_VIOLATION)
            return
        await ws.accept()
        session = state["session"]
        if state["connected"]:
            await ws.send_json({"type": "error", "code": "session_busy",
                                "message": "Another browser tab is already connected."})
            await ws.close(code=CLOSE_TRY_AGAIN_LATER)
            return
        state["connected"] = True
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def enqueue(item) -> None:  # called from worker threads
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                pass  # loop closed: the client is gone

        async def sender() -> None:
            while True:
                item = await queue.get()
                if isinstance(item, (bytes, bytearray)):
                    await ws.send_bytes(bytes(item))
                else:
                    await ws.send_json(item)

        sender_task = asyncio.create_task(sender())
        session.attach(enqueue, enqueue)
        try:
            await ws.send_json({"type": "config", **session.config()})
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    session.on_client_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    try:
                        command = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        enqueue({"type": "error", "code": "bad_command", "message": "invalid JSON"})
                        continue
                    if not isinstance(command, dict):
                        enqueue({"type": "error", "code": "bad_command", "message": "command must be an object"})
                        continue
                    session.handle(command)
        except WebSocketDisconnect:
            pass
        finally:
            try:
                session.handle({"type": "stop_listening"})
                session.handle({"type": "stop_speaking"})
            finally:
                session.detach()
                state["connected"] = False
                sender_task.cancel()

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.load()
    ssl = {}
    if settings.web_ssl_certfile and settings.web_ssl_keyfile:
        ssl = {"ssl_certfile": str(settings.web_ssl_certfile), "ssl_keyfile": str(settings.web_ssl_keyfile)}
    uvicorn.run(create_app(settings), host=settings.web_host, port=settings.web_port, log_level="info", **ssl)


if __name__ == "__main__":
    main()
