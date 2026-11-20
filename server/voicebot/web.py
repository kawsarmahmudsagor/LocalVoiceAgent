"""FastAPI app: /api/config, /api/catalog, the /ws WebSocket and the built UI. See spec §4.8 (v2: §4.9).

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

from fastapi import FastAPI, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
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


CATALOG = REPO_ROOT / "server" / "voicebot" / "catalog.toml"


def default_catalog(settings: Settings):
    """Without catalog.toml: the v1 models only (faster-whisper from STT_*, Kokoro)."""
    from voicebot.catalog import Entry

    s = settings
    return [Entry(slug="whisper-small-en", kind="stt", role="final", engine="faster_whisper", model=s.stt_model,
                  device=s.stt_device, family="faster-whisper", options={"compute_type": s.stt_compute_type}),
            Entry(slug="kokoro", kind="tts", role=None, engine="KokoroEngine", model="kokoro", device="cuda",
                  family="kokoro", voice=s.tts_voice)]


def session_callbacks(holder: list) -> dict[str, Callable]:
    """Recorder callbacks that forward to the session once holder[0] is filled.

    They return None: a truthy return from some RealtimeSTT callbacks aborts transcription. They never
    raise: RealtimeSTT's recording worker re-raises a callback's exception, which stops STT until a restart.
    """
    def to_session(name: str) -> Callable[..., None]:
        def call(*args) -> None:
            if holder:
                try:
                    getattr(holder[0], name)(*args)
                except Exception:
                    log.exception("session callback %s failed", name)
            return None
        return call

    return {
        "on_realtime_transcription_update": to_session("on_partial"),
        "on_recording_start": to_session("on_recording_start"),
        "on_recording_stop": to_session("on_recording_stop"),
        "on_turn_detection_start": to_session("on_speech_pause"),
        "on_turn_detection_stop": to_session("on_speech_resume"),
    }


def build_session(settings: Settings):
    """The real wiring: LLM backend, the catalog's STT recorder and TTS stream, timing callbacks."""
    from dataclasses import replace

    from voicebot.audio_io import BrowserAudio, ServerAudio, play_kwargs
    from voicebot.bench.clips import clip_callbacks, merge_callbacks
    from voicebot.catalog import load_catalog
    from voicebot.conversation import Conversation
    from voicebot.llm.base import GenerationParams
    from voicebot.llm.factory import make_backend
    from voicebot.models import ModelError, ModelManager
    from voicebot.session import VoiceSession
    from voicebot.stt import make_recorder, timing_callbacks
    from voicebot.timing import TurnTimer, append_jsonl, append_turn_text
    from voicebot.tts import list_voices, make_tts, output_format, warm_up

    s = settings
    entries = load_catalog(CATALOG) if CATALOG.exists() else default_catalog(s)
    log.info("catalog: %s", ", ".join(e.slug for e in entries))
    log.info("starting LLM backend: %s", s.llm_backend)
    backend = make_backend(s)
    backend.start()
    holder: list = []  # the session, once built: partials arrive through the recorder callbacks
    try:
        timer = TurnTimer()
        parts = [timing_callbacks(timer), session_callbacks(holder)]
        if s.clip_record_dir is not None:
            log.info("recording benchmark clips to %s", s.clip_record_dir)
            parts.append(clip_callbacks(s.clip_record_dir))
        callbacks = merge_callbacks(*parts)

        def tts_factory(entry):
            log.info("loading TTS %s", entry.slug)
            stream = make_tts(s, entry=entry)
            warm_up(stream)
            return stream

        def recorder_factory(final, realtime, **kw):
            log.info("loading STT %s (realtime: %s, audio mode: %s)", final.slug,
                     realtime.slug if realtime else "off", s.audio_io)
            return make_recorder(s, final=final, realtime=realtime, **kw)

        browser = s.audio_io == "browser"
        models = ModelManager(entries, s, recorder_factory=recorder_factory, tts_factory=tts_factory)
        try:
            recorder, stream = models.start(not browser, callbacks)
        except ModelError as e:
            log.warning("%s; falling back to whisper-small-en / off / kokoro", e)
            models.settings = replace(s, stt_final="whisper-small-en", stt_realtime="off", tts_model="kokoro")
            recorder, stream = models.start(not browser, callbacks)
        pk = play_kwargs(s)
        audio = BrowserAudio(recorder, output_format(stream), play_kwargs=pk) if browser else ServerAudio(recorder, play_kwargs=pk)
    except BaseException:
        backend.close()
        raise
    info = {"llm_backend": s.llm_backend, "audio_io": s.audio_io, "model_label": backend.model_label,
            "stt_model": s.stt_model, "stt_device": s.stt_device, "stt_compute_type": s.stt_compute_type,
            "stt_silence_sec": s.stt_silence_sec,
            # TTS pacing: logged per turn so latency_report --by can compare settings.
            "kokoro_trim_silence": s.kokoro_trim_silence, "kokoro_extra_end_ms": s.kokoro_extra_end_ms,
            "tts_first_fragment_words": s.tts_first_fragment_words,
            "tts_fast_sentence_fragment": s.tts_fast_sentence_fragment,
            "tts_sentence_silence_ms": s.tts_sentence_silence_ms, "kokoro_speed": s.kokoro_speed}
    try:
        voices = list_voices(stream) or ["default"]
    except Exception:
        voices = ["default"]
    tts_entry = models.entry("tts")
    voice = tts_entry.voice if tts_entry is not None and tts_entry.voice in voices else voices[0]
    rag = _build_rag(s, lambda ev: holder[0]._emit(ev) if holder else None)
    session = VoiceSession(
        backend=backend, audio=audio, stream=stream,
        conversation=Conversation(s.system_prompt, s.history_turns),
        params=GenerationParams(s.llm_temperature, s.llm_top_p, s.llm_max_tokens),
        voice=voice, voices=voices, info=info, timer=timer,
        metrics_sink=lambda record: (append_jsonl(TURNS_LOG, record), append_turn_text(TURNS_TEXT_LOG, record)),
        models=models, callbacks=callbacks, rag=rag,
        word_aligner=_build_word_aligner(s), word_align=s.word_align,
        diarizer_factory=_build_diarizer_factory(s))
    holder.append(session)
    _close_session = session.close
    # audio.recorder, not recorder: a model switch replaces it.
    session.close = lambda: shutdown_resources(_close_session, audio.recorder, backend, rag=rag)  # type: ignore[method-assign]
    return session


def _build_word_aligner(s: Settings):
    """The forced word aligner (its model loads when first switched on); None if it cannot be built."""
    try:
        from voicebot.aligner import WordAligner
        return WordAligner(device=s.word_align_device)
    except Exception:
        log.exception("word alignment is unavailable")
        return None


def _build_diarizer_factory(s: Settings) -> Callable | None:
    """The v4 diarizer factory (browser audio only, else None). It imports voicebot.diarizer when first
    called, so a server that never switches diarization on never imports it or transformers."""
    if s.audio_io != "browser":
        return None

    def factory(on_error: Callable[[str, str], None]):
        from voicebot.diarizer import Diarizer, NemotronModel
        return Diarizer(NemotronModel(device="cuda"), on_error=on_error)
    return factory


def _build_rag(s: Settings, on_event: Callable[[dict], None]):
    """The v3 RagService, started; None if it cannot be built, so the voice bot still runs."""
    try:
        from functools import partial

        from voicebot.rag.embedders import list_embedders, make_embedder
        from voicebot.rag.library import RagSettings
        from voicebot.rag.service import RagService

        rag = RagService(s.rag_dir, RagSettings(embedder=s.rag_embedder),
                         make_embedder=partial(make_embedder, device=s.rag_embed_device, ollama_url=s.ollama_url),
                         list_embedders=partial(list_embedders, s.ollama_url),
                         on_event=on_event, max_upload_mb=s.rag_max_upload_mb)
    except Exception:
        log.exception("RAG could not be built; running without it")
        return None
    try:
        rag.start()
    except Exception:
        log.exception("RAG could not start; running without it")
        try:
            rag.close()
        except Exception:
            log.exception("rag.close failed")
        return None
    return rag


def shutdown_resources(close_session: Callable[[], None], recorder, backend, *,
                       recorder_timeout: float = 10.0, rag=None) -> None:
    """Stop everything; the LLM backend (llama-server) is closed whatever else fails or hangs."""
    try:
        close_session()
    except Exception:
        log.exception("session.close failed")
    if rag is not None:
        _safe_call(rag.close, "rag.close")
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

    @app.get("/api/catalog")
    def get_catalog() -> list:
        return state["session"].catalog()

    def _rag():
        rag = getattr(state["session"], "rag", None)
        if rag is None:
            raise HTTPException(status_code=503, detail="RAG is not available")
        return rag

    @app.post("/api/rag/documents", status_code=202)
    async def rag_upload(files: list[UploadFile]) -> list:
        rag = _rag()
        results = []
        for f in files:
            data = await f.read()
            results.append(await asyncio.to_thread(rag.upload, f.filename or "file", data))
        return results

    @app.get("/api/rag/documents")
    def rag_documents() -> list:
        return _rag().status_event()["documents"]

    @app.delete("/api/rag/documents/{doc_id}", status_code=204)
    async def rag_delete(doc_id: str) -> Response:
        if not await asyncio.to_thread(_rag().delete, doc_id):
            raise HTTPException(status_code=404, detail="unknown document")
        return Response(status_code=204)

    @app.post("/api/rag/reindex", status_code=202)
    async def rag_reindex() -> dict:
        from voicebot.rag.service import RagError
        try:
            await asyncio.to_thread(_rag().reindex_all)
        except RagError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"state": "queued"}

    @app.get("/api/rag/embedders")
    async def rag_embedders() -> list:
        return (await asyncio.to_thread(_rag().config_event))["embedders"]

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
            await ws.send_json({"type": "catalog", "entries": session.catalog()})
            for event in await asyncio.to_thread(session.rag_events):
                await ws.send_json(event)
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
