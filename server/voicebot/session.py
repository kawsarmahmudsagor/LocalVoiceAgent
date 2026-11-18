"""VoiceSession: the listen/turn loop. See spec §4.7.

Threads:
- handle() is called from the web server's event loop and never blocks on
  STT, the LLM or TTS.
- The listen loop runs on its own thread; typed turns run on a short-lived one.
- One turn at a time, guarded by _turn_lock.

Per-turn timing marks (logged and sent in the metrics event):
  silence_end  VAD ended the utterance (RealtimeSTT on_recording_stop)
  stt_done     recorder.text() returned the transcript
  first_token  first filtered LLM text
  first_audio  first TTS audio chunk
"""
from __future__ import annotations

import logging
import string
import threading
import time
from dataclasses import replace
from typing import Callable

from voicebot.conversation import Conversation
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.speech_filter import filter_stream
from voicebot.tts import set_voice

log = logging.getLogger(__name__)

APOLOGY = "Sorry, something went wrong."
_STRIP_PUNCT = str.maketrans("", "", string.punctuation)


def is_noise(text: str) -> bool:
    """Whisper output that is not worth a turn: empty, fewer than 2 letters, or a lone "you"."""
    t = (text or "").strip()
    if sum(c.isalpha() for c in t) < 2:
        return True
    return t.lower().translate(_STRIP_PUNCT).strip() == "you"


def _ms(a: float | None, b: float | None) -> int | None:
    return None if a is None or b is None else round((b - a) * 1000)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class VoiceSession:
    def __init__(self, *, backend, audio, stream, conversation: Conversation, params: GenerationParams,
                 voice: str, voices: list[str], info: dict, clock: Callable[[], float] = time.perf_counter) -> None:
        self.backend, self.audio, self.stream, self.conversation = backend, audio, stream, conversation
        self.params, self.voice, self.voices, self.info = params, voice, list(voices), dict(info)
        self.clock = clock
        self._emit_fn: Callable[[dict], None] | None = None
        self._emit_lock = threading.Lock()
        self._state_lock = threading.Lock()   # params, voice, system prompt, current stop_event
        self._turn_lock = threading.Lock()    # one turn at a time
        self._stop_event: threading.Event | None = None
        self._pending_voice: str | None = None
        self._status: str | None = None
        self._listening = False
        self._listen_thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self._silence_end: float | None = None

    # ---------- wiring ----------

    def attach(self, emit: Callable[[dict], None], send_binary: Callable[[bytes], None]) -> None:
        with self._emit_lock:
            self._emit_fn = emit
            self._status = None
        self.audio.set_sink(send_binary)

    def detach(self) -> None:
        with self._emit_lock:
            self._emit_fn = None
        self.audio.set_sink(None)

    def _emit(self, event: dict) -> None:
        with self._emit_lock:
            fn = self._emit_fn
        if fn is None:
            return
        try:
            fn(event)
        except Exception:
            log.exception("emit failed")

    def _set_status(self, state: str) -> None:
        with self._emit_lock:
            if self._status == state:
                return
            self._status = state
        self._emit({"type": "status", "state": state})

    def _error(self, code: str, message: str) -> None:
        self._emit({"type": "error", "code": code, "message": message})

    def mark_silence_end(self) -> None:
        """RealtimeSTT on_recording_stop callback: VAD has ended the utterance."""
        self._silence_end = self.clock()

    def on_client_audio(self, data: bytes) -> None:
        self.audio.on_client_audio(data)

    def config(self) -> dict:
        with self._state_lock:
            p, prompt, voice = self.params, self.conversation.system_prompt, self.voice
        return {**self.info, "voices": list(self.voices), "output_format": self.audio.output_format,
                "defaults": {"system_prompt": prompt, "temperature": p.temperature, "top_p": p.top_p,
                             "max_tokens": p.max_tokens, "voice": voice}}

    # ---------- commands ----------

    def handle(self, command: dict) -> None:
        kind = command.get("type") if isinstance(command, dict) else None
        handler = {
            "start_listening": self._start_listening,
            "stop_listening": self._stop_listening,
            "stop_speaking": self._stop_speaking,
            "send_text": self._send_text,
            "update_settings": self._update_settings,
            "clear": self._clear,
        }.get(kind)  # type: ignore[arg-type]
        if handler is None:
            self._error("bad_command", f"unknown command type {kind!r}")
            return
        handler(command)

    def _start_listening(self, _cmd: dict) -> None:
        if self._listening:
            return
        self._listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, name="listen", daemon=True)
        self._listen_thread.start()

    def _stop_listening(self, _cmd: dict) -> None:
        if not self._listening:
            return
        self._listening = False
        # abort() can block until the recorder acknowledges, so keep it off the caller's thread.
        self._spawn(self.audio.recorder.abort, "stt-abort")
        if not self._turn_lock.locked():
            self._set_status("idle")

    def _stop_speaking(self, _cmd: dict) -> None:
        with self._state_lock:
            ev = self._stop_event
        if ev is None:
            return
        ev.set()
        try:
            self.stream.stop()
        except Exception:
            log.exception("stream.stop failed")

    def _send_text(self, cmd: dict) -> None:
        text = cmd.get("text")
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            return
        if not self._turn_lock.acquire(blocking=False):
            self._error("busy", "A reply is already in progress. Press Stop first.")
            return
        started = self.clock()

        def run() -> None:
            try:
                self._run_turn(text, start=started)
            finally:
                self._turn_lock.release()
        self._spawn(run, "typed-turn")

    def _update_settings(self, cmd: dict) -> None:
        bad = []
        if "temperature" in cmd and not (_is_number(cmd["temperature"]) and 0 <= cmd["temperature"] <= 2):
            bad.append("temperature must be a number from 0 to 2")
        if "top_p" in cmd and not (_is_number(cmd["top_p"]) and 0 <= cmd["top_p"] <= 1):
            bad.append("top_p must be a number from 0 to 1")
        if "max_tokens" in cmd and not (isinstance(cmd["max_tokens"], int) and not isinstance(cmd["max_tokens"], bool)
                                        and 32 <= cmd["max_tokens"] <= 1024):
            bad.append("max_tokens must be an integer from 32 to 1024")
        if "voice" in cmd and cmd["voice"] not in self.voices:
            bad.append(f"unknown voice {cmd['voice']!r}")
        if "system_prompt" in cmd and not isinstance(cmd["system_prompt"], str):
            bad.append("system_prompt must be a string")
        if bad:
            self._error("bad_settings", "; ".join(bad))
            return
        with self._state_lock:
            changes = {k: cmd[k] for k in ("temperature", "top_p", "max_tokens") if k in cmd}
            if changes:
                self.params = replace(self.params, **changes)
            if "system_prompt" in cmd:
                self.conversation.system_prompt = cmd["system_prompt"]
            if "voice" in cmd and cmd["voice"] != self.voice:
                self.voice = self._pending_voice = cmd["voice"]

    def _clear(self, _cmd: dict) -> None:
        self.conversation.clear()

    # ---------- loops ----------

    def _spawn(self, target: Callable[[], None], name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        self._threads = [x for x in self._threads if x.is_alive()] + [t]
        t.start()

    def _listen_loop(self) -> None:
        recorder = self.audio.recorder
        while self._listening:
            self._set_status("listening")
            self._silence_end = None
            try:
                text = recorder.text()
            except Exception as e:
                log.exception("STT failed")
                self._error("stt_error", str(e))
                break
            stt_done = self.clock()
            if not self._listening:
                break
            if is_noise(text):
                continue
            with self._turn_lock:
                self._run_turn(text.strip(), start=stt_done, silence_end=self._silence_end, stt_done=stt_done)
        self._listening = False

    def _run_turn(self, text: str, *, start: float, silence_end: float | None = None,
                  stt_done: float | None = None) -> None:
        stop_event = threading.Event()
        with self._state_lock:
            self._stop_event = stop_event
            params = self.params
            voice, self._pending_voice = self._pending_voice, None
            messages = self.conversation.messages(text)
        marks: dict[str, float | None] = {"first_token": None, "first_audio": None}
        parts: list[str] = []
        errors: list[BaseException] = []

        def tokens():
            # Exceptions are captured here, not raised: RealtimeTTS consumes this
            # iterator on its own thread, where a raise would be lost.
            try:
                for chunk in filter_stream(self.backend.stream(messages, params, stop_event)):
                    if stop_event.is_set():
                        break
                    if marks["first_token"] is None:
                        marks["first_token"] = self.clock()
                    parts.append(chunk)
                    self._emit({"type": "assistant_delta", "text": chunk})
                    yield chunk
            except Exception as e:
                errors.append(e)

        def on_first_audio() -> None:
            if marks["first_audio"] is None:
                marks["first_audio"] = self.clock()
            self._set_status("speaking")

        self._emit({"type": "user_transcript", "text": text})
        self.audio.mute_input(True)
        try:
            if voice:
                set_voice(self.stream, voice)
            self._set_status("thinking")
            self.stream.feed(tokens())
            self.audio.play(self.stream, on_first_audio)
            end = self.clock()

            if errors:
                e = errors[0]
                if not isinstance(e, LLMError):
                    log.error("unexpected error during turn", exc_info=e)
                else:
                    log.error("LLM error during turn: %s", e)
                self._error("llm_error", str(e))
                self.stream.feed(APOLOGY)
                self.audio.play(self.stream, lambda: self._set_status("speaking"))
                self._emit({"type": "assistant_done", "text": APOLOGY, "interrupted": False})
                return

            reply = "".join(parts)
            interrupted = stop_event.is_set()
            if reply:
                self.conversation.add_exchange(text, reply)
            self._emit({"type": "assistant_done", "text": reply, "interrupted": interrupted})
            metrics = {
                "type": "metrics",
                "stt_ms": _ms(silence_end, stt_done),
                "ttft_ms": _ms(start, marks["first_token"]),
                "first_audio_ms": _ms(start, marks["first_audio"]),
                "total_ms": _ms(start, end),
                "speech_to_audio_ms": _ms(silence_end, marks["first_audio"]),
            }
            self._emit(metrics)
            self._log_timing(start, silence_end, stt_done, marks, end, interrupted)
        except Exception as e:
            log.exception("turn failed")
            self._error("llm_error", str(e))
        finally:
            with self._state_lock:
                self._stop_event = None
            self.audio.mute_input(False)
            self._set_status("listening" if self._listening else "idle")

    def _log_timing(self, start, silence_end, stt_done, marks, end, interrupted) -> None:
        origin = silence_end if silence_end is not None else start
        since = "silence end" if silence_end is not None else "typed send"

        def rel(t):
            v = _ms(origin, t)
            return "-" if v is None else f"+{v}"
        log.info("turn timing (ms since %s): silence_end=%s stt_done=%s first_token=%s first_audio=%s done=%s%s",
                 since, rel(silence_end), rel(stt_done), rel(marks["first_token"]), rel(marks["first_audio"]),
                 rel(end), " (interrupted)" if interrupted else "")

    def close(self) -> None:
        self._listening = False
        try:
            self.audio.recorder.abort()
        except Exception:
            log.exception("recorder.abort failed")
        self._stop_speaking({})
        threads = [self._listen_thread, *self._threads]
        for t in threads:
            if t is not None and t is not threading.current_thread():
                t.join(timeout=5)
