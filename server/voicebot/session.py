"""VoiceSession: the listen/turn loop. See spec §4.7 and plan Task 9 / Task 7b.

Threads:
- handle() is called from the web server's event loop and never blocks on
  STT, the LLM or TTS.
- The listen loop runs on its own thread; typed turns run on a short-lived one.
- One turn at a time, guarded by _turn_lock.

Timing: a TurnTimer (Task 7b) collects stage marks. The recorder's callbacks
mark speech_end / endpoint / stt_start while listening; this module marks
stt_done, input, llm_first_token, first_text, llm_done, tts_first_audio,
playback_started and play_done. Each turn emits a metrics event, logs one
"turn timing" line and passes a record to metrics_sink (logs/turns.jsonl).
"""
from __future__ import annotations

import logging
import string
import threading
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable, Iterable, Iterator

from voicebot.conversation import Conversation
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.speech_filter import filter_stream
from voicebot.timing import TurnTimer
from voicebot.tts import drain_audio_queue, set_voice

log = logging.getLogger(__name__)

APOLOGY = "Sorry, something went wrong."
_STRIP_PUNCT = str.maketrans("", "", string.punctuation)
_LOG_SPANS = ("silence_wait_ms", "stt_ms", "llm_ttft_ms", "filter_ms", "tts_first_audio_ms",
              "playback_ms", "perceived_ms", "total_ms")


def is_noise(text: str) -> bool:
    """Whisper output that is not worth a turn: empty, fewer than 2 letters, or a lone "you"."""
    t = (text or "").strip()
    if sum(c.isalpha() for c in t) < 2:
        return True
    return t.lower().translate(_STRIP_PUNCT).strip() == "you"


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


class VoiceSession:
    def __init__(self, *, backend, audio, stream, conversation: Conversation, params: GenerationParams,
                 voice: str, voices: list[str], info: dict, timer: TurnTimer | None = None,
                 metrics_sink: Callable[[dict], None] | None = None, ack_timeout: float = 1.0) -> None:
        self.backend, self.audio, self.stream, self.conversation = backend, audio, stream, conversation
        self.params, self.voice, self.voices, self.info = params, voice, list(voices), dict(info)
        self.timer = timer if timer is not None else TurnTimer()
        self.metrics_sink = metrics_sink
        self.ack_timeout = ack_timeout
        self._emit_fn: Callable[[dict], None] | None = None
        self._emit_lock = threading.Lock()
        self._state_lock = threading.Lock()   # params, voice, system prompt, current turn
        self._turn_lock = threading.Lock()    # one turn at a time
        self._stop_event: threading.Event | None = None
        self._turn_id = 0
        self._current_turn: int | None = None
        self._ack = threading.Event()
        self._pending_voice: str | None = None
        self._status: str | None = None
        self._listening = False
        self._listen_gen = 0  # bumped per listen run; stale runs exit without touching state
        self._listen_thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self._stopper: threading.Thread | None = None  # in-flight stream.stop()

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
            "playback_started": self._playback_started,
        }.get(kind)  # type: ignore[arg-type]
        if handler is None:
            self._error("bad_command", f"unknown command type {kind!r}")
            return
        handler(command)

    def _start_listening(self, _cmd: dict) -> None:
        if self._listening:
            return
        self._listening = True
        self._listen_gen += 1
        previous = self._listen_thread
        self._listen_thread = threading.Thread(target=self._listen_loop, args=(self._listen_gen, previous),
                                               name="listen", daemon=True)
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
        self.audio.cut()
        # RealtimeTTS stream.stop() can block until synthesis winds down, and handle()
        # runs on the web event loop, so stop the stream on a helper thread.

        def stop_stream() -> None:
            # Silence the player first: stream.stop() sets RealtimeTTS's abort flags, after
            # which play() runs its own cleanup with a draining (non-immediate) player stop
            # that could otherwise take the player lock first and play out the buffer.
            player = getattr(self.stream, "player", None)
            if player is not None:
                try:
                    player.stop(immediate=True)
                except Exception:
                    log.exception("player.stop failed")
            try:
                self.stream.stop()
            except Exception:
                log.exception("stream.stop failed")
        self._stopper = threading.Thread(target=stop_stream, name="tts-stop", daemon=True)
        self._stopper.start()

    def _send_text(self, cmd: dict) -> None:
        text = cmd.get("text")
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            return
        if not self._turn_lock.acquire(blocking=False):
            self._error("busy", "A reply is already in progress. Press Stop first.")
            return
        self.timer.reset()
        self.timer.mark("input")

        def run() -> None:
            try:
                self._run_turn(text, source="text")
            finally:
                self._turn_lock.release()
        self._spawn(run, "typed-turn")

    def _playback_started(self, cmd: dict) -> None:
        with self._state_lock:
            current = self._current_turn
        if current is None or cmd.get("turn_id") != current:
            return  # late ack or an old turn: ignore silently
        self.timer.mark("playback_started")
        self._ack.set()

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

    def _listen_loop(self, gen: int, previous: threading.Thread | None) -> None:
        # A quick Mic off -> on can start this run while the previous one is still inside
        # recorder.text() (its abort runs on a helper thread) or a turn. RealtimeSTT is not
        # safe for two concurrent text() calls, so wait for it to finish first.
        if previous is not None and previous is not threading.current_thread():
            previous.join()
        recorder = self.audio.recorder

        def current() -> bool:
            return self._listening and gen == self._listen_gen

        while current():
            self._set_status("listening")
            try:
                text = recorder.text()
            except Exception as e:
                log.exception("STT failed")
                self._error("stt_error", str(e))
                break
            if not current():
                break
            if is_noise(text):
                self.timer.reset()
                continue
            with self._turn_lock:
                self.timer.mark("stt_done")
                self._run_turn(text.strip(), source="voice")
        if gen == self._listen_gen:  # an older run must not switch off a newer one
            if self._listening:
                self._listening = False
                if not self._turn_lock.locked():
                    self._set_status("idle")

    def _marked(self, chunks: Iterable[str]) -> Iterator[str]:
        """Mark llm_first_token on the first raw chunk and llm_done when the backend stream ends."""
        for chunk in chunks:
            self.timer.mark("llm_first_token")
            yield chunk
        self.timer.mark("llm_done")

    def _run_turn(self, text: str, *, source: str) -> None:
        stop_event = threading.Event()
        with self._state_lock:
            self._turn_id += 1
            turn_id = self._turn_id
            self._current_turn = turn_id
            self._stop_event = stop_event
            self._ack.clear()
            params = self.params
            voice, self._pending_voice = self._pending_voice, None
            messages = self.conversation.messages(text)
        parts: list[str] = []
        errors: list[BaseException] = []
        browser = self.audio.output_format is not None

        def tokens():
            # Exceptions are captured here, not raised: RealtimeTTS consumes this
            # iterator on its own thread, where a raise would be lost.
            try:
                for chunk in filter_stream(self._marked(self.backend.stream(messages, params, stop_event))):
                    if stop_event.is_set():
                        break
                    self.timer.mark("first_text")
                    parts.append(chunk)
                    self._emit({"type": "assistant_delta", "text": chunk})
                    yield chunk
            except Exception as e:
                errors.append(e)

        def on_first_audio() -> None:
            self.timer.mark("tts_first_audio")
            if not browser:
                self.timer.mark("playback_started")  # server mode: the PC speakers start now
            self._set_status("speaking")

        stopper = self._stopper
        if stopper is not None:
            stopper.join(timeout=5)  # let the previous turn's stop finish before playing again
        self._emit({"type": "user_transcript", "text": text, "turn_id": turn_id})
        self.audio.mute_input(True)
        try:
            dropped = drain_audio_queue(self.stream)
            if dropped:
                log.info("dropped %d stale TTS audio chunks before turn %d", dropped, turn_id)
            if voice:
                set_voice(self.stream, voice)
            self._set_status("thinking")
            self.stream.feed(tokens())
            self.audio.play(self.stream, on_first_audio)
            self.timer.mark("play_done")

            if errors:
                e = errors[0]
                if isinstance(e, LLMError):
                    log.error("LLM error during turn %d: %s", turn_id, e)
                else:
                    log.error("unexpected error during turn %d", turn_id, exc_info=e)
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
            if browser and self.timer.has("tts_first_audio") and not self.timer.has("playback_started"):
                self._ack.wait(self.ack_timeout)
            self._finish_metrics(turn_id, source, interrupted, params, len(reply))
        except Exception as e:
            log.exception("turn %d failed", turn_id)
            self._error("llm_error", str(e))
        finally:
            with self._state_lock:
                self._stop_event = None
                self._current_turn = None
            self.timer.reset()
            self.audio.mute_input(False)
            self._set_status("listening" if self._listening else "idle")

    def _finish_metrics(self, turn_id: int, source: str, interrupted: bool, params: GenerationParams,
                        reply_chars: int) -> None:
        report = self.timer.report()
        metrics = {"type": "metrics", "turn_id": turn_id, "source": source, "interrupted": interrupted, **report}
        self._emit(metrics)
        spans = report["spans"]
        log.info("turn timing #%d (%s%s): %s", turn_id, source, ", interrupted" if interrupted else "",
                 " ".join(f"{k}={'-' if spans[k] is None else spans[k]}" for k in _LOG_SPANS))
        if self.metrics_sink is None:
            return
        record = {k: v for k, v in metrics.items() if k != "type"}
        record.update(ts=datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **self.info,
                      voice=self.voice, temperature=params.temperature, top_p=params.top_p,
                      max_tokens=params.max_tokens, reply_chars=reply_chars)
        try:
            self.metrics_sink(record)
        except Exception:
            log.exception("metrics_sink failed")

    def close(self) -> None:
        self._listening = False
        self._listen_gen += 1
        # Abort only to wake a listen thread blocked in recorder.text(). RealtimeSTT's abort()
        # waits forever when no text() is running and the recorder was left 'transcribing'
        # (after any Mic on -> off), so never call it unguarded and never wait unbounded.
        listener = self._listen_thread
        if listener is not None and listener.is_alive():
            def abort() -> None:
                try:
                    self.audio.recorder.abort()
                except Exception:
                    log.exception("recorder.abort failed")
            aborter = threading.Thread(target=abort, name="stt-abort-close", daemon=True)
            aborter.start()
            aborter.join(timeout=2)
        self._stop_speaking({})
        for t in [self._listen_thread, self._stopper, *self._threads]:
            if t is not None and t is not threading.current_thread():
                t.join(timeout=5)
