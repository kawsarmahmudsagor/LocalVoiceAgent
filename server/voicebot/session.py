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

RAG (v3, spec §4.3): with an optional RagService that is switched on, each turn
marks retrieval_start / retrieval_done around rag.retrieve(text) before the LLM
call. The context goes into that request's system message only, never into
history. A retrieval failure is reported as rag_error and the turn answers
without context. Without RAG, turns run exactly as before.
"""
from __future__ import annotations

import logging
import string
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

from voicebot.conversation import Conversation
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.speech_filter import filter_stream
from voicebot.timing import TurnTimer
from voicebot.tts import drain_audio_queue, list_voices, output_format as tts_output_format, set_voice

if TYPE_CHECKING:  # the RAG package imports chromadb; keep it out of the session's import path
    from voicebot.diarizer import Diarizer
    from voicebot.rag.service import RagService

log = logging.getLogger(__name__)

APOLOGY = "Sorry, something went wrong."
PLAYBACK_MARGIN_S = 0.25  # browser playback slack after the estimated end of the audio
SOURCE_PREVIEW_CHARS = 160  # spec §3.2: the Sources line shows the first 160 characters of each chunk
_STRIP_PUNCT = str.maketrans("", "", string.punctuation)
# Copied from voicebot.diarizer.MODES: importing that module here would pull numpy and the model code into
# a server that runs with diarization off.
DIAR_MODES = ("ultra_low_latency", "very_low_latency", "low_latency")
DIAR_UNAVAILABLE = "Speaker diarization is not available on this server (browser audio only)."
BARGE_SHARE = 0.5      # a barge-in fires when Speaker 0 is active in at least this share of the onset's voiced frames
BARGE_POLL_S = 0.04    # the barge-in check polls a lagging diarizer this often
_LOG_SPANS = ("silence_wait_ms", "stt_ms", "diar_wait_ms", "retrieval_ms", "llm_ttft_ms", "filter_ms", "tts_first_audio_ms",
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
                 metrics_sink: Callable[[dict], None] | None = None, ack_timeout: float = 1.0,
                 models=None, callbacks: dict | None = None, rag: RagService | None = None,
                 word_aligner=None, word_align: bool = False, barge_in_mode: str = "off",
                 barge_in_min_ms: int = 200,
                 diarizer_factory: Callable[[Callable[[str, str], None]], Diarizer] | None = None) -> None:
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
        # playback_cut ack for the current turn (spec §3 "History cut"), cleared at turn start. The browser
        # can send it before the turn reaches its wait, so it is stored, not just signalled.
        self._cut = threading.Event()
        self._cut_heard: str | None = None   # guarded by _state_lock
        self._pending_voice: str | None = None
        self._status: str | None = None
        self._listening = False
        self._listen_gen = 0  # bumped per listen run; stale runs exit without touching state
        self._listen_thread: threading.Thread | None = None
        self._threads: list[threading.Thread] = []
        self._stopper: threading.Thread | None = None  # in-flight stream.stop()
        self.models = models                  # v2 ModelManager, or None (v1: fixed models)
        self.callbacks = dict(callbacks or {})
        self._switching = threading.Lock()    # one select_model at a time
        self.rag = rag                        # v3 RagService, or None (RAG not configured)
        self.barge_in_mode = barge_in_mode    # off | onset | words (guarded by _state_lock)
        self.barge_in_min_ms = barge_in_min_ms
        # Barge-in arming state for the running turn (spec §2), all guarded by _state_lock. The recorder
        # callbacks below run on RealtimeSTT's threads and check it; a turn that is not armed has
        # _barge_turn None.
        self._recording = False
        self._barge_turn: int | None = None
        self._barge_mode = "off"
        self._barge_min_ms = barge_in_min_ms
        self._barge_fired = False
        self._barge_false = 0
        self._barge_timer: threading.Timer | None = None
        # A barge-in that fired, kept until the listen loop's next transcript settles it (spec §2 step 3):
        # a noise transcript makes it a false fire. Guarded by _state_lock.
        self._barge_pending: dict | None = None
        # Forced word alignment for TTS engines without word timings: a voicebot.aligner.WordAligner (or None
        # when unavailable), switched on and off from the UI. Its model loads on first use.
        self._word_aligner = word_aligner
        self.word_align = False
        self._align_wanted = False
        if word_align:
            self._set_word_align(True)
        # v4 speaker diarization. diarizer_factory(on_error) builds the Diarizer on first switch-on (None:
        # unavailable). _diar_lock guards diarization, _diar_wanted, _diar_loading, _diarizer, diar_mode and
        # audio.diarizer. It is only held for quick flag flips, set_mode/start and emits, never across a model
        # load, stop() or a model switch, so _on_diar_error (on the audio or diarizer thread) can always take it.
        self.diarizer_factory = diarizer_factory
        self._diar_lock = threading.Lock()
        self._diarizer = None
        self._diar_wanted = False    # the latest on/off request; a load that finishes checks it
        self._diar_loading = False   # a switch-on worker is running: another "on" does not start a second one
        self._diar_stopper: threading.Thread | None = None   # the stop() _on_diar_error spawned
        self.diarization = False     # on and attached to the audio
        self.diar_mode = "ultra_low_latency"
        self.diar_wait_ms = 600      # guarded by _state_lock
        # The current utterance on the diarizer's timeline (sample indices at 16 kHz; None: unknown), set by
        # the recorder callbacks. Guarded by _state_lock.
        self._utt_start: int | None = None
        self._utt_end: int | None = None
        # The last completed utterance (start, end, epoch), snapshotted at on_recording_stop so a following
        # on_recording_start cannot clobber it before the decision. Every diarizer reset bumps _diar_epoch and
        # clears both windows: indices on the old clock are meaningless on the new one. Guarded by _state_lock.
        # Lock order: _invalidate_window takes _state_lock inside _diar_lock; never take _diar_lock under _state_lock.
        self._last_window: tuple[int | None, int | None, int] | None = None
        self._diar_epoch = 0
        self._diar_rejects = 0       # barge-ins rejected as another speaker this turn (Task 8); _state_lock
        # The barge-in speaker check (Task 8), all guarded by _state_lock. _barge_onset_sample is where the
        # marked barge_onset starts on the diarizer's timeline (None: unknown, the check fails open);
        # _barge_onset_gen counts onset marks, so a check can tell its onset was replaced by a newer one.
        # _barge_checking: a check thread runs; _barge_recheck: the turn a confirm asked for meanwhile.
        # _barge_words_rejected: words mode rejected this recording, so its later partials do not trigger.
        self._barge_onset_sample: int | None = None
        self._barge_onset_gen = 0
        self._barge_checking = False
        self._barge_recheck: int | None = None
        self._barge_words_rejected = False

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
            template = self.conversation.rag_template
            barge_mode, barge_min = self.barge_in_mode, self.barge_in_min_ms
            diar_wait = self.diar_wait_ms
        with self._diar_lock:
            diar_on, diar_mode = self.diarization, self.diar_mode
        cfg = {**self.info, "voices": list(self.voices), "output_format": self.audio.output_format,
               "defaults": {"system_prompt": prompt, "temperature": p.temperature, "top_p": p.top_p,
                            "max_tokens": p.max_tokens, "voice": voice, "rag_template": template,
                            "word_align": self.word_align, "barge_in_mode": barge_mode,
                            "barge_in_min_ms": barge_min, "diarization": diar_on, "diar_mode": diar_mode,
                            "diar_wait_ms": diar_wait},
               "barge_words_available": self.barge_words_available(),
               "diarization_available": self.diarization_available()}
        if self.models is not None:
            m, st = self.models.current, self.models.settings
            cfg.update(stt_final=m.get("stt_final"), stt_realtime=m.get("stt_realtime", "off"), tts=m.get("tts"),
                       gpu_total_mb=st.gpu_total_mb, llm_vram_mb=st.llm_vram_mb)
        return cfg

    def barge_words_available(self) -> bool:
        return self.models is not None and self.models.current.get("stt_realtime", "off") != "off"

    def diarization_available(self) -> bool:
        return self.diarizer_factory is not None and self.audio.output_format is not None

    def catalog(self) -> list[dict]:
        if self.models is None:
            return []
        from dataclasses import asdict
        return [asdict(e) for e in self.models.entries.values()]

    def rag_events(self) -> list[dict]:
        """The RAG state a newly connected client needs: [] without RAG, else rag_config and rag_status."""
        if self.rag is None:
            return []
        return [self.rag.config_event(), self.rag.status_event()]

    # ---------- commands ----------

    def handle(self, command: dict) -> None:
        kind = command.get("type") if isinstance(command, dict) else None
        handler = {
            "start_listening": self._start_listening,
            "stop_listening": self._stop_listening,
            "stop_speaking": self._stop_command,
            "send_text": self._send_text,
            "update_settings": self._update_settings,
            "clear": self._clear,
            "playback_started": self._playback_started,
            "playback_cut": self._playback_cut,
            "select_model": self._select_model,
            "update_rag": self._update_rag,
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
        self._disarm()
        with self._state_lock:
            self._recording = False   # abort() may not fire on_recording_stop
            self._barge_pending = None
        # abort() can block until the recorder acknowledges, so keep it off the caller's thread.
        self._spawn(self.audio.recorder.abort, "stt-abort")
        if not self._turn_lock.locked():
            self._set_status("idle")

    def _stop_command(self, cmd: dict) -> None:
        """The user's Stop. It disarms the running turn's barge-in first, so a later onset cannot fire on a
        reply Stop already cut; a pending onset is dropped with its mark (a Stop cut has no onset). The stop
        event is set under the lock too, so a turn that arms after this sees it and stays disarmed."""
        with self._state_lock:
            ev = self._stop_event
            if ev is not None:
                ev.set()
                if self._barge_turn is not None and not self._barge_fired:
                    self._barge_fired = True
                    if self._barge_timer is not None:
                        self._barge_timer.cancel()
                        self._barge_timer = None
                    self.timer.unmark("barge_onset")
        self._stop_speaking(cmd)

    def _stop_speaking(self, _cmd: dict, *, turn_id: int | None = None) -> None:
        """Stop the running reply. With turn_id, only if that turn is still the one running."""
        with self._state_lock:
            ev = self._stop_event
            if turn_id is not None and self._current_turn != turn_id:
                return
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

    def _playback_cut(self, cmd: dict) -> None:
        """The browser stopped this turn's audio; heard is the reply text it had revealed (spec §3)."""
        turn_id, heard = cmd.get("turn_id"), cmd.get("heard")
        if not isinstance(turn_id, int) or isinstance(turn_id, bool) or not isinstance(heard, str):
            return
        with self._state_lock:
            if self._current_turn is None or turn_id != self._current_turn or self._cut.is_set():
                return  # late, stale or repeated ack: ignore silently
            self._cut_heard = heard
            # Marked under the lock: the turn clears _current_turn under it before timer.reset().
            self.timer.mark("playback_cut")
            self._cut.set()

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
        if "rag_template" in cmd and not (isinstance(cmd["rag_template"], str) and "{context}" in cmd["rag_template"]):
            bad.append("rag_template must be a string containing {context}")
        if "word_align" in cmd and not isinstance(cmd["word_align"], bool):
            bad.append("word_align must be true or false")
        if "barge_in_mode" in cmd:
            if cmd["barge_in_mode"] not in ("off", "onset", "words"):
                bad.append("barge_in_mode must be off, onset or words")
            elif cmd["barge_in_mode"] == "words" and not self.barge_words_available():
                bad.append("barge_in_mode words needs a realtime STT model")
        if "barge_in_min_ms" in cmd and not (isinstance(cmd["barge_in_min_ms"], int)
                                             and not isinstance(cmd["barge_in_min_ms"], bool)
                                             and 0 <= cmd["barge_in_min_ms"] <= 1000):
            bad.append("barge_in_min_ms must be an integer from 0 to 1000")
        if "diarization" in cmd and not isinstance(cmd["diarization"], bool):
            bad.append("diarization must be true or false")
        if "diar_mode" in cmd and cmd["diar_mode"] not in DIAR_MODES:
            bad.append("diar_mode must be ultra_low_latency, very_low_latency or low_latency")
        if "diar_wait_ms" in cmd and not (isinstance(cmd["diar_wait_ms"], int)
                                          and not isinstance(cmd["diar_wait_ms"], bool)
                                          and 0 <= cmd["diar_wait_ms"] <= 2000):
            bad.append("diar_wait_ms must be an integer from 0 to 2000")
        if bad:
            self._error("bad_settings", "; ".join(bad))
            return
        if "word_align" in cmd:
            self._set_word_align(cmd["word_align"])
        if "diar_mode" in cmd:   # before switching on, so a combined update starts in the new mode
            self._set_diar_mode(cmd["diar_mode"])
        if "diarization" in cmd:
            self._set_diarization(cmd["diarization"])
        stale = False
        with self._state_lock:
            changes = {k: cmd[k] for k in ("temperature", "top_p", "max_tokens") if k in cmd}
            if changes:
                self.params = replace(self.params, **changes)
            if "system_prompt" in cmd:
                self.conversation.system_prompt = cmd["system_prompt"]
            if "rag_template" in cmd:
                self.conversation.rag_template = cmd["rag_template"]
            if "voice" in cmd and cmd["voice"] != self.voice:
                self.voice = self._pending_voice = cmd["voice"]
            if "barge_in_mode" in cmd:
                if cmd["barge_in_mode"] == "words" and not self.barge_words_available():
                    stale = True   # realtime STT was unloaded since validation: keep the old mode
                else:
                    self.barge_in_mode = cmd["barge_in_mode"]
            if "barge_in_min_ms" in cmd:
                self.barge_in_min_ms = cmd["barge_in_min_ms"]
            if "diar_wait_ms" in cmd:
                self.diar_wait_ms = cmd["diar_wait_ms"]
        if stale:
            self._error("bad_settings", "barge_in_mode words needs a realtime STT model")

    # ---------- forced word alignment ----------

    def _set_word_align(self, on: bool) -> None:
        """Switch forced word alignment. The first switch-on loads the model (about 1.2 GB) off the event loop;
        the browser hears {"type": "word_align", "state": "loading" | "ready" | "off" | "error"}."""
        self._align_wanted = on
        if not on:
            self.word_align = False
            self.audio.word_aligner = None
            self._emit({"type": "word_align", "state": "off"})
            return
        aligner = self._word_aligner
        if aligner is None:
            self._emit({"type": "word_align", "state": "error",
                        "message": "Word alignment is not available on this server; see the server log."})
            return
        if aligner.ready:
            self._enable_word_align(aligner)
            return
        self._emit({"type": "word_align", "state": "loading"})

        def load() -> None:
            try:
                aligner.load()
            except Exception as e:
                log.exception("loading the word alignment model failed")
                self._align_wanted = False
                self._emit({"type": "word_align", "state": "error", "message": f"Loading the alignment model failed: {e}"})
                return
            if self._align_wanted:  # still wanted: it may have been switched off while loading
                self._enable_word_align(aligner)
        self._spawn(load, "word-align-load")

    def _enable_word_align(self, aligner) -> None:
        self.audio.word_aligner = aligner
        self.word_align = True
        self._emit({"type": "word_align", "state": "ready"})

    # ---------- v4: speaker diarization ----------

    def _active_diarizer(self):
        """The diarizer when diarization is on and attached to the audio, else None."""
        d = self._diarizer
        if self.diarization and d is not None and getattr(self.audio, "diarizer", None) is d:
            return d
        return None

    def _set_diar_mode(self, mode: str) -> None:
        with self._diar_lock:
            self.diar_mode = mode
            d = self._active_diarizer()
            if d is not None:
                self._invalidate_window()
                try:
                    d.set_mode(mode)   # resets the timeline
                except Exception:
                    log.exception("diarizer set_mode failed")
                self._invalidate_window()

    def _set_diarization(self, on: bool) -> None:
        """Switch speaker diarization. The first switch-on loads the model off the event loop; the browser
        hears {"type": "diarization", "state": "loading" | "ready" | "off" | "error"}, like word_align.

        Every switch-on runs on a _switch_on worker. _diar_wanted records the latest request, and the worker
        enables the diarizer only if it is still set, so on -> off (or on -> off -> on -> off) during one load
        ends off. _diar_loading makes a second "on" while a worker runs wait for it instead of starting another."""
        if not on:
            self._diarization_off()
            return
        if not self.diarization_available():
            self._emit({"type": "diarization", "state": "error", "message": DIAR_UNAVAILABLE})
            return
        with self._diar_lock:
            self._diar_wanted = True
            if self.diarization:
                self._emit({"type": "diarization", "state": "ready"})
                return
            if self._diar_loading:   # the running worker emits ready (or error) for this request too
                if not self._diarizer.ready:
                    self._emit({"type": "diarization", "state": "loading"})
                return
            d = self._diarizer
            if d is None:
                try:
                    d = self._diarizer = self.diarizer_factory(self._on_diar_error)
                except Exception as e:
                    log.exception("building the diarizer failed")
                    self._diar_wanted = False
                    self._emit({"type": "diarization", "state": "error",
                                "message": f"Building the diarizer failed: {e}"})
                    return
            self._diar_loading = True
            if not d.ready:
                self._emit({"type": "diarization", "state": "loading"})
        self._spawn(lambda: self._switch_on(d), "diarizer-on")

    def _switch_on(self, d) -> None:
        """The switch-on worker (off the event loop): stop any previous run, load if needed, then enable.

        A run that ended with on_error keeps its thread until a stop: Diarizer.start() returns early while
        that thread is alive and would not re-arm its once-per-start error report. So wait for the stop
        _on_diar_error spawned, and stop() again (a no-op without a thread) before set_mode() and start()."""
        with self._diar_lock:
            stopper = self._diar_stopper
        if stopper is not None:
            stopper.join()
        try:
            d.stop()
        except Exception:
            log.exception("stopping the previous diarizer run failed")
        t0 = time.monotonic()
        load_ms = 0
        if not d.ready:
            try:
                d.load()
            except Exception as e:
                log.exception("loading the diarization model failed")
                with self._diar_lock:
                    self._diar_loading = False
                    wanted, self._diar_wanted = self._diar_wanted, False
                    if wanted:
                        self._emit({"type": "diarization", "state": "error",
                                    "message": f"Loading the diarization model failed: {e}"})
                if wanted:
                    self._write_event({"event": "diarizer_off", "reason": "error"})
                return
            load_ms = round((time.monotonic() - t0) * 1000)
        self._enable_diarizer(d, load_ms)

    def _enable_diarizer(self, d, load_ms: int) -> None:
        """Start and attach d if diarization is still wanted (it may have been switched off while loading).
        Clears _diar_loading in the same critical section, so an "on" that arrives meanwhile is never lost."""
        with self._diar_lock:
            self._diar_loading = False
            if not self._diar_wanted or self.diarization:
                return
            mode = self.diar_mode
            self._invalidate_window()   # windows from before this run are on another clock
            try:
                d.set_mode(mode)   # resets: a restart after an error gets a fresh stream
                d.start()
            except Exception as e:
                log.exception("starting the diarizer failed")
                self._diar_wanted = False
                self._emit({"type": "diarization", "state": "error", "message": f"Starting the diarizer failed: {e}"})
                return
            self.audio.diarizer = d
            self.diarization = True
            self._emit({"type": "diarization", "state": "ready"})
        log.info("diarizer on: mode=%s load_ms=%d", mode, load_ms)
        self._write_event({"event": "diarizer_on", "diar_mode": mode, "load_ms": load_ms})

    def _diarization_off(self) -> None:
        with self._diar_lock:
            self._diar_wanted = False
            was_on, d = self.diarization, self._diarizer
            if was_on:
                self.audio.diarizer = None
                self.diarization = False
            self._emit({"type": "diarization", "state": "off"})
        if not was_on or d is None:
            return
        # Outside _diar_lock: stop() joins the diarizer thread, whose on_error may be waiting for that lock.
        try:
            d.stop()
            d.reset()
        except Exception:
            log.exception("stopping the diarizer failed")
        self._invalidate_window()
        log.info("diarizer off: user")
        self._write_event({"event": "diarizer_off", "reason": "user"})

    def _on_diar_error(self, reason: str, message: str) -> None:
        """The Diarizer gave up: "overflow" (on the websocket audio thread, inside feed) or "error" (on the
        diarizer thread). Quick and non-blocking: detach and report. stop() joins the diarizer thread, so it
        runs on a thread of its own; after an overflow that thread is still alive and must be stopped, or a
        later start() would return early and never report another error. _switch_on waits for it."""
        with self._diar_lock:
            if not self.diarization:
                return
            d = self._diarizer
            self.audio.diarizer = None
            self.diarization = False
            self._diar_wanted = False
            self._emit({"type": "diarization", "state": "error", "message": message})
            self._diar_stopper = self._spawn(lambda: self._stop_quietly(d), "diarizer-stop")
        log.warning("diarizer off: %s (%s)", reason, message)
        self._write_event({"event": "diarizer_off", "reason": reason})

    @staticmethod
    def _stop_quietly(d) -> None:
        try:
            d.stop()
        except Exception:
            log.exception("stopping the diarizer failed")

    def _reset_diarizer(self) -> None:
        """Clear and model switches: the next voice heard becomes Speaker 0."""
        d = self._active_diarizer()
        if d is None:
            return
        # Before and after: a decision that snapshotted before the reset sees the first bump, and a
        # position() read on the old clock between the two is cleared by the second.
        self._invalidate_window()
        try:
            d.reset()
        except Exception:
            log.exception("diarizer reset failed")
        self._invalidate_window()

    def _invalidate_window(self) -> None:
        """The diarizer's clock restarted: no window recorded so far can be decided on."""
        with self._state_lock:
            self._diar_epoch += 1
            self._utt_start = self._utt_end = None
            self._last_window = None
            self._barge_onset_sample = None   # a barge check on the old clock fails open

    def _pre_samples(self) -> int:
        """RealtimeSTT's pre-roll (the audio kept from before the VAD start) in samples at 16 kHz."""
        return int(getattr(self.audio.recorder, "pre_recording_buffer_duration", 0.2) * 16000)

    @staticmethod
    def _diar_position(d) -> int | None:
        try:
            return d.position()
        except Exception:
            log.exception("diarizer position failed")
            return None

    def _decide_speaker(self) -> dict | None:
        """Who spoke the utterance just transcribed (spec: turn check). None when diarization is off.

        Waits (polling every 20 ms) up to diar_wait_ms for the diarizer's timeline to cover the utterance.
        Fails open: a timeout, a diarizer that fails or is switched off meanwhile, a window that a reset made
        invalid, or any diarizer exception gives "unknown", which gets a turn."""
        d = self._active_diarizer()
        if d is None:
            return None
        with self._state_lock:
            window, epoch, wait_ms = self._last_window, self._diar_epoch, self.diar_wait_ms
            self._last_window = None   # each completed window is decided once
        start, end = (None, None) if window is None or window[2] != epoch else window[:2]
        self.timer.mark("diar_start")
        speaker: int | str = "unknown"
        lag = 0
        try:
            if start is not None and end is not None and start < end:
                covered = d.covered()
                lag = max(0, end - covered)
                deadline = time.monotonic() + wait_ms / 1000
                # The epoch is read without a lock (an int): a reset during the wait ends it at once.
                while (covered < end and time.monotonic() < deadline and self._active_diarizer() is d
                       and self._diar_epoch == epoch):
                    time.sleep(0.02)
                    covered = d.covered()
                if covered >= end and self._active_diarizer() is d and self._diar_epoch == epoch:
                    found = d.speaker_between(start, end)
                    # A reset that raced speaker_between may have answered from the new clock.
                    if found is not None and self._diar_epoch == epoch:
                        speaker = found
        except Exception:
            log.exception("speaker decision failed; passing the turn as unknown")
            speaker = "unknown"
        self.timer.mark("diar_done")
        return {"diarization": True, "diar_mode": d.mode, "diar_speaker": speaker, "diar_lag_ms": round(lag / 16)}

    def _diar_turn_fields(self) -> dict:
        """The diarization fields of a turn record with no speaker decision (typed turns, or off)."""
        d = self._active_diarizer()
        rtf = None
        if d is not None:
            try:
                rtf = d.rtf()
            except Exception:
                log.exception("diarizer rtf failed")
        with self._state_lock:
            rejects = self._diar_rejects
        return {"diarization": d is not None, "diar_mode": d.mode if d is not None else None,
                "diar_speaker": None, "diar_lag_ms": None, "diar_rejects": rejects, "diar_rtf": rtf}

    # ---------- v3: RAG settings (spec §4.5) ----------

    def _update_rag(self, cmd: dict) -> None:
        rag = self.rag
        if rag is None:
            self._error("bad_command", "RAG is not available")
            return
        if self._turn_lock.locked():
            self._error("rag_busy", "Wait for the reply to finish before changing RAG settings.")
            # resync the RAG tab's drafts; off the event loop (config_event probes Ollama)
            self._spawn(lambda: self._emit_rag_config(rag), "rag-config")
            return
        patch = {k: v for k, v in cmd.items() if k != "type"}

        def run() -> None:
            # Off the event loop: loading an embedder takes seconds.
            from voicebot.rag.service import RagError
            # The service emits rag_config itself after a successful update and after a
            # failed switch (RagError); the session adds one where the service did not.
            try:
                rag.update(patch)
            except ValueError as e:
                self._error("bad_settings", str(e))
                self._emit_rag_config(rag)
            except RagError as e:  # the switch failed; the previous embedder stays
                self._error("rag_error", str(e))
            except Exception as e:
                log.exception("update_rag failed")
                self._error("rag_error", str(e))
                self._emit_rag_config(rag)
        self._spawn(run, "rag-update")

    def _emit_rag_config(self, rag) -> None:
        try:
            self._emit(rag.config_event())
        except Exception:
            log.exception("rag config_event failed")

    # ---------- v2: model switching (spec §4.4) ----------

    def on_partial(self, text: str) -> None:
        """Realtime STT partial (recorder's on_realtime_transcription_update). Dropped during a turn,
        after the words-mode barge-in check: the first partial that is not noise fires it."""
        with self._state_lock:
            fire = (self._barge_turn if self._barge_turn is not None and not self._barge_fired
                    and self._barge_mode == "words" and not self._barge_words_rejected
                    and not is_noise(text) else None)
        if fire is not None:
            self._confirm_barge(fire)
        if self._turn_lock.locked():
            return
        self._emit({"type": "user_partial", "text": text})

    def _select_model(self, cmd: dict) -> None:
        role, slug = cmd.get("role"), cmd.get("slug")
        if self._turn_lock.locked() or self._switching.locked():
            self._error("busy", "Models can only be switched while idle.")
            return
        if self.models is None:
            self._error("bad_model", "model switching is not available")
            return
        from voicebot.models import ModelError
        try:
            self.models.validate(role, slug)
        except ModelError as e:
            self._error("bad_model", str(e))
            return
        if not self._switching.acquire(blocking=False):
            self._error("busy", "A model switch is already in progress.")
            return
        self._spawn(lambda: self._run_switch(role, slug), "model-switch")

    def _pause_listening(self) -> bool:
        """Stop the listen loop and wait for it, so the recorder can be replaced. Returns whether it was on."""
        was = self._listening
        listener = self._listen_thread
        self._listening = False
        self._listen_gen += 1
        with self._state_lock:
            self._recording = False   # the recorder is aborted and replaced: its on_recording_stop never comes
        if listener is not None and listener.is_alive():
            recorder = self.audio.recorder

            def abort() -> None:
                try:
                    recorder.abort()
                except Exception:
                    log.exception("recorder.abort failed")
            aborter = threading.Thread(target=abort, name="stt-abort-switch", daemon=True)
            aborter.start()
            aborter.join(timeout=2)
            listener.join(timeout=10)
        return was

    def _run_switch(self, role: str, slug: str) -> None:
        from voicebot.models import ModelError
        t0 = time.monotonic()
        was_listening = False
        with self._state_lock:
            self._barge_pending = None   # the recorder is replaced: no transcript will settle it
        try:
            self._emit({"type": "model_status", "role": role, "slug": slug, "state": "loading"})
            if role != "tts":
                was_listening = self._pause_listening()
            with self._turn_lock:
                try:
                    if role == "tts":
                        self._install_stream(self.models.switch_tts(slug, self.stream))
                    else:
                        self.audio.set_recorder(self.models.switch_stt(role, slug, self.audio.recorder))
                except ModelError as e:
                    if e.restored is not None:
                        if role == "tts":
                            self._install_stream(e.restored)
                        else:
                            self.audio.set_recorder(e.restored)
                    self._emit({"type": "model_status", "role": role, "slug": slug, "state": "failed",
                                "message": str(e)})
                    return
            self._emit({"type": "model_status", "role": role, "slug": slug, "state": "ready",
                        "load_ms": round((time.monotonic() - t0) * 1000)})
            if role != "tts":
                with self._state_lock:
                    if self.barge_in_mode == "words" and not self.barge_words_available():
                        self.barge_in_mode = "onset"
                self._emit({"type": "config", **self.config()})
        except Exception as e:
            log.exception("model switch %s=%s failed", role, slug)
            self._emit({"type": "model_status", "role": role, "slug": slug, "state": "failed", "message": str(e)})
        finally:
            self._reset_diarizer()   # after any switch, success or failure
            self._switching.release()
            if was_listening:
                self._start_listening({})
            elif not self._turn_lock.locked():
                self._set_status("idle")

    def _install_stream(self, stream) -> None:
        """Use a new TTS stream: refresh voices and, in browser mode, the output format (spec §4.4)."""
        self.stream = stream
        try:
            voices = list_voices(stream)
        except Exception:
            voices = []
        entry = self.models.entry("tts") if self.models is not None else None
        with self._state_lock:
            self.voices = voices or ["default"]
            preferred = entry.voice if entry is not None else None
            self.voice = preferred if preferred in self.voices else self.voices[0]
            self._pending_voice = None
        if self.audio.output_format is not None:
            self.audio.set_output_format(tts_output_format(stream))
            self._emit({"type": "config", **self.config()})

    def _clear(self, _cmd: dict) -> None:
        self._disarm()
        with self._state_lock:
            self._barge_pending = None
        self.conversation.clear()
        self._reset_diarizer()

    # ---------- barge-in (spec §2) ----------
    # Public recorder callbacks, called from RealtimeSTT's threads at any time. Each one reads and
    # changes the arming state under _state_lock, and fires (which emits and calls _stop_speaking,
    # both taking locks of their own) only after releasing it.

    def on_recording_start(self) -> None:
        d = self._active_diarizer()
        with self._state_lock:
            self._recording = True
            # The utterance window starts before the VAD start by the recorder's pre-roll. A stale window
            # from before diarization was switched on must not be reused, so it is cleared when off.
            pos = self._diar_position(d) if d is not None else None
            self._utt_start = max(0, pos - self._pre_samples()) if pos is not None else None
            self._utt_end = None
            self._barge_words_rejected = False   # a new recording may be another speaker
            if self._barge_turn is None or self._barge_fired:
                return
            turn = self._barge_turn
            self._mark_onset_locked(self._utt_start)
            fire = self._barge_mode == "onset" and self._start_barge_timer_locked()
        if fire:
            self._confirm_barge(turn)

    def on_recording_stop(self) -> None:
        d = self._active_diarizer()
        with self._state_lock:
            self._recording = False
            self._utt_end = self._diar_position(d) if d is not None else None
            self._last_window = (self._utt_start, self._utt_end, self._diar_epoch)
            if self._barge_turn is None or self._barge_fired:
                return
            if self._barge_mode == "words" and self.timer.has("barge_onset"):
                self._barge_false += 1   # the recording ended without a real word: a cough or a noise
                self.timer.unmark("barge_onset")
            elif self._barge_mode == "onset" and self._barge_timer is not None:
                # The recording ended before min_ms without a pause callback. A pause that came first
                # already cancelled the timer and counted it, so this cannot count twice.
                self._barge_timer.cancel()
                self._barge_timer = None
                self.timer.unmark("barge_onset")
                self._barge_false += 1

    def on_speech_pause(self) -> None:
        """Recorder on_turn_detection_start: the speech paused before the onset timer ran out."""
        with self._state_lock:
            t = self._barge_timer
            if self._barge_turn is None or self._barge_fired or self._barge_mode != "onset" or t is None:
                return
            t.cancel()
            self._barge_timer = None
            self.timer.unmark("barge_onset")
            self._barge_false += 1

    def on_speech_resume(self) -> None:
        """Recorder on_turn_detection_stop: speech resumed inside the same recording, so arm the onset
        again (RealtimeSTT fires no second on_recording_start). It also fires after a recording ends;
        _recording is False then and nothing happens. A running speaker check keeps its onset."""
        d = self._active_diarizer()
        with self._state_lock:
            if (self._barge_turn is None or self._barge_fired or self._barge_mode != "onset"
                    or not self._recording or self._barge_timer is not None or self._barge_checking):
                return
            turn = self._barge_turn
            # Like on_recording_start: VAD reports the resume late, so the onset reaches back by the pre-roll
            # (an onset at position() would leave the check an empty window, which never decides).
            pos = self._diar_position(d) if d is not None else None
            self._mark_onset_locked(max(0, pos - self._pre_samples()) if pos is not None else None)
            fire = self._start_barge_timer_locked()
        if fire:
            self._confirm_barge(turn)

    def _mark_onset_locked(self, sample: int | None) -> None:
        """Mark barge_onset and where it starts on the diarizer's timeline (caller holds _state_lock)."""
        self.timer.mark("barge_onset")
        self._barge_onset_sample = sample
        self._barge_onset_gen += 1

    def _start_barge_timer_locked(self) -> bool:
        """Start the onset timer (caller holds _state_lock). Returns True when it must fire at once."""
        if self._barge_timer is not None:
            return False
        if self._barge_min_ms <= 0:
            return True
        turn = self._barge_turn

        def due() -> None:
            with self._state_lock:
                if self._barge_timer is not t:   # cancelled, disarmed or replaced: a late callback
                    return
                self._barge_timer = None
            self._confirm_barge(turn)
        t = threading.Timer(self._barge_min_ms / 1000, due)
        t.daemon = True
        self._barge_timer = t
        t.start()
        return False

    def _confirm_barge(self, turn_id: int | None) -> None:
        """A barge-in trigger. Diarization off: fire at once on this thread (v3). On: only Speaker 0 may
        interrupt, so a check runs on its own thread (never blocking RealtimeSTT's callback threads). One
        check at a time; a trigger that comes during one is remembered and checked after it if it still
        applies (a newer onset), else dropped."""
        d = self._active_diarizer()
        if d is None:
            self._fire_barge(turn_id)
            return
        with self._state_lock:
            if turn_id is None or self._barge_turn != turn_id or self._barge_fired:
                return
            if self._barge_checking:
                self._barge_recheck = turn_id
                return
            self._barge_checking = True
            self._barge_recheck = None
        try:
            self._spawn(lambda: self._barge_check(turn_id, d), "barge-check")
        except Exception:
            log.exception("starting the barge-in check failed; firing (fail open)")
            with self._state_lock:
                self._barge_checking = False
            self._fire_barge(turn_id)

    def _barge_check(self, turn_id: int, d) -> None:
        """The check thread: decide, then fire, reject or stop. Always clears _barge_checking."""
        try:
            while True:
                verdict, gen = self._barge_verdict(turn_id, d)
                if verdict == "fire":
                    self._fire_barge(turn_id)
                    return
                with self._state_lock:
                    if verdict == "reject" and self._barge_turn == turn_id and not self._barge_fired:
                        self._diar_rejects += 1
                        if self._barge_onset_gen == gen:   # still the onset that was checked
                            self.timer.unmark("barge_onset")
                            if self._barge_mode == "words":
                                self._barge_words_rejected = True
                            self._barge_recheck = None     # later triggers of this onset are the same speaker
                    nxt, self._barge_recheck = self._barge_recheck, None
                    if nxt is None or nxt != self._barge_turn or self._barge_fired:
                        return
                    turn_id = nxt                          # a newer onset was triggered meanwhile: check it
        finally:
            with self._state_lock:
                self._barge_checking = False
                self._barge_recheck = None

    def _barge_verdict(self, turn_id: int, d) -> tuple[str, int]:
        """("fire" | "reject" | "stop", onset generation). Fixes the window end once, then polls every
        BARGE_POLL_S up to diar_wait_ms while the diarizer has not covered it. Fails open: an unknown
        onset, a timeout, a diarizer that is gone or reset, or any diarizer exception gives "fire"."""
        with self._state_lock:
            onset, gen = self._barge_onset_sample, self._barge_onset_gen
            epoch, wait_ms = self._diar_epoch, self.diar_wait_ms
        try:
            if onset is None:
                return "fire", gen
            end = d.position()
            deadline = time.monotonic() + wait_ms / 1000
            while True:
                with self._state_lock:
                    if self._barge_turn != turn_id or self._barge_fired:
                        return "stop", gen
                # The epoch is read without a lock (an int), as in _decide_speaker.
                if self._active_diarizer() is not d or self._diar_epoch != epoch:
                    return "fire", gen
                share = d.share(0, onset, end)
                if self._diar_epoch != epoch:   # a reset raced share(): its answer may be on the new clock
                    return "fire", gen
                if share is not None:
                    if share >= BARGE_SHARE:
                        return "fire", gen
                    log.info("barge-in on turn %d rejected: Speaker 0 share %.2f", turn_id, share)
                    return "reject", gen
                if time.monotonic() >= deadline:
                    return "fire", gen
                time.sleep(BARGE_POLL_S)
        except Exception:
            log.exception("barge-in speaker check failed; firing (fail open)")
            return "fire", gen

    def _fire_barge(self, turn_id: int | None) -> None:
        """Fire the barge-in once per armed turn: mark barge_in, tell the browser, stop the reply."""
        with self._state_lock:
            if turn_id is None or self._barge_turn != turn_id or self._barge_fired:
                return
            self._barge_fired = True
            if self._barge_timer is not None:
                self._barge_timer.cancel()
                self._barge_timer = None
            # Marked under the lock: the turn disarms under it before timer.reset(), so this
            # mark can never land on the next turn's timer.
            self.timer.mark("barge_in")
            self._barge_pending = {"turn_id": turn_id, "barge_in_mode": self._barge_mode,
                                   "barge_in_min_ms": self._barge_min_ms}
        # Cut first: audio or reveal events sent before barge_in would reach the browser after its flush.
        self._stop_speaking({}, turn_id=turn_id)
        self._emit({"type": "barge_in", "turn_id": turn_id})

    def _disarm(self) -> None:
        with self._state_lock:
            if self._barge_timer is not None:
                self._barge_timer.cancel()
                self._barge_timer = None
            self._barge_turn = None

    # ---------- loops ----------

    def _spawn(self, target: Callable[[], None], name: str) -> threading.Thread:
        t = threading.Thread(target=target, name=name, daemon=True)
        self._threads = [x for x in self._threads if x.is_alive()] + [t]
        t.start()
        return t

    def _listen_loop(self, gen: int, previous: threading.Thread | None) -> None:
        # A quick Mic off -> on can start this run while the previous one is still inside
        # recorder.text() (its abort runs on a helper thread) or a turn. RealtimeSTT is not
        # safe for two concurrent text() calls, so wait for it to finish first.
        if previous is not None and previous is not threading.current_thread():
            previous.join()
        if gen == self._listen_gen:
            # No recording can be in progress before this run's first text(): drop a flag left by an
            # earlier run whose abort() raced an on_recording_start.
            with self._state_lock:
                self._recording = False
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
            noise = is_noise(text)
            with self._state_lock:
                pending, self._barge_pending = self._barge_pending, None
            if pending is not None and noise:
                self._log_false_fire(pending)
            if noise:
                # Under _turn_lock: an armed typed turn leaves the mic open, and its marks must survive.
                with self._turn_lock:
                    self.timer.reset()
                continue
            with self._turn_lock:
                self.timer.mark("stt_done")
                text = text.strip()
                diar = self._decide_speaker()
                if diar is not None and diar["diar_speaker"] not in (0, "unknown"):
                    if pending is not None:
                        # The barge-in check passed (or failed open) but the turn check says another speaker.
                        log.info("barge-in on turn %d fired, but its utterance was speaker %s: ignored",
                                 pending["turn_id"], diar["diar_speaker"])
                    self._ignore_speaker(text, diar)
                    continue
                self._run_turn(text, source="voice", diar_fields=diar)
        if gen == self._listen_gen:  # an older run must not switch off a newer one
            if self._listening:
                self._listening = False
                if not self._turn_lock.locked():
                    self._set_status("idle")

    def _ignore_speaker(self, text: str, diar: dict) -> None:
        """Another speaker's utterance (caller holds _turn_lock): shown as ignored, logged, no turn."""
        speaker = diar["diar_speaker"]
        start, done = self.timer.get("diar_start"), self.timer.get("diar_done")
        wait_ms = round((done - start) * 1000) if start is not None and done is not None else None
        self._emit({"type": "user_transcript", "text": text, "turn_id": None, "speaker": speaker, "ignored": True})
        log.info("ignored speaker %d: %r", speaker, text)
        self._write_event({"event": "diar_ignored", "speaker": speaker, "text": text, "diar_wait_ms": wait_ms,
                           "diar_mode": diar["diar_mode"]})
        self.timer.reset()

    def _log_false_fire(self, pending: dict) -> None:
        """A fired barge-in whose utterance transcribed as noise: one event record (spec §2 step 3, §4)."""
        log.info("barge-in on turn %d was a false fire (noise transcript)", pending["turn_id"])
        self._write_event({"event": "barge_false_fire", **pending})

    def _write_event(self, record: dict) -> None:
        """One event record for metrics_sink: the record plus ts, the server info and the model slugs."""
        if self.metrics_sink is None:
            return
        record = {"event": record.get("event"), "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  **record, **self.info}
        if self.models is not None:
            m = self.models.current
            record.update(stt_final=m.get("stt_final"), stt_realtime=m.get("stt_realtime", "off"),
                          tts_model=m.get("tts"))
        try:
            self.metrics_sink(record)
        except Exception:
            log.exception("metrics_sink failed")

    def _marked(self, chunks: Iterable[str]) -> Iterator[str]:
        """Mark llm_first_token on the first raw chunk and llm_done when the backend stream ends."""
        for chunk in chunks:
            self.timer.mark("llm_first_token")
            yield chunk
        self.timer.mark("llm_done")

    def _run_turn(self, text: str, *, source: str, diar_fields: dict | None = None) -> None:
        stop_event = threading.Event()
        with self._state_lock:
            self._turn_id += 1
            turn_id = self._turn_id
            self._current_turn = turn_id
            self._stop_event = stop_event
            self._ack.clear()
            self._cut.clear()
            self._cut_heard = None
            params = self.params
            voice, self._pending_voice = self._pending_voice, None
            messages = self.conversation.messages(text)
        rag_settings = self.rag.settings if self.rag is not None else None  # the RAG switch at turn start
        diar_record = self._diar_turn_fields()   # the diarization switch at turn start
        if diar_fields is not None:   # a voice turn's speaker decision
            diar_record.update(diar_fields)
        rag_on = rag_settings is not None and rag_settings.enabled
        rag_fields: dict = {
            "rag_enabled": rag_on,
            "rag_embedder": rag_settings.embedder if rag_on else None,
            "rag_top_k": rag_settings.top_k if rag_on else None,
            "rag_min_score": rag_settings.min_score if rag_on else None,
            "rag_max_context_chars": rag_settings.max_context_chars if rag_on else None,
            "rag_chunk_size": rag_settings.chunk_size if rag_on else None,
            "rag_chunk_overlap": rag_settings.chunk_overlap if rag_on else None,
            "rag_chunks_used": 0,
            "rag_context_chars": 0,
        }
        parts: list[str] = []
        errors: list[BaseException] = []
        consumed: list[bool] = []  # set once TTS starts reading the LLM stream
        browser = self.audio.output_format is not None

        def tokens():
            # Exceptions are captured here, not raised: RealtimeTTS consumes this
            # iterator on its own thread, where a raise would be lost.
            consumed.append(True)
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

        def on_sentence(sentence: str, at_s: float | None) -> None:
            # The browser shows each sentence when its playback reaches at_s (None: show at once).
            if not stop_event.is_set():
                self._emit({"type": "assistant_segment", "turn_id": turn_id, "text": sentence, "at_s": at_s})

        def on_words(sentence: str, word_s: list[float] | None, end_s: float) -> None:
            # After the sentence above is synthesized: when each of its words starts on the reply clock
            # (None: the engine gave none, so the browser spreads the words up to end_s).
            if not stop_event.is_set():
                self._emit({"type": "assistant_words", "turn_id": turn_id, "word_s": word_s, "end_s": end_s})

        stopper = self._stopper
        if stopper is not None:
            stopper.join(timeout=5)  # let the previous turn's stop finish before playing again
        transcript_event = {"type": "user_transcript", "text": text, "turn_id": turn_id}
        if diar_fields is not None:
            transcript_event["speaker"] = diar_fields["diar_speaker"]
        self._emit(transcript_event)
        fire_now = False
        with self._state_lock:   # barge-in settings are read once per turn (spec §2)
            barge_mode, barge_min_ms = self.barge_in_mode, self.barge_in_min_ms
            armed = barge_mode != "off" and (source == "voice" or self._listening)
            if armed:
                # Before the turn is armed: a callback that sees it armed may mark at once, and the
                # barge utterance's STT marks belong to the next turn.
                self.timer.hold()
            self._barge_turn = turn_id if armed else None
            self._barge_mode = barge_mode if armed else "off"
            self._barge_min_ms = barge_min_ms
            self._barge_fired = stop_event.is_set()   # a Stop that came before arming keeps it disarmed
            self._barge_false = 0
            self._diar_rejects = 0
            self._barge_timer = None
            if armed and self._recording and not self._barge_fired:
                # The user is already mid-utterance (e.g. during a typed turn): no on_recording_start
                # will come for it, so the onset starts now (words mode: the mark only, no timer).
                self._mark_onset_locked(self._utt_start)
                if barge_mode == "onset":
                    fire_now = self._start_barge_timer_locked()
        if not armed:
            self.audio.mute_input(True)
        try:
            if fire_now:
                self._confirm_barge(turn_id)
            dropped = drain_audio_queue(self.stream)
            if dropped:
                log.info("dropped %d stale TTS audio chunks before turn %d", dropped, turn_id)
            if voice:
                set_voice(self.stream, voice)
            self._set_status("thinking")
            if rag_on:
                messages = self._retrieve(text, turn_id, rag_fields) or messages
            self.stream.feed(tokens())
            audio_seconds = self.audio.play(self.stream, on_first_audio, on_sentence, on_words)
            if not consumed and not stop_event.is_set():
                # RealtimeTTS failed on its own thread before reading the reply (it only prints the
                # error there). Without this the turn would end silently as an empty reply.
                log.error("turn %d: text-to-speech never read the LLM stream; see the RealtimeTTS error above", turn_id)
                self._error("tts_error", "Text-to-speech failed before speaking; see the server log.")
                self._emit({"type": "assistant_done", "text": "", "interrupted": False, "heard": None})
                return
            if not errors:
                self._wait_for_browser_playback(audio_seconds, stop_event)
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
                self._emit({"type": "assistant_done", "text": APOLOGY, "interrupted": False, "heard": None})
                return

            reply = "".join(parts)
            interrupted = stop_event.is_set()
            heard = self._await_cut() if interrupted and browser else None
            # History keeps only what the user heard (spec §3): the acked text, nothing for an empty
            # ack, or the full reply when no ack came.
            stored = reply if heard is None else heard
            if stored:
                self.conversation.add_exchange(text, stored)
            self._emit({"type": "assistant_done", "text": reply, "interrupted": interrupted, "heard": heard})
            with self._state_lock:
                # Barge-in settings and counts, plus heard (also set by a Stop-button cut with barge-in off).
                interrupt_fields = {"barge_in_mode": self._barge_mode if armed else "off",
                                    "barge_in_min_ms": barge_min_ms, "barge_false_triggers": self._barge_false,
                                    "heard": heard}
                diar_record["diar_rejects"] = self._diar_rejects
            self._finish_metrics(turn_id, source, interrupted, params, reply, text, rag_fields, interrupt_fields,
                                 diar_record)
        except Exception as e:
            log.exception("turn %d failed", turn_id)
            self._error("llm_error", str(e))
        finally:
            self._disarm()
            with self._state_lock:
                self._stop_event = None
                self._current_turn = None
            self.timer.reset()
            if not armed:
                self.audio.mute_input(False)
            self._set_status("listening" if self._listening else "idle")

    def _retrieve(self, text: str, turn_id: int, rag_fields: dict) -> list[dict] | None:
        """Retrieve context for this turn (spec §4.3). Returns the messages with the context in the
        system message, or None when no context was found. Never raises: a failure is reported
        as rag_error and the turn answers without context."""
        assert self.rag is not None
        self.timer.mark("retrieval_start")
        try:
            result = self.rag.retrieve(text)
            error = None
        except Exception as e:
            result, error = None, e
        self.timer.mark("retrieval_done")
        # Re-read the settings now: an embedder switch that overlapped this turn may have
        # changed what retrieve() actually used, and the log must name that.
        s = self.rag.settings
        rag_fields.update(rag_embedder=s.embedder, rag_top_k=s.top_k, rag_min_score=s.min_score,
                          rag_max_context_chars=s.max_context_chars, rag_chunk_size=s.chunk_size,
                          rag_chunk_overlap=s.chunk_overlap)
        if error is not None:
            log.warning("turn %d: RAG retrieval failed: %s", turn_id, error)
            rag_fields["rag_error"] = str(error)
            self._error("rag_error", str(error))
            return None
        assert result is not None
        if not result.context or not result.chunks:
            return None
        rag_fields.update(rag_chunks_used=len(result.chunks), rag_context_chars=len(result.context))
        with self._state_lock:
            messages = self.conversation.messages(text, context=result.context)
        self._emit({"type": "rag_sources", "turn_id": turn_id, "sources": [
            {"name": h.name, "chunk": h.chunk, "score": h.score, "preview": h.text[:SOURCE_PREVIEW_CHARS]}
            for h in result.chunks]})
        return messages

    def _await_cut(self) -> str | None:
        """Wait up to ack_timeout for this turn's playback_cut; the heard text, or None without an ack."""
        self._cut.wait(self.ack_timeout)
        with self._state_lock:
            return self._cut_heard

    def _wait_for_browser_playback(self, audio_seconds: float | None, stop_event: threading.Event) -> None:
        """Browser mode: Kokoro synthesizes faster than real time, so play() returns while the
        browser still has audio queued. Keep the turn open (mic muted, status "speaking",
        Stop enabled) until that audio has played; Stop ends the wait at once."""
        if not audio_seconds or stop_event.is_set():
            return
        if not self.timer.has("playback_started"):
            self._ack.wait(self.ack_timeout)
        start = self.timer.get("playback_started") or self.timer.get("tts_first_audio")
        if start is None:
            return
        remaining = start + audio_seconds + PLAYBACK_MARGIN_S - self.timer.now()
        if remaining > 0:
            stop_event.wait(remaining)

    def _finish_metrics(self, turn_id: int, source: str, interrupted: bool, params: GenerationParams,
                        reply: str, transcript: str, rag_fields: dict, interrupt_fields: dict,
                        diar_record: dict) -> None:
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
                      max_tokens=params.max_tokens, reply_chars=len(reply), transcript=transcript, reply=reply)
        if self.models is not None:
            m = self.models.current
            record.update(stt_final=m.get("stt_final"), stt_realtime=m.get("stt_realtime", "off"),
                          tts_model=m.get("tts"), runtime=self.models.runtime())
        record.update(rag_fields)
        record.update(interrupt_fields)
        record.update(diar_record)
        record["word_align"] = self.word_align
        try:
            self.metrics_sink(record)
        except Exception:
            log.exception("metrics_sink failed")

    def close(self) -> None:
        self._listening = False
        self._listen_gen += 1
        self._disarm()
        with self._state_lock:
            self._recording = False
            self._barge_pending = None
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
        with self._diar_lock:
            self._diar_wanted = False   # a load still running will not enable it
            d = self._diarizer
            if self.diarization:
                self.audio.diarizer = None
                self.diarization = False
        if d is not None:
            try:
                d.stop()
            except Exception:
                log.exception("stopping the diarizer failed")
        for t in [self._listen_thread, self._stopper, *self._threads]:
            if t is not None and t is not threading.current_thread():
                t.join(timeout=5)
