"""Streaming speaker diarization on a copy of the browser mic audio.

The model runs on its own thread. `feed` (called from the websocket side)
only enqueues and never blocks. The result is a speaker timeline with one
byte per 10 ms frame: bit k is set when speaker k has probability >=
DIAR_THRESHOLD. Frame i covers samples [i*160, (i+1)*160) of the audio fed
since the last reset, so session code can ask about any sample window.

Speaker 0 is the first voice heard since the last reset.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 160
DIAR_THRESHOLD = 0.5
MODES = ("ultra_low_latency", "very_low_latency", "low_latency")
DROP_LIMIT = 50
QUEUE_MAX = 500
_SPEAKERS = 8
_BIT_WEIGHTS = (1 << np.arange(_SPEAKERS)).astype(np.uint16)


def _ceil_frames(samples: int) -> int:
    return -(-samples // FRAME_SAMPLES)


class Diarizer:
    def __init__(
        self,
        model: Any,
        *,
        on_error: Callable[[str, str], None] | None = None,
        mode: str = "ultra_low_latency",
        queue_max: int = QUEUE_MAX,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown diarization mode: {mode!r}")
        self._model = model
        self._on_error = on_error
        self._mode = mode
        self._queue: queue.Queue = queue.Queue(maxsize=queue_max)
        self._lock = threading.Lock()
        # Everything below is guarded by _lock.
        self._gen = 0
        self._timeline = bytearray()
        self._position = 0      # samples fed (queued or dropped) since reset
        self._pushed = 0        # samples pushed to the stream since reset
        self._stream: Any = None
        self._drops = 0
        self._reported = False
        self._compute_s = 0.0
        self._audio_s = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    # ----- model ---------------------------------------------------------
    def load(self) -> None:
        self._model.load()

    @property
    def ready(self) -> bool:
        return bool(getattr(self._model, "ready", False))

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown diarization mode: {mode!r}")
        with self._lock:
            self._mode = mode
            self._reset_locked()

    # ----- thread --------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._reported = False
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._thread = threading.Thread(
                target=self._run, args=(stop_event,), name="diarizer", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread, stop_event = self._thread, self._stop_event
            self._thread = self._stop_event = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self.process(0.05)
            except Exception as e:  # noqa: BLE001 - any model failure ends the thread
                log.exception("diarizer failed")
                self._report("error", str(e))
                return

    def _report(self, reason: str, message: str) -> None:
        with self._lock:
            if self._reported:
                return
            self._reported = True
            callback = self._on_error
        if callback is not None:
            try:
                callback(reason, message)
            except Exception:  # noqa: BLE001 - never let the callback break feed/thread
                log.exception("diarizer on_error callback failed")

    # ----- audio in ------------------------------------------------------
    def feed(self, pcm: bytes) -> None:
        """Queue a copy of a 16 kHz mono int16 frame. Never blocks."""
        data = bytes(pcm)
        n = len(data) // 2
        if n == 0:
            return
        overflow = False
        with self._lock:
            start = self._position
            self._position += n
            try:
                self._queue.put_nowait((self._gen, start, data[: n * 2]))
                self._drops = 0
            except queue.Full:
                self._drops += 1
                overflow = self._drops >= DROP_LIMIT
        if overflow:
            self._report("overflow", f"diarizer dropped {DROP_LIMIT} frames in a row")

    def process(self, timeout: float = 0.0) -> bool:
        """Handle one queued frame. False when the queue stayed empty."""
        try:
            if timeout > 0:
                gen, start, data = self._queue.get(timeout=timeout)
            else:
                gen, start, data = self._queue.get_nowait()
        except queue.Empty:
            return False
        with self._lock:
            if gen != self._gen:
                return True
            stream, mode, pushed = self._stream, self._mode, self._pushed
        if stream is None:
            stream = self._model.open(mode)
        frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        gap = max(0, start - pushed)   # frames dropped on overflow: fill with silence
        chunks = [np.zeros(gap, dtype=np.float32), frame] if gap else [frame]
        t0 = time.perf_counter()
        bits = b"".join(self._bits(stream.push(chunk)) for chunk in chunks)
        elapsed = time.perf_counter() - t0
        n_audio = gap + len(frame)
        with self._lock:
            if gen != self._gen:   # reset while the model ran: drop the result
                return True
            self._stream = stream
            self._pushed = start + len(frame)
            self._timeline.extend(bits)
            self._compute_s += elapsed
            self._audio_s += n_audio / SAMPLE_RATE
        return True

    @staticmethod
    def _bits(probs: Any) -> bytes:
        """(n, 8) speaker probabilities -> n timeline bytes."""
        probs = np.asarray(probs, dtype=np.float32)
        if probs.size == 0:
            return b""
        active = probs.reshape(-1, probs.shape[-1])[:, :_SPEAKERS] >= DIAR_THRESHOLD
        return (active.astype(np.uint16) @ _BIT_WEIGHTS[: active.shape[1]]).astype(np.uint8).tobytes()

    # ----- queries -------------------------------------------------------
    def position(self) -> int:
        with self._lock:
            return self._position

    def covered(self) -> int:
        with self._lock:
            return len(self._timeline) * FRAME_SAMPLES

    def _frames(self, start: int, end: int, need_covered: bool) -> np.ndarray | None:
        if start >= end:
            return None
        with self._lock:
            covered = len(self._timeline) * FRAME_SAMPLES
            if need_covered and covered < end:
                return None
            f0 = max(start, 0) // FRAME_SAMPLES
            f1 = _ceil_frames(min(end, covered))
            if f1 <= f0:
                return None
            return np.frombuffer(bytes(self._timeline[f0:f1]), dtype=np.uint8)

    def speaker_between(self, start: int, end: int) -> int | None:
        """Speaker active in the most covered frames of [start, end); ties to the lower index."""
        frames = self._frames(start, end, need_covered=False)
        if frames is None:
            return None
        counts = np.unpackbits(frames[:, None], axis=1, bitorder="little").sum(axis=0)
        if counts.max() == 0:
            return None
        return int(np.argmax(counts))

    def share(self, speaker: int, start: int, end: int) -> float | None:
        """Fraction of the window's voiced frames where `speaker` is active."""
        if not 0 <= speaker < _SPEAKERS:
            return None
        frames = self._frames(start, end, need_covered=True)
        if frames is None:
            return None
        voiced = int(np.count_nonzero(frames))
        if voiced == 0:
            return None
        return int(np.count_nonzero(frames & (1 << speaker))) / voiced

    def rtf(self) -> float | None:
        """Model compute time over audio time pushed since the last call."""
        with self._lock:
            compute, audio = self._compute_s, self._audio_s
            self._compute_s = self._audio_s = 0.0
        if audio <= 0:
            return None
        return compute / audio

    # ----- reset ---------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        self._gen += 1
        self._timeline = bytearray()
        self._position = 0
        self._pushed = 0
        self._stream = None
        self._drops = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# ----- Nemotron adapter ---------------------------------------------------
MODEL_ID = "nvidia/Nemotron-3-Diarization"
# Measured on the VM (Task 1): the model returns logits (about -45 to +7.7).
OUTPUT_IS_LOGITS = True


class NemotronStream:
    """Feeds audio to the model chunk by chunk, carrying the speaker cache."""

    def __init__(self, processor: Any, model: Any) -> None:
        self._proc = processor
        self._model = model
        self._buf = np.zeros(0, dtype=np.float32)
        self._base = 0          # absolute sample index of _buf[0]
        self._first = True
        self._mel = 0
        self._cache: Any = None

    def push(self, audio: np.ndarray) -> np.ndarray:
        import torch

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            self._buf = np.concatenate([self._buf, audio])
        proc = self._proc
        out: list[np.ndarray] = []
        while True:
            if self._first:
                start, size = 0, proc.num_samples_first_audio_chunk
            else:
                start, size = proc.audio_chunk_start(self._mel), proc.num_samples_per_audio_chunk
            lo = start - self._base
            if lo < 0 or lo + size > len(self._buf):
                break
            x = proc(self._buf[lo:lo + size], sampling_rate=SAMPLE_RATE,
                     is_streaming=True, is_first_audio_chunk=self._first)
            with torch.inference_mode():
                o = self._model(**x.to(self._model.device, dtype=self._model.dtype),
                                speaker_cache=self._cache)
            self._cache = o.speaker_cache
            logits = o.logits[0].float().cpu().numpy()
            out.append(1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0))) if OUTPUT_IS_LOGITS else logits)
            if self._first:
                self._first = False
                self._mel = proc.num_mel_frames_per_step
            else:
                self._mel += proc.num_mel_frames_per_step
            # Chunks overlap (look-back): keep samples from the next chunk start.
            cut = proc.audio_chunk_start(self._mel) - self._base
            if cut > 0:
                cut = min(cut, len(self._buf))
                self._buf = self._buf[cut:]
                self._base += cut
        if not out:
            return np.zeros((0, _SPEAKERS), dtype=np.float32)
        return np.concatenate(out, axis=0)


class NemotronModel:
    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID, *,
                 processor: Any = None, model: Any = None) -> None:
        self._device = device
        self._model_id = model_id
        self._proc = processor
        self._net = model
        self.ready = False

    def load(self) -> None:
        if self._proc is None or self._net is None:
            import torch
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor

            device = self._device
            if device.startswith("cuda") and not torch.cuda.is_available():
                log.warning("CUDA unavailable; running the diarizer on CPU")
                device = "cpu"
            self._proc = AutoProcessor.from_pretrained(self._model_id)
            self._net = AutoModelForAudioFrameClassification.from_pretrained(
                self._model_id, device_map=device, dtype=torch.bfloat16)
        if hasattr(self._net, "eval"):
            self._net.eval()
        self.ready = True

    def open(self, mode: str) -> NemotronStream:
        self._proc.set_streaming_mode(mode)
        return NemotronStream(self._proc, self._net)


def _smoke_feed(d: Diarizer, pcm: bytes) -> None:
    """Feed 20 ms pieces like live audio: drain after each, so the queue never overflows."""
    step = FRAME_SAMPLES * 2 * 2    # 20 ms of int16
    for i in range(0, len(pcm), step):
        d.feed(pcm[i:i + step])
        while d.process(0):
            pass


def _smoke(path: str) -> None:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="int16", always_2d=True)
    assert sr == SAMPLE_RATE and audio.shape[1] == 1, f"need 16 kHz mono, got {sr} Hz x{audio.shape[1]}"
    pcm = audio[:, 0].tobytes()
    t0 = time.perf_counter()
    model = NemotronModel()
    d = Diarizer(model, mode="ultra_low_latency")
    d.load()
    load_s = time.perf_counter() - t0
    _smoke_feed(d, pcm)
    seconds = len(pcm) / 2 / SAMPLE_RATE
    print(f"load {load_s:.1f}s  audio {seconds:.1f}s  rtf {d.rtf()}  covered {d.covered() / SAMPLE_RATE:.2f}s")
    for s in range(int(seconds)):
        print(f"{s:3d}s speaker {d.speaker_between(s * SAMPLE_RATE, (s + 1) * SAMPLE_RATE)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Nemotron diarizer smoke test")
    ap.add_argument("--smoke", metavar="WAV", required=True)
    _smoke(ap.parse_args().smoke)
