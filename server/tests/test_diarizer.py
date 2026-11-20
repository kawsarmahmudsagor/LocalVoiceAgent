import numpy as np, pytest, threading, time
from voicebot.diarizer import Diarizer, DIAR_THRESHOLD, FRAME_SAMPLES

def pcm(n_samples, value=1000):
    return (np.full(n_samples, value, dtype=np.int16)).tobytes()

class FakeStream:
    """One frame per 160 samples; script(frame_index) -> {speaker: prob}. Holds back `delay` samples like look-ahead."""
    def __init__(self, script, delay=0): self.script, self.delay, self.buf, self.done, self.pushed = script, delay, 0, 0, []
    def push(self, audio):
        self.pushed.append(audio.copy()); self.buf += len(audio)
        n = max(0, (self.buf - self.delay) // FRAME_SAMPLES - self.done)
        out = np.zeros((n, 8), dtype=np.float32)
        for j in range(n):
            for spk, p in self.script(self.done + j).items(): out[j, spk] = p
        self.done += n; return out

class FakeModel:
    def __init__(self, script=lambda i: {0: 0.9}, delay=0, fail=None):
        self.script, self.delay, self.fail, self.ready, self.opened = script, delay, fail, False, []
    def load(self): self.ready = True
    def open(self, mode):
        self.opened.append(mode); s = FakeStream(self.script, self.delay)
        if self.fail:
            def boom(a): raise RuntimeError(self.fail)
            s.push = boom
        self.stream = s; return s

def drain(d):
    while d.process(0): pass

def test_timeline_maps_frames_to_samples():
    d = Diarizer(FakeModel(lambda i: {0: 0.9} if i < 50 else {1: 0.8}))
    d.feed(pcm(16000)); drain(d)
    assert d.position() == 16000 and d.covered() == 16000
    assert d.speaker_between(0, 8000) == 0 and d.speaker_between(8000, 16000) == 1
    assert d.share(0, 0, 16000) == pytest.approx(0.5) and d.share(1, 8000, 16000) == 1.0

def test_threshold_and_silence():
    d = Diarizer(FakeModel(lambda i: {0: DIAR_THRESHOLD - 0.01} if i < 10 else {}))
    d.feed(pcm(3200)); drain(d)
    assert d.speaker_between(0, 3200) is None and d.share(0, 0, 3200) is None

def test_share_is_none_until_covered():
    d = Diarizer(FakeModel(delay=1600))
    d.feed(pcm(3200)); drain(d)
    assert d.covered() == 1600 and d.share(0, 0, 3200) is None and d.share(0, 0, 1600) == 1.0

def test_queries_on_bad_windows():                                   # Review Focus 2
    d = Diarizer(FakeModel()); d.feed(pcm(1600)); drain(d)
    assert d.speaker_between(1600, 0) is None and d.speaker_between(-500, -1) is None
    assert d.speaker_between(50_000, 60_000) is None

def test_reset_clears_timeline_clock_queue_and_stream():
    m = FakeModel(); d = Diarizer(m); d.feed(pcm(1600)); drain(d); first = m.stream
    d.feed(pcm(1600)); d.reset()
    assert d.position() == 0 and d.covered() == 0 and not d.process(0)   # the queued frame is gone
    d.feed(pcm(320)); drain(d)
    assert m.stream is not first and d.covered() == 320

def test_set_mode_resets_and_opens_in_the_new_mode():
    m = FakeModel(); d = Diarizer(m); d.feed(pcm(320)); drain(d)
    d.set_mode("low_latency"); assert d.mode == "low_latency" and d.position() == 0
    d.feed(pcm(320)); drain(d); assert m.opened[-1] == "low_latency"
    with pytest.raises(ValueError): d.set_mode("fast")

def test_dropped_frames_keep_clock_aligned():                        # Review Focus 4
    m = FakeModel(); d = Diarizer(m, queue_max=1)
    d.feed(pcm(160)); d.feed(pcm(160)); d.feed(pcm(160))             # 2nd and 3rd dropped
    drain(d); d.feed(pcm(160)); drain(d)
    assert d.position() == 640 and d.covered() == 640                # gap filled with zeros
    assert sum(len(a) for a in m.stream.pushed) == 640 and not m.stream.pushed[1].any()

def test_overflow_reports_once():
    got = []; d = Diarizer(FakeModel(), on_error=lambda r, m: got.append(r), queue_max=1)
    for _ in range(1 + 50 + 10): d.feed(pcm(160))
    assert got == ["overflow"]

def test_model_error_reports_and_stops_the_thread():
    got = threading.Event(); reasons = []
    d = Diarizer(FakeModel(fail="cuda oom"), on_error=lambda r, m: (reasons.append((r, m)), got.set()))
    d.start(); d.feed(pcm(320)); assert got.wait(2)
    assert reasons == [("error", "cuda oom")]
    d.stop()

def test_thread_processes_fed_audio_and_stop_is_safe_from_on_error():
    d = Diarizer(FakeModel()); d.start(); d.feed(pcm(1600))
    end = time.monotonic() + 2
    while d.covered() < 1600 and time.monotonic() < end: time.sleep(0.01)
    assert d.covered() == 1600; d.stop(); d.stop()

def test_rtf_since_last_call():
    d = Diarizer(FakeModel()); assert d.rtf() is None
    d.feed(pcm(16000)); drain(d); r = d.rtf()
    assert r is not None and r >= 0 and d.rtf() is None

def test_module_imports_no_torch():
    import subprocess, sys
    code = "import voicebot.diarizer, sys; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

def test_stop_called_from_on_error_does_not_join_itself():
    returned = threading.Event(); holder = {}
    def on_error(r, m):
        holder["d"].stop(); returned.set()                       # only reached if stop() returned
    d = Diarizer(FakeModel(fail="boom"), on_error=on_error); holder["d"] = d
    d.start(); d.feed(pcm(320))
    assert returned.wait(2)
    end = time.monotonic() + 2
    while any(t.name == "diarizer" and t.is_alive() for t in threading.enumerate()) and time.monotonic() < end:
        time.sleep(0.01)
    assert not any(t.name == "diarizer" and t.is_alive() for t in threading.enumerate())
