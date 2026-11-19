import pytest
from types import SimpleNamespace as NS
from voicebot.config import Settings
from voicebot import stt, tts
from voicebot.audio_io import BrowserAudio, ServerAudio, make_audio_io

class Rec:
    def __init__(self, **kw): self.kw, self.fed, self.mic = kw, [], []
    def feed_audio(self, b, original_sample_rate): self.fed.append((b, original_sample_rate))
    def set_microphone(self, on): self.mic.append(on)

class Stream:
    def __init__(self): self.play_kw = None; self.engine = NS(get_stream_info=lambda: (8, 1, 24000),
        get_voices=lambda: [NS(name="af_heart"), NS(name="am_adam")], set_voice=lambda n: setattr(self, "voice", n))
    def play(self, **kw):
        self.play_kw = kw
        for c in (b"a", b"b"):
            if kw.get("on_audio_chunk"): kw["on_audio_chunk"](c)

def test_make_recorder_kwargs():
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec)
    assert r.kw == dict(model="small.en", language="en", device="cuda", compute_type="int8", use_microphone=False,
                        post_speech_silence_duration=0.6, spinner=False, enable_realtime_transcription=False)
    cb = {"on_recording_stop": lambda: None}
    assert stt.make_recorder(Settings.load(env={}), use_microphone=False, callbacks=cb, recorder_cls=Rec).kw["on_recording_stop"] is cb["on_recording_stop"]

def test_timing_callbacks_mark_stages():                                  # Review Focus 6
    from voicebot.timing import TurnTimer
    tm = TurnTimer(); cb = stt.timing_callbacks(tm)
    cb["on_turn_detection_start"](); cb["on_turn_detection_stop"](); assert not tm.has("speech_end")
    cb["on_turn_detection_start"](); cb["on_recording_stop"]()
    assert cb["on_transcription_start"](b"audio") is None                  # truthy would abort transcription
    assert all(tm.has(s) for s in ("speech_end", "endpoint", "stt_start"))

def test_tts_helpers():
    s = Stream()
    assert tts.output_format(s) == {"sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"}
    assert tts.list_voices(s) == ["af_heart", "am_adam"]; tts.set_voice(s, "am_adam"); assert s.voice == "am_adam"

def test_browser_audio():
    r, s, sent, first = Rec(), Stream(), [], []
    a = BrowserAudio(r, {"sample_rate": 24000}); a.set_sink(sent.append)
    a.on_client_audio(b"\x01\x02"); a.mute_input(True); a.on_client_audio(b"x")
    assert r.fed == [(b"\x01\x02", 16000)]
    a.play(s, lambda: first.append(1))
    assert s.play_kw["muted"] is True and sent == [b"a", b"b"] and first == [1]
    a.set_sink(None); a.play(s, lambda: None)            # no sink: chunks dropped, no error

def test_server_audio():
    r, s, first = Rec(), Stream(), []
    a = ServerAudio(r); a.on_client_audio(b"x"); a.mute_input(True); a.mute_input(False); a.play(s, lambda: first.append(1))
    assert r.fed == [] and r.mic == [False, True] and not s.play_kw.get("muted") and first == [1] and a.output_format is None

def test_make_audio_io_selects_mode():
    calls = []
    f = lambda settings, use_microphone, callbacks=None: calls.append(use_microphone) or Rec()
    assert isinstance(make_audio_io(Settings.load(env={"AUDIO_IO": "browser"}), Stream(), recorder_factory=f), BrowserAudio)
    assert isinstance(make_audio_io(Settings.load(env={"AUDIO_IO": "server"}, platform="win32"), Stream(), recorder_factory=f), ServerAudio)
    assert calls == [False, True]

def test_expose_cuda_dlls_prepends_torch_lib(tmp_path, monkeypatch):
    # faster-whisper (CTranslate2) needs cuBLAS/cuDNN DLLs that ship inside torch\lib.
    lib = tmp_path / "torch" / "lib"; lib.mkdir(parents=True)
    monkeypatch.setenv("PATH", "C:\other")
    added = []
    stt.expose_cuda_dlls(torch_dir=tmp_path / "torch", add_dll_directory=added.append)
    import os
    assert os.environ["PATH"].split(os.pathsep)[0] == str(lib) and added == [str(lib)]
    stt.expose_cuda_dlls(torch_dir=tmp_path / "torch", add_dll_directory=added.append)   # idempotent
    assert os.environ["PATH"].split(os.pathsep).count(str(lib)) == 1

def test_preflight_reports_missing_faster_whisper():
    import pytest
    def no_module(name): raise ImportError(name)
    with pytest.raises(stt.STTError, match="faster-whisper"):
        stt.preflight(Settings.load(env={}), import_module=no_module, load_dll=lambda n: None, windows=True)

def test_preflight_reports_missing_cublas_and_skips_on_cpu():
    import pytest
    def no_dll(name): raise OSError(name)
    with pytest.raises(stt.STTError, match="STT_DEVICE=cpu"):
        stt.preflight(Settings.load(env={}), import_module=lambda n: None, load_dll=no_dll, windows=True)
    stt.preflight(Settings.load(env={"STT_DEVICE": "cpu"}), import_module=lambda n: None, load_dll=no_dll, windows=True)

def test_browser_audio_cut_drops_later_chunks():
    r, sent = Rec(), []
    a = BrowserAudio(r, {"sample_rate": 24000}); a.set_sink(sent.append)
    class CutMidway(Stream):
        def play(self, **kw):
            kw["on_audio_chunk"](b"a"); a.cut(); kw["on_audio_chunk"](b"b")
    a.play(CutMidway(), lambda: None)
    assert sent == [b"a"]                                   # audio after Stop never reaches the browser
    a.play(Stream(), lambda: None); assert sent == [b"a", b"a", b"b"]   # next play forwards again

def test_serialize_method_prevents_overlap():
    # RealtimeTTS StreamPlayer.stop() is not thread-safe: Stop (helper thread) and play()'s
    # cleanup can close the same PyAudio stream together and corrupt the heap.
    import threading, time
    class Player:
        active = 0; overlap = False
        def stop(self, immediate=False):
            Player.active += 1; Player.overlap |= Player.active > 1
            time.sleep(0.05); Player.active -= 1
            return immediate
    tts.serialize_method(Player, "stop"); tts.serialize_method(Player, "stop")    # idempotent
    p = Player(); ts = [threading.Thread(target=p.stop) for _ in range(3)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not Player.overlap and p.stop(immediate=True) is True

def test_drain_audio_queue_empties_engine_and_wrapped_queues():
    import queue
    inner, outer = queue.Queue(), queue.Queue()
    for q in (inner, outer): q.put(b"old"); q.put(b"older")
    outer.wrapped = inner
    s = Stream(); s.engine.queue = outer
    assert tts.drain_audio_queue(s) == 4 and inner.empty() and outer.empty()
    tts.drain_audio_queue(Stream())                     # engine without a queue: no error

def test_realtimetts_shim_serializes_close_not_player_stop():
    # Locking StreamPlayer.stop made Stop wait behind play()'s draining stop (speakers
    # played out the reply). Only the PyAudio close needs serializing.
    from RealtimeTTS.stream_player import AudioStream, StreamPlayer
    tts.install_realtimetts_shims()
    assert getattr(AudioStream.close_stream, "_voicebot_serialized", False)
    assert not getattr(StreamPlayer.stop, "_voicebot_serialized", False)

def test_timing_callbacks_keep_speech_end_through_real_endpoint_order():
    # RealtimeSTT ends an utterance with: turn_detection_start -> recording_stop (stop())
    # -> turn_detection_stop. The final turn_detection_stop must not erase speech_end.
    from voicebot.timing import TurnTimer
    tm = TurnTimer(); cb = stt.timing_callbacks(tm)
    cb["on_turn_detection_start"](); cb["on_recording_stop"](); cb["on_turn_detection_stop"]()
    assert tm.has("speech_end") and tm.has("endpoint")

def test_browser_audio_play_returns_seconds_sent():
    r, sent = Rec(), []
    a = BrowserAudio(r, {"sample_rate": 24000, "channels": 1}); a.set_sink(sent.append)
    class OneSecond(Stream):
        def play(self, **kw):
            for _ in range(4): kw["on_audio_chunk"](b"\x00" * 12000)   # 4 x 6000 samples
    assert a.play(OneSecond(), lambda: None) == 1.0
    a.set_sink(None); assert a.play(OneSecond(), lambda: None) == 0.0  # nothing reached a browser

from voicebot.config import REPO_ROOT
from voicebot.catalog import by_slug, load_catalog
POOL = REPO_ROOT / "bench" / "pool.toml"

def test_make_recorder_from_entries():
    e = by_slug(load_catalog(POOL))
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec, final=e["sherpa-parakeet-v3"], realtime=e["nemotron-rt"])
    assert r.kw["transcription_engine"] == "sherpa_onnx_parakeet" and "compute_type" not in r.kw
    assert r.kw["enable_realtime_transcription"] is True and r.kw["realtime_transcription_engine"] == "sherpa_onnx_nemotron"

def test_make_recorder_v1_path_unchanged():
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec)
    assert r.kw["model"] == "small.en" and r.kw["enable_realtime_transcription"] is False

def test_make_recorder_entry_paths_and_compute_type():
    e = by_slug(load_catalog(POOL))
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec, final=e["whisper-small-en"])
    assert r.kw["compute_type"] == "float16" and r.kw["model"] == "small.en"
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec, final=e["sherpa-parakeet-v3"])
    assert r.kw["download_root"] == str(REPO_ROOT / "models/sherpa-onnx") and "download_root" not in r.kw["transcription_engine_options"]
    conf = REPO_ROOT / "assets/sherpa-cpu-session.conf"
    assert r.kw["transcription_engine_options"]["provider"] == f"cpu:{conf}"
    assert "allow_spinning=0" in conf.read_text()  # sherpa-onnx.md, "CPU performance and installation"
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec, final=e["transcribe-cpp-parakeet-q4"])
    assert r.kw["model"] == str(REPO_ROOT / "models/transcribe-cpp/parakeet-tdt-0.6b-v3-Q4_K_M.gguf")

def test_make_tts_from_entry():
    e = by_slug(load_catalog(POOL))
    built = {}
    mod = NS(KokoroEngine=lambda **k: built.update(k) or "engine")
    s = tts.make_tts(Settings.load(env={}), stream_cls=lambda eng, **kw: ("stream", eng), entry=e["kokoro"], module=mod)
    assert s == ("stream", "engine") and built == {"voice": "af_heart", "trim_silence": False, "extra_end_ms": 200,
                                                       "default_speed": 1.0}

def test_set_recorder_and_output_format():
    a, r2 = BrowserAudio(Rec(), {"sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"}), Rec()
    a.set_recorder(r2); a.on_client_audio(b"x"); assert r2.fed == [(b"x", 16000)]
    a.set_output_format({"sample_rate": 22050, "channels": 1, "encoding": "pcm_s16le"})
    a.set_sink(lambda b: None); assert a.play(Stream(), lambda: None) == pytest.approx(2 / (2 * 22050))
    s = ServerAudio(Rec()); s.set_recorder(r2); assert s.recorder is r2

def test_realtime_download_root_reaches_recorder():                       # final review
    from dataclasses import replace
    e = by_slug(load_catalog(POOL))
    rt = replace(e["nemotron-rt"], options={**e["nemotron-rt"].options, "download_root": "models/sherpa-onnx"})
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec, final=e["whisper-small-en"], realtime=rt)
    assert r.kw["download_root"] == str(REPO_ROOT / "models/sherpa-onnx")
    assert "download_root" not in r.kw["realtime_transcription_engine_options"]

def test_make_tts_kokoro_pacing_v1_and_entry():
    seen = []
    eng = lambda **k: seen.append(k) or "engine"
    tts.make_tts(Settings.load(env={}), engine_cls=eng, stream_cls=lambda e, **kw: e)
    assert seen[-1] == {"voice": "af_heart", "trim_silence": False, "extra_end_ms": 200, "default_speed": 1.0}
    e = by_slug(load_catalog(POOL))
    tts.make_tts(Settings.load(env={"KOKORO_TRIM_SILENCE": "true"}), stream_cls=lambda x, **kw: x, entry=e["kokoro"],
                 module=NS(KokoroEngine=eng))
    assert seen[-1] == {"voice": "af_heart", "trim_silence": True, "extra_end_ms": 200, "default_speed": 1.0}

def test_play_forces_first_fragment_after_words():
    s = Stream(); a = BrowserAudio(Rec(), {"sample_rate": 24000, "channels": 1}, play_kwargs={"force_first_fragment_after_words": 8})
    a.set_sink(lambda b: None); a.play(s, lambda: None); assert s.play_kw["force_first_fragment_after_words"] == 8
    sa = ServerAudio(Rec(), play_kwargs={"force_first_fragment_after_words": 8}); sa.play(s, lambda: None)
    assert s.play_kw["force_first_fragment_after_words"] == 8
    b = make_audio_io(Settings.load(env={}), Stream(), recorder_factory=lambda st, **k: Rec())
    assert b.play_kwargs == {"force_first_fragment_after_words": 30, "fast_sentence_fragment": False,
                             "sentence_silence_duration": 0.0}
    env = {"TTS_FAST_SENTENCE_FRAGMENT": "true", "TTS_SENTENCE_SILENCE_MS": "250"}
    b = make_audio_io(Settings.load(env=env), Stream(), recorder_factory=lambda st, **k: Rec())
    assert (b.play_kwargs["fast_sentence_fragment"], b.play_kwargs["sentence_silence_duration"]) == (True, 0.25)

def test_browser_mode_stream_opens_no_output_device():                    # e2e: VM has no sound card
    made = []
    sc = lambda eng, **kw: made.append(kw) or eng
    tts.make_tts(Settings.load(env={}), engine_cls=lambda **k: "e", stream_cls=sc)
    assert made[-1] == {"muted": True}                  # browser mode: RealtimeTTS skips open_stream()
    tts.make_tts(Settings.load(env={"AUDIO_IO": "server"}, platform="win32"), engine_cls=lambda **k: "e", stream_cls=sc)
    assert made[-1] == {}                               # server mode plays on this PC's speakers
    e = by_slug(load_catalog(POOL))
    tts.make_tts(Settings.load(env={}), stream_cls=sc, entry=e["kokoro"], module=NS(KokoroEngine=lambda **k: "e"))
    assert made[-1] == {"muted": True}

class QueuedStream(Stream):
    """Like muted RealtimeTTS: sentences are synthesized in order into engine.queue, and the player thread
    sends chunks later. Here every sentence is queued before the first chunk is sent."""
    def __init__(self, sentences):
        import queue
        super().__init__(); self.sentences = sentences; self.engine.queue = queue.Queue()
    def play(self, **kw):
        self.play_kw = kw
        for text, nbytes in self.sentences:
            kw["before_sentence_synthesized"](text)
            self.engine.queue.put(b"\0" * nbytes)
        while not self.engine.queue.empty():
            kw["on_audio_chunk"](self.engine.queue.get())

def test_browser_audio_reports_sentence_audio_offsets():
    # 24000 Hz mono s16le: 4800 bytes = 0.1 s. Offsets count queued audio, not sent audio.
    s, seen = QueuedStream([("One.", 4800), ("Two.", 9600), ("Three.", 0)]), []
    a = BrowserAudio(Rec(), {"sample_rate": 24000, "channels": 1}); a.set_sink(lambda b: None)
    a.play(s, lambda: None, on_sentence=lambda t, at: seen.append((t, at)))
    assert seen == [("One.", 0.0), ("Two.", 0.1), ("Three.", pytest.approx(0.3))]
    seen.clear(); a.play(s, lambda: None, on_sentence=lambda t, at: seen.append((t, at)))
    assert seen[0] == ("One.", 0.0)                         # the count restarts with every reply

def test_browser_audio_drops_sentences_after_cut():
    s, seen = QueuedStream([("One.", 4800), ("Two.", 4800)]), []
    a = BrowserAudio(Rec(), {"sample_rate": 24000, "channels": 1}); a.set_sink(lambda b: None)
    orig = s.engine.queue.put
    s.engine.queue.put = lambda c, *x, **k: (orig(c, *x, **k), a.cut())  # Stop pressed after the first sentence
    a.play(s, lambda: None, on_sentence=lambda t, at: seen.append(t))
    assert seen == ["One."]

def test_server_audio_reports_sentences_without_offset():
    s, seen = QueuedStream([("One.", 4800)]), []
    ServerAudio(Rec()).play(s, lambda: None, on_sentence=lambda t, at: seen.append((t, at)))
    assert seen == [("One.", None)]
