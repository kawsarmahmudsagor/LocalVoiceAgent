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

def test_make_recorder_passes_recording_stop_callback():     # per-stage timing: silence end
    cb = lambda: None
    r = stt.make_recorder(Settings.load(env={}), use_microphone=True, recorder_cls=Rec, on_recording_stop=cb)
    assert r.kw["on_recording_stop"] is cb

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
    f = lambda settings, use_microphone: calls.append(use_microphone) or Rec()
    assert isinstance(make_audio_io(Settings.load(env={"AUDIO_IO": "browser"}), Stream(), recorder_factory=f), BrowserAudio)
    assert isinstance(make_audio_io(Settings.load(env={"AUDIO_IO": "server"}), Stream(), recorder_factory=f), ServerAudio)
    assert calls == [False, True]
