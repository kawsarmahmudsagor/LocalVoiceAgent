import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from voicebot.config import REPO_ROOT
from voicebot.catalog import by_slug, load_catalog
from voicebot.bench.gpu import GpuSampler, parse_mem_line
from voicebot.bench.run_stt import bench_command, server_command
from voicebot.bench.run_tts import measure
from voicebot.tts_engines import engine_kwargs

POOL = REPO_ROOT / "bench" / "pool.toml"

def test_parse_mem_line(): assert parse_mem_line("7123\n") == 7123 and parse_mem_line("[N/A]") is None

def test_gpu_sampler_peak():
    class P:
        def __init__(self, *a, **k): self.stdout = iter(["100\n", "[N/A]\n", "7123\n", "500\n"]); self.terminated = False
        def terminate(self): self.terminated = True
        def wait(self, timeout=None): return 0
    with GpuSampler(popen=P) as g: pass
    assert g.peak_mb == 7123

def test_server_and_bench_commands():
    e = by_slug(load_catalog(POOL))
    cmd = server_command(e["whisper-small-en"], 8766, e["nemotron-rt"], "py")
    assert cmd[cmd.index("--engine") + 1] == "faster_whisper" and cmd[cmd.index("--realtime-engine") + 1] == "sherpa_onnx_nemotron"
    assert json.loads(cmd[cmd.index("--engine-options") + 1]) == {"compute_type": "float16"}
    b = bench_command("whisper-small-en", 8766, Path("m.json"), Path("o.json"), False, "py")
    assert "whisper-small-en=http://127.0.0.1:8766" in b and b[-2:] == ["--repetitions", "3"]
    assert "--pace" in bench_command("nemotron-rt", 8766, Path("m.json"), Path("o.json"), True, "py")

def test_measure_first_audio_and_rtf():
    clock = iter([0.0, 0.1, 0.5]).__next__              # t0, first chunk, end
    class S:
        engine = NS(get_stream_info=lambda: (8, 1, 24000))
        def feed(self, s): pass
        def play(self, muted, on_audio_chunk): on_audio_chunk(b"\0" * 24000); on_audio_chunk(b"\0" * 24000)
    row = measure(S(), ["Hi."], clock=clock)[0]
    assert row["first_audio_ms"] == 100 and row["audio_ms"] == 1000 and row["rtf"] == pytest.approx(0.5)

def test_engine_kwargs_build_voice_objects(tmp_path):
    e = by_slug(load_catalog(POOL))
    mod = NS(QwenVoice=lambda **k: ("qv", k), PiperVoice=lambda **k: ("pv", k), ChatterboxVoice=lambda **k: ("cv", k))
    q = engine_kwargs(e["qwen-cuda"], mod, root=REPO_ROOT)
    assert q["voice"][0] == "qv" and q["voice"][1]["ref_text"].startswith("The morning train")
    assert set(q) == {"voice"}
    p = engine_kwargs(e["piper"], mod, root=tmp_path, piper_path="/x/piper")
    assert p["voice"][1]["model_file"] == str(tmp_path / "models/piper/en_US-lessac-medium.onnx") and p["piper_path"] == "/x/piper"
    c = engine_kwargs(e["chatterbox"], mod, root=tmp_path)
    assert c == {"device": "cuda", "voice": ("cv", {"audio_prompt_path": str(tmp_path / "assets/ref_voice.wav")})}
    assert engine_kwargs(e["kokoro"], mod, root=tmp_path) == {"voice": "af_heart"}
