import wave
import numpy as np
import pytest
from voicebot.config import Settings
from voicebot.bench.clips import clip_callbacks, merge_callbacks
from voicebot.bench.manifest import build_manifest

def test_merge_calls_both_and_returns_none():
    seen = []; a = {"on_x": lambda *v: seen.append("a") or "truthy"}; b = {"on_x": lambda *v: seen.append("b")}
    assert merge_callbacks(a, b)["on_x"](1) is None and seen == ["a", "b"]

def test_clip_callbacks_write_numbered_wavs(tmp_path):
    (tmp_path / "clip-004.wav").write_bytes(b"")
    fn = clip_callbacks(tmp_path)["on_transcription_start"]
    assert fn(np.array([0.0, 0.5, -2.0], dtype=np.float32)) is None
    with wave.open(str(tmp_path / "clip-005.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert list(np.frombuffer(w.readframes(3), dtype=np.int16)) == [0, 16384, -32767]

def test_manifest_pairs_and_rejects_mismatch(tmp_path):                  # Review Focus 5
    for i in (1, 2): (tmp_path / f"clip-00{i}.wav").write_bytes(b"")
    (tmp_path / "p.txt").write_text("Hi there\n\nWhat time is it\n")
    assert build_manifest(tmp_path, tmp_path / "p.txt")[1] == {"wav": "clip-002.wav", "text": "What time is it", "id": "clip-002"}
    (tmp_path / "p.txt").write_text("only one\n")
    with pytest.raises(ValueError, match="2 clips but 1 prompts"): build_manifest(tmp_path, tmp_path / "p.txt")

def test_manifest_main_writes_nothing_on_mismatch(tmp_path):              # Review Focus 5
    from voicebot.bench.manifest import main
    (tmp_path / "clip-001.wav").write_bytes(b""); (tmp_path / "p.txt").write_text("a\nb\n")
    assert main([str(tmp_path), str(tmp_path / "p.txt")]) != 0 and not (tmp_path / "manifest.json").exists()

def test_clip_record_dir_setting():
    assert Settings.load(env={}).clip_record_dir is None
    assert Settings.load(env={"CLIP_RECORD_DIR": "bench/clips"}).clip_record_dir.name == "clips"
