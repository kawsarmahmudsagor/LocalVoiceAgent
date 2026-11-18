import pytest
from voicebot.config import REPO_ROOT
from voicebot.catalog import by_slug, load_catalog
from voicebot.bench.fetch import fetch_command

POOL = REPO_ROOT / "bench" / "pool.toml"

def test_fetch_commands(tmp_path):
    e = by_slug(load_catalog(POOL))
    assert fetch_command(e["whisper-small-en"], tmp_path, "py") is None
    assert fetch_command(e["sherpa-parakeet-v3"], tmp_path, "py")[-2:] == ["--model", "parakeet"]
    assert fetch_command(e["piper"], tmp_path, "py")[:3] == ["py", "-m", "piper.download_voices"]
    with pytest.raises(SystemExit, match="transcribe-cpp.md"): fetch_command(e["transcribe-cpp-parakeet-q4"], tmp_path, "py")
