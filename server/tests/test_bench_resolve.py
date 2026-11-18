from voicebot.bench.resolve import minimal_worker_set

def test_minimal_worker_set():
    conflict = {"chatterbox"}
    resolves = lambda s: not (conflict & set(s))
    assert minimal_worker_set(["whisper-small-en", "kokoro", "qwen-cuda"], resolves) == set()
    assert minimal_worker_set(["whisper-small-en", "kokoro", "chatterbox", "qwen-cuda"], resolves) == {"chatterbox"}
    pair = lambda s: not {"a", "b"} <= set(s)
    assert minimal_worker_set(["kokoro", "a", "b"], pair) == {"a"}

def test_catalog_toml_round_trips(tmp_path):
    from voicebot.catalog import by_slug, load_catalog
    from voicebot.config import REPO_ROOT
    from voicebot.bench.resolve import catalog_toml
    pool = by_slug(load_catalog(REPO_ROOT / "bench" / "pool.toml"))
    text = catalog_toml([pool["whisper-small-en"], pool["nemotron-rt"], pool["kokoro"], pool["chatterbox"]],
                        vram={"whisper-small-en": 900, "kokoro": 400}, workers={"chatterbox"})
    p = tmp_path / "c.toml"; p.write_text(text); e = by_slug(load_catalog(p))
    assert e["whisper-small-en"].vram_mb == 900 and e["nemotron-rt"].streaming and e["kokoro"].voice == "af_heart"
    assert (e["chatterbox"].runtime, e["chatterbox"].group) == ("worker", "bench-chatterbox")
    assert e["nemotron-rt"].options == {"provider": "cpu", "num_threads": 4, "download_root": "models/sherpa-onnx"}
