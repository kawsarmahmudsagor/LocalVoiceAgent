import pytest
from voicebot.config import REPO_ROOT
from voicebot.catalog import CatalogError, by_slug, for_role, load_catalog

POOL = REPO_ROOT / "bench" / "pool.toml"
def write(tmp_path, body): p = tmp_path / "c.toml"; p.write_text(body); return p
BASE = 'slug="{s}"\nkind="stt"\nrole="{r}"\nengine="faster_whisper"\nmodel="tiny.en"\ndevice="cuda"\nfamily="fw"\n'

def test_pool_is_valid():
    e = load_catalog(POOL); assert len(e) == 14
    assert {x.slug for x in for_role(e, "stt", "realtime")} == {"nemotron-rt", "kroko-en-128-rt", "whisper-tiny-en-rt"}
    assert by_slug(e)["nemotron-rt"].streaming and by_slug(e)["kokoro"].voice == "af_heart"

@pytest.mark.parametrize("slug,role,msg", [("whisper-x", "realtime", "-rt"), ("whisper-x-rt", "final", "-rt"),
                                           ("Bad_Slug", "final", "slug")])
def test_slug_rules(tmp_path, slug, role, msg):
    with pytest.raises(CatalogError, match=msg): load_catalog(write(tmp_path, "[[entry]]\n" + BASE.format(s=slug, r=role)))

def test_duplicate_and_worker_group(tmp_path):
    one = "[[entry]]\n" + BASE.format(s="a-rt", r="realtime")
    with pytest.raises(CatalogError, match="duplicate"): load_catalog(write(tmp_path, one + one))
    with pytest.raises(CatalogError, match="group"): load_catalog(write(tmp_path, one + 'runtime="worker"\n'))

def test_bad_fetch(tmp_path):
    with pytest.raises(CatalogError, match="fetch"):
        load_catalog(write(tmp_path, "[[entry]]\n" + BASE.format(s="a", r="final") + 'fetch="ftp:x"\n'))

def test_sherpa_entries_point_at_fetched_models():                         # final review
    e = by_slug(load_catalog(POOL))
    for slug in ("sherpa-parakeet-v3", "nemotron-rt"):
        assert e[slug].options.get("download_root") == "models/sherpa-onnx", slug

def test_user_requested_entries_in_pool():
    e = by_slug(load_catalog(POOL))
    q, c = e["qwen3-asr-1-7b"], e["coqui-xtts-v2"]
    assert (q.kind, q.role, q.engine, q.model) == ("stt", "final", "qwen3_asr", "Qwen/Qwen3-ASR-1.7B")
    assert (c.kind, c.engine, c.family) == ("tts", "CoquiEngine", "coqui")
