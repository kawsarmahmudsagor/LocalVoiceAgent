import pytest
from voicebot.catalog import Entry, by_slug, load_catalog
from voicebot.config import REPO_ROOT, Settings
from voicebot.models import ModelError, ModelManager

POOL = REPO_ROOT / "bench" / "pool.toml"

@pytest.fixture
def entries():
    e = by_slug(load_catalog(POOL))
    boom = Entry(slug="boom", kind="stt", role="final", engine="x", model="x", device="cpu", family="x")
    return [e["whisper-small-en"], e["nemotron-rt"], e["kokoro"], e["piper"], boom]

@pytest.fixture
def settings(): return Settings.load(env={})

# factories record what they built; "boom" slugs raise
def test_switch_rebuilds_and_restores_on_failure(entries, settings):
    built = []
    def rf(final, realtime, **kw):
        if final.slug == "boom": raise RuntimeError("load failed")
        built.append((final.slug, realtime and realtime.slug)); return object()
    m = ModelManager(entries, settings, recorder_factory=rf, tts_factory=lambda e: object(), unload=lambda o: None)
    rec, _ = m.start(False, {}); new = m.switch_stt("stt_realtime", "nemotron-rt", rec)
    assert built[-1] == ("whisper-small-en", "nemotron-rt") and m.current["stt_realtime"] == "nemotron-rt"
    with pytest.raises(ModelError, match="load failed"): m.switch_stt("stt_final", "boom", new)
    assert built[-1] == ("whisper-small-en", "nemotron-rt") and m.current["stt_final"] == "whisper-small-en"

def test_validate_roles(entries, settings):
    m = ModelManager(entries, settings, recorder_factory=None, tts_factory=None)
    assert m.validate("stt_realtime", "off") is None
    for role, slug in [("stt_final", "nemotron-rt"), ("tts", "whisper-small-en"), ("stt_final", "nope")]:
        with pytest.raises(ModelError, match="bad_model"): m.validate(role, slug)

def test_failed_switch_hands_back_restored_object(entries, settings):
    made = []
    def tf(e):
        if e.slug == "piper": raise RuntimeError("no piper")
        made.append(e.slug); return e.slug
    m = ModelManager(entries, settings, recorder_factory=lambda f, r, **k: object(), tts_factory=tf, unload=lambda o: None)
    _, stream = m.start(False, {})
    with pytest.raises(ModelError) as err: m.switch_tts("piper", stream)
    assert err.value.restored == "kokoro" and m.current["tts"] == "kokoro" and made == ["kokoro", "kokoro"]

def test_start_rejects_unknown_slug(entries):
    m = ModelManager(entries, Settings.load(env={"TTS_MODEL": "nope"}), recorder_factory=lambda f, r, **k: 1,
                     tts_factory=lambda e: 2)
    with pytest.raises(ModelError, match="nope"): m.start(False, {})
