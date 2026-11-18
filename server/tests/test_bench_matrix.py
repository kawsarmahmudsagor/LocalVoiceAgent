import json
from voicebot.config import REPO_ROOT
from voicebot.bench.matrix import Row, _stt_metrics, load_rows, render, shortlist

POOL = REPO_ROOT / "bench" / "pool.toml"

def r(slug, role, **kw): return Row(slug=slug, kind="tts" if role == "tts" else "stt", role=None if role == "tts" else role,
                                     state=kw.pop("state", "ok"), **{f: kw.get(f) for f in
                                     ("latency_median_ms", "latency_p90_ms", "wer", "first_partial_ms", "verdict", "peak_vram_mb", "note")})

def test_shortlist_rule():
    rows = [r("whisper-small-en", "final", wer=0.08, latency_median_ms=300),
            r("a", "final", wer=0.095, latency_median_ms=100), r("b", "final", wer=0.11, latency_median_ms=50),
            r("c", "final", wer=0.07, latency_median_ms=200), r("d", "final", state="dropped", wer=0.01, latency_median_ms=1),
            r("x-rt", "realtime", wer=0.2, first_partial_ms=300), r("y-rt", "realtime", wer=0.3, first_partial_ms=200),
            r("z-rt", "realtime", wer=None, first_partial_ms=10),
            r("kokoro", "tts", verdict="ok", latency_median_ms=150), r("q", "tts", verdict="ok", latency_median_ms=40),
            r("p", "tts", verdict="poor", latency_median_ms=10)]
    assert shortlist(rows) == {"stt_final": ["whisper-small-en", "a", "c"], "stt_realtime": ["y-rt", "x-rt"], "tts": ["kokoro", "q"]}

def test_render_has_tables_and_shortlist():
    md = render([r("whisper-small-en", "final", wer=0.08, latency_median_ms=300)], {"stt_final": ["whisper-small-en"]})
    assert "| whisper-small-en |" in md and "## Shortlist" in md

def test_stt_metrics_from_both_report_shapes():
    ab = {"targets": [{"accuracy": {"wer": 0.05}, "latency": {"final_client_seconds": {"median": 0.1234, "p90": 0.2}}}]}
    assert _stt_metrics(ab) == {"wer": 0.05, "latency_median_ms": 123, "latency_p90_ms": 200, "first_partial_ms": None}
    st = {"metrics": {"accuracy": {"overall": {"wer": 0.2}},
                      "latency": {"first_partial_s": {"median": 0.31, "p90": 0.5}, "final_s": {"median": 0.9, "p90": 1.1}}}}
    assert _stt_metrics(st) == {"wer": 0.2, "latency_median_ms": 900, "latency_p90_ms": 1100, "first_partial_ms": 310}

def test_load_rows_states(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "status.json").write_text(json.dumps({"chatterbox": {"state": "dropped", "reason": "torch 2.6.0 != 2.7.1+cu128"},
                                                      "kokoro": {"state": "ok"}}))
    (tmp_path / "results" / "kokoro.json").write_text(json.dumps(
        {"slug": "kokoro", "peak_vram_mb": 9000, "rows": [{"first_audio_ms": m} for m in (100, 300, 200)]}))
    (tmp_path / "v.toml").write_text('[verdicts]\nkokoro = "ok"\n')
    rows = {x.slug: x for x in load_rows(POOL, tmp_path / "status.json", tmp_path / "results", tmp_path / "v.toml")}
    assert rows["chatterbox"].state == "dropped" and "torch" in rows["chatterbox"].note
    assert (rows["kokoro"].state, rows["kokoro"].latency_median_ms, rows["kokoro"].verdict, rows["kokoro"].peak_vram_mb) == ("ok", 200, "ok", 9000)
    assert rows["piper"].state == "missing"

def test_parse_shortlist_round_trip():
    from voicebot.bench.matrix import parse_shortlist
    picks = {"stt_final": ["whisper-small-en", "a"], "stt_realtime": [], "tts": ["kokoro"]}
    assert parse_shortlist(render([], picks)) == picks

def test_forced_picks_join_the_shortlist():
    rows = [r("whisper-small-en", "final", wer=0.08, latency_median_ms=300),
            r("qwen3-asr-1-7b", "final", wer=0.5, latency_median_ms=900),
            r("kokoro", "tts", verdict="ok", latency_median_ms=150),
            r("coqui-xtts-v2", "tts", state="dropped", note="import failed")]
    picks = shortlist(rows, include={"stt_final": ["qwen3-asr-1-7b"], "tts": ["coqui-xtts-v2"]})
    assert picks["stt_final"] == ["whisper-small-en", "qwen3-asr-1-7b"]   # forced despite WER
    assert picks["tts"] == ["kokoro"]                                    # dropped entries cannot run
    md = render(rows, picks, include={"stt_final": ["qwen3-asr-1-7b"], "tts": ["coqui-xtts-v2"]}, reason="user request")
    over = md.split("## Overrides")[1]
    assert "qwen3-asr-1-7b" in over and "coqui-xtts-v2" in over and "not included" in over

def test_overrides_file(tmp_path):
    from voicebot.bench.matrix import load_overrides
    p = tmp_path / "o.toml"; p.write_text('reason = "user"\n[include]\ntts = ["coqui-xtts-v2"]\n')
    assert load_overrides(p) == ({"tts": ["coqui-xtts-v2"]}, "user")
    assert load_overrides(tmp_path / "missing.toml") == ({}, None)
