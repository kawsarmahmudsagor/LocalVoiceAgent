import json, pytest
from voicebot.timing import TurnTimer, append_jsonl

class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t

def at(tm, c, stage, t): c.t = t; tm.mark(stage)

def test_voice_turn_spans():
    c = Clock(); tm = TurnTimer(clock=c)
    for st, t in [("speech_end", 100.0), ("endpoint", 100.6), ("stt_start", 100.61), ("stt_done", 100.8),
                  ("llm_first_token", 101.1), ("first_text", 101.12), ("tts_first_audio", 101.3),
                  ("playback_started", 101.34), ("llm_done", 101.9), ("play_done", 104.0)]:
        at(tm, c, st, t)
    r = tm.report()
    assert r["marks"]["speech_end"] == 0 and r["marks"]["endpoint"] == 600 and r["marks"]["input"] is None
    assert r["spans"] == {"silence_wait_ms": 600, "stt_ms": 200, "diar_wait_ms": None, "retrieval_ms": None, "llm_ttft_ms": 300, "filter_ms": 20,
                          "tts_first_audio_ms": 180, "playback_ms": 40, "perceived_ms": 1340,
                          "llm_total_ms": 1100, "total_ms": 4000,
                          "barge_detect_ms": None, "barge_cut_ms": None}

def test_typed_turn_uses_input_origin_and_null_spans():
    c = Clock(); tm = TurnTimer(clock=c)
    for st, t in [("input", 50.0), ("llm_first_token", 50.25), ("first_text", 50.25), ("tts_first_audio", 50.5), ("play_done", 52.0)]:
        at(tm, c, st, t)
    s = tm.report()["spans"]
    assert s["llm_ttft_ms"] == 250 and s["silence_wait_ms"] is None and s["stt_ms"] is None
    assert s["playback_ms"] is None and s["perceived_ms"] == 500        # falls back to tts_first_audio

def test_first_mark_wins_unmark_and_reset():                            # Review Focus 6
    c = Clock(); tm = TurnTimer(clock=c)
    at(tm, c, "speech_end", 1.0); at(tm, c, "speech_end", 2.0); assert tm.get("speech_end") == 1.0
    tm.unmark("speech_end"); at(tm, c, "speech_end", 3.0); assert tm.get("speech_end") == 3.0
    tm.reset(); assert not tm.has("speech_end") and tm.report()["spans"]["total_ms"] is None

def test_unknown_stage_raises():
    with pytest.raises(ValueError): TurnTimer().mark("nope")

def test_append_jsonl(tmp_path):
    p = tmp_path / "logs" / "t.jsonl"; append_jsonl(p, {"a": 1}); append_jsonl(p, {"a": 2})
    assert [json.loads(l) for l in p.read_text().splitlines()] == [{"a": 1}, {"a": 2}]

def test_fmt_ms_shows_seconds_and_ms():
    from voicebot.timing import fmt_ms
    assert fmt_ms(1234) == "1.234 s (1234 ms)"
    assert fmt_ms(5) == "0.005 s (5 ms)"
    assert fmt_ms(None) == "-"

def test_format_turn_is_readable():
    from voicebot.timing import SPANS, format_turn
    rec = {"turn_id": 3, "source": "voice", "interrupted": True, "ts": "2026-11-18T10:22:31.123+00:00",
           "llm_backend": "llamacpp", "model_label": "qwen", "audio_io": "browser", "stt_model": "small.en",
           "voice": "af_heart", "temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "reply_chars": 42,
           "spans": {s: None for s in SPANS} | {"perceived_ms": 1234, "stt_ms": 87}}
    text = format_turn(rec)
    assert text.startswith("=== Turn 3 | voice | interrupted | 2026-11-18 10:22:31.123 UTC ===")
    assert "llm_backend=llamacpp" in text and "voice=af_heart" in text
    lines = text.splitlines()
    assert any(l.split()[:1] == ["perceived"] and "1.234 s (1234 ms)" in l for l in lines)
    assert any(l.split()[:1] == ["stt"] and "0.087 s (87 ms)" in l for l in lines)
    assert any(l.split()[:1] == ["silence_wait"] and l.rstrip().endswith("-") for l in lines)
    assert text.endswith("\n\n")

def test_append_turn_text(tmp_path):
    from voicebot.timing import SPANS, append_turn_text
    p = tmp_path / "logs" / "turns.log"
    rec = {"turn_id": 1, "source": "text", "interrupted": False, "ts": "2026-11-18T10:00:00.000+00:00",
           "spans": {s: 100 for s in SPANS}}
    append_turn_text(p, rec); append_turn_text(p, rec | {"turn_id": 2})
    t = p.read_text()
    assert t.count("=== Turn ") == 2 and "0.100 s (100 ms)" in t

def test_format_turn_shows_models_and_text():
    from voicebot.timing import format_turn
    rec = {"turn_id": 5, "source": "voice", "interrupted": False, "ts": "2026-11-19T10:00:00.000+00:00",
           "stt_final": "qwen3-asr-1-7b", "stt_realtime": "nemotron-rt", "tts_model": "piper",
           "transcript": "What is the capital of France?", "reply": "Paris.", "spans": {}}
    lines = format_turn(rec).splitlines()
    assert lines[1].split() == ["stt_final=qwen3-asr-1-7b", "stt_realtime=nemotron-rt", "tts_model=piper"]
    assert '  heard: "What is the capital of France?"' in lines
    assert '  reply: "Paris."' in lines

def test_retrieval_span_and_pure_llm_ttft():
    t = iter([0.0, 1.0, 1.2, 1.5]); tm = TurnTimer(clock=lambda: next(t))
    for s in ("input", "retrieval_start", "retrieval_done", "llm_first_token"): tm.mark(s)
    sp = tm.report()["spans"]; assert (sp["retrieval_ms"], sp["llm_ttft_ms"]) == (200, 300)

def test_llm_total_starts_at_retrieval_done():
    c = Clock(); tm = TurnTimer(clock=c)
    for st, t in [("input", 10.0), ("retrieval_start", 10.1), ("retrieval_done", 10.4),
                  ("llm_first_token", 10.7), ("llm_done", 12.4)]:
        at(tm, c, st, t)
    sp = tm.report()["spans"]
    assert (sp["retrieval_ms"], sp["llm_ttft_ms"], sp["llm_total_ms"]) == (300, 300, 2000)


def test_barge_spans():
    c = Clock(); tm = TurnTimer(clock=c)
    at(tm, c, "input", 0.0); at(tm, c, "barge_onset", 1.0)
    at(tm, c, "barge_in", 1.2); at(tm, c, "playback_cut", 1.35)
    s = tm.report()["spans"]
    assert s["barge_detect_ms"] == 200 and s["barge_cut_ms"] == 350


def test_barge_spans_none_without_onset():
    c = Clock(); tm = TurnTimer(clock=c)
    tm.mark("input"); tm.mark("playback_cut")
    s = tm.report()["spans"]
    assert s["barge_detect_ms"] is None and s["barge_cut_ms"] is None


def test_hold_keeps_input_marks_for_next_turn():
    c = Clock(); tm = TurnTimer(clock=c)
    at(tm, c, "input", 0.0)
    tm.hold()
    at(tm, c, "speech_end", 0.5)
    assert tm.has("speech_end") and tm.report()["marks"]["speech_end"] is None   # pending, not in this turn
    tm.unmark("speech_end"); assert not tm.has("speech_end")
    at(tm, c, "speech_end", 0.6); tm.mark("endpoint")
    tm.reset()
    assert tm.get("speech_end") == 0.6 and tm.get("endpoint") == 0.6 and not tm.has("input")
    tm.reset(); assert not tm.has("speech_end")          # the hold ended with the first reset


def test_format_turn_shows_barge_lines_only_when_present():
    from voicebot.timing import format_turn
    base = {"turn_id": 1, "source": "voice", "spans": {}}
    assert "barge_detect" not in format_turn(base)
    out = format_turn({**base, "interrupted": True, "heard": "One.", "barge_in_mode": "onset",
                       "spans": {"barge_detect_ms": 200, "barge_cut_ms": 260}})
    assert 'reply heard: "One."' in out
    assert "barge_detect" in out and "0.260 s (260 ms)" in out and "barge_in_mode=onset" in out


def test_event_record_is_one_line(tmp_path):
    from voicebot.timing import append_turn_text, format_turn
    ev = {"event": "barge_false_fire", "ts": "2026-11-20T10:00:00.000+00:00", "turn_id": 3,
          "barge_in_mode": "onset", "barge_in_min_ms": 200, "llm_backend": "llamacpp"}
    assert format_turn(ev) == "=== barge_false_fire | turn 3 | onset 200 ms | 2026-11-20 10:00:00.000 UTC ===\n\n"
    p = tmp_path / "turns.log"; append_turn_text(p, ev)
    assert p.read_text().count("\n") == 2 and "Turn" not in p.read_text()


def test_diar_wait_span_and_llm_starts_after_the_decision():
    from voicebot.timing import SPANS
    clock = iter([0.0, 0.1, 0.2, 0.5, 0.9, 1.4]).__next__
    t = TurnTimer(clock=clock)
    for s in ("speech_end", "endpoint", "stt_done", "diar_start", "diar_done", "llm_first_token"):
        t.mark(s)
    sp = t.report()["spans"]
    assert sp["diar_wait_ms"] == 400 and sp["llm_ttft_ms"] == 500   # LLM time starts at diar_done
    assert SPANS.index("diar_wait_ms") == SPANS.index("stt_ms") + 1


def test_format_turn_prints_diarization_fields_and_events():
    from voicebot.timing import format_turn
    rec = {"turn_id": 3, "source": "voice", "spans": {}, "diarization": True, "diar_mode": "ultra_low_latency",
           "diar_speaker": 0, "diar_lag_ms": 120, "diar_rejects": 1, "diar_rtf": 0.08}
    text = format_turn(rec)
    for part in ("diarization=True", "diar_mode=ultra_low_latency", "diar_speaker=0", "diar_lag_ms=120",
                 "diar_rejects=1", "diar_rtf=0.08", "diar_wait"):
        assert part in text
    on = format_turn({"event": "diarizer_on", "ts": "2026-11-20T10:00:00.000+00:00", "diar_mode": "low_latency", "load_ms": 4200})
    assert on.startswith("=== diarizer_on | low_latency | load 4200 ms |") and "turn" not in on
    off = format_turn({"event": "diarizer_off", "ts": "x", "reason": "overflow"})
    assert "diarizer_off | overflow" in off
    ign = format_turn({"event": "diar_ignored", "ts": "x", "speaker": 1, "text": "hello there", "diar_mode": "low_latency"})
    assert 'speaker 1 | "hello there"' in ign
    ff = format_turn({"event": "barge_false_fire", "ts": "x", "turn_id": 2, "barge_in_mode": "onset", "barge_in_min_ms": 200})
    assert "barge_false_fire | turn 2 | onset 200 ms" in ff          # v3 line unchanged
