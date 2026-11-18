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
    assert r["spans"] == {"silence_wait_ms": 600, "stt_ms": 200, "llm_ttft_ms": 300, "filter_ms": 20,
                          "tts_first_audio_ms": 180, "playback_ms": 40, "perceived_ms": 1340,
                          "llm_total_ms": 1100, "total_ms": 4000}

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
