import json
from voicebot.latency_report import summarize, main

def rec(backend, perceived, total, interrupted=False, source="voice"):
    return {"llm_backend": backend, "audio_io": "browser", "stt_model": "small.en", "source": source,
            "interrupted": interrupted, "spans": {"perceived_ms": perceived, "total_ms": total, "stt_ms": None}}

def test_summarize_groups_and_skips():
    rows = summarize([rec("llamacpp", 1000, 3000), rec("llamacpp", 1200, 3100), rec("llamacpp", 1400, 3200),
                      rec("llamacpp", 9999, 1, interrupted=True), rec("transformers", 2000, 5000)], group_by=("llm_backend",))
    g = {r["group"]["llm_backend"]: r for r in rows}
    assert g["llamacpp"]["n"] == 3 and g["llamacpp"]["spans"]["perceived_ms"] == {"median": 1200, "p90": 1400}
    assert g["llamacpp"]["spans"]["stt_ms"] is None and g["transformers"]["n"] == 1

def test_main_prints_table(tmp_path, capsys):
    p = tmp_path / "t.jsonl"; p.write_text("".join(json.dumps(rec("llamacpp", 1000, 3000)) + "\n" for _ in range(2)))
    main([str(p), "--by", "llm_backend"]); out = capsys.readouterr().out
    assert "llamacpp" in out and "perceived_ms" in out and "1000" in out


def test_summarize_barge_groups_interrupted_turns():
    from voicebot.latency_report import summarize_barge
    r = lambda mode, d, c, ft: {"interrupted": True, "barge_in_mode": mode, "barge_false_triggers": ft,
                                "spans": {"barge_detect_ms": d, "barge_cut_ms": c}}
    rows = summarize_barge([r("onset", 200, 260, 1), r("onset", 220, 300, 0), r("words", 500, 560, 2),
                            {"interrupted": False, "barge_in_mode": "onset", "barge_false_triggers": 2,
                             "spans": {"barge_detect_ms": None}},
                            {"interrupted": False, "barge_in_mode": "silent", "barge_false_triggers": None,
                             "spans": {}},
                            {"interrupted": True, "barge_in_mode": "off", "spans": {"barge_detect_ms": None}},
                            {"interrupted": True, "spans": {"barge_detect_ms": None}}],   # Stop button / old log
                           group_by=("barge_in_mode",))
    g = {row["group"]["barge_in_mode"]: row for row in rows}
    assert g["onset"]["armed"] == 3 and g["onset"]["n"] == 2 and g["onset"]["false_triggers"] == 3
    assert g["onset"]["spans"]["barge_detect_ms"] == {"median": 210, "p90": 220}
    assert g["words"]["spans"]["barge_cut_ms"] == {"median": 560, "p90": 560}
    assert g["silent"]["armed"] == 1 and g["silent"]["n"] == 0 and g["silent"]["false_triggers"] == 0
    assert g["silent"]["spans"] == {"barge_detect_ms": None, "barge_cut_ms": None}
    assert len(rows) == 3


FALSE_FIRE = {"event": "barge_false_fire", "ts": "2026-11-20T10:00:00.000+00:00", "turn_id": 3,
              "barge_in_mode": "onset", "barge_in_min_ms": 200, "llm_backend": "llamacpp"}

def test_summarize_skips_event_records():
    rows = summarize([rec("llamacpp", 1000, 3000), FALSE_FIRE], group_by=("llm_backend",))
    assert len(rows) == 1 and rows[0]["n"] == 1

def test_summarize_barge_counts_false_fires():
    from voicebot.latency_report import summarize_barge
    armed = {"interrupted": True, "barge_in_mode": "onset", "barge_false_triggers": 0,
             "spans": {"barge_detect_ms": 200, "barge_cut_ms": 260}}
    rows = summarize_barge([armed, FALSE_FIRE, FALSE_FIRE, {**FALSE_FIRE, "barge_in_mode": "words"}],
                           group_by=("barge_in_mode",))
    g = {row["group"]["barge_in_mode"]: row for row in rows}
    assert g["onset"]["armed"] == 1 and g["onset"]["n"] == 1 and g["onset"]["false_fires"] == 2
    assert g["words"] == {"group": {"barge_in_mode": "words"}, "armed": 0, "n": 0, "false_triggers": 0,
                          "false_fires": 1, "spans": {"barge_detect_ms": None, "barge_cut_ms": None}}

def test_main_prints_false_fires(tmp_path, capsys):
    p = tmp_path / "t.jsonl"; p.write_text(json.dumps(FALSE_FIRE) + "\n")
    main([str(p), "--by", "barge_in_mode"]); out = capsys.readouterr().out
    assert "false_fires=1" in out
