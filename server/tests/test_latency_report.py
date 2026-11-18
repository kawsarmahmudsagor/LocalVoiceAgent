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
