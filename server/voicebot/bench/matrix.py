"""Benchmark results matrix and the shortlist rule (v2 spec §3.3, §3.4; plan Task 9).

    python -m voicebot.bench.matrix      -> docs/v2-model-matrix.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tomllib
from dataclasses import astuple, dataclass, fields
from pathlib import Path

from voicebot.catalog import load_catalog
from voicebot.config import REPO_ROOT

BASELINE_FINAL = "whisper-small-en"
BASELINE_TTS = "kokoro"
WER_MARGIN = 0.02  # 2 percentage points (WER is a fraction)
ROLES = ("stt_final", "stt_realtime", "tts")
ROLE_TITLES = {"stt_final": "Final STT", "stt_realtime": "Realtime STT", "tts": "TTS"}

DEFAULT_POOL = REPO_ROOT / "bench" / "pool.toml"
DEFAULT_STATUS = REPO_ROOT / "bench" / "status.json"
DEFAULT_RESULTS = REPO_ROOT / "bench" / "results"
DEFAULT_VERDICTS = REPO_ROOT / "bench" / "verdicts.toml"
DEFAULT_OUT = REPO_ROOT / "docs" / "v2-model-matrix.md"
DEFAULT_OVERRIDES = REPO_ROOT / "bench" / "overrides.toml"


@dataclass
class Row:
    slug: str
    kind: str
    role: str | None
    state: str
    latency_median_ms: int | float | None = None
    latency_p90_ms: int | float | None = None
    wer: float | None = None
    first_partial_ms: int | float | None = None
    verdict: str | None = None
    peak_vram_mb: int | None = None
    note: str | None = None


def role_key(row: Row) -> str:
    return "tts" if row.kind == "tts" else f"stt_{row.role}"


def _ms(seconds) -> int | None:
    return None if seconds is None else round(seconds * 1000)


def _stt_metrics(report: dict) -> dict:
    """Map a benchmark_asr_ab.py or benchmark_asr_streaming.py report to Row fields.

    ab:        targets[0].accuracy.wer, targets[0].latency.final_client_seconds.{median,p90}
    streaming: metrics.accuracy.overall.wer, metrics.latency.final_s.{median,p90}, first_partial_s.median
    """
    if "targets" in report:
        t = report["targets"][0]
        lat = t["latency"]["final_client_seconds"]
        return {"wer": t["accuracy"].get("wer"), "latency_median_ms": _ms(lat.get("median")),
                "latency_p90_ms": _ms(lat.get("p90")), "first_partial_ms": None}
    m = report["metrics"]
    final, first = m["latency"]["final_s"], m["latency"]["first_partial_s"]
    return {"wer": m["accuracy"]["overall"].get("wer"), "latency_median_ms": _ms(final.get("median")),
            "latency_p90_ms": _ms(final.get("p90")), "first_partial_ms": _ms(first.get("median"))}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_rows(pool: Path, status_path: Path, results_dir: Path, verdicts_path: Path) -> list[Row]:
    status = _read_json(Path(status_path)) or {}
    verdicts = {}
    if Path(verdicts_path).exists():
        verdicts = tomllib.loads(Path(verdicts_path).read_text(encoding="utf-8")).get("verdicts", {})
    rows = []
    for e in load_catalog(pool):
        row = Row(slug=e.slug, kind=e.kind, role=e.role, state="missing")
        st = status.get(e.slug, {})
        report = _read_json(Path(results_dir) / f"{e.slug}.json")
        meta = _read_json(Path(results_dir) / f"{e.slug}.meta.json") or {}
        if st.get("state") == "dropped":
            row.state, row.note = "dropped", st.get("reason")
        elif report is not None and meta.get("returncode", 0) == 0:
            row.state = "ok"
            if e.kind == "stt":
                for k, v in _stt_metrics(report).items():
                    setattr(row, k, v)
                row.peak_vram_mb = meta.get("peak_vram_mb")
            else:
                firsts = [r["first_audio_ms"] for r in report.get("rows", []) if r.get("first_audio_ms") is not None]
                if firsts:
                    s = sorted(firsts)
                    row.latency_median_ms = round(statistics.median(s))
                    row.latency_p90_ms = s[max(0, -(-9 * len(s) // 10) - 1)]
                row.peak_vram_mb = report.get("peak_vram_mb")
                row.verdict = verdicts.get(e.slug)
        elif report is not None:
            row.state, row.note = "failed", f"benchmark returncode {meta.get('returncode')}"
        rows.append(row)
    return rows


def _sort_key(value):
    return (value is None, value if value is not None else 0)


def load_overrides(path: Path) -> tuple[dict[str, list[str]], str | None]:
    """bench/overrides.toml: `reason` and an [include] table of role -> slugs forced onto the shortlist."""
    if not Path(path).exists():
        return {}, None
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return {role: list(slugs) for role, slugs in data.get("include", {}).items()}, data.get("reason")


def shortlist(rows: list[Row], include: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    picks = _rule(rows)
    ok = {r.slug for r in rows if r.state == "ok"}
    for role, slugs in (include or {}).items():
        for slug in slugs:
            if slug in ok and slug not in picks.setdefault(role, []):
                picks[role].append(slug)  # a user override: added beyond the rule's limits
    return picks


def _rule(rows: list[Row]) -> dict[str, list[str]]:
    by = {r.slug: r for r in rows}
    ok = [r for r in rows if r.state == "ok"]

    base = by.get(BASELINE_FINAL)
    finals = [BASELINE_FINAL]
    if base is not None and base.wer is not None:
        cands = [r for r in ok if role_key(r) == "stt_final" and r.slug != BASELINE_FINAL
                 and r.wer is not None and r.wer <= base.wer + WER_MARGIN + 1e-12]
        finals += [r.slug for r in sorted(cands, key=lambda r: _sort_key(r.latency_median_ms))][:2]

    rts = [r for r in ok if role_key(r) == "stt_realtime" and r.wer is not None]
    realtime = [r.slug for r in sorted(rts, key=lambda r: _sort_key(r.first_partial_ms))][:2]

    tts = [BASELINE_TTS]
    cands = [r for r in ok if role_key(r) == "tts" and r.slug != BASELINE_TTS and r.verdict == "ok"]
    tts += [r.slug for r in sorted(cands, key=lambda r: _sort_key(r.latency_median_ms))][:2]
    return {"stt_final": finals, "stt_realtime": realtime, "tts": tts}


def _cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v).replace("|", "\\|")


def render(rows: list[Row], picks: dict[str, list[str]], include: dict[str, list[str]] | None = None,
           reason: str | None = None) -> str:
    cols = [f.name for f in fields(Row)]
    out = ["# v2 benchmark matrix", "",
           "Generated by `python -m voicebot.bench.matrix` (v2 spec §3.3). Latency in ms; WER is a fraction "
           "(0.05 = 5 %). Final STT latency is the client time to the final transcript; realtime latency is the "
           "final text time, with time to first partial in its own column; TTS latency is time to first audio.", ""]
    for role in ROLES:
        role_rows = [r for r in rows if role_key(r) == role]
        if not role_rows:
            continue
        out += [f"## {ROLE_TITLES[role]}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(_cell(v) for v in astuple(r)) + " |" for r in role_rows]
        out.append("")
    out += ["## Shortlist", ""]
    out += [f"- {role}: {', '.join(picks.get(role, [])) or '(none)'}" for role in ROLES]
    out += ["", "## Overrides", ""]
    forced = [(role, slug) for role, slugs in (include or {}).items() for slug in slugs]
    if not forced:
        out.append("(none)")
    else:
        if reason:
            out += [reason, ""]
        states = {r.slug: r for r in rows}
        for role, slug in forced:
            row = states.get(slug)
            if slug in picks.get(role, []):
                out.append(f"- {role}: {slug} (forced onto the shortlist)")
            else:
                why = f"{row.state}: {row.note}" if row is not None and row.note else (row.state if row else "not in pool")
                out.append(f"- {role}: {slug} not included ({why})")
    out.append("")
    return "\n".join(out)


def parse_shortlist(markdown: str) -> dict[str, list[str]]:
    """Read the '## Shortlist' section back (used by Task 11's resolve)."""
    picks: dict[str, list[str]] = {}
    in_section = False
    for line in markdown.splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Shortlist"
            continue
        if in_section and line.startswith("- ") and ":" in line:
            role, _, slugs = line[2:].partition(":")
            if role.strip() in ROLES:
                picks[role.strip()] = [s.strip() for s in slugs.split(",") if s.strip() and s.strip() != "(none)"]
    return picks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicebot.bench.matrix", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    rows = load_rows(DEFAULT_POOL, DEFAULT_STATUS, DEFAULT_RESULTS, DEFAULT_VERDICTS)
    include, reason = load_overrides(DEFAULT_OVERRIDES)
    picks = shortlist(rows, include)
    args.out.write_text(render(rows, picks, include, reason), encoding="utf-8")
    print(f"wrote {args.out}")
    for role in ROLES:
        print(f"  {role}: {', '.join(picks[role]) or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
