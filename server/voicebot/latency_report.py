"""Summarize logs/turns.jsonl: median and p90 of each latency span, per group.

    python -m voicebot.latency_report [path] [--by k1,k2] [--last N] [--source voice|text|all]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from voicebot.config import REPO_ROOT
from voicebot.timing import SPANS

DEFAULT_PATH = REPO_ROOT / "logs" / "turns.jsonl"
DEFAULT_BY = ("llm_backend", "audio_io", "stt_model")


def _p90(values: list[int]) -> int:
    v = sorted(values)
    return v[math.ceil(0.9 * len(v)) - 1]


def summarize(records: list[dict], group_by: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r.get("interrupted"):
            continue
        key = tuple(r.get(k) for k in group_by)
        groups.setdefault(key, []).append(r)
    rows = []
    for key, recs in groups.items():
        spans: dict[str, dict | None] = {}
        for span in SPANS:
            vals = [r.get("spans", {}).get(span) for r in recs]
            vals = [v for v in vals if v is not None]
            spans[span] = {"median": round(statistics.median(vals)), "p90": _p90(vals)} if vals else None
        rows.append({"group": dict(zip(group_by, key)), "n": len(recs), "spans": spans})
    return rows


def _load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicebot.latency_report", description=__doc__.splitlines()[0])
    ap.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--by", default=",".join(DEFAULT_BY), help="comma-separated record fields to group by")
    ap.add_argument("--last", type=int, default=None, help="only the last N records")
    ap.add_argument("--source", choices=("voice", "text", "all"), default="all")
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"no turn log at {args.path}", file=sys.stderr)
        return 1
    records = _load(args.path)
    if args.source != "all":
        records = [r for r in records if r.get("source") == args.source]
    if args.last:
        records = records[-args.last:]
    group_by = tuple(k.strip() for k in args.by.split(",") if k.strip())

    for row in summarize(records, group_by):
        title = "  ".join(f"{k}={v}" for k, v in row["group"].items())
        print(f"{title}  (n={row['n']})")
        print(f"  {'span':<20}{'median':>8}{'p90':>8}")
        for span in SPANS:
            s = row["spans"][span]
            med, p90 = (s["median"], s["p90"]) if s else ("-", "-")
            print(f"  {span:<20}{med:>8}{p90:>8}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
