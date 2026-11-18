"""Pair recorded clips with their prompts into the repo benchmark tools' manifest (plan Task 7).

    python -m voicebot.bench.manifest bench/clips bench/prompts.txt

Writes <clip_dir>/manifest.json only when the clip count matches the prompt count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_manifest(clip_dir: Path, prompts_path: Path) -> list[dict]:
    clips = sorted(Path(clip_dir).glob("clip-*.wav"))
    prompts = [l.strip() for l in Path(prompts_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(clips) != len(prompts):
        raise ValueError(f"{len(clips)} clips but {len(prompts)} prompts")
    return [{"wav": c.name, "text": t, "id": c.stem} for c, t in zip(clips, prompts)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicebot.bench.manifest", description=__doc__.splitlines()[0])
    ap.add_argument("clip_dir", type=Path)
    ap.add_argument("prompts", type=Path)
    args = ap.parse_args(argv)
    try:
        records = build_manifest(args.clip_dir, args.prompts)
    except ValueError as e:
        print(f"manifest not written: {e}", file=sys.stderr)
        return 1
    out = args.clip_dir / "manifest.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(records)} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
