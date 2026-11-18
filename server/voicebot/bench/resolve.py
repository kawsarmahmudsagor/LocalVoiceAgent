"""Pick in-process or worker runtime for the shortlist and write catalog.toml (v2 spec §4.2, plan Task 11).

    .venv/bin/python -m voicebot.bench.resolve [--idle-mb N]

1. Reads the approved shortlist from docs/v2-model-matrix.md ("## Shortlist").
2. Checks with `uv pip compile` whether every shortlisted family's requirements resolve together
   with the server's; the smallest set of entries whose removal makes it resolve become workers.
3. Writes server/voicebot/catalog.toml: the shortlisted pool entries with vram_mb = measured peak
   minus the idle baseline (only the LLM loaded), and runtime/group set.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from voicebot.bench.matrix import BASELINE_FINAL, BASELINE_TTS, DEFAULT_OUT as MATRIX, DEFAULT_RESULTS, parse_shortlist
from voicebot.catalog import Entry, by_slug, load_catalog
from voicebot.config import REPO_ROOT

POOL = REPO_ROOT / "bench" / "pool.toml"
FAMILIES = REPO_ROOT / "bench" / "families"
CATALOG = REPO_ROOT / "server" / "voicebot" / "catalog.toml"
BASELINES = (BASELINE_FINAL, BASELINE_TTS)


def minimal_worker_set(slugs: list[str], resolves: Callable[[list[str]], bool]) -> set[str]:
    removable = [s for s in slugs if s not in BASELINES]
    for k in range(len(removable) + 1):
        for removed in itertools.combinations(removable, k):
            rest = [s for s in slugs if s not in removed]
            if resolves(rest):
                return set(removed)
    raise RuntimeError("the baselines alone do not resolve with the server's requirements")


def family_requirements(family: str) -> list[str]:
    lines = (FAMILIES / f"{family}.txt").read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def uv_resolves(requirements: list[str]) -> bool:
    """True if the v1 server requirements plus `requirements` resolve for Python 3.12."""
    lines = ["-e ./server[dev]", *requirements]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=REPO_ROOT) as f:
        f.write("\n".join(dict.fromkeys(lines)) + "\n")
        path = f.name
    try:
        return subprocess.run(["uv", "pip", "compile", "--python-version", "3.12", path, "-o", "/dev/null"],
                              cwd=REPO_ROOT, capture_output=True).returncode == 0
    finally:
        Path(path).unlink(missing_ok=True)


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    return json.dumps(str(v))  # a JSON string is a valid TOML basic string


def catalog_toml(entries: list[Entry], *, vram: dict[str, int], workers: set[str]) -> str:
    out = ["# Shortlisted engines (v2 spec §4.1). Written by `python -m voicebot.bench.resolve`.", ""]
    for e in entries:
        fields = {"slug": e.slug, "kind": e.kind, "role": e.role, "engine": e.engine, "model": e.model,
                  "device": e.device, "family": e.family, "fetch": e.fetch, "streaming": e.streaming,
                  "vram_mb": vram.get(e.slug, e.vram_mb), "voice": e.voice, "options": e.options}
        if e.slug in workers:
            fields.update(runtime="worker", group=f"bench-{e.family}")
        out.append("[[entry]]")
        out += [f"{k} = {_toml_value(v)}" for k, v in fields.items() if v is not None and v != {}]
        out.append("")
    return "\n".join(out)


def _idle_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def _peak_mb(slug: str) -> int | None:
    for name in (f"{slug}.meta.json", f"{slug}.json"):
        try:
            peak = json.loads((DEFAULT_RESULTS / name).read_text(encoding="utf-8")).get("peak_vram_mb")
        except (OSError, json.JSONDecodeError):
            continue
        if peak is not None:
            return peak
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicebot.bench.resolve", description=__doc__.splitlines()[0])
    ap.add_argument("--idle-mb", type=int, default=None,
                    help="GPU memory with only the LLM loaded (default: read nvidia-smi now)")
    args = ap.parse_args(argv)

    picks = parse_shortlist(MATRIX.read_text(encoding="utf-8"))
    slugs = list(dict.fromkeys(s for role in ("stt_final", "stt_realtime", "tts") for s in picks.get(role, [])))
    pool = by_slug(load_catalog(POOL))
    entries = [pool[s] for s in slugs]
    reqs = {e.slug: family_requirements(e.family) for e in entries}
    workers = minimal_worker_set(slugs, lambda keep: uv_resolves([r for s in keep for r in reqs[s]]))
    idle = args.idle_mb if args.idle_mb is not None else _idle_mb()
    vram = {}
    for e in entries:
        peak = _peak_mb(e.slug)
        vram[e.slug] = max(0, peak - idle) if (peak is not None and e.device == "cuda") else 0
    CATALOG.write_text(catalog_toml(entries, vram=vram, workers=workers), encoding="utf-8")
    load_catalog(CATALOG)  # validate what was written
    print(f"idle baseline: {idle} MB")
    print(f"worker set: {', '.join(sorted(workers)) or '(empty: everything in-process)'}")
    print(f"wrote {CATALOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
