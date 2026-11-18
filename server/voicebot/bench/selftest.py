"""Import self-test for one bench family's venv (v2 spec §3.3, plan Task 6).

    python -m voicebot.bench.selftest --family F [--pool bench/pool.toml] [--status bench/status.json]

It checks torch (version and sm_120 kernels) and that every engine class of the family imports.
No model weights are loaded. Results are merged into bench/status.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Callable

from voicebot.catalog import Entry, load_catalog
from voicebot.config import REPO_ROOT

TORCH_VERSION = "2.7.1+cu128"
DEFAULT_POOL = REPO_ROOT / "bench" / "pool.toml"
DEFAULT_STATUS = REPO_ROOT / "bench" / "status.json"


def check_torch(torch_module) -> str | None:
    if torch_module.__version__ != TORCH_VERSION:
        return f"torch {torch_module.__version__} != {TORCH_VERSION}"
    if torch_module.cuda.is_available() and "sm_120" not in torch_module.cuda.get_arch_list():
        return "no sm_120 in torch arch list"
    return None


def check_imports(entries: list[Entry], importer: Callable = importlib.import_module) -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    for e in entries:
        try:
            if e.kind == "stt":
                factory = importer("RealtimeSTT.transcription_engines.factory")
                factory._load_engine_class(e.engine)
            else:
                getattr(importer("RealtimeTTS"), e.engine)
            results[e.slug] = None
        except Exception as exc:  # any import-time failure drops the entry
            results[e.slug] = str(exc) or type(exc).__name__
    return results


def update_status(path: Path, results: dict[str, str | None]) -> None:
    path = Path(path)
    status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for slug, problem in results.items():
        status[slug] = {"state": "ok"} if problem is None else {"state": "dropped", "reason": problem}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicebot.bench.selftest", description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    args = ap.parse_args(argv)

    entries = [e for e in load_catalog(args.pool) if e.family == args.family]
    if not entries:
        print(f"no pool entries in family {args.family!r}", file=sys.stderr)
        return 2
    try:
        import torch  # noqa: PLC0415  (only families that ship torch have it)
        torch_problem = check_torch(torch)
    except ImportError:
        torch_problem = None
    results = check_imports(entries)
    if torch_problem:
        results = {slug: torch_problem for slug in results}
    update_status(args.status, results)
    for slug, problem in results.items():
        print(f"{slug}: {'ok' if problem is None else 'DROPPED: ' + problem}")
    return 0 if all(p is None for p in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
