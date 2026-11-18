"""Download the weights of one pool entry (v2 spec §3.3, plan Task 6).

    python -m voicebot.bench.fetch --slug S [--pool bench/pool.toml]

The command runs with the entry family's venv python (venvs/bench-<family>/bin/python).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from voicebot.catalog import Entry, by_slug, load_catalog
from voicebot.config import REPO_ROOT

DEFAULT_POOL = REPO_ROOT / "bench" / "pool.toml"
MODELS_ROOT = REPO_ROOT / "models"
KROKO_REPO = "Banafo/Kroko-ASR"


def fetch_command(entry: Entry, models_root: Path, python: str) -> list[str] | None:
    kind, _, arg = entry.fetch.partition(":")
    if kind == "auto":
        return None
    if kind == "sherpa":
        return [python, "-m", "RealtimeSTT.install_sherpa_models", "--root", str(models_root / "sherpa-onnx"),
                "--model", arg]
    if kind == "hf":
        repo, _, file = arg.partition(":")
        cmd = [python, "-m", "huggingface_hub.commands.huggingface_cli", "download", repo]
        if file:
            cmd.append(file)
        return cmd + ["--local-dir", str(models_root / "hf" / repo.split("/")[-1])]
    if kind == "kroko":
        # kroko-onnx.md: public Community .data files come from Banafo/Kroko-ASR on Hugging Face.
        code = ("from huggingface_hub import hf_hub_download; "
                f"print(hf_hub_download(repo_id={KROKO_REPO!r}, filename={arg!r}, "
                f"local_dir={str(models_root / 'kroko-onnx')!r}))")
        return [python, "-c", code]
    if kind == "piper":
        return [python, "-m", "piper.download_voices", arg, "--data-dir", str(models_root / "piper")]
    if kind == "manual":
        raise SystemExit(f"{entry.slug}: manual download: {arg}")
    raise SystemExit(f"{entry.slug}: unknown fetch {entry.fetch!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicebot.bench.fetch", description=__doc__.splitlines()[0])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    args = ap.parse_args(argv)
    entry = by_slug(load_catalog(args.pool)).get(args.slug)
    if entry is None:
        raise SystemExit(f"unknown slug {args.slug!r}")
    python = REPO_ROOT / "venvs" / f"bench-{entry.family}" / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"{python} not found: run scripts/bench_install.sh {entry.family} first")
    cmd = fetch_command(entry, MODELS_ROOT, str(python))
    if cmd is None:
        print(f"{entry.slug}: fetch=auto, the library downloads the weights on first load")
        return 0
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
