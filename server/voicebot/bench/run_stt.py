"""Benchmark one STT pool entry with the repo's own tools (v2 spec §3.3, plan Task 8).

    python -m voicebot.bench.run_stt --slug S [--realtime] [--port 8766] [--pool ...] [--manifest ...]

Starts stt-server-production from venvs/bench-<family>, runs benchmark_asr_ab.py (final) or
benchmark_asr_streaming.py (realtime) against it while sampling GPU memory, and writes
bench/results/<slug>.json (the tool's report) and <slug>.meta.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from voicebot.bench.gpu import GpuSampler
from voicebot.catalog import Entry, by_slug, load_catalog
from voicebot.config import REPO_ROOT

DEFAULT_POOL = REPO_ROOT / "bench" / "pool.toml"
DEFAULT_MANIFEST = REPO_ROOT / "bench" / "clips" / "manifest.json"
RESULTS = REPO_ROOT / "bench" / "results"
BASELINE_FINAL = "whisper-small-en"
HEALTH_TIMEOUT_S = 600


def _engine_options(entry: Entry) -> dict:
    """Engine options without download_root, which the server takes as its own --download-root flag."""
    return {k: v for k, v in entry.options.items() if k != "download_root"}


def server_command(final: Entry, port: int, realtime: Entry | None, python: str) -> list[str]:
    cmd = [python, "-m", "RealtimeSTT_server.production_server", "--host", "127.0.0.1", "--port", str(port),
           "--engine", final.engine, "--model", final.model, "--device", final.device, "--language", "en",
           "--engine-options", json.dumps(_engine_options(final))]
    # compute_type and download_root are recorder-level settings: the reference server reads them from
    # --compute-type / --download-root, not from --engine-options (example_fastapi_server/server.py).
    if "compute_type" in final.options:
        cmd += ["--compute-type", str(final.options["compute_type"])]
    roots = [e.options["download_root"] for e in (final, realtime) if e is not None and "download_root" in e.options]
    if roots:
        cmd += ["--download-root", str(REPO_ROOT / roots[0])]
    if realtime is not None:
        cmd += ["--realtime-engine", realtime.engine, "--realtime-model", realtime.model,
                "--realtime-engine-options", json.dumps(_engine_options(realtime))]
    return cmd


def bench_command(slug: str, port: int, manifest: Path, out: Path, realtime: bool, python: str) -> list[str]:
    if realtime:
        return [python, "RealtimeSTT/tools/benchmarks/benchmark_asr_streaming.py", "--url", f"http://127.0.0.1:{port}",
                "--manifest", str(manifest), "--output", str(out), "--pace", "1.0"]
    return [python, "RealtimeSTT/tools/benchmarks/benchmark_asr_ab.py", "--manifest", str(manifest),
            "--target", f"{slug}=http://127.0.0.1:{port}", "--output", str(out), "--repetitions", "3"]


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicebot.bench.run_stt", description=__doc__.splitlines()[0])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args(argv)

    entries = by_slug(load_catalog(args.pool))
    entry = entries[args.slug]
    if args.realtime != (entry.role == "realtime"):
        raise SystemExit(f"{entry.slug} has role {entry.role}; use --realtime only for realtime slugs")
    final, realtime = (entries[BASELINE_FINAL], entry) if args.realtime else (entry, None)
    python = str(REPO_ROOT / "venvs" / f"bench-{entry.family}" / "bin" / "python")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out, meta = RESULTS / f"{entry.slug}.json", RESULTS / f"{entry.slug}.meta.json"
    log_path = RESULTS / f"{entry.slug}.server.log"

    with open(log_path, "wb") as log:
        server = subprocess.Popen(server_command(final, args.port, realtime, python), cwd=REPO_ROOT,
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + HEALTH_TIMEOUT_S
            while not _healthy(args.port):
                if server.poll() is not None:
                    raise SystemExit(f"stt-server-production exited ({server.returncode}); see {log_path}")
                if time.monotonic() > deadline:
                    raise SystemExit(f"stt-server-production not healthy after {HEALTH_TIMEOUT_S}s; see {log_path}")
                time.sleep(1)
            with GpuSampler() as gpu:
                rc = subprocess.run(bench_command(entry.slug, args.port, args.manifest, out, args.realtime, python),
                                    cwd=REPO_ROOT).returncode
        finally:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
    meta.write_text(json.dumps({"slug": entry.slug, "peak_vram_mb": gpu.peak_mb, "returncode": rc}, indent=2) + "\n")
    print(f"{entry.slug}: returncode={rc} peak_vram_mb={gpu.peak_mb} -> {out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
