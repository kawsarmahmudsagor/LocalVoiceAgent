"""VM prerequisite check (v2 spec §2.2). Stdlib only: it runs before any venv exists.

    python3 server/voicebot/vmcheck.py      (or scripts/check_vm.sh)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_DRIVER = 570
MIN_CUDA = (12, 8)
DOCKER_TEST = ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.8.0-base-ubuntu24.04", "nvidia-smi"]


def check_gpu(query_rc: int, query_out: str, header_out: str) -> list[str]:
    """Problems found in `nvidia-smi --query-gpu=name,driver_version` output and plain `nvidia-smi` output."""
    if "Driver/library version mismatch" in query_out + header_out:
        return ["NVIDIA driver/library version mismatch: reboot the VM (sudo reboot), then rerun."]
    if query_rc != 0:
        first = next((l.strip() for l in query_out.splitlines() if l.strip()), "(no output)")
        return [f"nvidia-smi failed: {first}"]
    problems = []
    line = next((l for l in query_out.splitlines() if l.strip()), "")
    name, _, driver = (x.strip() for x in line.partition(","))
    try:
        major = int(driver.split(".")[0])
        if major < MIN_DRIVER:
            problems.append(f"driver {driver} is older than {MIN_DRIVER}")
    except ValueError:
        problems.append(f"could not read the driver version from: {line!r}")
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", header_out)
    if m and (int(m[1]), int(m[2])) < MIN_CUDA:
        problems.append(f"CUDA {m[1]}.{m[2]} is older than {MIN_CUDA[0]}.{MIN_CUDA[1]}")
    elif not m:
        problems.append("could not read 'CUDA Version' from nvidia-smi")
    if "5090" not in name:
        problems.append(f"expected an RTX 5090, found {name or '(none)'}")
    return problems


def check_disk(free_bytes: int, need_gb: int = 45) -> list[str]:
    free_gb = free_bytes / 1024**3
    return [] if free_gb >= need_gb else [f"only {free_gb:.0f} GB free on {REPO_ROOT}; need at least {need_gb} GB"]


def _run(cmd: list[str], timeout: float = 120) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]}: timed out after {timeout:g}s"


def main() -> int:
    problems: list[str] = []
    rc, out, err = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    _, header, herr = _run(["nvidia-smi"])
    problems += check_gpu(rc, out + err, header + herr)
    if shutil.which("docker") is None:
        problems.append("docker not found on PATH")
    else:
        rc, _, err = _run(DOCKER_TEST, timeout=600)
        if rc != 0:
            last = next((l.strip() for l in reversed(err.splitlines()) if l.strip()), f"exit {rc}")
            problems.append(f"docker GPU access failed: {last}")
    if shutil.which("uv") is None:
        problems.append("uv not found on PATH (https://docs.astral.sh/uv/)")
    problems += check_disk(shutil.disk_usage(REPO_ROOT).free)
    for p in problems:
        print(p)
    if not problems:
        print("OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
