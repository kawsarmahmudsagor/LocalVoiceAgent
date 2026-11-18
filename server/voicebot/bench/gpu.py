"""Peak GPU memory during a benchmark run, sampled with nvidia-smi every 200 ms (v2 spec §3.3)."""
from __future__ import annotations

import subprocess
import threading

DEFAULT_CMD = ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "200")


def parse_mem_line(line: str) -> int | None:
    try:
        return int(line.strip().split(",")[0])
    except ValueError:
        return None


class GpuSampler:
    def __init__(self, cmd=DEFAULT_CMD, popen=subprocess.Popen) -> None:
        self.cmd, self._popen = list(cmd), popen
        self.peak_mb = 0
        self._proc = None
        self._thread: threading.Thread | None = None

    def _read(self) -> None:
        for line in self._proc.stdout:
            mb = parse_mem_line(line)
            if mb is not None and mb > self.peak_mb:
                self.peak_mb = mb

    def __enter__(self) -> "GpuSampler":
        self._proc = self._popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True, name="gpu-sampler")
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
