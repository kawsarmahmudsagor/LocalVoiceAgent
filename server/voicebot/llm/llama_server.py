"""Starts, health-checks and stops llama-server: a local binary (v1 spec §4.2) or the CUDA Docker image (v2 spec §2.4)."""
from __future__ import annotations

import atexit
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from voicebot.config import REPO_ROOT, Settings
from voicebot.llm.base import LLMError

LOG_TAIL_LINES = 30
CONTAINER = "voicebot-llama"
SM120_HINT = "no kernel image is available"


class LlamaServerError(LLMError):
    pass


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def http_status(url: str, timeout: float = 2.0) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def wait_for_health(base_url: str, timeout: float, *, status=http_status, sleep=time.sleep,
                    clock=time.monotonic, is_alive: Callable[[], bool] = lambda: True) -> None:
    """Poll {base_url}/health every second until it returns 200.

    llama-server answers 503 while the model loads, so anything but 200 means keep waiting.
    """
    deadline = clock() + timeout
    url = f"{base_url}/health"
    while True:
        if not is_alive():
            raise LlamaServerError("llama-server exited before becoming healthy")
        if status(url) == 200:
            return
        if clock() >= deadline:
            raise LlamaServerError(f"timeout: {url} not healthy after {timeout:g}s (LLAMA_START_TIMEOUT)")
        sleep(1.0)


class LlamaServer:
    def __init__(self, settings: Settings, *, popen=subprocess.Popen, status=http_status, sleep=time.sleep,
                 clock=time.monotonic, port_check=port_in_use,
                 log_path: Path = REPO_ROOT / "logs/llama-server.log", run=subprocess.run,
                 which=shutil.which, home: Path | None = None) -> None:
        self.settings = settings
        self._popen, self._status, self._sleep, self._clock = popen, status, sleep, clock
        self._run, self._which = run, which
        self._home = Path.home() if home is None else home
        self._port_check = port_check
        self.log_path = log_path
        self._proc = None
        self._log = None

    @property
    def docker(self) -> bool:
        return self.settings.llama_runtime == "docker"

    def command(self) -> list[str]:
        s = self.settings
        if self.docker:
            return ["docker", "run", "--rm", "--name", CONTAINER, "--gpus", "all",
                    "-p", f"127.0.0.1:{s.llama_port}:8080",
                    "-v", f"{self._home}/.cache/llama.cpp:/root/.cache/llama.cpp", s.llama_image,
                    "-hf", s.llama_hf_repo, "--no-mmproj", "--jinja", "-ngl", "99", "-np", "1",
                    "-c", str(s.llama_ctx), "--host", "0.0.0.0", "--port", "8080"]
        return [str(s.llama_server_path), "-hf", s.llama_hf_repo, "--no-mmproj", "--jinja",
                "-ngl", "99", "-np", "1", "-c", str(s.llama_ctx), "--host", s.llama_host, "--port", str(s.llama_port)]

    def start(self) -> None:
        s = self.settings
        if self.docker:
            if self._which("docker") is None:
                raise LlamaServerError("docker not found on PATH: install Docker with the NVIDIA Container Toolkit, "
                                       "or set LLAMA_RUNTIME=binary")
            self._run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        elif not Path(s.llama_server_path).is_file():
            raise LlamaServerError(f"llama-server not found at {s.llama_server_path}: run scripts/fetch_llama_server.ps1")
        if self._port_check(s.llama_host, s.llama_port):
            raise LlamaServerError(
                f"port {s.llama_host}:{s.llama_port} is already in use: change LLAMA_PORT, "
                "or set LLAMA_AUTOSTART=false to use the server already running there")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        proc = self._popen(self.command(), stdout=self._log, stderr=subprocess.STDOUT)
        self._proc = proc
        atexit.register(self.close)
        try:
            wait_for_health(s.llama_base_url, s.llama_start_timeout, status=self._status, sleep=self._sleep,
                            clock=self._clock, is_alive=lambda: proc.poll() is None)
        except LlamaServerError as e:
            self.close()
            tail = self._log_tail()
            hint = ""
            if self.docker and SM120_HINT in tail:
                hint = ("\nThe image has no sm_120 kernels: run scripts/build_llama_image.sh "
                        "and set LLAMA_IMAGE=voicebot/llama-server:cuda12.8")
            raise LlamaServerError(f"{e}{hint}\n--- last {LOG_TAIL_LINES} log lines ---\n{tail}") from None

    def _log_tail(self) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "(no log)"
        return "\n".join(lines[-LOG_TAIL_LINES:])

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and self.docker:
            self._run(["docker", "stop", "-t", "5", CONTAINER], capture_output=True)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        elif proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        if self._log is not None:
            self._log.close()
            self._log = None
