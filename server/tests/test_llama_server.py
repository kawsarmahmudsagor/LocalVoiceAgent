import subprocess, pytest
from voicebot.config import Settings
from voicebot.llm.llama_server import LlamaServer, LlamaServerError

class FakeProc:
    def __init__(self, exit_code=None, hang=False): self.exit_code, self.hang = exit_code, hang; self.terminated = self.killed = False
    def poll(self): return self.exit_code
    def terminate(self): self.terminated = True
    def kill(self): self.killed = True
    def wait(self, timeout=None):
        if self.hang and not self.killed: raise subprocess.TimeoutExpired("llama-server", timeout)
        return 0

class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def sleep(self, s): self.t += s

@pytest.fixture
def exe(tmp_path):
    p = tmp_path / "llama-server.exe"; p.write_text(""); return p

def make(exe, tmp_path, statuses=(200,), proc=None, port_busy=False, **env):
    s = Settings.load(env={"LLAMA_SERVER_PATH": str(exe), **env}); clock = Clock(); seq = list(statuses)
    srv = LlamaServer(s, popen=lambda *a, **k: proc or FakeProc(), status=lambda url: seq.pop(0) if len(seq) > 1 else seq[0],
                      sleep=clock.sleep, clock=clock, port_check=lambda h, p: port_busy, log_path=tmp_path / "ls.log")
    return srv, clock

def test_command(exe, tmp_path):
    srv, _ = make(exe, tmp_path)
    assert srv.command() == [str(exe), "-hf", "unsloth/Qwen3.5-9B-GGUF:Q4_K_M", "--no-mmproj", "--jinja",
                             "-ngl", "99", "-c", "8192", "--host", "127.0.0.1", "--port", "8081"]

def test_missing_binary(tmp_path):
    srv, _ = make(tmp_path / "nope.exe", tmp_path)
    with pytest.raises(LlamaServerError, match="fetch_llama_server.ps1"): srv.start()

def test_port_busy(exe, tmp_path):
    srv, _ = make(exe, tmp_path, port_busy=True)
    with pytest.raises(LlamaServerError, match="LLAMA_PORT"): srv.start()

def test_polls_until_healthy(exe, tmp_path):
    srv, clock = make(exe, tmp_path, statuses=(None, 503, 503, 200)); srv.start()
    assert clock.t == 3.0

def test_timeout_kills_and_shows_log_tail(exe, tmp_path):
    proc = FakeProc(); srv, _ = make(exe, tmp_path, statuses=(503,), proc=proc, LLAMA_START_TIMEOUT="5")
    (tmp_path / "ls.log").write_text("\n".join(f"line {i}" for i in range(40)))
    with pytest.raises(LlamaServerError, match="timeout") as e: srv.start()
    assert "line 39" in str(e.value) and "line 5\n" not in str(e.value) and proc.terminated

def test_early_exit(exe, tmp_path):
    srv, _ = make(exe, tmp_path, statuses=(None,), proc=FakeProc(exit_code=1))
    with pytest.raises(LlamaServerError, match="exited"): srv.start()

def test_close_terminates_then_kills(exe, tmp_path):
    proc = FakeProc(hang=True); srv, _ = make(exe, tmp_path, proc=proc); srv.start(); srv.close()
    assert proc.terminated and proc.killed

def test_close_without_start_is_noop(exe, tmp_path):
    make(exe, tmp_path)[0].close()
