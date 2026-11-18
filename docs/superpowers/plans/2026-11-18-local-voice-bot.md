# Local Voice Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local voice bot web app: RealtimeSTT → Qwen3.5-9B → Kokoro TTS, with a playground-style React UI. `.env` selects the LLM backend (transformers 4-bit or llama.cpp GGUF) and the audio mode (browser or server).

**Architecture:** A Python 3.12 FastAPI server owns one `VoiceSession` (STT recorder, LLM backend, Kokoro stream, conversation). A single WebSocket carries JSON commands and events plus binary PCM audio. A Vite + React + TypeScript UI talks to it. The LLM backends share one `LLMBackend` protocol and are imported lazily, so the llamacpp path never loads torch.

**Tech Stack:** Python 3.12 (uv venv), FastAPI, uvicorn, RealtimeSTT (faster-whisper), RealtimeTTS (KokoroEngine), transformers + bitsandbytes, llama.cpp `llama-server.exe`, openai client, pytest; npm, Vite, React 18, TypeScript, vitest.

**Spec:** `docs/superpowers/specs/2026-11-18-local-voice-bot-design.md`. Read it before starting any task. Section numbers below (§) refer to it.

## Global Constraints

- **Python:** 3.12 only, in `.venv` at the repo root, created with `uv venv -p 3.12 .venv`. Never use the system Python 3.14.
- **Torch:** install from `RealtimeSTT/requirements-gpu-torch.txt` (`torch==2.7.1+cu128`, `torchaudio==2.7.1+cu128`).
- **Models:**
  - transformers: `Qwen/Qwen3.5-9B`
  - llama.cpp: `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`
  - STT: `small.en`
  - TTS voice: `af_heart`
- **llama-server flags:** `-hf <repo> --no-mmproj --jinja -ngl 99 -c <ctx> --host <host> --port <port>`.
- **Thinking off** on both backends:
  - transformers: `enable_thinking=False`
  - llama.cpp: `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- **Lazy imports:** the llamacpp code path must never import `torch`, `transformers` or `bitsandbytes`.
- **Defaults** for every `.env` variable are exactly the spec §4.1 table. The WebSocket protocol is exactly spec §4.9.
- **Ports:** FastAPI `8000`, llama-server `8081`, Vite dev `5173`.
- **Tests:** every Python and frontend unit test runs without a GPU, model downloads or network.
- **Vendored code:** do not edit `RealtimeSTT/` or `RealtimeTTS/`. If a patch is unavoidable, record it in `VENDORED.md`.
- **UI:** dark playground-style layout, original design, no OpenAI branding or assets. Plain CSS.
- **Commits:**
  - Gitleaks hooks are active. Never use `--no-verify`.
  - Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Python test command**, run from `server/`: `../.venv/Scripts/python -m pytest <path> -v`
- **Frontend commands**, run from `web/`: `npm run test`, `npm run typecheck`, `npm run build`

## Review Focus

These are inputs the spec implies but doesn't spell out. Each one has a test in the task named after it.

1. **Bad values in `update_settings`** (e.g. `temperature: "hot"`, `max_tokens: 0`, an unknown voice) → reply with `error{code:"bad_settings"}` and keep the old settings. Task 9.
2. **Stop pressed while nothing is playing, or during `thinking` before any audio** → no crash. A running turn ends with `interrupted: true` and the mic is unmuted. Task 9.
3. **Browser disconnects mid-turn** → the server stops listening, detaches the session, and later events are dropped without raising. Tasks 9 and 10.
4. **llama-server dies after startup** → the next turn raises `LLMError` mentioning llama-server, the user hears the apology, and the session keeps working. Task 5 (backend) and Task 9 (apology).
5. **Empty or whitespace `send_text`** → ignored, no turn starts. Task 9.
6. **Stage timing edge cases:**
   - The user pauses and then keeps talking: `speech_end` is cleared and marked again, so the silence wait is measured from the real end of speech. Tasks 7b and 8.
   - A `playback_started` ack that arrives late or names an old turn is ignored.
   - A missing stage gives `null` spans, not an error. Tasks 7b and 9.

---

## File Structure

```
.env.example                         Task 1
scripts/setup_venv.ps1               Task 1 (npm line used from Task 11)
scripts/fetch_llama_server.ps1       Task 7
scripts/dev.ps1                      Task 13
README.md                            Task 13
server/pyproject.toml                Task 1
server/voicebot/__init__.py          Task 1 (empty)
server/voicebot/config.py            Task 1
server/voicebot/speech_filter.py     Task 2
server/voicebot/conversation.py      Task 3
server/voicebot/llm/__init__.py      Task 4 (empty; must not import submodules)
server/voicebot/llm/base.py          Task 4
server/voicebot/llm/llama_server.py  Task 4
server/voicebot/llm/llamacpp_backend.py      Task 5
server/voicebot/llm/transformers_backend.py  Task 6
server/voicebot/llm/factory.py       Task 6
server/voicebot/smoke_llm.py         Task 7
server/voicebot/timing.py  latency_report.py  Task 7b
server/voicebot/stt.py  tts.py  audio_io.py   Task 8
server/voicebot/session.py           Task 9
server/voicebot/web.py               Task 10
server/tests/test_<module>.py        one per module, in its task
web/ (package.json, vite.config.ts, tsconfig.json, index.html, src/main.tsx,
      src/protocol.ts, src/state.ts, src/audio/pcm.ts)       Task 11
web/src/ws.ts, src/audio/{mic,player,mic-worklet,player-worklet}.ts,
      src/components/*.tsx, src/App.tsx, src/styles.css      Task 12
```

---

### Task 1: Scaffold, setup script, `config.py`

**Files:**
- Create: `scripts/setup_venv.ps1`, `server/pyproject.toml`, `server/voicebot/__init__.py`, `server/voicebot/config.py`, `.env.example`
- Test: `server/tests/test_config.py`

**Interfaces:**
- Produces:
  - `voicebot.config.REPO_ROOT: Path`, which is `Path(__file__).resolve().parents[2]`.
  - `class ConfigError(ValueError)`. Its message always contains the variable name.
  - `@dataclass(frozen=True) class Settings` with these fields, one per spec §4.1 variable, snake_cased:
    - `llm_backend: str`, `audio_io: str`
    - `hf_model_id: str`, `hf_load_in_4bit: bool`
    - `llama_autostart: bool`, `llama_server_path: Path`, `llama_hf_repo: str`
    - `llama_host: str`, `llama_port: int`, `llama_base_url: str`, `llama_ctx: int`, `llama_start_timeout: float`
    - `stt_model: str`, `stt_device: str`, `stt_compute_type: str`, `stt_silence_sec: float`
    - `tts_voice: str`, `system_prompt: str`
    - `llm_temperature: float`, `llm_top_p: float`, `llm_max_tokens: int`, `history_turns: int`
    - `web_host: str`, `web_port: int`
  - `Settings.load(env: Mapping[str, str] | None = None, env_file: Path | None = None) -> Settings`
    - When `env` is given, it is the only source.
    - Otherwise it reads `dotenv_values(env_file or REPO_ROOT / ".env")`, and `os.environ` overrides those values.
    - Relative `LLAMA_SERVER_PATH` values resolve against `REPO_ROOT`.
    - Bools accept `true/false/1/0/yes/no`, case-insensitive.
    - `LLAMA_BASE_URL` defaults to `f"http://{host}:{port}"`.

- [ ] **Step 1: Write `server/pyproject.toml`**
  - Project `voicebot`, `requires-python = ">=3.12,<3.13"`.
  - Dependencies: `fastapi`, `uvicorn[standard]`, `python-dotenv`, `openai>=1.40`.
  - Extras: `transformers = ["transformers", "accelerate", "bitsandbytes>=0.45"]` (unpinned until Task 7 pins them) and `dev = ["pytest", "httpx"]`.
  - Packages: `voicebot*`.
  - `[tool.pytest.ini_options] testpaths = ["tests"]`.

- [ ] **Step 2: Write `scripts/setup_venv.ps1`.** It runs these steps in order and stops at the first failure (`$ErrorActionPreference = "Stop"`, and check `$LASTEXITCODE` after each native call):
  1. `uv venv -p 3.12 .venv` (skip if `.venv` exists)
  2. `uv pip install --python .venv -r RealtimeSTT/requirements-gpu-torch.txt`
  3. `uv pip install --python .venv -e RealtimeSTT -e "RealtimeTTS[kokoro]" -e "server[dev,transformers]"`
  4. Run step 2 again, so a CPU torch pulled in by step 3 gets replaced.
  5. `.venv\Scripts\python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch not active'; print(torch.__version__, torch.cuda.get_device_name(0))"`
  6. `git config core.hooksPath .githooks`
  7. `if (Test-Path web/package.json) { npm --prefix web install }`

- [ ] **Step 3: Run `scripts/setup_venv.ps1`**

  Expected: it ends by printing `2.7.1+cu128 NVIDIA GeForce RTX 4060`, and `git config core.hooksPath` prints `.githooks`.
  If a wheel is missing for Python 3.12, stop and report the package name. Do not switch Python versions.

- [ ] **Step 4: Write the failing test `server/tests/test_config.py`**

```python
import pytest
from voicebot.config import Settings, ConfigError, REPO_ROOT

def test_defaults_match_spec():
    s = Settings.load(env={})
    assert (s.llm_backend, s.audio_io) == ("llamacpp", "browser")
    assert (s.hf_model_id, s.hf_load_in_4bit) == ("Qwen/Qwen3.5-9B", True)
    assert s.llama_autostart is True
    assert s.llama_server_path == REPO_ROOT / "tools/llama/llama-server.exe"
    assert s.llama_hf_repo == "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
    assert (s.llama_host, s.llama_port, s.llama_ctx, s.llama_start_timeout) == ("127.0.0.1", 8081, 8192, 900)
    assert s.llama_base_url == "http://127.0.0.1:8081"
    assert (s.stt_model, s.stt_device, s.stt_compute_type, s.stt_silence_sec) == ("small.en", "cuda", "int8", 0.6)
    assert s.tts_voice == "af_heart"
    assert s.system_prompt == ("You are a friendly voice assistant. Reply in one to three short spoken "
                               "sentences. Never use lists, markdown, code blocks or emoji.")
    assert (s.llm_temperature, s.llm_top_p, s.llm_max_tokens, s.history_turns) == (0.7, 0.8, 256, 10)
    assert (s.web_host, s.web_port) == ("127.0.0.1", 8000)

def test_overrides_and_derived_base_url():
    s = Settings.load(env={"LLM_BACKEND": "transformers", "LLAMA_PORT": "9000", "LLAMA_AUTOSTART": "No"})
    assert s.llm_backend == "transformers" and s.llama_port == 9000 and s.llama_autostart is False
    assert s.llama_base_url == "http://127.0.0.1:9000"
    assert Settings.load(env={"LLAMA_BASE_URL": "http://box:1234"}).llama_base_url == "http://box:1234"

@pytest.mark.parametrize("var,val", [("LLM_BACKEND", "ollama"), ("AUDIO_IO", "phone"),
    ("LLAMA_PORT", "abc"), ("LLM_TEMPERATURE", "hot"), ("HF_LOAD_IN_4BIT", "maybe")])
def test_invalid_value_names_variable(var, val):
    with pytest.raises(ConfigError, match=var):
        Settings.load(env={var: val})

def test_reads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIO_IO", raising=False)
    (tmp_path / ".env").write_text("AUDIO_IO=server\n")
    assert Settings.load(env_file=tmp_path / ".env").audio_io == "server"
```

- [ ] **Step 5: Run it to see it fail**

  Run: `../.venv/Scripts/python -m pytest tests/test_config.py -v`
  Expected: FAIL with `ModuleNotFoundError: voicebot.config`.

- [ ] **Step 6: Implement `config.py`**, following the Interfaces block above.

- [ ] **Step 7: Run it to see it pass**

  Same command. Expected: 8 passed.

- [ ] **Step 8: Write `.env.example`.** List every spec §4.1 variable with its default and a one-line comment. Put `LLM_BACKEND` and `AUDIO_IO` first, with their allowed values. Add the comment `# LLAMA_BASE_URL is derived from LLAMA_HOST/LLAMA_PORT unless set`.

- [ ] **Step 9: Commit**

```bash
git add scripts/setup_venv.ps1 server/pyproject.toml server/voicebot server/tests .env.example
git commit -m "feat(server): scaffold voicebot package, setup script and settings"
```

---

### Task 2: `speech_filter.py`

**Files:**
- Create: `server/voicebot/speech_filter.py`
- Test: `server/tests/test_speech_filter.py`

**Interfaces:**
- Produces: `filter_stream(chunks: Iterable[str]) -> Iterator[str]`. It is a lazy generator: it yields as soon as text is safe to emit and never yields `""`.

- [ ] **Step 1: Write the failing test**

```python
from voicebot.speech_filter import filter_stream

def run(chunks):
    return "".join(filter_stream(iter(chunks)))

def test_plain_text_passes():            assert run(["Hello ", "world."]) == "Hello world."
def test_think_block_removed():          assert run(["<think>secret</think>Hi"]) == "Hi"
def test_think_tags_split():             assert run(["<th", "ink>sec", "ret</thi", "nk>Hel", "lo"]) == "Hello"
def test_empty_think_block():            assert run(["<think>\n\n</think>\n\n", "Sure."]).strip() == "Sure."
def test_unclosed_think_drops_rest():    assert run(["Ok. <think>reasoning", " more"]) == "Ok. "
def test_markdown_symbols_removed():     assert run(["**Bold** and `code` and # head"]) == "Bold and code and  head"
def test_line_start_bullets_removed():   assert run(["Items:\n- one\n", "- two"]) == "Items:\none\ntwo"
def test_inner_hyphens_kept():           assert run(["well-known - fine"]) == "well-known - fine"
def test_lone_angle_bracket_kept():      assert run(["a < b", " and c"]) == "a < b and c"
def test_never_yields_empty():
    assert all(c != "" for c in filter_stream(iter(["<think>", "x", "</think>", "", "**", "ok"])))
def test_is_lazy():
    gen = filter_stream(iter(["First. ", "<think>"]))
    assert next(gen) == "First. "
```

- [ ] **Step 2: Run it to see it fail**

  Run: `../.venv/Scripts/python -m pytest tests/test_speech_filter.py -v`
  Expected: FAIL (import error).

- [ ] **Step 3: Implement `filter_stream`.** Approach:
  - Keep a `pending` buffer, an `in_think` flag and an `at_line_start` flag.
  - Append each chunk to `pending`. Repeatedly look for the next relevant tag: `<think>` when outside, `</think>` when inside.
  - Emit (when outside) or drop (when inside) the text before the tag, and switch state at the tag.
  - Hold back only a trailing suffix of `pending` that is a prefix of the tag being looked for.
  - Clean emitted text: remove `*`, `#` and backticks, and drop `"- "` when `at_line_start`. Hold back a lone `-` at line start until the next character arrives.
  - At end of input:
    - outside a think block: emit any held text;
    - inside one: drop it.

- [ ] **Step 4: Run it to see it pass.** Expected: 11 passed.

- [ ] **Step 5: Commit** with the message `feat(server): add speech_filter for think tags and markdown`.

---

### Task 3: `conversation.py`

**Files:**
- Create: `server/voicebot/conversation.py`
- Test: `server/tests/test_conversation.py`

**Interfaces:**
- Produces `class Conversation(system_prompt: str, history_turns: int)` with:
  - attribute `system_prompt: str`, writable
  - `messages(user_text: str) -> list[dict[str, str]]`
  - `add_exchange(user: str, assistant: str) -> None`
  - `clear() -> None`
  - property `turns -> list[tuple[str, str]]`, returned as a copy

- [ ] **Step 1: Write the failing test**

```python
from voicebot.conversation import Conversation

def test_messages_shape():
    c = Conversation("SYS", history_turns=2); c.add_exchange("u1", "a1")
    assert c.messages("u2") == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "u1"},
                                {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]

def test_history_trimmed_to_last_n_pairs():
    c = Conversation("S", history_turns=2)
    for i in range(1, 4): c.add_exchange(f"u{i}", f"a{i}")
    assert [m["content"] for m in c.messages("new")] == ["S", "u2", "a2", "u3", "a3", "new"]

def test_clear_and_mutable_prompt():
    c = Conversation("S", 10); c.add_exchange("u", "a"); c.clear(); c.system_prompt = "NEW"
    assert c.turns == [] and c.messages("x")[0]["content"] == "NEW"
```

- [ ] **Step 2: Run it to see it fail.**
- [ ] **Step 3: Implement `Conversation`.** Store all turns and slice the last `history_turns` inside `messages`.
- [ ] **Step 4: Run it to see it pass.** Expected: 3 passed.
- [ ] **Step 5: Commit** with the message `feat(server): add conversation history`.

---

### Task 4: LLM base types and `LlamaServer`

**Files:**
- Create: `server/voicebot/llm/__init__.py` (empty), `server/voicebot/llm/base.py`, `server/voicebot/llm/llama_server.py`
- Test: `server/tests/test_llama_server.py`

**Interfaces:**
- Consumes: `Settings`, `REPO_ROOT` (Task 1).
- Produces:
  - `base.py`:
    - `@dataclass(frozen=True) class GenerationParams: temperature: float; top_p: float; max_tokens: int`
    - `class LLMError(RuntimeError)`
    - `class LLMBackend(Protocol)` with `name: str`, `model_label: str`, `start() -> None`, `stream(messages: list[dict], params: GenerationParams, stop_event: threading.Event) -> Iterator[str]`, `close() -> None`
  - `llama_server.py`:
    - `class LlamaServerError(LLMError)`
    - `port_in_use(host: str, port: int) -> bool` (a socket connect test)
    - `http_status(url: str, timeout: float = 2.0) -> int | None` (urllib; returns `None` on connection error)
    - `wait_for_health(base_url: str, timeout: float, *, status=http_status, sleep=time.sleep, clock=time.monotonic, is_alive: Callable[[], bool] = lambda: True) -> None`. It polls `f"{base_url}/health"` every 1 s until 200. It raises `LlamaServerError` containing `"exited"` if `not is_alive()`, or `"timeout"` past the deadline.
    - `class LlamaServer(settings, *, popen=subprocess.Popen, status=http_status, sleep=time.sleep, clock=time.monotonic, port_check=port_in_use, log_path: Path = REPO_ROOT / "logs/llama-server.log")` with `command() -> list[str]`, `start() -> None` and `close() -> None`.

- [ ] **Step 1: Write the failing test** (fake process and clock, a real tmp binary file)

```python
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
```

  The fake log file is read only at error time. `start()` opens `log_path` for writing and passes it to `popen`. The test writes the log content before calling `start()`, so `start()` must open the log in append mode (`"ab"`) and not truncate it.

- [ ] **Step 2: Run it to see it fail.**

- [ ] **Step 3: Implement.** `start()` does the following:
  1. Check the binary exists. If not, raise with `"run scripts/fetch_llama_server.ps1"`.
  2. Check the port. If busy, raise with `"LLAMA_PORT ... or set LLAMA_AUTOSTART=false"`.
  3. `mkdir` the log dir and open the log with `"ab"`.
  4. `popen(self.command(), stdout=log, stderr=subprocess.STDOUT)`.
  5. Register `atexit.register(self.close)`.
  6. `wait_for_health(..., is_alive=lambda: proc.poll() is None)`.
  7. On `LlamaServerError`: `close()`, then re-raise with `"\n--- last 30 log lines ---\n" + tail`.

  `close()` calls `terminate`, then `wait(5)`, and on `TimeoutExpired` calls `kill`.

- [ ] **Step 4: Run it to see it pass.** Expected: 8 passed.
- [ ] **Step 5: Commit** with the message `feat(server): add LLM base types and llama-server manager`.

---

### Task 5: `LlamaCppBackend`

**Files:**
- Create: `server/voicebot/llm/llamacpp_backend.py`
- Test: `server/tests/test_llamacpp_backend.py`

**Interfaces:**
- Consumes: `GenerationParams`, `LLMError` (Task 4), `LlamaServer`, `wait_for_health`, `http_status` (Task 4).
- Produces: `class LlamaCppBackend(settings, *, client=None, server_factory=LlamaServer, status=http_status, sleep=time.sleep, clock=time.monotonic)`
  - `name = "llamacpp"`, `model_label = settings.llama_hf_repo`
  - `start()`:
    - With autostart: `server_factory(settings).start()`.
    - Otherwise: `wait_for_health(settings.llama_base_url, settings.llama_start_timeout, ...)`.
    - When `client` is None: create `openai.OpenAI(base_url=f"{base}/v1", api_key="local")`.
  - `stream(...)` and `close()` as below.

- [ ] **Step 1: Write the failing test**

```python
import threading, httpx, openai, pytest
from types import SimpleNamespace as NS
from voicebot.config import Settings
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.llm.llamacpp_backend import LlamaCppBackend

class FakeStream:
    def __init__(self, deltas): self.deltas, self.closed = deltas, False
    def __iter__(self):
        for d in self.deltas: yield NS(choices=[NS(delta=NS(content=d))])
    def close(self): self.closed = True

class FakeClient:
    def __init__(self, deltas=(), error=None):
        self.kwargs, self.stream_obj, self.error = None, FakeStream(list(deltas)), error
        self.chat = NS(completions=NS(create=self.create))
    def create(self, **kw):
        if self.error: raise self.error
        self.kwargs = kw; return self.stream_obj

P = GenerationParams(temperature=0.3, top_p=0.9, max_tokens=64)
MSGS = [{"role": "user", "content": "hi"}]

def backend(client, **env):
    return LlamaCppBackend(Settings.load(env={"LLAMA_AUTOSTART": "false", **env}), client=client,
                           status=lambda u: 200, sleep=lambda s: None)

def test_sends_thinking_off_and_params():
    c = FakeClient(["Hi"]); list(backend(c).stream(MSGS, P, threading.Event()))
    assert c.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert (c.kwargs["stream"], c.kwargs["temperature"], c.kwargs["top_p"], c.kwargs["max_tokens"]) == (True, 0.3, 0.9, 64)
    assert c.kwargs["messages"] == MSGS

def test_skips_empty_deltas():
    assert list(backend(FakeClient(["Hi", None, "", " there"])).stream(MSGS, P, threading.Event())) == ["Hi", " there"]

def test_stop_event_closes_stream():
    c, ev = FakeClient(["Hi", " more", " text"]), threading.Event(); out = []
    for t in backend(c).stream(MSGS, P, ev): out.append(t); ev.set()
    assert out == ["Hi"] and c.stream_obj.closed

def test_connection_error_becomes_llm_error():   # Review Focus 4
    err = openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8081/v1/chat/completions"))
    with pytest.raises(LLMError, match="llama-server"):
        list(backend(FakeClient(error=err)).stream(MSGS, P, threading.Event()))

def test_start_autostart_uses_server_and_close_stops_it():
    calls = []
    class FakeServer:
        def __init__(self, s): calls.append("init")
        def start(self): calls.append("start")
        def close(self): calls.append("close")
    b = LlamaCppBackend(Settings.load(env={}), client=FakeClient(), server_factory=FakeServer)
    b.start(); b.close()
    assert calls == ["init", "start", "close"]

def test_start_external_waits_for_health():
    seq = [503, 200]
    b = LlamaCppBackend(Settings.load(env={"LLAMA_AUTOSTART": "false"}), client=FakeClient(),
                        server_factory=lambda s: pytest.fail("must not start server"),
                        status=lambda u: seq.pop(0), sleep=lambda s: None)
    b.start(); assert seq == []
```

- [ ] **Step 2: Run it to see it fail.**
- [ ] **Step 3: Implement.**
  - `stream` calls `client.chat.completions.create(model="local", messages=..., stream=True, temperature=..., top_p=..., max_tokens=..., extra_body=...)`.
  - It iterates the response. When `stop_event.is_set()` it calls `response.close()` and returns. It yields `delta.content` when truthy.
  - It wraps `openai.APIConnectionError` and `openai.APIError` in `LLMError(f"llama-server at {base} failed: {e}")`.
- [ ] **Step 4: Run it to see it pass.** Expected: 6 passed.
- [ ] **Step 5: Commit** with the message `feat(server): add llama.cpp streaming backend`.

---

### Task 6: `TransformersBackend` and `factory.py`

**Files:**
- Create: `server/voicebot/llm/transformers_backend.py`, `server/voicebot/llm/factory.py`
- Test: `server/tests/test_transformers_backend.py`, `server/tests/test_factory.py`

**Interfaces:**
- Consumes: `GenerationParams`, `LLMError` (Task 4), `LlamaCppBackend` (Task 5).
- Produces:
  - `transformers_backend.py`:
    - `class StopOnEvent(StoppingCriteria)` with `__init__(event: threading.Event)`; `__call__(input_ids, scores, **kw) -> bool` returns `event.is_set()`.
    - `build_inputs(tokenizer, messages: list[dict])`, which calls `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt")`.
    - `class TransformersBackend(settings, *, loader: Callable[[], tuple[model, tokenizer]] | None = None)`:
      - `name = "transformers"`, `model_label = f"{hf_model_id} (4-bit)"` when 4-bit, else `hf_model_id`.
      - The constructor must not load anything.
      - `start()` calls `loader or self._default_loader`.
  - `factory.py`: `make_backend(settings) -> LLMBackend`. It imports the selected backend module inside the function.

- [ ] **Step 1: Write the failing tests**

```python
# test_transformers_backend.py
import threading, pytest, torch
from voicebot.config import Settings
from voicebot.llm.base import GenerationParams, LLMError
from voicebot.llm.transformers_backend import TransformersBackend, StopOnEvent, build_inputs

class FakeInputs(dict):
    def to(self, device): return self

class FakeTokenizer:
    def __init__(self): self.kwargs = None
    def apply_chat_template(self, messages, **kw): self.kwargs = kw; return FakeInputs(input_ids=[[1, 2]])

class FakeModel:
    device = "cpu"
    def __init__(self, texts=("Hel", "lo"), error=None): self.texts, self.error, self.kwargs = texts, error, None
    def generate(self, **kw):
        self.kwargs = kw
        if self.error: raise self.error
        for i, t in enumerate(self.texts): kw["streamer"].on_finalized_text(t, stream_end=i == len(self.texts) - 1)

P = GenerationParams(0.4, 0.85, 64)

def test_build_inputs_disables_thinking():
    tok = FakeTokenizer(); build_inputs(tok, [{"role": "user", "content": "x"}])
    assert tok.kwargs["enable_thinking"] is False and tok.kwargs["add_generation_prompt"] is True and tok.kwargs["return_dict"] is True

def test_stop_on_event():
    ev = threading.Event(); s = StopOnEvent(ev); assert s(None, None) is False; ev.set(); assert s(None, None) is True

def test_stream_yields_text_and_passes_params():
    model = FakeModel(); b = TransformersBackend(Settings.load(env={}), loader=lambda: (model, FakeTokenizer())); b.start()
    assert "".join(b.stream([{"role": "user", "content": "x"}], P, threading.Event())) == "Hello"
    assert (model.kwargs["max_new_tokens"], model.kwargs["temperature"], model.kwargs["top_p"], model.kwargs["do_sample"]) == (64, 0.4, 0.85, True)
    assert any(isinstance(c, StopOnEvent) for c in model.kwargs["stopping_criteria"])

def test_generate_error_raises_without_hanging():
    b = TransformersBackend(Settings.load(env={}), loader=lambda: (FakeModel(error=RuntimeError("boom")), FakeTokenizer())); b.start()
    with pytest.raises(LLMError, match="boom"): list(b.stream([], P, threading.Event()))

def test_oom_hint():
    def loader(): raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(LLMError, match="STT_DEVICE=cpu"): TransformersBackend(Settings.load(env={}), loader=loader).start()
```

```python
# test_factory.py
import subprocess, sys
from voicebot.config import Settings
from voicebot.llm.factory import make_backend

def test_picks_backend_class():
    assert type(make_backend(Settings.load(env={"LLM_BACKEND": "llamacpp"}))).__name__ == "LlamaCppBackend"
    assert type(make_backend(Settings.load(env={"LLM_BACKEND": "transformers"}))).__name__ == "TransformersBackend"

def test_llamacpp_path_never_imports_torch():
    code = ("import sys; from voicebot.config import Settings; from voicebot.llm.factory import make_backend; "
            "make_backend(Settings.load(env={'LLM_BACKEND': 'llamacpp'})); "
            "bad = {'torch', 'transformers', 'bitsandbytes'} & set(sys.modules); assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run both test files to see them fail.**

- [ ] **Step 3: Implement.**

  `stream()`:
  - `inputs = build_inputs(tok, messages).to(model.device)`
  - `TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)`
  - Run `model.generate(**inputs, streamer=..., max_new_tokens=..., do_sample=True, temperature=..., top_p=..., stopping_criteria=StoppingCriteriaList([StopOnEvent(stop_event)]))` on a daemon thread. The thread wrapper stores any exception and calls `streamer.end()` in `finally`.
  - Yield non-empty text, join the thread, and re-raise a stored exception as `LLMError(str(e))`.

  `_default_loader()`:
  - Raise `LLMError` if `not torch.cuda.is_available()`.
  - Import `bitsandbytes`, raising `LLMError` mentioning it if that fails.
  - Load `AutoProcessor` and `AutoModelForImageTextToText` with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` when `hf_load_in_4bit`, and `device_map="cuda:0"`.
  - Return `(model, processor.tokenizer)`.

  `start()` wraps `torch.cuda.OutOfMemoryError` as `LLMError("CUDA out of memory loading the model: set STT_DEVICE=cpu or use LLM_BACKEND=llamacpp")`.

- [ ] **Step 4: Run both to see them pass.** Expected: 7 passed.
- [ ] **Step 5: Commit** with the message `feat(server): add transformers 4-bit backend and lazy backend factory`.

---

### Task 7: `smoke_llm`, `fetch_llama_server.ps1`, real-model check

**Files:**
- Create: `server/voicebot/smoke_llm.py`, `scripts/fetch_llama_server.ps1`
- Modify: `server/pyproject.toml` (pin versions), the spec §4.2 open item (record the outcome)
- Test: `server/tests/test_smoke_llm.py`

**Interfaces:**
- Consumes: `Settings.load`, `make_backend`, `GenerationParams`.
- Produces: `smoke_llm.main(argv: list[str] | None = None, *, backend_factory=make_backend) -> int`.
  - It prints `model_label`, the streamed reply, and `ttft_ms=<n> total_ms=<n>`.
  - It returns 1 if the raw output contains `<think>`, otherwise 0.
  - The backend is always closed.
  - The prompt is `[{"role": "system", "content": settings.system_prompt}, {"role": "user", "content": "In one sentence, what is a voice assistant?"}]`.
  - Run it with `python -m voicebot.smoke_llm`.

- [ ] **Step 1: Write the failing test**

```python
from voicebot import smoke_llm

class FakeBackend:
    name, model_label = "fake", "fake-model"
    def __init__(self, chunks): self.chunks, self.closed = chunks, False
    def start(self): pass
    def stream(self, messages, params, stop_event): yield from self.chunks
    def close(self): self.closed = True

def test_ok(capsys):
    b = FakeBackend(["A voice ", "assistant."])
    assert smoke_llm.main([], backend_factory=lambda s: b) == 0 and b.closed
    out = capsys.readouterr().out
    assert "fake-model" in out and "A voice assistant." in out and "ttft_ms=" in out

def test_think_leak_fails():
    assert smoke_llm.main([], backend_factory=lambda s: FakeBackend(["<think>x</think>Hi"])) == 1
```

- [ ] **Step 2: Run it to see it fail. Step 3: Implement. Step 4: Run it to see it pass** (2 passed).

- [ ] **Step 5: Write `scripts/fetch_llama_server.ps1`**
  1. GET `https://api.github.com/repos/ggml-org/llama.cpp/releases/latest`.
  2. From its assets, pick the highest CUDA 12.x pair: `llama-*-bin-win-cuda-12.*-x64.zip` and the matching `cudart-llama-bin-win-cuda-12.*-x64.zip`.
  3. Download both to `%TEMP%` and extract both into `tools/llama/`, replacing what's there.
  4. Print the release tag.
  5. Exit with an error if no matching asset exists.

- [ ] **Step 6: Run it and verify**

  Run: `scripts/fetch_llama_server.ps1; tools/llama/llama-server.exe --version`
  Expected: a build number is printed, and CUDA is listed among the backends.

- [ ] **Step 7: Manual llamacpp check.** Set `LLM_BACKEND=llamacpp` in `.env` and run `.venv/Scripts/python -m voicebot.smoke_llm` from `server/`.
  - Expected: one English sentence and exit code 0.
  - The first run downloads about 5.7 GB. Raise `LLAMA_START_TIMEOUT` if the connection is slow.
  - `nvidia-smi` during the run should show roughly 6 GB or less in use.
  - If llama.cpp reports an unknown architecture, stop and report the release tag.

- [ ] **Step 8: Manual transformers check.** Set `LLM_BACKEND=transformers` and run the same command.
  - Expected: a sentence and exit code 0. The first run downloads 19.3 GB.
  - If you get `KeyError`/`ValueError` for `qwen3_5`, run `uv pip install --python .venv -U transformers` and retry.

- [ ] **Step 9: Resolve the spec §4.2 open item.** In `.venv/Scripts/python`, try loading `Qwen/Qwen3.5-9B` with `AutoModelForCausalLM` and the same 4-bit config, then generate 10 tokens.
  - If it works and uses less VRAM (compare `torch.cuda.memory_allocated()`), switch `_default_loader` to it, use `AutoTokenizer`, and keep the Task 6 tests passing.
  - Either way, replace the open-item sentence in the spec with the measured result.

- [ ] **Step 10: Pin versions.** In `server/pyproject.toml`, pin `transformers`, `accelerate` and `bitsandbytes` to the versions that just worked (`==`, read with `uv pip show`).

- [ ] **Step 11: Commit**

```bash
git add server/voicebot/smoke_llm.py server/tests/test_smoke_llm.py scripts/fetch_llama_server.ps1 server/pyproject.toml docs/superpowers/specs server/voicebot/llm/transformers_backend.py
git commit -m "feat(server): add LLM smoke check and llama.cpp fetch script; pin verified versions"
```

---

### Task 7b: `timing.py` and `latency_report.py` — stage timing

Every turn records a timestamp at each pipeline stage. The `metrics` event, the UI and `logs/turns.jsonl` all report per-stage durations, so you can see where the latency goes.

**Stages**, in pipeline order:

| Stage | Marked when | Marked by |
|---|---|---|
| `speech_end` | The VAD first hears silence after speech (RealtimeSTT `on_turn_detection_start`). Cleared if speech resumes (`on_turn_detection_stop`). | recorder callback (Task 8) |
| `endpoint` | The silence wait (`STT_SILENCE_SEC`) is over and recording stops (`on_recording_stop`) | recorder callback |
| `stt_start` | Final transcription starts (`on_transcription_start`) | recorder callback |
| `stt_done` | `recorder.text()` returns | session (Task 9) |
| `input` | A typed `send_text` is received (typed turns only) | session |
| `llm_first_token` | The backend yields its first raw chunk | session |
| `first_text` | `filter_stream` yields its first speakable chunk | session |
| `tts_first_audio` | The first TTS audio chunk is produced | session (`on_first_audio`) |
| `playback_started` | Browser mode: the browser's player starts outputting audio and acks. Server mode: same moment as `tts_first_audio`. | session |
| `llm_done` | The backend stream ends | session |
| `play_done` | `audio.play()` returns | session |

Note: `on_vad_stop` is **not** used. RealtimeSTT fires it after the silence wait, not when speech ends (`RealtimeSTT/RealtimeSTT/core/recording.py:474-479`). `speech_end` itself lags the true end of speech by the VAD's detection delay, which is a few tens of ms.

**Spans** (ms; `null` when either end is missing). The origin is the first present stage of `speech_end`, `endpoint`, `stt_done` or `input`. "Input" below means `stt_done` for voice turns and `input` for typed turns.

| Span | Formula | What it shows |
|---|---|---|
| `silence_wait_ms` | `endpoint − speech_end` | the end-of-turn wait (≈ `STT_SILENCE_SEC`) |
| `stt_ms` | `stt_done − endpoint` | final transcription |
| `llm_ttft_ms` | `llm_first_token − input` | LLM time to first token |
| `filter_ms` | `first_text − llm_first_token` | text dropped by the filter (`<think>`, markdown) |
| `tts_first_audio_ms` | `tts_first_audio − first_text` | TTS time to first audio |
| `playback_ms` | `playback_started − tts_first_audio` | WebSocket + browser player buffering |
| `perceived_ms` | `(playback_started, or else tts_first_audio) − origin` | **what the user feels**: end of speech to first sound |
| `llm_total_ms` | `llm_done − input` | full generation time |
| `total_ms` | `play_done − origin` | whole turn |

**Files:**
- Create: `server/voicebot/timing.py`, `server/voicebot/latency_report.py`
- Test: `server/tests/test_timing.py`, `server/tests/test_latency_report.py`

**Interfaces:**
- Produces:
  - **`timing.py`** (stdlib only):
    - `STAGES: tuple[str, ...]`, the 11 stages above, in order
    - `SPANS: tuple[str, ...]`, the 9 spans above, in order
    - `class TurnTimer(clock: Callable[[], float] = time.perf_counter)`. All methods are thread-safe behind one lock.
      - `mark(stage)`: records `clock()` only if the stage is not already set, so the first mark wins. An unknown stage raises `ValueError`.
      - `unmark(stage)`, `get(stage) -> float | None`, `has(stage) -> bool`, `reset()`.
      - `report() -> {"marks": {stage: int | None}, "spans": {span: int | None}}`. Marks are ms since the origin. Every value is `round(seconds * 1000)`. With no origin, everything is `None`.
    - `append_jsonl(path: Path, record: dict) -> None`: creates the parent dirs and appends one compact JSON line. An `OSError` is logged and swallowed, so a turn never fails because of logging.
  - **`latency_report.py`**: `python -m voicebot.latency_report [path] [--by k1,k2] [--last N] [--source voice|text|all]`
    - Default `path`: `REPO_ROOT/logs/turns.jsonl`. Default `--by`: `llm_backend,audio_io,stt_model`. Default `--source`: `all`.
    - `summarize(records, group_by: tuple[str, ...]) -> list[dict]`:
      - Skips records with `interrupted: true`.
      - Groups by the `group_by` fields.
      - Each row is `{"group": {...}, "n": int, "spans": {span: {"median": int, "p90": int} | None}}`.
      - Medians use `round(statistics.median(...))`. p90 uses nearest rank: `sorted(v)[ceil(0.9 * len(v)) - 1]`. `None` values are excluded. A span with no values is `None`.
    - `main(argv: list[str] | None = None)`: applies `--source` and `--last`, then prints one plain-text table per group: the group values, `n`, and one line per span with its median and p90.

- [ ] **Step 1: Write the failing tests**

```python
# test_timing.py
import json, pytest
from voicebot.timing import TurnTimer, append_jsonl

class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t

def at(tm, c, stage, t): c.t = t; tm.mark(stage)

def test_voice_turn_spans():
    c = Clock(); tm = TurnTimer(clock=c)
    for st, t in [("speech_end", 100.0), ("endpoint", 100.6), ("stt_start", 100.61), ("stt_done", 100.8),
                  ("llm_first_token", 101.1), ("first_text", 101.12), ("tts_first_audio", 101.3),
                  ("playback_started", 101.34), ("llm_done", 101.9), ("play_done", 104.0)]:
        at(tm, c, st, t)
    r = tm.report()
    assert r["marks"]["speech_end"] == 0 and r["marks"]["endpoint"] == 600 and r["marks"]["input"] is None
    assert r["spans"] == {"silence_wait_ms": 600, "stt_ms": 200, "llm_ttft_ms": 300, "filter_ms": 20,
                          "tts_first_audio_ms": 180, "playback_ms": 40, "perceived_ms": 1340,
                          "llm_total_ms": 1100, "total_ms": 4000}

def test_typed_turn_uses_input_origin_and_null_spans():
    c = Clock(); tm = TurnTimer(clock=c)
    for st, t in [("input", 50.0), ("llm_first_token", 50.25), ("first_text", 50.25), ("tts_first_audio", 50.5), ("play_done", 52.0)]:
        at(tm, c, st, t)
    s = tm.report()["spans"]
    assert s["llm_ttft_ms"] == 250 and s["silence_wait_ms"] is None and s["stt_ms"] is None
    assert s["playback_ms"] is None and s["perceived_ms"] == 500        # falls back to tts_first_audio

def test_first_mark_wins_unmark_and_reset():                            # Review Focus 6
    c = Clock(); tm = TurnTimer(clock=c)
    at(tm, c, "speech_end", 1.0); at(tm, c, "speech_end", 2.0); assert tm.get("speech_end") == 1.0
    tm.unmark("speech_end"); at(tm, c, "speech_end", 3.0); assert tm.get("speech_end") == 3.0
    tm.reset(); assert not tm.has("speech_end") and tm.report()["spans"]["total_ms"] is None

def test_unknown_stage_raises():
    with pytest.raises(ValueError): TurnTimer().mark("nope")

def test_append_jsonl(tmp_path):
    p = tmp_path / "logs" / "t.jsonl"; append_jsonl(p, {"a": 1}); append_jsonl(p, {"a": 2})
    assert [json.loads(l) for l in p.read_text().splitlines()] == [{"a": 1}, {"a": 2}]
```

```python
# test_latency_report.py
import json
from voicebot.latency_report import summarize, main

def rec(backend, perceived, total, interrupted=False, source="voice"):
    return {"llm_backend": backend, "audio_io": "browser", "stt_model": "small.en", "source": source,
            "interrupted": interrupted, "spans": {"perceived_ms": perceived, "total_ms": total, "stt_ms": None}}

def test_summarize_groups_and_skips():
    rows = summarize([rec("llamacpp", 1000, 3000), rec("llamacpp", 1200, 3100), rec("llamacpp", 1400, 3200),
                      rec("llamacpp", 9999, 1, interrupted=True), rec("transformers", 2000, 5000)], group_by=("llm_backend",))
    g = {r["group"]["llm_backend"]: r for r in rows}
    assert g["llamacpp"]["n"] == 3 and g["llamacpp"]["spans"]["perceived_ms"] == {"median": 1200, "p90": 1400}
    assert g["llamacpp"]["spans"]["stt_ms"] is None and g["transformers"]["n"] == 1

def test_main_prints_table(tmp_path, capsys):
    p = tmp_path / "t.jsonl"; p.write_text("".join(json.dumps(rec("llamacpp", 1000, 3000)) + "\n" for _ in range(2)))
    main([str(p), "--by", "llm_backend"]); out = capsys.readouterr().out
    assert "llamacpp" in out and "perceived_ms" in out and "1000" in out
```

- [ ] **Step 2: Run them to see them fail. Step 3: Implement. Step 4: Run them to see them pass** (7 passed).
- [ ] **Step 5: Commit** with the message `feat(server): add per-turn stage timer and latency report`.

---

### Task 8: `stt.py`, `tts.py`, `audio_io.py`

**Files:**
- Create: `server/voicebot/stt.py`, `server/voicebot/tts.py`, `server/voicebot/audio_io.py`
- Test: `server/tests/test_audio_io.py`

**Interfaces:**
- Consumes: `Settings`; `TurnTimer` (Task 7b).
- Produces:
  - **`stt.py`**:
    - `make_recorder(settings, *, use_microphone: bool, callbacks: dict | None = None, recorder_cls=None) -> AudioToTextRecorder`.
      - Imports `RealtimeSTT` lazily when `recorder_cls` is None.
      - Passes exactly: `model=stt_model, language="en", device=stt_device, compute_type=stt_compute_type, use_microphone=use_microphone, post_speech_silence_duration=stt_silence_sec, spinner=False, enable_realtime_transcription=False`, plus every entry of `callbacks`.
    - `timing_callbacks(timer) -> dict` returns:
      - `on_turn_detection_start` → `timer.mark("speech_end")`
      - `on_turn_detection_stop` → `timer.unmark("speech_end")` (the user resumed speaking)
      - `on_recording_stop` → `timer.mark("endpoint")`
      - `on_transcription_start(audio)` → `timer.mark("stt_start")`. It **must return `None`**, because RealtimeSTT aborts the transcription when this callback returns a truthy value (`core/transcription_api.py:53-54`).
  - **`tts.py`**:
    - `make_tts(settings, *, engine_cls=None, stream_cls=None) -> TextToAudioStream` (`KokoroEngine(voice=tts_voice)`)
    - `warm_up(stream) -> None`: `stream.feed("Warming up."); stream.play(muted=True)`
    - `list_voices(stream) -> list[str]`: `getattr(v, "name", str(v))` over `stream.engine.get_voices()`
    - `set_voice(stream, name: str) -> None`: `stream.engine.set_voice(name)`
    - `output_format(stream) -> dict`: `{"sample_rate": rate, "channels": ch, "encoding": "pcm_s16le"}`, from `stream.engine.get_stream_info()`
  - **`audio_io.py`**:
    - `class AudioIO(Protocol)` with `recorder`, `output_format: dict | None`, `on_client_audio(data: bytes)`, `mute_input(muted: bool)`, `set_sink(send_binary: Callable[[bytes], None] | None)`, `play(stream, on_first_audio: Callable[[], None]) -> None`
    - `class BrowserAudio(recorder, output_format: dict)`
    - `class ServerAudio(recorder)`
    - `make_audio_io(settings, stream, *, callbacks: dict | None = None, recorder_factory=make_recorder) -> AudioIO`. It calls `recorder_factory(settings, use_microphone=..., callbacks=callbacks)`.

- [ ] **Step 1: Write the failing test**

```python
from types import SimpleNamespace as NS
from voicebot.config import Settings
from voicebot import stt, tts
from voicebot.audio_io import BrowserAudio, ServerAudio, make_audio_io

class Rec:
    def __init__(self, **kw): self.kw, self.fed, self.mic = kw, [], []
    def feed_audio(self, b, original_sample_rate): self.fed.append((b, original_sample_rate))
    def set_microphone(self, on): self.mic.append(on)

class Stream:
    def __init__(self): self.play_kw = None; self.engine = NS(get_stream_info=lambda: (8, 1, 24000),
        get_voices=lambda: [NS(name="af_heart"), NS(name="am_adam")], set_voice=lambda n: setattr(self, "voice", n))
    def play(self, **kw):
        self.play_kw = kw
        for c in (b"a", b"b"):
            if kw.get("on_audio_chunk"): kw["on_audio_chunk"](c)

def test_make_recorder_kwargs():
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec)
    assert r.kw == dict(model="small.en", language="en", device="cuda", compute_type="int8", use_microphone=False,
                        post_speech_silence_duration=0.6, spinner=False, enable_realtime_transcription=False)
    cb = {"on_recording_stop": lambda: None}
    assert stt.make_recorder(Settings.load(env={}), use_microphone=False, callbacks=cb, recorder_cls=Rec).kw["on_recording_stop"] is cb["on_recording_stop"]

def test_timing_callbacks_mark_stages():                                  # Review Focus 6
    from voicebot.timing import TurnTimer
    tm = TurnTimer(); cb = stt.timing_callbacks(tm)
    cb["on_turn_detection_start"](); cb["on_turn_detection_stop"](); assert not tm.has("speech_end")
    cb["on_turn_detection_start"](); cb["on_recording_stop"]()
    assert cb["on_transcription_start"](b"audio") is None                  # truthy would abort transcription
    assert all(tm.has(s) for s in ("speech_end", "endpoint", "stt_start"))

def test_tts_helpers():
    s = Stream()
    assert tts.output_format(s) == {"sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"}
    assert tts.list_voices(s) == ["af_heart", "am_adam"]; tts.set_voice(s, "am_adam"); assert s.voice == "am_adam"

def test_browser_audio():
    r, s, sent, first = Rec(), Stream(), [], []
    a = BrowserAudio(r, {"sample_rate": 24000}); a.set_sink(sent.append)
    a.on_client_audio(b"\x01\x02"); a.mute_input(True); a.on_client_audio(b"x")
    assert r.fed == [(b"\x01\x02", 16000)]
    a.play(s, lambda: first.append(1))
    assert s.play_kw["muted"] is True and sent == [b"a", b"b"] and first == [1]
    a.set_sink(None); a.play(s, lambda: None)            # no sink: chunks dropped, no error

def test_server_audio():
    r, s, first = Rec(), Stream(), []
    a = ServerAudio(r); a.on_client_audio(b"x"); a.mute_input(True); a.mute_input(False); a.play(s, lambda: first.append(1))
    assert r.fed == [] and r.mic == [False, True] and not s.play_kw.get("muted") and first == [1] and a.output_format is None

def test_make_audio_io_selects_mode():
    calls = []
    f = lambda settings, use_microphone, callbacks=None: calls.append(use_microphone) or Rec()
    assert isinstance(make_audio_io(Settings.load(env={"AUDIO_IO": "browser"}), Stream(), recorder_factory=f), BrowserAudio)
    assert isinstance(make_audio_io(Settings.load(env={"AUDIO_IO": "server"}), Stream(), recorder_factory=f), ServerAudio)
    assert calls == [False, True]
```

- [ ] **Step 2: Run it to see it fail. Step 3: Implement. Step 4: Run it to see it pass** (6 passed).

- [ ] **Step 5: Manual hardware check**, from `server/`:
  `.venv/Scripts/python -c "from voicebot.config import Settings; from voicebot import tts; s=tts.make_tts(Settings.load()); tts.warm_up(s); s.feed('Kokoro is working.'); s.play()"`
  - Expected: the sentence is heard on the PC speakers.
  - If Kokoro complains about espeak-ng, install it (`winget install eSpeak-NG.eSpeak-NG`) and add the step to `scripts/setup_venv.ps1`.

- [ ] **Step 6: Commit** with the message `feat(server): add STT, TTS and audio I/O adapters`.

---

### Task 9: `session.py` — VoiceSession

**Files:**
- Create: `server/voicebot/session.py`
- Test: `server/tests/test_session.py`

**Interfaces:**
- Consumes: `LLMBackend`, `GenerationParams`, `LLMError` (Task 4); `filter_stream` (Task 2); `Conversation` (Task 3); `TurnTimer` (Task 7b); `AudioIO` (Task 8); `set_voice` (Task 8).
- Produces:
  - `is_noise(text: str) -> bool`: true for empty or whitespace text, fewer than 2 letters, or text that becomes `"you"` after lowercasing and stripping punctuation.
  - `APOLOGY = "Sorry, something went wrong."`
  - `class VoiceSession(*, backend, audio, stream, conversation, params: GenerationParams, voice: str, voices: list[str], info: dict, timer: TurnTimer | None = None, metrics_sink: Callable[[dict], None] | None = None)`.
    - `info` holds `llm_backend`, `audio_io` and `model_label`, plus `stt_model`, `stt_device`, `stt_compute_type` and `stt_silence_sec` for the metrics log.
    - `timer` defaults to a new `TurnTimer()`. The real wiring passes the same timer whose callbacks went into the recorder (Task 10).
    - `attach(emit: Callable[[dict], None], send_binary: Callable[[bytes], None]) -> None`: calls `audio.set_sink(send_binary)`.
    - `detach() -> None`: emit becomes a no-op, and calls `audio.set_sink(None)`.
    - `handle(command: dict) -> None`: dispatches on `command["type"]` per spec §4.7. An unknown type emits `error{code:"bad_command"}`.
    - `on_client_audio(data: bytes) -> None`: delegates to `audio`.
    - `config() -> dict`: `{**info, "voices": ..., "output_format": audio.output_format, "defaults": {"system_prompt", "temperature", "top_p", "max_tokens", "voice"}}`, using the current values.
    - `close() -> None`: stops listening, stops speaking, joins threads.
  - **Event dicts emitted** (spec §4.9): `status`, `user_transcript` (`text`, `turn_id`), `assistant_delta`, `assistant_done`, `metrics`, `error` (codes `busy`, `bad_settings`, `bad_command`, `llm_error`).
    - `metrics` = `{"type": "metrics", "turn_id": int, "source": "voice" | "text", "interrupted": bool, **timer.report()}`, which adds `marks` and `spans` (Task 7b).
    - `turn_id` counts up from 1 for each turn the session runs.
  - **Stage marks** (Task 7b table):
    - `stt_done` when `recorder.text()` returns; `input` on `send_text`.
    - `llm_first_token` on the first raw backend chunk (wrap `backend.stream(...)` before `filter_stream`), `first_text` on the first filtered chunk, `llm_done` when the backend iterator ends.
    - `tts_first_audio` in `on_first_audio`. In server mode (`audio.output_format is None`), also `playback_started` at the same moment.
    - `play_done` after `audio.play` returns.
  - **`playback_started` command** `{type, turn_id}` (browser mode): marks `playback_started` only if `turn_id` is the running turn's id. Otherwise it is ignored with no event. After `play` returns in browser mode, the turn waits up to 1.0 s for this ack before emitting `metrics`.
  - **Timer lifecycle:**
    - A typed turn calls `timer.reset()` and then marks `input`.
    - A voice turn keeps the marks the recorder callbacks made while listening.
    - Every turn calls `timer.reset()` in `finally`, after `metrics` is emitted. A skipped noise transcript also resets the timer.
  - **Metrics log:** after emitting `metrics`, call `metrics_sink(record)` if it is set. `record` is the metrics dict without `type`, plus `ts` (ISO 8601 UTC), the `info` fields, the current `voice`, `temperature`, `top_p`, `max_tokens` and `reply_chars`. Sink exceptions are logged and swallowed.
  - **Event order for one turn:** `user_transcript`, `status thinking`, `assistant_delta`×N, `status speaking` (at first audio), `assistant_done`, `metrics`, then `status listening` if the listen loop is active, else `status idle`.
  - **`update_settings` validation:** `temperature` must be a number in [0, 2], `top_p` a number in [0, 1], `max_tokens` an int in [32, 1024], `voice` must be in `voices`, and `system_prompt` must be a str. If anything is invalid, emit `bad_settings` and apply nothing.

- [ ] **Step 1: Write the failing test.** Fakes (in the test file):
  - `FakeRecorder`: `texts` queue. `text()` pops the next item, or blocks on an Event until `abort()`, then returns `""`.
  - `FakeStream`: `feed(x)` stores the iterator. A plain `str` is also stored in `fed_text`. `text()` returns what was consumed; `stop()` sets `stopped`. It has `engine` with a fake `set_voice` that sets `engine.voice`.
  - `FakeAudio`: logs `mute:True`, `play` and `mute:False` to a shared `log`. `play(stream, on_first_audio)` consumes `stream`'s iterator, calls `on_first_audio()` after the first chunk, and stops early if `stream.stopped`.
  - `FakeBackend`: records `(messages, params)` and yields chunks. It can block between chunks on a `threading.Event` (`gate`) or raise `LLMError`.
  - Helpers:
    - `wait_for(pred, timeout=2.0)` polls every 10 ms and returns the first truthy value of `pred()`, or fails.
    - `types(events) -> list[str]`.

```python
def test_typed_turn_event_order_and_history(session, events, conv):
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: types(events)[-1:] == ["status"] and events[-1]["state"] == "idle")
    assert types(events)[:3] == ["user_transcript", "status", "assistant_delta"] and events[1]["state"] == "thinking"
    done = next(e for e in events if e["type"] == "assistant_done")
    assert done == {"type": "assistant_done", "text": "Hello there.", "interrupted": False}
    assert conv.turns == [("Hi", "Hello there.")]
    m = next(e for e in events if e["type"] == "metrics")
    assert m["source"] == "text" and m["turn_id"] == events[0]["turn_id"] == 1 and m["interrupted"] is False
    assert m["spans"]["total_ms"] >= 0 and m["spans"]["llm_ttft_ms"] is not None and m["spans"]["stt_ms"] is None

def test_voice_turn_metrics_and_sink(session_with_texts, sink, events):     # sink fixture: list.append
    session_with_texts.handle({"type": "start_listening"}); wait_for(lambda: sink)
    assert sink[0]["source"] == "voice" and sink[0]["marks"]["stt_done"] is not None
    assert sink[0]["stt_model"] == "small.en" and "ts" in sink[0] and "type" not in sink[0]

def test_playback_ack_only_for_current_turn(gated_session, events, timer):   # Review Focus 6
    gated_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: "user_transcript" in types(events))
    gated_session.handle({"type": "playback_started", "turn_id": 99}); assert not timer.has("playback_started")
    gated_session.handle({"type": "playback_started", "turn_id": events[0]["turn_id"]}); assert timer.has("playback_started")

def test_mute_before_play_unmute_after(session, audio):
    session.handle({"type": "send_text", "text": "Hi"}); wait_for(lambda: "mute:False" in audio.log)
    assert audio.log == ["mute:True", "play", "mute:False"]

def test_voice_turn_and_stop_listening(session_with_texts, recorder, backend):   # recorder.texts = ["What time is it?"]
    session_with_texts.handle({"type": "start_listening"}); wait_for(lambda: backend.calls)
    session_with_texts.handle({"type": "stop_listening"}); assert recorder.aborted

def test_noise_skipped(session_with_texts, backend):          # texts = ["", ".", "you", "a"]
    session_with_texts.handle({"type": "start_listening"}); time.sleep(0.2); assert backend.calls == []

def test_stop_speaking_marks_interrupted_and_keeps_partial(gated_session, backend, stream, conv, events):
    gated_session.handle({"type": "send_text", "text": "Tell me"}); wait_for(lambda: "assistant_delta" in types(events))
    gated_session.handle({"type": "stop_speaking"}); backend.gate.set()
    done = wait_for(lambda: next((e for e in events if e["type"] == "assistant_done"), None))
    assert done["interrupted"] is True and stream.stopped and conv.turns[0][1] == done["text"] != ""

def test_stop_during_thinking_unmutes(gated_session, audio, events):          # Review Focus 2
    gated_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: "mute:True" in audio.log)
    gated_session.handle({"type": "stop_speaking"}); wait_for(lambda: audio.log[-1] == "mute:False")

def test_stop_when_idle_is_noop(session, events):                            # Review Focus 2
    session.handle({"type": "stop_speaking"}); assert events == []

def test_busy(gated_session, events):
    gated_session.handle({"type": "send_text", "text": "a"}); gated_session.handle({"type": "send_text", "text": "b"})
    assert any(e["type"] == "error" and e["code"] == "busy" for e in events)

def test_empty_send_text_ignored(session, backend, events):                  # Review Focus 5
    session.handle({"type": "send_text", "text": "   "}); time.sleep(0.1); assert backend.calls == [] and events == []

def test_llm_error_apologizes_and_recovers(failing_session, stream, events):  # Review Focus 4
    failing_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: events and events[-1].get("state") == "idle")
    assert any(e["type"] == "error" and e["code"] == "llm_error" for e in events) and stream.fed_text == APOLOGY

def test_update_settings_next_turn(session, backend, stream):
    session.handle({"type": "update_settings", "temperature": 0.2, "max_tokens": 100, "system_prompt": "S", "voice": "am_adam"})
    session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: backend.calls)
    msgs, params = backend.calls[0]
    assert params == GenerationParams(0.2, 0.8, 100) and msgs[0]["content"] == "S" and stream.engine.voice == "am_adam"

@pytest.mark.parametrize("patch", [{"temperature": "hot"}, {"max_tokens": 0}, {"voice": "nope"}, {"top_p": 1.5}])
def test_update_settings_rejects_invalid(session, events, patch):            # Review Focus 1
    session.handle({"type": "update_settings", "temperature": 0.1, **patch})
    assert events[-1]["code"] == "bad_settings" and session.config()["defaults"]["temperature"] == 0.7

def test_emit_after_detach_is_safe(session, events):                          # Review Focus 3
    session.detach(); session.handle({"type": "send_text", "text": "Hi"}); time.sleep(0.2); assert events == []

def test_unknown_command_and_clear(session, events, conv):
    conv.add_exchange("u", "a"); session.handle({"type": "clear"}); session.handle({"type": "dance"})
    assert conv.turns == [] and events[-1]["code"] == "bad_command"
```

  Fixtures: `session` (the backend yields `["Hello", " there."]`), `session_with_texts`, `gated_session` (the backend waits on `gate` after its first chunk), `failing_session` (the backend raises `LLMError("down")`). `events` is a list filled by the attached `emit`. `stream.engine.voice` is set by the fake `set_voice`. The default params are `GenerationParams(0.7, 0.8, 256)`.
  - Every session fixture gets a real `TurnTimer` (exposed as the `timer` fixture), `metrics_sink=sink.append` (the `sink` fixture, a list), and an `info` with the Task 1 default STT values.
  - `FakeAudio.output_format` is `{"sample_rate": 24000}` (browser mode). To keep the tests fast, the session's playback-ack wait is a constructor argument `ack_timeout: float = 1.0`, set to `0.05` in the fixtures.

- [ ] **Step 2: Run it to see it fail.**

- [ ] **Step 3: Implement**, following spec §4.7 and the Interfaces above. Approach:
  - `_turn_lock = threading.Lock()`. `send_text` runs `_run_turn` on a daemon thread. If `_turn_lock.acquire(blocking=False)` fails, emit `busy`. The listen loop acquires it blocking.
  - In `_run_turn`, the backend iterator is `filter_stream(marked(backend.stream(conversation.messages(text), params, stop_event)))`, wrapped in a generator that emits `assistant_delta` and marks `first_text`.
    - `marked` is a generator that marks `llm_first_token` on the first chunk and `llm_done` when the iterator ends.
    - Then `stream.feed(...)` and `audio.play(stream, on_first_audio)`.
    - `on_first_audio` emits `status speaking` and marks `tts_first_audio` (and `playback_started` in server mode).
  - The playback ack uses a `threading.Event`. The `playback_started` handler sets it, and the turn waits on it with `ack_timeout`.
  - `stop_speaking` sets the current `stop_event` (if a turn is running) and calls `stream.stop()`.
  - On `LLMError`:
    1. Emit `error{code:"llm_error", message:str(e)}`.
    2. `stream.feed(APOLOGY)` and `audio.play(...)`.
    3. Emit `assistant_done{text: APOLOGY, interrupted: False}`.
    4. Emit no `metrics`, and do not add the exchange to history.
  - `audio.mute_input(False)` goes in `finally`.
  - `emit` and the voice changes are guarded by a lock.

- [ ] **Step 4: Run it to see it pass.** Expected: all pass, 19 tests including the parametrized ones. Run it 3 times to check for flakiness.

- [ ] **Step 5: Commit** with the message `feat(server): add VoiceSession turn loop with events and settings`.

---

### Task 10: `web.py` — FastAPI app

**Files:**
- Create: `server/voicebot/web.py`
- Test: `server/tests/test_web.py`

**Interfaces:**
- Consumes: `Settings`, `make_backend`, `make_tts`, `warm_up`, `list_voices`, `make_audio_io`, `timing_callbacks`, `TurnTimer`, `append_jsonl`, `Conversation`, `GenerationParams`, `VoiceSession`.
- Produces:
  - `create_app(settings: Settings | None = None, *, session_factory: Callable[[Settings], Any] | None = None) -> FastAPI`. The session is created in `lifespan` startup and `close()`d at shutdown.
  - `build_session(settings) -> VoiceSession`, the real wiring:
    1. `backend = make_backend(s); backend.start()`
    2. `stream = make_tts(s); warm_up(stream)`
    3. `timer = TurnTimer(); audio = make_audio_io(s, stream, callbacks=timing_callbacks(timer))`
    4. Construct `VoiceSession` with:
       - `params` from `llm_*`
       - `info` from settings plus `backend.model_label`
       - the same `timer`
       - `metrics_sink=lambda r: append_jsonl(REPO_ROOT / "logs" / "turns.jsonl", r)`
  - `main()` runs `uvicorn` on `web_host:web_port`. `if __name__ == "__main__": main()`, so `python -m voicebot.web` works.
  - Routes:
    - `GET /api/config` → `session.config()`.
    - `WS /ws`:
      - If a client is already connected: accept, send `{"type":"error","code":"session_busy","message":...}`, close with code 1013.
      - Otherwise: accept and `session.attach(emit, send_binary)`. Both callbacks use `loop.call_soon_threadsafe(queue.put_nowait, item)`, and a sender task drains the queue (dict → `send_json`, bytes → `send_bytes`).
      - Send `{"type":"config", **session.config()}`.
      - Receive loop: text frames are parsed as JSON and passed to `session.handle(...)` (invalid JSON → `error bad_command`); bytes frames go to `session.on_client_audio(...)`.
      - On disconnect, in `finally`: `handle({"type":"stop_listening"})`, `handle({"type":"stop_speaking"})`, `detach()`, clear the connected flag, cancel the sender.
  - `web/dist` is mounted with `StaticFiles(html=True)` at `/` (after the routes) only if it exists.

- [ ] **Step 1: Write the failing test**

```python
import threading, pytest
from fastapi.testclient import TestClient
from voicebot.config import Settings
from voicebot.web import create_app

class FakeSession:
    def __init__(self): self.handled, self.audio, self.emit, self.send_binary, self.detached, self.closed = [], [], None, None, False, False
    def attach(self, emit, send_binary): self.emit, self.send_binary, self.detached = emit, send_binary, False
    def detach(self): self.detached = True
    def handle(self, cmd): self.handled.append(cmd)
    def on_client_audio(self, data): self.audio.append(data)
    def config(self): return {"llm_backend": "llamacpp", "audio_io": "browser", "model_label": "m", "voices": ["af_heart"],
                              "output_format": None, "defaults": {}}
    def close(self): self.closed = True

@pytest.fixture
def ctx():
    fake = FakeSession()
    with TestClient(create_app(Settings.load(env={}), session_factory=lambda s: fake)) as client:
        yield client, fake
    assert fake.closed

def test_config_endpoint(ctx):
    client, _ = ctx; assert client.get("/api/config").json()["model_label"] == "m"

def test_ws_config_commands_binary(ctx):
    client, fake = ctx
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "config"
        ws.send_json({"type": "send_text", "text": "hi"}); ws.send_bytes(b"\x00\x01"); ws.send_text("{bad")
        assert ws.receive_json()["code"] == "bad_command"
        assert {"type": "send_text", "text": "hi"} in fake.handled and fake.audio == [b"\x00\x01"]

def test_events_from_worker_thread(ctx):
    client, fake = ctx
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        t = threading.Thread(target=lambda: (fake.emit({"type": "status", "state": "idle"}), fake.send_binary(b"pcm"))); t.start(); t.join()
        assert ws.receive_json() == {"type": "status", "state": "idle"} and ws.receive_bytes() == b"pcm"

def test_second_client_busy(ctx):
    client, _ = ctx
    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json()
        with client.websocket_connect("/ws") as ws2:
            assert ws2.receive_json()["code"] == "session_busy"

def test_disconnect_detaches_and_stops(ctx):                 # Review Focus 3
    client, fake = ctx
    with client.websocket_connect("/ws") as ws: ws.receive_json()
    assert fake.detached and {"type": "stop_listening"} in fake.handled and {"type": "stop_speaking"} in fake.handled
```

- [ ] **Step 2: Run it to see it fail. Step 3: Implement. Step 4: Run it to see it pass** (5 passed). Then run the full suite with `../.venv/Scripts/python -m pytest -v`. Everything should pass.
- [ ] **Step 5: Commit** with the message `feat(server): add FastAPI app with single-client WebSocket`.

---

### Task 11: Frontend scaffold — protocol, state, PCM

**Files:**
- Create: `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/index.html`, `web/src/main.tsx` (renders a placeholder `<div>` until Task 12), `web/src/protocol.ts`, `web/src/state.ts`, `web/src/audio/pcm.ts`
- Test: `web/src/state.test.ts`, `web/src/audio/pcm.test.ts`

**Interfaces:**
- Produces:
  - **`protocol.ts`**, mirroring spec §4.9:

```ts
export type Status = "idle" | "listening" | "thinking" | "speaking";
export interface OutputFormat { sample_rate: number; channels: number; encoding: "pcm_s16le" }
export interface Defaults { system_prompt: string; temperature: number; top_p: number; max_tokens: number; voice: string }
export interface ServerConfig { llm_backend: "transformers" | "llamacpp"; audio_io: "browser" | "server"; model_label: string;
  voices: string[]; output_format: OutputFormat | null; defaults: Defaults }
export const SPANS = ["silence_wait_ms", "stt_ms", "llm_ttft_ms", "filter_ms", "tts_first_audio_ms",
  "playback_ms", "perceived_ms", "llm_total_ms", "total_ms"] as const;
export type SpanName = (typeof SPANS)[number];
export interface Metrics { turn_id: number; source: "voice" | "text"; interrupted: boolean;
  marks: Record<string, number | null>; spans: Record<SpanName, number | null> }
export type ServerEvent = ({ type: "config" } & ServerConfig) | { type: "status"; state: Status }
  | { type: "user_transcript"; text: string; turn_id: number }
  | { type: "assistant_delta"; text: string } | { type: "assistant_done"; text: string; interrupted: boolean }
  | ({ type: "metrics" } & Metrics) | { type: "error"; code: string; message: string };
export type ClientCommand = { type: "start_listening" } | { type: "stop_listening" } | { type: "stop_speaking" }
  | { type: "send_text"; text: string } | ({ type: "update_settings" } & Partial<Defaults>) | { type: "clear" }
  | { type: "playback_started"; turn_id: number };
```

  - **`state.ts`**:
    - `interface ChatMessage { id: number; role: "user" | "assistant"; text: string; streaming: boolean; interrupted: boolean }`
    - `interface AppState { connected: boolean; listening: boolean; status: Status; config: ServerConfig | null; settings: Defaults | null; messages: ChatMessage[]; metrics: Metrics | null; turnId: number | null; error: string | null }`. `user_transcript` sets `turnId`.
    - `type Action = { kind: "server"; event: ServerEvent } | { kind: "connected"; value: boolean } | { kind: "listening"; value: boolean } | { kind: "settings"; patch: Partial<Defaults> } | { kind: "error"; message: string } | { kind: "dismiss_error" } | { kind: "clear" }`
    - `initialState: AppState`; `reducer(state: AppState, action: Action): AppState`
  - **`pcm.ts`**:
    - `floatTo16BitPCM(f: Float32Array): Int16Array`: clamps to [-1, 1], uses `s < 0 ? s * 32768 : s * 32767`, then `Math.round`.
    - `resample(input: Float32Array, inRate: number, outRate: number): Float32Array`: linear interpolation, output length `Math.round(input.length * outRate / inRate)`, and returns a copy when the rates are equal.
    - `pcm16ToFloat(buf: ArrayBuffer): Float32Array`: divides by 32768.

- [ ] **Step 1: Scaffold the project.**
  - `package.json` scripts: `"dev": "vite"`, `"build": "tsc --noEmit && vite build"`, `"typecheck": "tsc --noEmit"`, `"test": "vitest run"`.
  - Dependencies: `react@^18`, `react-dom@^18`.
  - Dev dependencies: `vite`, `@vitejs/plugin-react`, `typescript`, `vitest`, `@types/react`, `@types/react-dom`, `@types/audioworklet`.
  - `tsconfig.json`: `"strict": true`, `"jsx": "react-jsx"`, `"lib": ["ES2022", "DOM"]`, `"moduleResolution": "bundler"`.
  - `vite.config.ts`: the react plugin, plus `server.proxy`: `"/api": "http://127.0.0.1:8000"` and `"/ws": { target: "ws://127.0.0.1:8000", ws: true }`.
  - Then run `npm install` in `web/`.

- [ ] **Step 2: Write the failing tests**

```ts
// pcm.test.ts
import { describe, it, expect } from "vitest";
import { floatTo16BitPCM, resample, pcm16ToFloat } from "./pcm";
describe("pcm", () => {
  it("converts and clamps", () => {
    expect(Array.from(floatTo16BitPCM(new Float32Array([0, 1, -1, 2, -2, 0.5])))).toEqual([0, 32767, -32768, 32767, -32768, 16384]);
  });
  it("resamples 48k to 16k", () => {
    const out = resample(new Float32Array(480).fill(0.25), 48000, 16000);
    expect(out.length).toBe(160); expect(out.every(v => Math.abs(v - 0.25) < 1e-6)).toBe(true);
  });
  it("same rate copies", () => { const a = new Float32Array([0.1, 0.2]); const b = resample(a, 16000, 16000); expect(b).toEqual(a); expect(b).not.toBe(a); });
  it("pcm16 to float", () => { expect(Array.from(pcm16ToFloat(new Int16Array([0, 16384, -32768]).buffer))).toEqual([0, 0.5, -1]); });
});
```

```ts
// state.test.ts — one test per rule
import { describe, it, expect } from "vitest";
import { reducer, initialState } from "./state";
const cfg = { type: "config", llm_backend: "llamacpp", audio_io: "browser", model_label: "m", voices: ["af_heart"], output_format: null,
  defaults: { system_prompt: "S", temperature: 0.7, top_p: 0.8, max_tokens: 256, voice: "af_heart" } } as const;
const srv = (s: any, event: any) => reducer(s, { kind: "server", event });
describe("reducer", () => {
  it("config sets config and settings once", () => {
    const s = srv(initialState, cfg); expect(s.settings?.temperature).toBe(0.7);
    const s2 = srv(reducer(s, { kind: "settings", patch: { temperature: 0.1 } }), cfg); expect(s2.settings?.temperature).toBe(0.1);
  });
  it("streams assistant deltas into one message and finalizes", () => {
    let s = srv(initialState, { type: "user_transcript", text: "hi", turn_id: 3 }); expect(s.turnId).toBe(3);
    s = srv(s, { type: "assistant_delta", text: "Hel" }); s = srv(s, { type: "assistant_delta", text: "lo" });
    expect(s.messages.map(m => [m.role, m.text, m.streaming])).toEqual([["user", "hi", false], ["assistant", "Hello", true]]);
    s = srv(s, { type: "assistant_done", text: "Hello", interrupted: true });
    expect(s.messages[1]).toMatchObject({ streaming: false, interrupted: true });
  });
  it("assistant_done without deltas adds a message if text non-empty", () => {
    expect(srv(initialState, { type: "assistant_done", text: "Sorry.", interrupted: false }).messages).toHaveLength(1);
    expect(srv(initialState, { type: "assistant_done", text: "", interrupted: false }).messages).toHaveLength(0);
  });
  it("status, metrics, error, clear, dismiss", () => {
    let s = srv(initialState, { type: "status", state: "thinking" }); expect(s.status).toBe("thinking");
    s = srv(s, { type: "metrics", turn_id: 1, source: "text", interrupted: false, marks: {},
      spans: { silence_wait_ms: null, stt_ms: null, llm_ttft_ms: 300, filter_ms: 0, tts_first_audio_ms: 200,
               playback_ms: 30, perceived_ms: 530, llm_total_ms: 900, total_ms: 2000 } });
    expect(s.metrics?.spans.perceived_ms).toBe(530);
    s = srv(s, { type: "error", code: "busy", message: "Busy" }); expect(s.error).toBe("Busy");
    s = reducer(reducer(s, { kind: "dismiss_error" }), { kind: "clear" }); expect(s.error).toBeNull(); expect(s.messages).toEqual([]);
  });
});
```

- [ ] **Step 3: Run `npm run test` to see it fail. Step 4: Implement. Step 5: Run `npm run test` (all pass) and `npm run typecheck` (no errors).**
- [ ] **Step 6: Commit** `web/` without `node_modules` (it's already in `.gitignore`), with the message `feat(web): scaffold Vite React TS app with protocol, state and PCM utils`.

---

### Task 12: Frontend audio, WebSocket client and playground UI

**Files:**
- Create: `web/src/ws.ts`, `web/src/audio/mic.ts`, `web/src/audio/mic-worklet.ts`, `web/src/audio/player.ts`, `web/src/audio/player-worklet.ts`, `web/src/components/{SettingsPanel,ChatTranscript,Composer,StatusBar}.tsx`, `web/src/App.tsx`, `web/src/styles.css`
- Modify: `web/src/main.tsx` (render `<App/>`)
- Test: `web/src/ws.test.ts`

**Interfaces:**
- Consumes: `protocol.ts`, `state.ts`, `pcm.ts` (Task 11).
- Produces:
  - **`ws.ts`**:
    - `nextDelay(attempt: number): number` = `Math.min(500 * 2 ** attempt, 8000)`
    - `class VoiceSocket(url: string, handlers: { onEvent(e: ServerEvent): void; onAudio(buf: ArrayBuffer): void; onOpen(): void; onClose(): void })` with `send(cmd: ClientCommand)`, `sendAudio(pcm: Int16Array)` and `close()`. It uses `binaryType = "arraybuffer"`, reconnects with `nextDelay` until `close()` is called, and resets the attempt count on open.
    - URL: `` `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws` ``.
  - **`mic.ts`**: `startMic(onFrame: (pcm: Int16Array) => void): Promise<{ stop(): void }>`.
    - `getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } })`.
    - The worklet is loaded with `import url from "./mic-worklet.ts?worker&url"`.
    - The worklet resamples from `sampleRate` to 16000 using `resample` and `floatTo16BitPCM`, buffers 640 samples (40 ms), and posts each `Int16Array` with a transfer.
  - **`player.ts`**: `createPlayer(format: OutputFormat, onStarted: () => void): { push(buf: ArrayBuffer): void; flush(): void; close(): void }`.
    - `push` converts with `pcm16ToFloat` and resamples to `ctx.sampleRate`, then posts the result to the player worklet.
    - The worklet keeps a FIFO of Float32 chunks and fills each `process()` output from it, writing silence when empty. A `"flush"` message clears the FIFO.
    - When `process()` outputs samples after the FIFO was empty, the worklet posts `"started"` to the main thread, which calls `onStarted()`. The device output latency (`ctx.outputLatency`, typically 10–40 ms) is not included.
  - **Components:**
    - `SettingsPanel({ config, settings, onChange(patch: Partial<Defaults>) })`: badges for `llm_backend` / `audio_io` / `model_label`; system prompt textarea; temperature 0–2 (step 0.05), top_p 0–1 (0.05), max tokens 32–1024 (step 16); voice `<select>`.
    - `ChatTranscript({ messages })`: auto-scrolls to the bottom. Streaming messages show a cursor; interrupted ones show an "interrupted" tag.
    - `Composer({ status, listening, connected, onToggleMic(), onSend(text), onStop(), onClear() })`: Enter sends and Shift+Enter adds a newline. Stop is enabled while `thinking` or `speaking`.
    - `StatusBar({ connected, status, metrics, error, onDismissError() })`
      - It shows `perceived_ms` prominently as "Response: 1340 ms".
      - Below that is one breakdown line built from `spans`: `silence 600 · stt 200 · llm 300 · filter 20 · tts 180 · play 40`, skipping null spans.
      - The full `marks` and `spans` are in a hover tooltip.
  - **`App.tsx`:**
    - `useReducer(reducer, initialState)`; one `VoiceSocket` in a `useEffect`.
    - Settings changes dispatch locally at once, and `update_settings` is sent after a 300 ms debounce.
    - **Mic toggle in browser mode:** `startMic(pcm => socket.sendAudio(pcm))`, then `start_listening`. If permission is refused, dispatch `{kind:"error", message:"Microphone permission denied — typing still works."}`.
    - **Mic toggle in server mode:** send `start_listening` / `stop_listening` only.
    - `onAudio` calls `player.push` when `config.output_format` is set. The player is created on the first config event.
    - The player's `onStarted` sends `{type:"playback_started", turn_id: state.turnId}` once per turn. Keep the last acked id in a ref, and skip the ack when `turnId` is null or already acked.
    - Stop sends `stop_speaking` and calls `player.flush()`. Clear sends `clear`, dispatches clear, and flushes.
  - **`styles.css`:**
    - Color tokens in `:root` with a dark theme.
    - Grid of a 300px sidebar plus the main column; below 800px they stack.
    - Body background set explicitly. No external fonts or branding.

- [ ] **Step 1: Write the failing test**

```ts
// ws.test.ts
import { it, expect } from "vitest";
import { nextDelay } from "./ws";
it("backs off exponentially with cap", () => {
  expect([0, 1, 2, 3, 4, 5].map(nextDelay)).toEqual([500, 1000, 2000, 4000, 8000, 8000]);
});
```

- [ ] **Step 2: Run `npm run test` to see it fail. Step 3: Implement all files listed. Step 4: Run `npm run test`, `npm run typecheck` and `npm run build`. All three must succeed.**
- [ ] **Step 5: Commit** with the message `feat(web): add playground UI, mic capture, audio playback and socket client`.

---

### Task 13: `dev.ps1`, README, end-to-end checks

**Files:**
- Create: `scripts/dev.ps1`, `README.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write `scripts/dev.ps1`.**
  - Start the API in a new window: `Start-Process` running `.venv\Scripts\python -m voicebot.web` with working dir `server`.
  - Then run `npm --prefix web run dev` in the current window.
  - Print `Open http://localhost:5173`.

- [ ] **Step 2: Write `README.md`.** Sections:
  - What it is (2 lines)
  - Requirements: Windows, NVIDIA GPU with 8 GB, uv, Node 20+, gitleaks
  - Setup: the spec §8 steps, including `scripts/setup_venv.ps1`, which also enables the gitleaks hooks, and `scripts/fetch_llama_server.ps1`
  - The `.env` switches (`LLM_BACKEND`, `AUDIO_IO`), with a pointer to `.env.example`
  - Run: dev, and production (`npm --prefix web run build`, then `python -m voicebot.web`, then http://127.0.0.1:8000)
  - Troubleshooting: the spec §6 table, condensed
  - The vendored code note, linking `VENDORED.md`

- [ ] **Step 3: End-to-end matrix.** For each `LLM_BACKEND` × `AUDIO_IO` combination (4 runs), restart via `scripts/dev.ps1`, then:
  1. Ask "What's the capital of France?" by voice.
  2. Ask a typed follow-up, "And its population?".
  3. Press Stop during a long answer ("Tell me a long story").
  4. Change the voice and the temperature, then ask again.

  Pass criteria:
  - Spoken answers are heard in the right place (browser or PC speakers).
  - The transcript streams.
  - No `<think>` text appears.
  - Stop cuts the audio within about 0.5 s and the reply is tagged "interrupted".
  - The follow-up uses context.
  - The StatusBar shows the response time and the stage breakdown after each turn.
  - `logs/turns.jsonl` gains one line per turn.

- [ ] **Step 4: Latency baseline.** For each `LLM_BACKEND` × `AUDIO_IO` combination, ask 10 short spoken questions (the same 10 for every run, listed in the README). Then run, from `server/`:
  `../.venv/Scripts/python -m voicebot.latency_report --by llm_backend,audio_io --source voice`
  - Pass criterion: the median `perceived_ms` with `llamacpp` is about 2000 or less.
  - Paste the output into a "Measured latency" section of the README, with the date, GPU and `.env` values.
  - Note which span is the largest. That tells you where to optimise next.

- [ ] **Step 5: Production check.** Run `npm --prefix web run build`, then start only `python -m voicebot.web` and open http://127.0.0.1:8000. The UI loads and one voice turn works.

- [ ] **Step 6: Final verification.** All of these must pass: `../.venv/Scripts/python -m pytest -v` from `server/`, plus `npm run test`, `npm run typecheck` and `npm run build` from `web/`.

- [ ] **Step 7: Commit** with the message `docs: add README, dev script and measured latency`.
