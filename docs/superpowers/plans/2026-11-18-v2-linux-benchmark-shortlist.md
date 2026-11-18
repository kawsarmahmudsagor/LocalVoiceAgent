# Local Voice Bot v2 — Linux VM, Benchmark Pool and Switchable Shortlist — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the v1 voice bot on an Ubuntu 24.04 VM with an RTX 5090, benchmark a curated pool of STT and TTS engines with the repos' own tools, then make the 3–5 STT and 2–3 TTS winners switchable from the web UI.

**Architecture:** There are three parts.

- **Part 1: Linux port of v1.** Settings, llama-server in Docker, HTTPS option, a VM check script and Linux shell scripts.
- **Part 2: Phase A.** A catalog schema shared by the pool and the shortlist, per-family bench venvs, clip recording, runners that drive `stt-server-production` with the repo's benchmark scripts, a TTS timing script, and a matrix with the shortlist rule.
- **Part 3: Phase B.**
  - A `ModelManager` builds and swaps the recorder and TTS stream in-process.
  - The session gains `select_model` and `user_partial`, and the UI gains pickers.
  - Worker processes are built only if the dependency check in Task 11 says a shortlisted engine cannot share the main venv.

**Tech Stack:** Python 3.12 (uv), FastAPI, RealtimeSTT, RealtimeTTS, Docker (`ghcr.io/ggml-org/llama.cpp:server-cuda`), torch 2.7.1+cu128, pytest; React 18 + TypeScript + vitest (from v1).

**Spec:** `docs/superpowers/specs/2026-11-18-v2-model-switching-design.md`. Read it before starting any task. Section numbers below (§) refer to it. The v1 plan `docs/superpowers/plans/2026-11-18-local-voice-bot.md` is the base this plan modifies.

## Global Constraints

- **Start condition:** every v1 plan task (1–13, including 7b) is complete and committed on `feat/voice-bot` before Task 1 here starts.
- **Target:** Ubuntu 24.04, glibc 2.39, RTX 5090 (`sm_120`), driver ≥ 570, CUDA ≥ 12.8, Docker with the NVIDIA Container Toolkit. Windows support for v2 is out of scope, but v1's Windows unit tests must still pass.
- **Python:** 3.12 everywhere, via `uv`. `.venv` is at the repo root.
- **Torch:** exactly `torch==2.7.1+cu128` from `https://download.pytorch.org/whl/cu128` in every venv that has torch. A pool entry that needs a different torch version is dropped (§3.1).
- **No host CUDA compiling:** never use the host `nvcc` (12.0). CUDA code is built only inside containers.
- **LLM and audio on the VM:** `LLM_BACKEND=llamacpp` and `AUDIO_IO=browser` only.
- **Disk:** the total for v2 stays at about 35 GB (§3.2).
- **v1 Task 7b interfaces:** keep `TurnTimer`, `timing_callbacks(timer)`, `append_jsonl`, the `metrics` event shape and `latency_report` CLI unchanged. Only *add* record fields.
- **Vendored code:** do not edit `RealtimeSTT/` or `RealtimeTTS/`.
- **Tests:** unit tests run without a GPU, Docker, network or models.
  - Python, from `server/`: `../.venv/bin/python -m pytest <path> -v`. On Windows, `../.venv/Scripts/python`.
  - Frontend, from `web/`: `npm run test`, `npm run typecheck`, `npm run build`.
- **Commits:**
  - Stage files by explicit path.
  - Gitleaks hooks are active; never use `--no-verify`.
  - Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Repo-tool facts used below (verified):**
  - `stt-server-production` flags: `--host --port --engine --model --device --engine-options --realtime-engine --realtime-model --realtime-engine-options`. Routes: `GET /health`, `POST /transcribe-pcm16`, `WS /api/v1/ws`.
  - `RealtimeSTT/tools/benchmarks/benchmark_asr_ab.py --manifest M --target LABEL=URL --output OUT.json [--repetitions N]`.
  - `RealtimeSTT/tools/benchmarks/benchmark_asr_streaming.py --url BASE --manifest M --output OUT.json [--pace 1.0] [--chunk-ms 100]`.
  - Both tools compute WER and percentiles themselves. Their manifest is a JSON list of `{"wav": path, "text": reference, "id": str}`.

## Review Focus

1. **`select_model` while listening but idle.** The listen loop is blocked in `recorder.text()`. The switch must abort it, swap the recorder and restart listening, with no lost or duplicate turn. Task 13.
2. **The NVIDIA "Driver/library version mismatch" output.** `check_vm.sh` must say "reboot the VM", not show a parse error. Task 3.
3. **A stale `voicebot-llama` container from a crashed run.** It is removed before `docker run`, so start never fails with "name already in use". Task 2.
4. **A TTS switch that changes the sample rate mid-session.** The server sends a new `config` with the new `output_format`, and the browser rebuilds its player. Audio is never played at the wrong rate. Tasks 13 and 15.
5. **Clip count not matching the prompt count when building the manifest.** Stop with both counts, and do not write a partial manifest. Task 7.

---

## File Structure

```
server/voicebot/config.py                 T1, T12 (modify)
server/voicebot/web.py                    T1, T7, T14 (modify)
server/voicebot/llm/llama_server.py       T2 (modify)
server/voicebot/vmcheck.py                T3 (stdlib only; runs before any venv exists)
scripts/check_vm.sh                       T3
scripts/install_system_deps.sh  setup_venv.sh  dev.sh  make_cert.sh  build_llama_image.sh   T4
server/voicebot/catalog.py                T5 (stdlib only)
bench/pool.toml                           T5
bench/families/<family>.txt               T6
server/voicebot/bench/__init__.py  selftest.py  fetch.py        T6
scripts/bench_install.sh  bench_fetch.sh  T6
server/voicebot/bench/clips.py  manifest.py  ref_voice.py       T7
bench/prompts.txt  bench/sentences.txt  assets/ref_voice.txt    T7
server/voicebot/bench/gpu.py  run_stt.py  run_tts.py            T8
server/voicebot/bench/matrix.py           T9
docs/v2-model-matrix.md                   T10 (generated, then committed)
server/voicebot/bench/resolve.py          T11
server/voicebot/catalog.toml              T11
scripts/bench_clean.sh                    T11
server/voicebot/stt.py  tts.py            T12 (modify, additive)
server/voicebot/models.py                 T13
server/voicebot/session.py  audio_io.py   T13 (modify)
web/src/protocol.ts state.ts components/* audio/player.ts App.tsx   T15 (modify)
worker/…  server/voicebot/workers/…       T16a/T16b (only if Task 11 finds workers)
README.md                                 T4, T17 (modify)
```

---

# Part 1 — Linux port of v1

### Task 1: Linux settings and HTTPS option

**Files:**
- Modify: `server/voicebot/config.py`, `server/voicebot/web.py`, `.env.example`
- Test: `server/tests/test_config.py`, `server/tests/test_web.py` (extend)

**Interfaces:**
- Produces new `Settings` fields:
  - `llama_runtime: str`. Allowed values `docker` / `binary`. Default `docker` when `platform.startswith("linux")`, else `binary`.
  - `llama_image: str`, default `ghcr.io/ggml-org/llama.cpp:server-cuda`.
  - `web_ssl_certfile: Path | None` and `web_ssl_keyfile: Path | None`, from `WEB_SSL_CERTFILE` / `WEB_SSL_KEYFILE`. Empty means `None`. Relative paths resolve against `REPO_ROOT`.
- `Settings.load(env=None, env_file=None, *, platform: str = sys.platform)`. Rules:
  - With `platform` starting with `linux`, `AUDIO_IO=server` raises `ConfigError` whose message contains `server audio is not supported on the VM; use browser mode`.
  - Setting only one of the two SSL variables raises `ConfigError` naming the missing one.
- `web.main()` passes `ssl_certfile` and `ssl_keyfile` to `uvicorn.run` only when both are set.

- [ ] **Step 1: Write the failing tests**

```python
def test_linux_defaults_and_server_audio_rejected():
    s = Settings.load(env={}, platform="linux")
    assert s.llama_runtime == "docker" and s.llama_image == "ghcr.io/ggml-org/llama.cpp:server-cuda"
    assert Settings.load(env={}, platform="win32").llama_runtime == "binary"
    with pytest.raises(ConfigError, match="server audio is not supported on the VM"):
        Settings.load(env={"AUDIO_IO": "server"}, platform="linux")

def test_ssl_pair_required():
    with pytest.raises(ConfigError, match="WEB_SSL_KEYFILE"):
        Settings.load(env={"WEB_SSL_CERTFILE": "certs/cert.pem"}, platform="linux")
    s = Settings.load(env={"WEB_SSL_CERTFILE": "certs/cert.pem", "WEB_SSL_KEYFILE": "certs/key.pem"}, platform="linux")
    assert s.web_ssl_certfile.name == "cert.pem" and s.web_ssl_certfile.is_absolute()
```

```python
# test_web.py
def test_main_passes_ssl(monkeypatch):
    seen = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr("voicebot.web.Settings.load", classmethod(lambda cls, **k: Settings.load(
        env={"WEB_SSL_CERTFILE": "c.pem", "WEB_SSL_KEYFILE": "k.pem"}, platform="linux")))
    web.main(); assert seen["ssl_certfile"].endswith("c.pem") and seen["ssl_keyfile"].endswith("k.pem")
```

- [ ] **Step 2: Run them to see them fail. Step 3: Implement. Step 4: Run the full suite** (`../.venv/bin/python -m pytest -v`) to confirm v1 tests still pass.
- [ ] **Step 5:** Add the new variables to `.env.example` with one-line comments, and commit with `feat(server): add Linux runtime settings and optional HTTPS`.

---

### Task 2: llama-server in Docker

**Files:**
- Modify: `server/voicebot/llm/llama_server.py`
- Test: `server/tests/test_llama_server.py` (extend)

**Interfaces:**
- `CONTAINER = "voicebot-llama"` and `SM120_HINT = "no kernel image is available"` (module constants).
- `LlamaServer(settings, *, popen=subprocess.Popen, status=http_status, sleep=time.sleep, clock=..., log_path=..., run=subprocess.run, which=shutil.which, home: Path = Path.home())`. The existing parameters are unchanged.
- `command() -> list[str]`:
  - With `llama_runtime == "binary"`: unchanged from v1.
  - With `docker`:
    ```
    ["docker","run","--rm","--name",CONTAINER,"--gpus","all","-p",f"127.0.0.1:{port}:8080",
     "-v",f"{home}/.cache/llama.cpp:/root/.cache/llama.cpp", settings.llama_image,
     "-hf",repo,"--no-mmproj","--jinja","-ngl","99","-c",str(ctx),"--host","0.0.0.0","--port","8080"]
    ```
- `start()` in Docker mode:
  1. If `which("docker")` is None, raise `LlamaServerError("docker not found …")`.
  2. Run `run(["docker","rm","-f",CONTAINER], capture_output=True)` and ignore the result.
  3. Then do the v1 port check, popen with stdout/stderr to `log_path`, and health polling.
  4. On failure, if the log tail contains `SM120_HINT`, the error message adds `run scripts/build_llama_image.sh and set LLAMA_IMAGE=voicebot/llama-server:cuda12.8`.
- `close()` in Docker mode runs `run(["docker","stop","-t","5",CONTAINER])`, then waits for the popen process (up to 10 s) and kills it if it is still alive.

- [ ] **Step 1: Write the failing tests** (fakes for `popen`, `run`, `which`, `status`):

```python
def docker_settings(**env): return Settings.load(env={"LLAMA_RUNTIME": "docker", **env}, platform="linux")

def test_docker_command(tmp_path):
    srv = LlamaServer(docker_settings(), home=tmp_path)
    cmd = srv.command()
    assert cmd[:5] == ["docker", "run", "--rm", "--name", "voicebot-llama"] and "127.0.0.1:8081:8080" in cmd
    assert f"{tmp_path}/.cache/llama.cpp:/root/.cache/llama.cpp" in cmd
    assert cmd[cmd.index("-hf") + 1] == "unsloth/Qwen3.5-9B-GGUF:Q4_K_M" and "--no-mmproj" in cmd

def test_stale_container_removed_before_run(tmp_path):                    # Review Focus 3
    calls = []; fake_run = lambda c, **k: calls.append(c)
    srv = LlamaServer(docker_settings(), run=fake_run, which=lambda n: "/usr/bin/docker",
                      popen=FakePopen(), status=lambda u, **k: 200, log_path=tmp_path / "l.log", home=tmp_path)
    srv.start(); assert calls[0] == ["docker", "rm", "-f", "voicebot-llama"]
    srv.close(); assert ["docker", "stop", "-t", "5", "voicebot-llama"] in calls

def test_missing_docker():
    with pytest.raises(LlamaServerError, match="docker not found"):
        LlamaServer(docker_settings(), which=lambda n: None).start()

def test_sm120_hint(tmp_path):
    log = tmp_path / "l.log"
    srv = LlamaServer(docker_settings(), run=lambda c, **k: None, which=lambda n: "x", log_path=log, home=tmp_path,
                      popen=FakePopen(exit_code=1, writes="CUDA error: no kernel image is available"), status=lambda u, **k: None)
    with pytest.raises(LlamaServerError, match="build_llama_image.sh"): srv.start()
```

`FakePopen` reuses the v1 test helper. Extend it so it writes `writes` to the stdout file it was given and reports `exit_code` from `poll()`.

- [ ] **Step 2: Run the tests to see them fail. Step 3: Implement. Step 4: Run the tests** (`test_llama_server.py`: all v1 tests and the 4 new ones pass).
- [ ] **Step 5: Commit** with `feat(server): run llama-server from the CUDA Docker image on Linux`.

---

### Task 3: VM check

**Files:**
- Create: `server/voicebot/vmcheck.py` (stdlib only, no package-relative imports), `scripts/check_vm.sh`
- Test: `server/tests/test_vmcheck.py`

**Interfaces:**
- `check_gpu(query_rc: int, query_out: str, header_out: str) -> list[str]`
  - `query_out` comes from `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`. `header_out` is plain `nvidia-smi` output.
  - It returns problem strings:
    - Output containing `Driver/library version mismatch` → `"NVIDIA driver/library version mismatch: reboot the VM (sudo reboot), then rerun."`
    - Any other non-zero `query_rc` → `"nvidia-smi failed: <first line>"`
    - Driver major version < 570 → `"driver <v> is older than 570"`
    - `CUDA Version: X.Y` below 12.8 → `"CUDA <v> is older than 12.8"`
    - A name without `5090` → `"expected an RTX 5090, found <name>"`
- `check_disk(free_bytes: int, need_gb: int = 45) -> list[str]`
- `main() -> int`. It runs:
  - the two `nvidia-smi` calls
  - `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`; a failure gives the problem `"docker GPU access failed: <last stderr line>"`
  - `shutil.which("uv")`
  - `shutil.disk_usage(REPO_ROOT)`

  It prints `OK` or each problem, and returns 1 if there are any problems.
- `scripts/check_vm.sh` runs `exec python3 "$(dirname "$0")/../server/voicebot/vmcheck.py" "$@"`.

- [ ] **Step 1: Write the failing tests**

```python
from voicebot.vmcheck import check_gpu, check_disk
HEADER = "| NVIDIA-SMI 580.178   Driver Version: 580.178   CUDA Version: 13.0 |"

def test_mismatch_says_reboot():                                         # Review Focus 2
    p = check_gpu(18, "Failed to initialize NVML: Driver/library version mismatch\nNVML library version: 580.178", "")
    assert p == ["NVIDIA driver/library version mismatch: reboot the VM (sudo reboot), then rerun."]

def test_healthy_5090(): assert check_gpu(0, "NVIDIA GeForce RTX 5090, 580.178", HEADER) == []

def test_old_driver_cuda_and_wrong_gpu():
    p = check_gpu(0, "NVIDIA GeForce RTX 4060, 550.54", HEADER.replace("13.0", "12.4"))
    assert any("older than 570" in x for x in p) and any("older than 12.8" in x for x in p) and any("RTX 5090" in x for x in p)

def test_disk(): assert check_disk(44 * 1024**3) and not check_disk(46 * 1024**3)
```

- [ ] **Step 2: Run them to see them fail. Step 3: Implement. Step 4: Run them to see them pass** (4 passed).
- [ ] **Step 5: Commit** with `feat(scripts): add VM prerequisite check`.

---

### Task 4: Linux scripts and first VM run

**Files:**
- Create:
  - `scripts/install_system_deps.sh`
  - `scripts/setup_venv.sh`
  - `scripts/dev.sh`
  - `scripts/make_cert.sh`
  - `scripts/build_llama_image.sh`
- Modify: `README.md` (add a "Linux VM" section)

Every script starts with `#!/usr/bin/env bash` and `set -euo pipefail`, and resolves paths from `$(dirname "$0")/..`.

- **`install_system_deps.sh`:** `sudo apt-get update && sudo apt-get install -y espeak-ng ffmpeg portaudio19-dev libasound2-dev build-essential`.
- **`setup_venv.sh`:**
  1. `uv venv -p 3.12 .venv`
  2. `uv pip install --python .venv -r RealtimeSTT/requirements-gpu-torch.txt`
  3. `uv pip install --python .venv -e RealtimeSTT -e "RealtimeTTS[kokoro]" -e "server[dev]"`
  4. `git config core.hooksPath .githooks` (as in v1)
  5. `npm --prefix web install`

  Do not install the `transformers` extra; the VM uses `llamacpp` only.
- **`dev.sh`:** starts `.venv/bin/python -m voicebot.web` in the background from `server/`, with `trap 'kill $API' EXIT`. Then it runs `npm --prefix web run dev` and prints the SSH tunnel command:
  `ssh -L 8000:localhost:8000 -L 5173:localhost:5173 $USER@<vm-ip>`
- **`make_cert.sh [ip]`:**
  - The `ip` argument defaults to the first address from `hostname -I`.
  - Runs `openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout certs/key.pem -out certs/cert.pem -subj "/CN=voicebot" -addext "subjectAltName=IP:$ip,DNS:localhost"`.
  - Add `certs/` to `.gitignore`.
- **`build_llama_image.sh`:**
  1. Shallow-clone `https://github.com/ggml-org/llama.cpp` into `tools/llama.cpp-src`.
  2. Run `docker build -f .devops/cuda.Dockerfile --target server --build-arg CUDA_VERSION=12.8.0 --build-arg CUDA_DOCKER_ARCH=120 -t voicebot/llama-server:cuda12.8 .`
  3. Print `set LLAMA_IMAGE=voicebot/llama-server:cuda12.8`.

- [ ] **Step 1: Write the five scripts and run `bash -n` on each** (no output means the syntax is OK). If `shellcheck` is installed, it must report no errors.
- [ ] **Step 2: README.** Add a "Linux VM" section:
  - the order of the scripts
  - the SSH tunnel command
  - the HTTPS option
  - that `transformers` and server audio are not available on the VM
- [ ] **Step 3: Manual VM run** (on the VM, by the user or an agent with SSH access):
  1. `scripts/check_vm.sh` prints `OK`. If it says to reboot, reboot first.
  2. `scripts/install_system_deps.sh`, then `scripts/setup_venv.sh`.
  3. `cp .env.example .env` (defaults: `llamacpp`, `browser`, `docker`).
  4. `scripts/dev.sh`. On the user's PC, open the tunnel and `http://localhost:5173`.
  5. Do one voice turn and one typed turn.
     - If the llama-server log shows `no kernel image is available`, run `scripts/build_llama_image.sh`, set `LLAMA_IMAGE`, and retry.
  6. Run the v1 latency baseline: 10 questions, then `../.venv/bin/python -m voicebot.latency_report --source voice`. Paste the result into the README under "Measured latency (VM)".
- [ ] **Step 4: Commit** the scripts, `.gitignore` and the README with `feat(scripts): add Linux setup, dev, cert and llama image scripts`.

---

# Part 2 — Phase A: benchmark the curated pool

### Task 5: Catalog schema and the pool file

**Files:**
- Create: `server/voicebot/catalog.py` (stdlib only: `tomllib`, `dataclasses`, `re`), `bench/pool.toml`
- Test: `server/tests/test_catalog.py`

**Interfaces:**
- `@dataclass(frozen=True) class Entry`, with these fields:

  | Field | Type | Default |
  |---|---|---|
  | `slug` | `str` | |
  | `kind` | `"stt"` or `"tts"` | |
  | `role` | `"final"`, `"realtime"` or `None` | |
  | `engine` | `str` | |
  | `model` | `str` | |
  | `options` | `dict` | `{}` |
  | `streaming` | `bool` | `False` |
  | `device` | `"cpu"` or `"cuda"` | |
  | `vram_mb` | `int` | `0` |
  | `family` | `str` | |
  | `fetch` | `str` | `"auto"` |
  | `runtime` | `"inprocess"` or `"worker"` | `"inprocess"` |
  | `group` | `str` or `None` | `None` |
  | `voice` | `str` or `None` | `None` (TTS default voice) |

- `class CatalogError(ValueError)`. Its message starts with the offending slug.
- `load_catalog(path: Path) -> list[Entry]`. It reads `[[entry]]` tables and validates:
  - the slug matches `^[a-z0-9]+(-[a-z0-9]+)*$`, and slugs are unique
  - `kind == "stt"` requires `role` in `{final, realtime}`; `kind == "tts"` requires `role is None`
  - the slug ends in `-rt` if and only if `role == "realtime"`
  - `streaming` is only true when `role == "realtime"`
  - `device` is in `{cpu, cuda}`
  - `runtime == "worker"` requires `group`
  - `fetch` matches `auto`, `hf:<repo>[:<file>]`, `sherpa:<parakeet|nemotron>`, `kroko:<file>`, `piper:<voice>` or `manual:<note>`
- `by_slug(entries) -> dict[str, Entry]`, `for_role(entries, kind, role=None) -> list[Entry]`

**`bench/pool.toml`** holds the 12 entries of spec §3.1 with these values. Engine option keys come from each engine's doc; the values here are the decisions.

| slug | engine | model | device | family | fetch | key options |
|---|---|---|---|---|---|---|
| `whisper-small-en` | `faster_whisper` | `small.en` | cuda | `faster-whisper` | `auto` | `compute_type="float16"` |
| `whisper-large-v3-turbo` | `faster_whisper` | `large-v3-turbo` | cuda | `faster-whisper` | `auto` | `compute_type="float16"` |
| `transcribe-cpp-parakeet-q4` | `transcribe_cpp` | `models/transcribe-cpp/<gguf from transcribe-cpp.md>` | cuda | `transcribe-cpp` | `manual:see RealtimeSTT/docs/engines/transcribe-cpp.md` | `backend="cuda"` |
| `sherpa-parakeet-v3` | `sherpa_onnx_parakeet` | `nvidia/parakeet-tdt-0.6b-v3` | cpu | `sherpa` | `sherpa:parakeet` | `provider="cpu", num_threads=7, download_root="models/sherpa-onnx"` |
| `nemotron-rt` | `sherpa_onnx_nemotron` | `sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11` | cpu | `sherpa` | `sherpa:nemotron` | `provider="cpu", num_threads=4`, `streaming=true` |
| `kroko-en-128-rt` | `kroko_onnx` | `Kroko-EN-Community-128-L-Streaming-001.data` | cpu | `kroko` | `kroko:Kroko-EN-Community-128-L-Streaming-001.data` | `streaming=true` |
| `whisper-tiny-en-rt` | `faster_whisper` | `tiny.en` | cuda | `faster-whisper` | `auto` | `compute_type="float16"` |
| `kokoro` | `KokoroEngine` | `kokoro` | cuda | `kokoro` | `auto` | `voice="af_heart"` |
| `qwen-cuda` | `QwenEngine` | `qwen-talker-0.6b-base-Q8_0` | cuda | `qwen` | `manual:see RealtimeTTS/docs/engines/qwen.md` | `talker_path` from qwen.md, `reference_audio="assets/ref_voice.wav"`, `reference_text_file="assets/ref_voice.txt"` |
| `inflect-cuda` | `InflectEngine` | `Inflect-Micro-v2` | cuda | `inflect` | `auto` | `backend="pytorch"` |
| `piper` | `PiperEngine` | `en_US-lessac-medium` | cpu | `piper` | `piper:en_US-lessac-medium` | `model_dir="models/piper"` |
| `chatterbox` | `ChatterboxEngine` | `chatterbox-tts` | cuda | `chatterbox` | `auto` | `audio_prompt_path="assets/ref_voice.wav"` |

- [ ] **Step 1: Write the failing tests**

```python
POOL = REPO_ROOT / "bench" / "pool.toml"
def write(tmp_path, body): p = tmp_path / "c.toml"; p.write_text(body); return p
BASE = 'slug="{s}"\nkind="stt"\nrole="{r}"\nengine="faster_whisper"\nmodel="tiny.en"\ndevice="cuda"\nfamily="fw"\n'

def test_pool_is_valid():
    e = load_catalog(POOL); assert len(e) == 12
    assert {x.slug for x in for_role(e, "stt", "realtime")} == {"nemotron-rt", "kroko-en-128-rt", "whisper-tiny-en-rt"}
    assert by_slug(e)["nemotron-rt"].streaming and by_slug(e)["kokoro"].voice == "af_heart"

@pytest.mark.parametrize("slug,role,msg", [("whisper-x", "realtime", "-rt"), ("whisper-x-rt", "final", "-rt"),
                                           ("Bad_Slug", "final", "slug")])
def test_slug_rules(tmp_path, slug, role, msg):
    with pytest.raises(CatalogError, match=msg): load_catalog(write(tmp_path, "[[entry]]\n" + BASE.format(s=slug, r=role)))

def test_duplicate_and_worker_group(tmp_path):
    one = "[[entry]]\n" + BASE.format(s="a-rt", r="realtime")
    with pytest.raises(CatalogError, match="duplicate"): load_catalog(write(tmp_path, one + one))
    with pytest.raises(CatalogError, match="group"): load_catalog(write(tmp_path, one + 'runtime="worker"\n'))

def test_bad_fetch(tmp_path):
    with pytest.raises(CatalogError, match="fetch"):
        load_catalog(write(tmp_path, "[[entry]]\n" + BASE.format(s="a", r="final") + 'fetch="ftp:x"\n'))
```

- [ ] **Step 2: Run them to see them fail. Step 3: Implement `catalog.py` and write `bench/pool.toml`. Step 4: Run them to see them pass** (6 passed).
- [ ] **Step 5: Commit** with `feat(bench): add catalog schema and curated benchmark pool`.

---

### Task 6: Bench venvs, self-test and weight fetching

**Files:**
- Create:
  - `bench/families/{faster-whisper,transcribe-cpp,sherpa,kroko,kokoro,qwen,inflect,piper,chatterbox}.txt`
  - `server/voicebot/bench/__init__.py` (empty), `server/voicebot/bench/selftest.py`, `server/voicebot/bench/fetch.py`
  - `scripts/bench_install.sh`, `scripts/bench_fetch.sh`
  - `.gitignore` additions: `venvs/`, `models/`, `bench/out/`, `bench/results/`, `bench/clips/*.wav`
- Test: `server/tests/test_bench_selftest.py`, `server/tests/test_bench_fetch.py`

**Family requirement files**
- Each file starts with `--extra-index-url https://download.pytorch.org/whl/cu128` and `torch==2.7.1+cu128` (plus `torchaudio==2.7.1+cu128` where the extra needs it).
- Then the editable RealtimeSTT or RealtimeTTS extra:
  - `faster-whisper` → `-e ./RealtimeSTT[faster-whisper]`
  - `kokoro` → `-e ./RealtimeTTS[kokoro]`
  - and so on, using each engine doc's extra name.
- `transcribe-cpp`, `sherpa`, `kroko` and `piper` have no torch line, unless their extra pulls torch in.
- STT families also install `RealtimeSTT[server]` for `stt-server-production`.

**Interfaces:**
- **`selftest.py`:** `python -m voicebot.bench.selftest --family F [--pool bench/pool.toml] [--status bench/status.json]`
  - `check_torch(torch_module) -> str | None` returns a problem string:
    - `"torch <v> != 2.7.1+cu128"`
    - `"no sm_120 in torch arch list"`, checked only when `torch.cuda.is_available()` and `"sm_120" not in torch.cuda.get_arch_list()`
  - `check_imports(entries, importer=importlib.import_module) -> dict[slug, str | None]`:
    - STT entries import `RealtimeSTT.transcription_engines.factory` and call `_load_engine_class(engine)` (the factory's lazy loader).
    - TTS entries import `RealtimeTTS` and `getattr` the engine class.
    - No weights are loaded.
  - `update_status(path, results: dict[slug, str | None])` merges into `bench/status.json` as `{slug: {"state": "ok"} | {"state": "dropped", "reason": str}}`.
- **`fetch.py`:** `python -m voicebot.bench.fetch --slug S [--pool …]`
  - `fetch_command(entry, models_root: Path, python: str) -> list[str] | None`:

    | `fetch` value | Command |
    |---|---|
    | `auto` | `None` (the library downloads on first load) |
    | `sherpa:<m>` | `[python, "-m", "RealtimeSTT.install_sherpa_models", "--root", str(models_root/"sherpa-onnx"), "--model", m]` |
    | `hf:<repo>` / `hf:<repo>:<file>` | `[python, "-m", "huggingface_hub.commands.huggingface_cli", "download", repo, (file), "--local-dir", str(models_root/"hf"/repo_name)]` |
    | `kroko:<file>` | `[python, "-c", <download via RealtimeSTT kroko helper per kroko-onnx.md>]` |
    | `piper:<voice>` | `[python, "-m", "piper.download_voices", voice, "--data-dir", str(models_root/"piper")]` |
    | `manual:<note>` | raises `SystemExit(note)` |

  - `main()` runs the command from the entry family's venv python (`venvs/bench-<family>/bin/python`).
- **`bench_install.sh <family>`:**
  1. `uv venv -p 3.12 venvs/bench-<family>`
  2. `uv pip install --python venvs/bench-<family> -r bench/families/<family>.txt`
  3. `uv pip install --python venvs/bench-<family> --no-deps -e server`
  4. `venvs/bench-<family>/bin/python -m voicebot.bench.selftest --family <family>`
- **`bench_fetch.sh <slug>`:** looks up the slug's family and runs `.venv/bin/python -m voicebot.bench.fetch --slug <slug>`.

- [ ] **Step 1: Write the failing tests**

```python
# test_bench_selftest.py
def fake_torch(ver="2.7.1+cu128", cuda=True, arch=("sm_90", "sm_120")):
    return NS(__version__=ver, cuda=NS(is_available=lambda: cuda, get_arch_list=lambda: list(arch)))

def test_check_torch():
    assert check_torch(fake_torch()) is None
    assert "2.6.0" in check_torch(fake_torch(ver="2.6.0+cu124"))
    assert "sm_120" in check_torch(fake_torch(arch=("sm_90",)))
    assert check_torch(fake_torch(cuda=False, arch=())) is None

def test_check_imports_records_failures():
    entries = [e for e in load_catalog(POOL) if e.family == "kokoro"]
    def importer(name):
        if name == "RealtimeTTS": raise ImportError("No module named kokoro")
    assert check_imports(entries, importer=importer) == {"kokoro": "No module named kokoro"}

def test_update_status_merges(tmp_path):
    p = tmp_path / "s.json"; update_status(p, {"a": None}); update_status(p, {"b": "torch 2.6.0 != 2.7.1+cu128"})
    assert json.loads(p.read_text()) == {"a": {"state": "ok"}, "b": {"state": "dropped", "reason": "torch 2.6.0 != 2.7.1+cu128"}}
```

```python
# test_bench_fetch.py
def test_fetch_commands(tmp_path):
    e = by_slug(load_catalog(POOL))
    assert fetch_command(e["whisper-small-en"], tmp_path, "py") is None
    assert fetch_command(e["sherpa-parakeet-v3"], tmp_path, "py")[-2:] == ["--model", "parakeet"]
    assert fetch_command(e["piper"], tmp_path, "py")[:3] == ["py", "-m", "piper.download_voices"]
    with pytest.raises(SystemExit, match="transcribe-cpp.md"): fetch_command(e["transcribe-cpp-parakeet-q4"], tmp_path, "py")
```

- [ ] **Step 2: Run them to see them fail. Step 3: Implement the modules, the requirement files and the two scripts. Step 4: Run them to see them pass** (4 passed), and run `bash -n` on both scripts.
- [ ] **Step 5: Commit** with `feat(bench): add per-family bench venvs, self-test and weight fetching`.

---

### Task 7: Test material: recorded clips, manifest, sentences, reference voice

**Files:**
- Create:
  - `server/voicebot/bench/clips.py`, `server/voicebot/bench/manifest.py`, `server/voicebot/bench/ref_voice.py`
  - `bench/prompts.txt` (20 lines), `bench/sentences.txt` (10 lines), `assets/ref_voice.txt` (1 line)
- Modify: `server/voicebot/config.py` (`CLIP_RECORD_DIR`), `server/voicebot/web.py` (`build_session`)
- Test: `server/tests/test_bench_clips.py`

**Interfaces:**
- `Settings.clip_record_dir: Path | None`, from `CLIP_RECORD_DIR`. Empty means off.
- **`clips.py`:**
  - `merge_callbacks(*dicts) -> dict`: for a key present in several dicts, the merged function calls each in order and returns `None`.
  - `clip_callbacks(out_dir: Path) -> dict`: `{"on_transcription_start": fn}`.
    - `fn(audio)` writes the next `clip-%03d.wav` (16 kHz mono int16; float input in [-1, 1] is scaled by 32767 and clipped) and returns `None`.
    - Numbering continues after the highest existing clip file.
- `build_session`: when `clip_record_dir` is set, it passes `merge_callbacks(timing_callbacks(timer), clip_callbacks(dir))` to `make_audio_io`. `timing_callbacks` is unchanged.
- **`manifest.py`:** `build_manifest(clip_dir: Path, prompts_path: Path) -> list[dict]` and `main()`.
  - Sorted clips pair with non-empty prompt lines as `{"wav": "<name>", "text": prompt, "id": "clip-NNN"}`.
  - A count mismatch raises `ValueError(f"{n_clips} clips but {n_prompts} prompts")`.
  - `main()` writes `clip_dir/manifest.json` only on success.
- **`ref_voice.py`:** `main()` runs in the `kokoro` bench venv. It synthesises `assets/ref_voice.txt` with `KokoroEngine(voice="af_heart")` into `assets/ref_voice.wav` (24 kHz mono int16) via `TextToAudioStream.play(muted=True, on_audio_chunk=…)`.
- **Content:**
  - `bench/prompts.txt`: 20 everyday spoken questions, 4–14 words each (2–8 s spoken).
  - `bench/sentences.txt`: 10 assistant replies of 1–3 sentences, with no markdown.
  - `assets/ref_voice.txt`: one neutral sentence of about 10 s.

- [ ] **Step 1: Write the failing tests**

```python
def test_merge_calls_both_and_returns_none():
    seen = []; a = {"on_x": lambda *v: seen.append("a") or "truthy"}; b = {"on_x": lambda *v: seen.append("b")}
    assert merge_callbacks(a, b)["on_x"](1) is None and seen == ["a", "b"]

def test_clip_callbacks_write_numbered_wavs(tmp_path):
    (tmp_path / "clip-004.wav").write_bytes(b"")
    fn = clip_callbacks(tmp_path)["on_transcription_start"]
    assert fn(np.array([0.0, 0.5, -2.0], dtype=np.float32)) is None
    with wave.open(str(tmp_path / "clip-005.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert list(np.frombuffer(w.readframes(3), dtype=np.int16)) == [0, 16384, -32767]

def test_manifest_pairs_and_rejects_mismatch(tmp_path):                  # Review Focus 5
    for i in (1, 2): (tmp_path / f"clip-00{i}.wav").write_bytes(b"")
    (tmp_path / "p.txt").write_text("Hi there\n\nWhat time is it\n")
    assert build_manifest(tmp_path, tmp_path / "p.txt")[1] == {"wav": "clip-002.wav", "text": "What time is it", "id": "clip-002"}
    (tmp_path / "p.txt").write_text("only one\n")
    with pytest.raises(ValueError, match="2 clips but 1 prompts"): build_manifest(tmp_path, tmp_path / "p.txt")
```

The `16384` comes from `round(0.5 * 32767)`. Use `np.round`, then `astype(np.int16)`.

- [ ] **Step 2: Run them to see them fail. Step 3: Implement and write the three text files. Step 4: Run them to see them pass** (3 passed), plus the full suite.
- [ ] **Step 5: Commit** with `feat(bench): add clip recording mode, manifest builder and reference voice`.

---

### Task 8: Runners: STT through the repo tools, TTS timing, GPU sampling

**Files:**
- Create: `server/voicebot/bench/gpu.py`, `server/voicebot/bench/run_stt.py`, `server/voicebot/bench/run_tts.py`
- Test: `server/tests/test_bench_runners.py`

**Interfaces:**
- **`gpu.py`:**
  - `class GpuSampler(cmd=("nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits","-lms","200"), popen=subprocess.Popen)`, a context manager.
    - It reads lines on a thread.
    - `peak_mb` is the max int seen; `0` if nothing was parsed.
    - On exit it terminates the process.
  - `parse_mem_line(line: str) -> int | None`
- **`run_stt.py`:** `python -m voicebot.bench.run_stt --slug S [--realtime] [--port 8766] [--pool …] [--manifest bench/clips/manifest.json]`
  - `server_command(final: Entry, port: int, realtime: Entry | None, python: str) -> list[str]`:
    - Base: `[python, "-m", "RealtimeSTT_server.production_server", "--host", "127.0.0.1", "--port", str(port), "--engine", final.engine, "--model", final.model, "--device", final.device, "--engine-options", json.dumps(final.options)]`.
    - With `realtime`, it appends `--realtime-engine`, `--realtime-model` and `--realtime-engine-options` from that entry.
    - `options` keys used by the bench only (`download_root`, `voice`) are kept; the engine doc decides validity.
  - `bench_command(slug, port, manifest, out: Path, realtime: bool, python: str) -> list[str]`:
    - Final:
      ```
      [python, "RealtimeSTT/tools/benchmarks/benchmark_asr_ab.py", "--manifest", str(manifest),
       "--target", f"{slug}=http://127.0.0.1:{port}", "--output", str(out), "--repetitions", "3"]
      ```
    - Realtime:
      ```
      [python, "RealtimeSTT/tools/benchmarks/benchmark_asr_streaming.py", "--url", f"http://127.0.0.1:{port}",
       "--manifest", str(manifest), "--output", str(out), "--pace", "1.0"]
      ```
  - `main()`:
    1. A realtime slug runs with `whisper-small-en` as its final engine.
    2. It starts the server from `venvs/bench-<family>/bin/python`, polls `/health` up to 600 s, then runs the benchmark inside `GpuSampler`.
    3. It writes `bench/results/<slug>.json` (the tool's report) and `bench/results/<slug>.meta.json` = `{"slug", "peak_vram_mb", "returncode"}`.
    4. It stops the server.
- **`run_tts.py`:** `python -m voicebot.bench.run_tts --slug S [--pool …] [--sentences bench/sentences.txt]`
  - `build_engine(entry) -> engine`. It imports the RealtimeTTS class named by `entry.engine` and passes `entry.options` (plus `voice` if set) as kwargs.
  - `measure(stream, sentences: list[str], clock=time.perf_counter, out_dir: Path | None = None) -> list[dict]`. For each sentence:
    1. `t0 = clock()`, then `stream.feed(s)` and `stream.play(muted=True, on_audio_chunk=cb)`.
    2. `cb` records the first chunk time and collects bytes.
    3. The row is `{"text", "first_audio_ms", "audio_ms", "rtf"}`:
       - `audio_ms` is the byte count over `(rate * 2 * channels)`, using `stream.engine.get_stream_info()`
       - `rtf` is `(t_end - t0) / (audio_ms / 1000)`
    4. If `out_dir` is set, it writes `NN.wav`.
  - `main()` builds the engine and stream, warms up once (excluded), then runs `measure` inside `GpuSampler`. It writes `bench/results/<slug>.json` = `{"slug", "rows", "peak_vram_mb"}` and saves WAVs to `bench/out/<slug>/`.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_mem_line(): assert parse_mem_line("7123\n") == 7123 and parse_mem_line("[N/A]") is None

def test_server_and_bench_commands():
    e = by_slug(load_catalog(POOL))
    cmd = server_command(e["whisper-small-en"], 8766, e["nemotron-rt"], "py")
    assert cmd[cmd.index("--engine") + 1] == "faster_whisper" and cmd[cmd.index("--realtime-engine") + 1] == "sherpa_onnx_nemotron"
    assert json.loads(cmd[cmd.index("--engine-options") + 1]) == {"compute_type": "float16"}
    b = bench_command("whisper-small-en", 8766, Path("m.json"), Path("o.json"), False, "py")
    assert "whisper-small-en=http://127.0.0.1:8766" in b and b[-2:] == ["--repetitions", "3"]
    assert "--pace" in bench_command("nemotron-rt", 8766, Path("m.json"), Path("o.json"), True, "py")

def test_measure_first_audio_and_rtf():
    clock = iter([0.0, 0.1, 0.5]).__next__              # t0, first chunk, end
    class S:
        engine = NS(get_stream_info=lambda: (8, 1, 24000))
        def feed(self, s): pass
        def play(self, muted, on_audio_chunk): on_audio_chunk(b"\0" * 24000); on_audio_chunk(b"\0" * 24000)
    row = measure(S(), ["Hi."], clock=clock)[0]
    assert row["first_audio_ms"] == 100 and row["audio_ms"] == 1000 and row["rtf"] == pytest.approx(0.5)
```

- [ ] **Step 2: Run them to see them fail. Step 3: Implement. Step 4: Run them to see them pass** (3 passed).
- [ ] **Step 5: Commit** with `feat(bench): add STT and TTS benchmark runners with GPU sampling`.

---

### Task 9: Matrix and shortlist rule

**Files:**
- Create: `server/voicebot/bench/matrix.py`
- Test: `server/tests/test_bench_matrix.py`

**Interfaces:**
- `@dataclass class Row`. Fields:
  - `slug`, `kind`, `role`, `state`
  - `latency_median_ms`, `latency_p90_ms`, `wer`, `first_partial_ms`, `verdict`, `peak_vram_mb`, `note` (all `int | float | str | None`)
- `load_rows(pool, status_path, results_dir, verdicts_path) -> list[Row]`:
  - It reads `bench/status.json`, `bench/results/*.json` and `bench/verdicts.toml` (`[verdicts] kokoro = "ok"`).
  - The latency and WER keys come from the report JSON. Read the key names from the report-building code in `benchmark_asr_ab.py` and `benchmark_asr_streaming.py` (verify against one real report from Task 10), then map them to `Row` fields in one function, `_stt_metrics(report: dict) -> dict`.
- `shortlist(rows) -> dict[str, list[str]]`, implementing spec §3.4 exactly:
  - **`stt_final`:** the baseline `whisper-small-en`, plus up to 2 more rows with `state == "ok"` and `wer <= baseline.wer + 0.02`, sorted by `latency_median_ms`. Maximum 3.
  - **`stt_realtime`:** up to 2 rows with `state == "ok"` and `wer` not None, sorted by `first_partial_ms`.
  - **`tts`:** the baseline `kokoro`, plus up to 2 more with `state == "ok"` and `verdict == "ok"`, sorted by `latency_median_ms` (the median of `first_audio_ms`).
- `render(rows, picks) -> str`: Markdown containing one table per role, columns in `Row` field order, and a "Shortlist" section listing `picks`.
- `main()` writes `docs/v2-model-matrix.md`.

- [ ] **Step 1: Write the failing tests**

```python
def r(slug, role, **kw): return Row(slug=slug, kind="tts" if role == "tts" else "stt", role=None if role == "tts" else role,
                                     state=kw.pop("state", "ok"), **{f: kw.get(f) for f in
                                     ("latency_median_ms", "latency_p90_ms", "wer", "first_partial_ms", "verdict", "peak_vram_mb", "note")})

def test_shortlist_rule():
    rows = [r("whisper-small-en", "final", wer=0.08, latency_median_ms=300),
            r("a", "final", wer=0.095, latency_median_ms=100), r("b", "final", wer=0.11, latency_median_ms=50),
            r("c", "final", wer=0.07, latency_median_ms=200), r("d", "final", state="dropped", wer=0.01, latency_median_ms=1),
            r("x-rt", "realtime", wer=0.2, first_partial_ms=300), r("y-rt", "realtime", wer=0.3, first_partial_ms=200),
            r("z-rt", "realtime", wer=None, first_partial_ms=10),
            r("kokoro", "tts", verdict="ok", latency_median_ms=150), r("q", "tts", verdict="ok", latency_median_ms=40),
            r("p", "tts", verdict="poor", latency_median_ms=10)]
    assert shortlist(rows) == {"stt_final": ["whisper-small-en", "a", "c"], "stt_realtime": ["y-rt", "x-rt"], "tts": ["kokoro", "q"]}

def test_render_has_tables_and_shortlist():
    md = render([r("whisper-small-en", "final", wer=0.08, latency_median_ms=300)], {"stt_final": ["whisper-small-en"]})
    assert "| whisper-small-en |" in md and "## Shortlist" in md
```

- [ ] **Step 2: Run them to see them fail. Step 3: Implement. Step 4: Run them to see them pass** (2 passed).
- [ ] **Step 5: Commit** with `feat(bench): add results matrix and shortlist rule`.

---

### Task 10: Run Phase A on the VM (manual)

No new code, unless a step fails and needs a fix in Tasks 5–9 (with a test).

- [ ] **Step 1: Material.**
  1. Set `CLIP_RECORD_DIR=bench/clips` and run `scripts/dev.sh`.
  2. Through the tunnel, read the 20 lines of `bench/prompts.txt` aloud, in order, one turn each.
  3. Then run `.venv/bin/python -m voicebot.bench.manifest bench/clips bench/prompts.txt`.
  4. Unset `CLIP_RECORD_DIR`.
- [ ] **Step 2: Families.** For each family:
  - `scripts/bench_install.sh <family>`, then `scripts/bench_fetch.sh <slug>` for each of its slugs.
  - For `manual:` entries, follow the printed doc.
  - After `kokoro`, run `venvs/bench-kokoro/bin/python -m voicebot.bench.ref_voice`.
  - Check `df -h ~` after each family. Stop and report if the total v2 usage passes 40 GB.
- [ ] **Step 3: Measure** with the llama-server container running (`scripts/dev.sh` in another terminal):
  - Every final STT slug: `run_stt --slug`.
  - Every realtime slug: `run_stt --slug --realtime`.
  - Every TTS slug: `run_tts --slug`.
- [ ] **Step 4: Listening check.**
  1. The user copies `bench/out/` to their PC (`scp -r xr23@<vm-ip>:<repo>/bench/out .`).
  2. They write `bench/verdicts.toml` with `ok` or `poor` for each TTS slug.
- [ ] **Step 5: Matrix.**
  1. Run `.venv/bin/python -m voicebot.bench.matrix` and check that the `_stt_metrics` key mapping matches the real reports.
  2. Show `docs/v2-model-matrix.md` to the user and get explicit approval of the shortlist, or record their overrides in the file's "Overrides" section.
- [ ] **Step 6: Commit** `docs/v2-model-matrix.md`, `bench/clips/manifest.json` and `bench/verdicts.toml` (not the WAVs) with `docs: add v2 benchmark matrix and approved shortlist`.

---

# Part 3 — Phase B: the switchable shortlist

### Task 11: Runtime decision, shortlist catalog and clean-up

**Files:**
- Create: `server/voicebot/bench/resolve.py`, `server/voicebot/catalog.toml`, `scripts/bench_clean.sh`
- Modify: `server/pyproject.toml` (an extra named `shortlist`, listing the shortlisted RealtimeSTT and RealtimeTTS extras, used only if every entry is in-process)
- Test: `server/tests/test_bench_resolve.py`

**Interfaces:**
- `minimal_worker_set(slugs: list[str], resolves: Callable[[list[str]], bool]) -> set[str]`:
  - It tries removal sets in order of size, k = 0..n, and within each size in the order `itertools.combinations` yields.
  - It returns the first set whose complement resolves.
  - Baselines (`whisper-small-en`, `kokoro`) are never removed; they already live in `.venv` from v1.
- `uv_resolves(requirements: list[str]) -> bool`: writes a temp requirements file (v1 server deps + the given extras) and returns whether `uv pip compile --python-version 3.12 <file> -o /dev/null` exits with 0.
- `main()`:
  1. Reads the approved shortlist from `docs/v2-model-matrix.md`'s "Shortlist" section.
  2. Prints the worker set.
  3. Writes `server/voicebot/catalog.toml`: the shortlisted pool entries, with `vram_mb` set to the measured peak minus the idle baseline measured with only the LLM loaded, and `runtime`/`group` set (`group` = the entry's family, reusing `venvs/bench-<family>`).
- `bench_clean.sh` removes `venvs/bench-<family>` for families with no shortlisted entry, and `models/<…>` directories used only by non-shortlisted slugs. It prints what it removes and asks `[y/N]` first.

- [ ] **Step 1: Write the failing test**

```python
def test_minimal_worker_set():
    conflict = {"chatterbox"}
    resolves = lambda s: not (conflict & set(s))
    assert minimal_worker_set(["whisper-small-en", "kokoro", "qwen-cuda"], resolves) == set()
    assert minimal_worker_set(["whisper-small-en", "kokoro", "chatterbox", "qwen-cuda"], resolves) == {"chatterbox"}
    pair = lambda s: not {"a", "b"} <= set(s)
    assert minimal_worker_set(["kokoro", "a", "b"], pair) == {"a"}
```

- [ ] **Step 2: Run it to see it fail. Step 3: Implement. Step 4: Run it to see it pass.**
- [ ] **Step 5: On the VM.**
  1. Run `resolve`.
  2. If the worker set is empty, install with `uv pip install --python .venv -e "server[shortlist]"`.
  3. Run `load_catalog(server/voicebot/catalog.toml)` to validate the file.
  4. Run `scripts/bench_clean.sh`.
  5. Record whether Task 16a/16b is needed (any `runtime = "worker"`).
- [ ] **Step 6: Commit** with `feat(bench): pick in-process/worker runtime and write shortlist catalog`.

---

### Task 12: Settings, recorder and TTS from catalog entries

**Files:**
- Modify: `server/voicebot/config.py`, `server/voicebot/stt.py`, `server/voicebot/tts.py`, `.env.example`
- Test: `server/tests/test_config.py`, `server/tests/test_audio_io.py` (extend)

**Interfaces:**
- **New `Settings` fields:**
  - `stt_final: str` (default `whisper-small-en`), `stt_realtime: str` (default `off`), `tts_model: str` (default `kokoro`)
  - `gpu_total_mb: int` (default `32607`), `llm_vram_mb: int` (default `7000`)
  - `stt_model`, `stt_device` and `stt_compute_type` stay as v1 fallbacks; they are used only when `catalog.toml` is missing.
- **`stt.make_recorder(settings, *, use_microphone, callbacks=None, recorder_cls=None, final: Entry | None = None, realtime: Entry | None = None)`.** This is additive; the v1 behaviour with `final=None` is unchanged.
  - With `final`: it passes `transcription_engine=final.engine, model=final.model, device=final.device, transcription_engine_options=final.options`, and leaves out `compute_type` unless it is in `final.options`.
  - With `realtime`: it also passes `enable_realtime_transcription=True, realtime_transcription_engine=realtime.engine, realtime_model_type=realtime.model, realtime_transcription_engine_options=realtime.options`.
  - Keys that are always passed: `language="en"`, `use_microphone`, `post_speech_silence_duration`, `spinner=False`, plus `callbacks`.
- **`tts.make_tts(settings, *, engine_cls=None, stream_cls=None, entry: Entry | None = None)`.** With `entry`, it imports `RealtimeTTS.<entry.engine>` lazily and constructs it with `**entry.options`, plus `voice=entry.voice` when set.

- [ ] **Step 1: Write the failing tests**

```python
def test_v2_settings_defaults():
    s = Settings.load(env={}, platform="linux")
    assert (s.stt_final, s.stt_realtime, s.tts_model, s.gpu_total_mb, s.llm_vram_mb) == ("whisper-small-en", "off", "kokoro", 32607, 7000)

def test_make_recorder_from_entries():
    e = by_slug(load_catalog(POOL))
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec, final=e["sherpa-parakeet-v3"], realtime=e["nemotron-rt"])
    assert r.kw["transcription_engine"] == "sherpa_onnx_parakeet" and "compute_type" not in r.kw
    assert r.kw["enable_realtime_transcription"] is True and r.kw["realtime_transcription_engine"] == "sherpa_onnx_nemotron"

def test_make_recorder_v1_path_unchanged():
    r = stt.make_recorder(Settings.load(env={}), use_microphone=False, recorder_cls=Rec)
    assert r.kw["model"] == "small.en" and r.kw["enable_realtime_transcription"] is False
```

- [ ] **Step 2: Run them to see them fail. Step 3: Implement. Step 4: Run the full suite.**
- [ ] **Step 5: Commit** with `feat(server): build recorder and TTS from catalog entries`.

---

### Task 13: ModelManager and in-process switching in the session

**Files:**
- Create: `server/voicebot/models.py`
- Modify: `server/voicebot/session.py`, `server/voicebot/audio_io.py`
- Test: `server/tests/test_models.py`, `server/tests/test_session.py` (extend)

**Interfaces:**
- **`audio_io`:** both classes gain `set_recorder(recorder)`. `BrowserAudio` gains `set_output_format(fmt: dict)`.
- **`models.py`:**
  - `ROLES = ("stt_final", "stt_realtime", "tts")`
  - `class ModelError(RuntimeError)`
  - `class ModelManager(entries: list[Entry], settings, *, recorder_factory, tts_factory, unload=default_unload)`.
    - `recorder_factory(final: Entry, realtime: Entry | None, *, use_microphone: bool, callbacks: dict)`. The real one is `partial(stt.make_recorder, settings)` with `final=` and `realtime=` keywords.
    - `tts_factory(entry: Entry)`. The real one is `lambda e: tts.make_tts(settings, entry=e)`.
    - Its members:
    - `current: dict[str, str]`, where `stt_realtime` may be `"off"`.
    - `start(audio_mode_use_microphone: bool, callbacks: dict) -> tuple[recorder, stream]` builds from `settings.stt_final`, `stt_realtime` and `tts_model`. It raises `ModelError` naming the slug if a slug is not in the catalog or has the wrong role. `web` catches this, falls back to `whisper-small-en` / `off` / `kokoro` and logs a warning.
    - `validate(role, slug) -> Entry | None` raises `ModelError("bad_model: …")`. It returns `None` for `stt_realtime` `"off"`.
    - `switch_stt(role, slug, old_recorder) -> recorder`:
      1. Call `unload(old_recorder)`.
      2. Build with the new pair.
      3. On exception, rebuild the previous pair and re-raise as `ModelError(str(e))`. `current` is unchanged on failure.
    - `switch_tts(slug, old_stream) -> stream`: the same, for TTS.
  - `default_unload(obj)`:
    1. Call `obj.shutdown()` if it exists.
    2. Run `gc.collect()`.
    3. If `"torch" in sys.modules`, run `torch.cuda.empty_cache()`.
    4. Any exception is logged and swallowed.
- **`session`:**
  - The constructor gains `models: ModelManager | None = None` and `callbacks: dict | None = None` (the recorder callbacks to reuse when rebuilding).
  - **`select_model{role, slug}`:**
    - During a turn: `error{code:"busy"}`.
    - An invalid role or slug: `error{code:"bad_model"}`.
    - Otherwise, on a worker thread:
      1. Emit `model_status loading`.
      2. If listening, `recorder.abort()` and join the listen loop.
      3. Switch, then `audio.set_recorder(new)` or replace `self.stream`.
      4. For TTS in browser mode: `audio.set_output_format(tts.output_format(new_stream))`, then emit a fresh `{"type":"config", **config()}`.
      5. Emit `model_status ready` with `load_ms`, or `failed` with `message`.
      6. Restart listening if it was on.
    - A second `select_model` while one is running gets `busy`.
  - **`user_partial{text}`:** emitted from `on_partial(text)`, which `web` wires as the recorder's `on_realtime_transcription_update` (merged with the timing callbacks via `merge_callbacks`). Partials arriving while a turn is running are dropped.
  - `config()` adds `stt_final`, `stt_realtime`, `tts`, `gpu_total_mb` and `llm_vram_mb`.
  - The metrics record adds `stt_final`, `stt_realtime`, `tts_model` and `runtime` (a dict of role → runtime). This is additive only.

- [ ] **Step 1: Write the failing tests**

```python
# test_models.py: factories record what they built; "boom" slugs raise
def test_switch_rebuilds_and_restores_on_failure(entries, settings):
    built = []
    def rf(final, realtime, **kw):
        if final.slug == "boom": raise RuntimeError("load failed")
        built.append((final.slug, realtime and realtime.slug)); return object()
    m = ModelManager(entries, settings, recorder_factory=rf, tts_factory=lambda e: object(), unload=lambda o: None)
    rec, _ = m.start(False, {}); new = m.switch_stt("stt_realtime", "nemotron-rt", rec)
    assert built[-1] == ("whisper-small-en", "nemotron-rt") and m.current["stt_realtime"] == "nemotron-rt"
    with pytest.raises(ModelError, match="load failed"): m.switch_stt("stt_final", "boom", new)
    assert built[-1] == ("whisper-small-en", "nemotron-rt") and m.current["stt_final"] == "whisper-small-en"

def test_validate_roles(entries, settings):
    m = ModelManager(entries, settings, recorder_factory=None, tts_factory=None)
    assert m.validate("stt_realtime", "off") is None
    for role, slug in [("stt_final", "nemotron-rt"), ("tts", "whisper-small-en"), ("stt_final", "nope")]:
        with pytest.raises(ModelError, match="bad_model"): m.validate(role, slug)
```

```python
# test_session.py additions
def test_select_model_busy_during_turn(gated_session, events):
    gated_session.handle({"type": "send_text", "text": "x"}); wait_for(lambda: "user_transcript" in types(events))
    gated_session.handle({"type": "select_model", "role": "tts", "slug": "kokoro"})
    assert events[-1]["code"] == "busy"

def test_select_model_while_listening_restarts_loop(session_with_models, recorder, events):   # Review Focus 1
    session_with_models.handle({"type": "start_listening"}); wait_for(lambda: recorder.waiting)
    session_with_models.handle({"type": "select_model", "role": "stt_realtime", "slug": "nemotron-rt"})
    st = wait_for(lambda: next((e for e in events if e["type"] == "model_status" and e["state"] == "ready"), None))
    assert recorder.aborted and st["role"] == "stt_realtime" and session_with_models.audio.recorder is not recorder
    assert wait_for(lambda: session_with_models.audio.recorder.waiting)          # listening again on the new recorder

def test_tts_switch_sends_new_output_format(session_with_models, events):     # Review Focus 4
    session_with_models.handle({"type": "select_model", "role": "tts", "slug": "piper"})   # fake stream reports 22050 Hz
    cfg = wait_for(lambda: next((e for e in events if e["type"] == "config"), None))
    assert cfg["output_format"]["sample_rate"] == 22050 and cfg["tts"] == "piper"

def test_partials_and_metrics_slugs(session_with_models, sink, events):
    session_with_models.on_partial("hel"); assert events[-1] == {"type": "user_partial", "text": "hel"}
    session_with_models.handle({"type": "send_text", "text": "x"}); wait_for(lambda: sink)
    assert sink[0]["stt_final"] == "whisper-small-en" and sink[0]["tts_model"] == "kokoro" and "runtime" in sink[0]
```

Fixture `session_with_models`:
- A `ModelManager` over a test catalog containing `whisper-small-en`, `nemotron-rt`, `kokoro` and `piper`.
- Fake factories that return a `FakeRecorder` (with `waiting`, set while blocked in `text()`) and a `FakeStream` whose `engine.get_stream_info()` is `(8, 1, 22050)` for piper and `(8, 1, 24000)` otherwise.

- [ ] **Step 2: Run them to see them fail. Step 3: Implement. Step 4: Run the full suite 3 times** to check for flakiness.
- [ ] **Step 5: Commit** with `feat(server): switch STT and TTS models in-process from the session`.

---

### Task 14: Web wiring: catalog event and endpoint

**Files:**
- Modify: `server/voicebot/web.py`
- Test: `server/tests/test_web.py` (extend)

**Interfaces:**
- `build_session`:
  - It loads `server/voicebot/catalog.toml` if it exists. Otherwise it uses a built-in two-entry catalog (`whisper-small-en`, `kokoro`) made from the v1 settings.
  - It creates the `ModelManager`, calls `start(...)` with `merge_callbacks(timing_callbacks(timer), {"on_realtime_transcription_update": lambda t: session.on_partial(t)}[, clip_callbacks])`, and passes `models` and `callbacks` to `VoiceSession`.
  - The lambda refers to `session` after it is constructed, through a one-element holder list.
- Session `catalog() -> list[dict]` returns the entries as dicts (`dataclasses.asdict`).
- `GET /api/catalog` returns `session.catalog()`.
- On WebSocket connect, the server sends `config` and then `{"type":"catalog","entries":[…]}`.

- [ ] **Step 1: Write the failing test.** `FakeSession` gains `catalog()`.

```python
def test_catalog_after_config(ctx):
    client, _ = ctx
    assert client.get("/api/catalog").json()[0]["slug"] == "whisper-small-en"
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "config" and ws.receive_json()["type"] == "catalog"
```

- [ ] **Step 2: Run it to see it fail. Step 3: Implement. Step 4: Run the full suite.**
- [ ] **Step 5: Commit** with `feat(server): expose shortlist catalog over HTTP and WebSocket`.

---

### Task 15: Frontend: pickers, VRAM meter, partials, player rebuild

**Files:**
- Modify: `web/src/protocol.ts`, `web/src/state.ts`, `web/src/components/SettingsPanel.tsx`, `web/src/components/ChatTranscript.tsx`, `web/src/components/StatusBar.tsx`, `web/src/App.tsx`, `web/src/styles.css`
- Create: `web/src/vram.ts`
- Test: `web/src/state.test.ts` (extend), `web/src/vram.test.ts`

**Interfaces:**
- **`protocol.ts` additions:**

```ts
export type Role = "stt_final" | "stt_realtime" | "tts";
export interface CatalogEntry { slug: string; kind: "stt" | "tts"; role: "final" | "realtime" | null; device: "cpu" | "cuda";
  vram_mb: number; runtime: "inprocess" | "worker"; streaming: boolean }
// ServerConfig gains: stt_final: string; stt_realtime: string; tts: string; gpu_total_mb: number; llm_vram_mb: number
// ServerEvent gains: { type: "catalog"; entries: CatalogEntry[] } | { type: "model_status"; role: Role; slug: string;
//   state: "loading" | "ready" | "failed"; message?: string; load_ms?: number } | { type: "user_partial"; text: string }
// ClientCommand gains: { type: "select_model"; role: Role; slug: string }
```

- **`state.ts`:** `AppState` gains `catalog: CatalogEntry[]`, `switching: Role | null` and `partial: string | null`.
  - `model_status loading` sets `switching`; `ready` / `failed` clears it, and `failed` also sets `error` to `message`.
  - `user_partial` sets `partial`. `user_transcript` clears it.
- **`vram.ts`:** `vramUsed(cfg: ServerConfig, catalog: CatalogEntry[]): { used: number; total: number; over: boolean }`. It sums `llm_vram_mb` and the `vram_mb` of the selected final, realtime (unless `off`) and TTS entries.
- **UI:**
  - **SettingsPanel:**
    - Three `<select>`s, `Final STT`, `Realtime STT` (first option `off`) and `TTS`. Option labels are `` `${slug} · ${device} · ${vram_mb} MB` ``.
    - The selects are disabled while `switching` is set or `status` is not `idle` / `listening`.
    - A change sends `select_model`.
    - A VRAM meter bar uses `vramUsed`; its amber class applies when `over`.
  - **ChatTranscript:** renders `partial` as a grey last line.
  - **StatusBar:** the status pill shows `loading <slug>` while switching. The tooltip adds the three active slugs.
  - **App:** when a `config` event arrives whose `output_format.sample_rate` differs from the current player's, it closes the player and creates a new one.

- [ ] **Step 1: Write the failing tests**

```ts
// vram.test.ts
it("sums selections and flags over budget", () => {
  const cat = [{ slug: "a", vram_mb: 1000 }, { slug: "b-rt", vram_mb: 500 }, { slug: "t", vram_mb: 2000 }] as any;
  const cfg = { stt_final: "a", stt_realtime: "off", tts: "t", gpu_total_mb: 10000, llm_vram_mb: 7000 } as any;
  expect(vramUsed(cfg, cat)).toEqual({ used: 10000, total: 10000, over: false });
  expect(vramUsed({ ...cfg, stt_realtime: "b-rt" }, cat).over).toBe(true);
});
```

```ts
// state.test.ts additions
it("model_status, partial and catalog", () => {
  let s = srv(initialState, { type: "catalog", entries: [{ slug: "kokoro" }] }); expect(s.catalog).toHaveLength(1);
  s = srv(s, { type: "model_status", role: "tts", slug: "piper", state: "loading" }); expect(s.switching).toBe("tts");
  s = srv(s, { type: "model_status", role: "tts", slug: "piper", state: "failed", message: "no" });
  expect(s.switching).toBeNull(); expect(s.error).toBe("no");
  s = srv(s, { type: "user_partial", text: "hel" }); expect(s.partial).toBe("hel");
  s = srv(s, { type: "user_transcript", text: "hello", turn_id: 1 }); expect(s.partial).toBeNull();
});
```

- [ ] **Step 2: Run `npm run test` to see them fail. Step 3: Implement. Step 4: Run `npm run test`, `npm run typecheck` and `npm run build`.** All must pass.
- [ ] **Step 5: Commit** with `feat(web): add model pickers, VRAM meter, live partials and player rebuild`.

---

### Task 16a / 16b: Worker mode (only if Task 11 found `runtime = "worker"` entries)

Skip both tasks if `catalog.toml` has no worker entries, and record "skipped: all in-process" in the matrix doc.

**16a: Worker package** (`worker/pyproject.toml`, `worker/voicebot_worker/{__main__,framing,stt_host,tts_host}.py`; tests in `worker/tests/`)
- **`framing.py`:**
  - `encode(header: dict, body: bytes = b"") -> bytes`, where the format is `>I` header length + header JSON + `>I` body length + body.
  - `read_frame(sock_file) -> tuple[dict, bytes]`, which raises `EOFError` on a clean close.
- **`__main__`:** `--slug --port 0 --catalog PATH`. It binds `127.0.0.1`, prints `PORT <n>` and flushes, then accepts one connection, loads the entry and sends `ready{info}` or `fatal{message}`, then serves messages per spec §4.5.
- **STT host:** builds the engine with `RealtimeSTT.transcription_engines.factory.create_transcription_engine`. Streaming slugs use `create_streaming_session`.
- **TTS host:** `TextToAudioStream(engine)` with `play(muted=True, on_audio_chunk=send_audio)`.
- **Tests:**
  - framing round trip, including an empty body and a frame split across two reads
  - one test per message type with a fake engine module injected through `--engine-module` (test-only flag)

**16b: Server side** (`server/voicebot/workers/{framing.py,manager.py,stt_proxy.py,tts_proxy.py}`; `framing.py` is a byte-identical copy, enforced by a test)
- **`WorkerManager.start(entry) -> WorkerHandle`:**
  - `popen([f"venvs/{entry.group}/bin/python","-m","voicebot_worker","--slug",entry.slug,"--port","0","--catalog",…], start_new_session=True, preexec_fn=_pdeathsig)`.
  - It reads `PORT`, connects and waits for `ready`, up to 300 s.
  - `_pdeathsig` calls `prctl(1, signal.SIGTERM)` through `ctypes.CDLL("libc.so.6")`.
- **`WorkerHandle.close()`:** sends `shutdown`, then after 5 s `os.killpg(pgid, SIGTERM)`, then `SIGKILL`.
- **Heartbeat:** `ping` every 2 s. No reply for 6 s marks the worker crashed, and it is restarted once.
  - A crash during a turn raises inside the proxy. The session treats it like an `LLMError`: `error` event plus the v1 spoken apology. If the TTS worker itself is down, the apology is sent as text only.
  - If the restart fails, `ModelManager` reverts that role to its default slug and emits `model_status failed`.
- **Proxies:**
  - `ProxyExecutor.transcribe(audio, language=None, use_prompt=True, **kw) -> TranscriptionResult`
  - `ProxyStreamingEngine` (`supports_streaming = True`, `create_streaming_session()`)
  - `ProxyTTSStream` (`feed`, `play(on_audio_chunk=None, muted=…)`, `stop`, `text`, and `engine` with `get_voices`, `set_voice` and `get_stream_info`)
- **`ModelManager`:**
  - For worker entries, it builds the recorder with `transcription_executor=` or `realtime_transcription_executor=` set to the proxy, and the TTS from `ProxyTTSStream`.
  - A worker switch starts the new worker before closing the old one.
- **Tests:** use a fake worker, a stdlib-only script speaking the protocol, for ready, switch, rollback on `fatal`, rollback on timeout, heartbeat timeout, and process-group kill on close (assert the pid is gone).
- Each sub-task runs Steps 1–4 as in the other tasks, and commits with `feat(worker): …` / `feat(server): …`.

---

### Task 17: Phase B check on the VM (manual)

- [ ] **Step 1: VRAM unload check.** For each in-process GPU entry:
  1. Record `nvidia-smi` memory with only the LLM loaded.
  2. Switch to the entry, then switch back to the baseline.
  3. The memory must return to within 300 MB of the start.

  An entry that fails moves to `runtime = "worker"` in `catalog.toml`. If this is the first worker entry, do Tasks 16a/16b, then repeat this step.
- [ ] **Step 2: Turn matrix.** For each shortlisted combination, start with the 3 final STT engines × `off`, then the best final with each realtime engine × each TTS engine:
  1. Select it in the UI.
  2. Ask the same 10 questions by voice.
  3. Run `.venv/bin/python -m voicebot.latency_report --by stt_final,stt_realtime,tts_model --source voice`.
- [ ] **Step 3:** Append the report output and a short "Fastest combination" note to `docs/v2-model-matrix.md`. Update the README's "Measured latency (VM)" section.
- [ ] **Step 4: Final verification.**
  - From `server/`: `../.venv/bin/python -m pytest -v`.
  - From `web/`: `npm run test`, `npm run typecheck`, `npm run build`.
  - If worker mode was built, the worker package's tests.
- [ ] **Step 5: Commit** with `docs: add v2 turn-level latency results`.
