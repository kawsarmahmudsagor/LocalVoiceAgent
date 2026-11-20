# LocalVoiceAgent — notes for Claude

This file is the handoff for any Claude session working in this repo, especially the one on the Linux VM that implements v2. Read it fully before doing anything.

## What this project is

A local voice bot for **measuring latency**. You speak, the speech-to-text model (STT) transcribes, Qwen3.5-9B answers, and the text-to-speech model (TTS) speaks the reply. A playground-style web UI shows the conversation and the per-stage timing.

The project's purpose is to find out where the latency goes, using **only the resources in the vendored `RealtimeSTT/` and `RealtimeTTS/` repos**: their engines and their benchmark tools. Keep every change aligned with that purpose. If a feature adds build work but produces no latency data, raise it with the user before building it.

## Where things stand

| Phase | Status | Documents |
|---|---|---|
| v1: the voice bot on Windows (RTX 4060, 8 GB) | Built on branch `feat/voice-bot`. Its manual end-to-end checks may still be open. | Spec `docs/superpowers/specs/2026-11-18-local-voice-bot-design.md`, plan `docs/superpowers/plans/2026-11-18-local-voice-bot.md` |
| v2: Linux VM port, benchmark a curated engine pool, make the winners switchable in the UI | **Not started. Implement it on the VM.** | Spec `docs/superpowers/specs/2026-11-18-v2-model-switching-design.md`, plan `docs/superpowers/plans/2026-11-18-v2-linux-benchmark-shortlist.md` |
| v3: voice barge-in | Built on branch `feat/v3-barge-In`. Manual checks on the VM still open. | Spec `docs/superpowers/specs/2026-11-20-barge-in-design.md`, plan `docs/superpowers/plans/2026-11-20-barge-in.md` |
| v4: speaker diarization | Built on branch `feat/v4-diarization`. Manual checks on the VM still open. | Spec `docs/superpowers/specs/2026-11-20-v4-diarization-design.md`, plan `docs/superpowers/plans/2026-11-20-v4-diarization.md` |

`docs/SPECIFICATIONS.md` is a plain-language overview of v1, written for humans.

**Read in this order:** this file, then the v2 spec, then the v2 plan. Read the v1 spec and plan for any section a v2 task refers to (`§` numbers and "v1 Task N").

**Check v1 is complete before starting v2 Task 1.** Each v1 plan task names its commit message. Compare them with `git log --oneline main..HEAD`. If a v1 task's commit is missing, tell the user before continuing.

**How to execute the v2 plan:** the user chose to run it from the VM. Use `superpowers:subagent-driven-development` if it is installed; that was the recommendation, because Tasks 12–16 touch v1's `session.py` and `audio_io.py`. Otherwise go task by task with the TDD steps exactly as the plan writes them. Ask the user which way they prefer if it's unclear.

## The VM

From a check run on 2026-11-18:

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0, glibc 2.39 |
| GPU | RTX 5090, 32 GB, Blackwell `sm_120`. The NVIDIA userspace was 580.178 |
| CPU / RAM / disk | 28 cores / 49 GB / about 211 GB free on `/` |
| Tools | Docker 29.1 with the NVIDIA Container Toolkit 1.19, `uv`, Python 3.12.3 |
| Host CUDA toolkit | `nvcc` 12.0. **Never use it**: it is too old for `sm_120`. Torch wheels and Docker images bring their own CUDA |
| User | `xr23` |

At the time of the check, `nvidia-smi` failed with **"Driver/library version mismatch"**. The VM needs `sudo reboot` after the driver update. Run `nvidia-smi` first. If it still fails, ask the user to reboot the VM; don't try to fix drivers yourself.

The user reaches the VM from a Windows PC on the same LAN. **You cannot use a browser or microphone on the VM.** Anything that needs the web UI, speaking or listening is done by the user through an SSH tunnel:
- `ssh -L 8000:localhost:8000 -L 5173:localhost:5173 xr23@<vm-ip>`
- then open `http://localhost:5173` (dev) or `http://localhost:8000`

Ask the user to do these steps, and wait for their results:
- recording the 20 benchmark clips
- voice turns
- listening verdicts on TTS output
- shortlist approval

## Decisions already made (with the reasons)

Don't re-open these unless the user asks.

- **Platform for v2:** Linux VM only. The v1 docs stay Windows. v2 spec §2 overrides v1 wherever they differ.
- **LLM:** `LLM_BACKEND=llamacpp` on the VM (`unsloth/Qwen3.5-9B-GGUF:Q4_K_M`).
- **Deferred: the transformers backend on the VM.** On 2026-11-19 the user had the `Qwen/Qwen3.5-9B` checkpoint downloaded into the Hugging Face cache and marked running and finishing the transformers backend (`server/voicebot/llm/`) on the VM for later. Until then it is a placeholder: do not start on it unless the user asks.
- **llama-server:** runs from the Docker image `ghcr.io/ggml-org/llama.cpp:server-cuda`. If it lacks `sm_120` kernels, build an image in a container (`scripts/build_llama_image.sh`, plan Task 4).
- **Audio:** browser mode only. The VM has no mic or speakers, so `AUDIO_IO=server` is rejected on Linux.
- **Disk:** the user set a budget of about 35 GB for v2, raised to 50 GB on 2026-11-18, and lifted it on 2026-11-19 ("the budget is of no issue").
- **Curated pool (14 entries):** listed in spec §3.1, with the reasons for each excluded engine in spec §3.2. 12 were chosen for the best latency per GB; `qwen3-asr-1-7b` and `coqui-xtts-v2` were added on user request and are forced onto the shortlist by `bench/overrides.toml`.
- **Torch:** exactly `torch==2.7.1+cu128` (it supports `sm_120`) in every venv. An engine that needs another torch version is dropped and recorded, not worked around. This also keeps disk low, because uv hardlinks one torch copy into every venv.
- **Shortlist rule (spec §3.4):**
  - Final STT: up to 3 engines, with WER at most 2 points worse than `whisper-small-en`, ranked by latency.
  - Realtime STT: up to 2 engines.
  - TTS: 2–3 engines with an "OK" listening verdict, ranked by time to first audio.
  - Baselines `whisper-small-en` and `kokoro` are always kept.
- **Runtime:** shortlisted engines run in the server process if they resolve together with the server's dependencies. Worker processes (plan Tasks 16a/16b) are built **only** if needed.

## Facts verified in the vendored code

- `post_speech_silence_duration` defaults to **0.6 s** in RealtimeSTT (`RealtimeSTT/docs/configuration.md`). It adds a fixed 0.6 s to every voice turn.
- RealtimeSTT's `on_vad_stop` fires *after* the silence wait (`RealtimeSTT/RealtimeSTT/core/recording.py`, around line 474). The real end of speech is `on_turn_detection_start`, which is what v1's timing uses.
- `on_transcription_start` must return `None`. A truthy return value aborts the transcription (`core/transcription_api.py`, around line 53).
- Repo benchmark tools:
  - `RealtimeSTT/tools/benchmarks/benchmark_asr_ab.py` (`--manifest --target LABEL=URL --output --repetitions`)
  - `RealtimeSTT/tools/benchmarks/benchmark_asr_streaming.py` (`--url --manifest --output --pace`)
  - Both talk to `stt-server-production` (`RealtimeSTT_server/production_server.py`), and both compute WER and percentiles themselves.
- RealtimeTTS has benchmarks only for Qwen, so v2 adds a small `run_tts.py` built on the public RealtimeTTS API.

## Working rules

- **Do not edit `RealtimeSTT/` or `RealtimeTTS/`.** They are vendored (see `VENDORED.md`). If a patch is unavoidable, record it there.
- **TDD:** follow the plan's steps in order. Write the failing test, see it fail, implement, see it pass, commit.
- **Tests run without a GPU, Docker, network or models.** From `server/`: `../.venv/bin/python -m pytest -v`. From `web/`: `npm run test && npm run typecheck && npm run build`.
- **Git:**
  - Work on `feat/voice-bot`, or a branch from it if the user asks.
  - Stage files **by explicit path**, never `git add -A`.
  - Gitleaks pre-commit and pre-push hooks are active. Run `git config core.hooksPath .githooks` once, and install gitleaks with `sudo apt install gitleaks`. **Never** use `--no-verify`.
  - End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Push only when the user asks.
- **Secrets:** `.env` is git-ignored. Create it from `.env.example`. Never commit tokens. Hugging Face downloads used here need no token.
- **Shell scripts** must have LF line endings; `.gitattributes` enforces this.
- **Ask before** any `sudo` command, deleting model weights or venvs (`scripts/bench_clean.sh` prompts first). Check `df -h ~` after each benchmark family.
- **Windows-only files** you can ignore on the VM: `scripts/*.ps1`, `tools/llama/`.

## The user

- They want recommendations, not long option lists. Give one recommended path and say why.
- They care about staying on the goal (latency testing with the repos' resources), not building infrastructure for its own sake.
- They answer questions quickly. Ask when a decision is genuinely theirs: scope, disk, anything outward-facing.
