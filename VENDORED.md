# Vendored upstream code

`RealtimeSTT/` and `RealtimeTTS/` are full copies of upstream repositories. They were committed directly into this repo (vendored), not added as submodules, so we can patch them freely.

| Directory | Upstream | Commit | Commit date | License |
|---|---|---|---|---|
| `RealtimeSTT/` | https://github.com/KoljaB/RealtimeSTT | `3574c247c531b8f6ff1f595b319eb400ba499f82` | 2026-11-11 | Apache-2.0 (see `RealtimeSTT/LICENSE`) |
| `RealtimeTTS/` | https://github.com/KoljaB/RealtimeTTS | `7443092f11bb52b71f85b11075472024ae894924` | 2026-11-10 | Apache-2.0 (see `RealtimeTTS/LICENSE` and `RealtimeTTS/LICENSING_ADDENDUM.md`) |

## Not committed

- `RealtimeTTS/deepspeed/*.whl` (49 MB). These DeepSpeed wheels are built for Python 3.10 and 3.11, and this project uses 3.12, so they are excluded in `.gitignore`.

## Updating from upstream

1. Clone upstream at the new commit into a temporary folder.
2. Diff it against the vendored directory and re-apply any local patches.
3. Copy it over, without the upstream `.git` folder.
4. Update the commit and date in the table above.
5. Commit with a message such as `vendor: update RealtimeSTT to <sha>`.

List every local patch to vendored code here.

## Local patches

No vendored file is edited. Two upstream issues are worked around from `server/voicebot/`:

| Upstream issue | Workaround |
|---|---|
| RealtimeTTS `AudioStream.close_stream()` closes the PyAudio stream without a lock. An immediate player stop from another thread (the Stop button) and the draining stop in a synchronous `play()`'s cleanup can close it twice and corrupt the heap (`Windows fatal exception: code 0xc0000374`). | `voicebot.tts.install_realtimetts_shims` wraps `AudioStream.close_stream` with a class-level lock at runtime (`serialize_method`); the second close then sees no stream. `StreamPlayer.stop` is deliberately not locked, so an immediate stop can still interrupt a draining one. |
| RealtimeSTT waits forever on `main_transcription_ready_event` if its transcription worker fails to load the model (for example, faster-whisper missing or cuBLAS not loadable), so startup hangs with no error. | `voicebot.stt.preflight` checks those causes first and fails with a fix hint; `voicebot.stt.expose_cuda_dlls` puts torch's CUDA 12 DLLs on the search path for CTranslate2. |

Re-check both after updating the vendored copies.
