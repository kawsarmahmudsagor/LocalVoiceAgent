# Local Voice Bot: Specifications

This document gives a readable overview of the local voice bot. For the full technical design, see [the design spec](superpowers/specs/2026-11-18-local-voice-bot-design.md). For the build steps, see [the implementation plan](superpowers/plans/2026-11-18-local-voice-bot.md).

## What it is

The voice bot is a test web app for talking to a local AI model by voice. You speak, the model answers, and the bot speaks the answer back to you. A browser page shows the conversation and lets you change how the model replies.

Everything runs on one Windows PC. No cloud service is used, and after the first download no internet connection is needed.

## How a conversation works

1. **You speak.** The bot listens to your microphone.
2. **The bot writes down what you said.** When you stop talking for 0.6 seconds, the speech-to-text model turns your speech into text.
3. **The model thinks of a reply.** The text goes to the language model with the recent conversation history.
4. **The bot speaks.** The reply is turned into speech as it is written, so the bot starts talking before the whole answer is ready.
5. **The bot listens again.** The microphone is muted while the bot speaks, so the bot does not hear itself.

You can also type a message instead of speaking. The **Stop** button cuts off the bot mid-reply.

## The three models

| Job | Model | Runs on | Notes |
|---|---|---|---|
| Speech to text | faster-whisper `small.en` (through RealtimeSTT) | GPU (can be moved to CPU) | English only. Uses int8 precision. |
| Language model | Qwen3.5-9B, 4-bit | GPU | Two ways to run it. See below. |
| Text to speech | Kokoro (through RealtimeTTS) | CPU | Default voice `af_heart`. Other voices are selectable in the UI. |

### Two ways to run the language model

You choose one in the `.env` file with `LLM_BACKEND`.

- **`llamacpp` (default).** The app starts a separate `llama-server.exe` program running `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`. This is the faster option. It needs a 5.7 GB download the first time.
- **`transformers`.** The model `Qwen/Qwen3.5-9B` loads inside the Python app and is compressed to 4-bit as it loads. This needs a 19.3 GB download the first time.

The model's internal reasoning (its "thinking") is turned off. If any reasoning text slips through, the app removes it, so it is never shown or spoken. Markdown symbols such as `*` and `#` are also removed, so the bot does not read them aloud.

## Hardware and memory

The target PC has an NVIDIA RTX 4060 with 8 GB of video memory (VRAM). Memory is shared like this:

| Component | Video memory used |
|---|---|
| Qwen3.5-9B, 4-bit | about 6 GB |
| faster-whisper `small.en` | about 0.5–1 GB |
| Kokoro | none (runs on the CPU) |

If the GPU runs out of memory, set `STT_DEVICE=cpu` to move speech-to-text off the GPU, or switch to the `llamacpp` backend.

The app needs **Python 3.12**. Newer versions do not work with RealtimeSTT.

## Two audio modes

You choose one in the `.env` file with `AUDIO_IO`.

- **`browser` (default).** The bot uses the microphone and speakers of the browser you open the app in. Browser echo cancellation is on. Headphones are still recommended.
- **`server`.** The bot uses the PC's own microphone and speakers. The browser page only shows the conversation.

Browsers only allow microphone access on `localhost` or over HTTPS. Open the app on the same PC, or use server mode.

## The web page

The page has a dark, playground-style layout.

- **Left sidebar: settings.**
  - Labels showing the model, the backend and the audio mode.
  - The system prompt, which is the instruction that tells the bot how to behave.
  - Sliders for temperature, top_p and maximum reply length.
  - A voice picker.
- **Centre: the conversation.** Your messages and the bot's replies. Replies appear word by word. Replies you stopped early are marked.
- **Bottom: controls.**
  - A microphone button, a text box with **Send**, a **Stop** button and a **Clear** button.
  - A status label: idle, listening, thinking or speaking.
  - Timing for the last reply.

Settings changes apply from the next reply.

## Default settings

All settings live in `.env`. Copy `.env.example` to `.env` to start. Restart the app after editing it.

| Setting | Default | What it does |
|---|---|---|
| `LLM_BACKEND` | `llamacpp` | How the language model runs: `llamacpp` or `transformers` |
| `AUDIO_IO` | `browser` | Whose mic and speakers to use: `browser` or `server` |
| `STT_MODEL` | `small.en` | Speech-to-text model |
| `STT_DEVICE` | `cuda` | `cuda` (GPU) or `cpu` |
| `STT_SILENCE_SEC` | `0.6` | Seconds of silence that end your turn |
| `TTS_VOICE` | `af_heart` | Kokoro voice |
| `SYSTEM_PROMPT` | "You are a friendly voice assistant. Reply in one to three short spoken sentences. Never use lists, markdown, code blocks or emoji." | Instruction for the bot |
| `LLM_TEMPERATURE` | `0.7` | Higher gives more varied replies |
| `LLM_TOP_P` | `0.8` | Limits word choice to the most likely options |
| `LLM_MAX_TOKENS` | `256` | Maximum reply length |
| `HISTORY_TURNS` | `10` | How many past exchanges the bot remembers |
| `WEB_PORT` | `8000` | Port of the app's server |
| `LLAMA_PORT` | `8081` | Port of `llama-server` |

`.env.example` lists every setting with a comment.

## Goals

The app is working when all of these are true:

1. With either backend and either audio mode, a spoken English question gets a spoken answer.
2. The bot starts speaking about 1–2 seconds after you stop talking, using the `llamacpp` backend.
3. Switching backend or audio mode needs only a `.env` edit and a restart.
4. The model's reasoning is never shown or spoken.
5. All automated tests pass without a GPU or downloaded models.

## Not included in version 1

- Interrupting the bot by talking over it. Use the **Stop** button instead.
- More than one user at a time. A second browser tab is refused.
- Languages other than English.
- Switching backend or audio mode while the app is running.
- Logins, HTTPS, or hosting beyond your own PC or local network.

## When something goes wrong

| Problem | What happens |
|---|---|
| A setting in `.env` is missing or invalid | The app does not start and names the setting |
| `llama-server.exe` is missing | The app tells you to run `scripts/fetch_llama_server.ps1` |
| The `llama-server` port is already in use | The app suggests a different `LLAMA_PORT` |
| `llama-server` does not start in time | The app stops it and shows the last 30 lines of its log |
| The GPU runs out of memory | The app suggests `STT_DEVICE=cpu` or `LLM_BACKEND=llamacpp` |
| The model fails during a reply | The bot says "Sorry, something went wrong." and keeps listening |
| The browser blocks the microphone | An error banner appears. Typing still works. |
| The connection drops | The page shows "disconnected" and reconnects |

## Setting it up

1. Run `scripts/setup_venv.ps1`. It creates a Python 3.12 environment and installs everything.
2. Run `scripts/fetch_llama_server.ps1`. It downloads `llama-server` into `tools/llama/`.
3. Copy `.env.example` to `.env`. Choose `LLM_BACKEND` and `AUDIO_IO`.
4. In the `web` folder, run `npm install`.
5. Run `scripts/dev.ps1`, then open http://localhost:5173.

The first start downloads the language model, which can take several minutes.

## Technology used

- **Server:** Python 3.12, FastAPI, RealtimeSTT, RealtimeTTS, Hugging Face transformers with bitsandbytes, and llama.cpp.
- **Web page:** React 18, TypeScript and Vite.
- **Connection:** one WebSocket carries the commands, the conversation events and the audio between the page and the server.
- **Tests:** pytest for the server, vitest for the web page.
