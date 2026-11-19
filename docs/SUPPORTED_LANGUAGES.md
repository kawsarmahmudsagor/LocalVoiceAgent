# Supported languages

This page lists the languages each speech engine supports, according to the two upstream repositories:

- RealtimeTTS: https://github.com/KoljaB/RealtimeTTS, commit `7443092`
- RealtimeSTT: https://github.com/KoljaB/RealtimeSTT, commit `3574c24`

Both commits were the upstream `HEAD` on 2026-11-18, and the vendored copies in this repo match them exactly.

Every entry comes from the repos' docs or source. "Not listed in the repo" means the repo leaves the language set to the model, the cloud provider or the installed voices, and gives no list of its own. Check the model card or the provider's docs for those.

## What this project uses today

| Stage | Setting (`.env`) | Languages |
|---|---|---|
| STT | `STT_MODEL=small.en` (faster-whisper) | **English only.** `.en` Whisper models are English-only. To transcribe other languages, use a multilingual model such as `small`. |
| TTS | `TTS_VOICE=af_heart` (Kokoro) | **American English.** Kokoro picks the language from the voice prefix, so choosing another voice changes the language (see the Kokoro row below). |
| LLM | Qwen3.5-9B | Not part of either repo. See the Qwen3.5 model card. |

## Text to speech (RealtimeTTS)

### Engines with an explicit language list in the repo

| Engine | Type | Supported languages | Source |
|---|---|---|---|
| `KokoroEngine` | Local | American English (`a`), British English (`b`), Japanese (`j`), Mandarin Chinese (`z`), Spanish (`e`), French (`f`), Hindi (`h`), Italian (`i`), Brazilian Portuguese (`p`). The voice prefix sets the language (`af`/`am`, `bf`/`bm`, `jf`/`jm`, `zf`/`zm`, `ef`/`em`, `ff_`, `hf`/`hm`, `if`/`im`, `pf`/`pm`). Japanese, Chinese and Korean need the `jp`, `zh`, `ko` extras, and some non-English voices need espeak-ng. | `RealtimeTTS/engines/kokoro_engine.py`, `docs/engines/kokoro.md` |
| `QwenEngine` (Qwen3-TTS) | Local | 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian. `Auto` detects the language per fragment with fastText. | `RealtimeTTS/language_router.py` (`SUPPORTED_QWEN_LANGUAGES`) |
| `CoquiEngine` (XTTS v2) | Local | XTTS v2.0.2, 16 languages: English (`en`), Spanish (`es`), French (`fr`), German (`de`), Italian (`it`), Portuguese (`pt`), Polish (`pl`), Turkish (`tr`), Russian (`ru`), Dutch (`nl`), Czech (`cs`), Arabic (`ar`), Chinese (`zh-cn`), Japanese (`ja`), Hungarian (`hu`), Korean (`ko`). XTTS v2.0.3 adds Hindi (`hi`). | `FAQ.md` |
| `ModelsLabEngine` | Cloud | Built-in voices for American English, British English, Spanish, French, German, Italian, Japanese, Hindi, Mandarin Chinese, Brazilian Portuguese. | `RealtimeTTS/engines/modelslab_engine.py` |
| `MiniMaxEngine` | Cloud | Built-in presets are tagged either `english` or `multilingual`. The repo doesn't list which languages `multilingual` covers. | `RealtimeTTS/engines/minimax_engine.py` |

### English-only engines

| Engine | Type | Note | Source |
|---|---|---|---|
| `InflectEngine` | Local | One fixed synthetic English male voice. | `docs/engines/inflect.md` |
| `SopranoEngine` | Local | Single-voice English engine. | `docs/engines/soprano.md` |
| `PocketTTSEngine` | Local (CPU) | The source describes it as English only. A `language` option selects an upstream Pocket TTS language configuration if one exists. | `RealtimeTTS/engines/pocket_engine.py` |
| `NeuTTSEngine` | Local | The default NeuTTS Nano backbones map to `en-us`. Other backbones accept a `language` override. | `RealtimeTTS/engines/neutts_engine.py` |
| `StyleTTSEngine` | Local | Its phonemizer is fixed to `en-us`. | `RealtimeTTS/engines/style_engine.py` |

### Engines whose language set comes from the model, the voices or the provider

| Engine | Type | How the language is chosen | Source |
|---|---|---|---|
| `OmniVoiceEngine` | Local | Described as "multilingual voice cloning". Takes a `language` parameter (default `en`), but the repo has no list. | `docs/engines/omnivoice.md` |
| `ZipVoiceEngine` | Local | Tokenizer `emilia`, `libritts`, `espeak` or `simple`. `language` is used only with `espeak`. No list in the repo. | `docs/engines/zipvoice.md` |
| `ChatterboxEngine` (Turbo) | Local | Takes a `language` argument but ignores it. Not listed in the repo. | `RealtimeTTS/engines/chatterbox_engine.py` |
| `SoproTTSEngine` | Local | Ignores `language`. Not listed in the repo. | `docs/engines/sopro.md` |
| `LuxTTSEngine` | Local | Voice cloning from a prompt WAV. Not listed in the repo. | `docs/engines/luxtts.md` |
| `MossTTSEngine` (MOSS-TTS-Nano) | Local | Not listed in the repo. | `docs/engines/moss-tts.md` |
| `ParlerEngine` | Local | Voice set by a text description. Not listed in the repo. | `docs/engines/parler.md` |
| `PiperEngine` | Local | Each Piper `.onnx` voice model has its own language. | `docs/engines/piper.md` |
| `OrpheusEngine` | Local server | Not listed in the repo. | `docs/engines/orpheus.md` |
| `HiggsEngine` (Higgs Audio v3) | Local server | Accepts `language` as an extra request field. Not listed in the repo. | `docs/engines/higgs.md` |
| `SystemEngine` (pyttsx3) | Local OS | Whatever voices are installed in the operating system. | `FAQ.md` |
| `EdgeEngine` | Cloud | Each voice has its own locale, such as `en-US-EmmaMultilingualNeural`. `get_voices()` lists them. | `docs/engines/edge.md` |
| `AzureEngine` | Cloud | Choose a voice with the locale you want: `AzureVoice(name, locale, gender)`. | `docs/engines/azure.md`, `FAQ.md` |
| `GTTSEngine` (Google TTS) | Cloud | Every language in `gtts.lang.tts_langs()`, with regional variants through `tld`. | `docs/engines/gtts.md` |
| `OpenAIEngine` | Cloud | Feed text in the language you want. | `FAQ.md` |
| `ElevenlabsEngine` | Cloud | Use a multilingual model such as `eleven_multilingual_v2`. | `docs/engines/elevenlabs.md` |
| `SpeechifyEngine` | Cloud | `simba-3.2` (the default) is English. For other languages use `simba-3.0` with a `language` such as `fr-FR`. | `docs/engines/speechify.md` |
| `CartesiaEngine` | Cloud | Optional `language` field per request or voice. | `docs/engines/cartesia.md` |
| `CambEngine` | Cloud | "Multilingual"; `language` parameter, default `en-us`. | `docs/engines/camb.md` |
| `TypecastEngine` | Cloud | `language` parameter. | `docs/engines/typecast.md` |

## Speech to text (RealtimeSTT)

### Engines with an explicit language list in the repo

| Engine | Model | Supported languages | Source |
|---|---|---|---|
| `qwen3_asr` | `Qwen/Qwen3-ASR-1.7B` (default) | Language codes the adapter maps: Arabic (`ar`), German (`de`), English (`en`), Spanish (`es`), French (`fr`), Italian (`it`), Japanese (`ja`), Korean (`ko`), Portuguese (`pt`), Russian (`ru`), Chinese (`zh`). Other language names are passed through unchanged. | `RealtimeSTT/transcription_engines/qwen3_asr_engine.py` |
| Production server (`stt-server-production`) | Any engine | Default allowed request languages: `en`, `de`, `fr`, `es`, `it`, `pt`, `ru`. Configurable with `--supported-languages`. | `RealtimeSTT_server/production_server.py` |

### English-only engines

| Engine | Note | Source |
|---|---|---|
| `faster_whisper`, `whisper_cpp`, `openai_whisper` with a `.en` model (`tiny.en`, `base.en`, `small.en`, `medium.en`) | `.en` models are English only. **This project uses `small.en`.** | `docs/engines/faster-whisper.md`, `docs/engines/whisper-cpp.md` |
| `moonshine`, `moonshine_streaming` | English only in this adapter; other languages raise an error. | `docs/engines/moonshine.md` |
| `sherpa_onnx_moonshine` | English only in the adapter (`moonshine-tiny-en-int8`). | `docs/engines/sherpa-onnx.md` |
| `kroko_fp32` | Fixed to English. `en-US` and similar are normalized to `en`. `auto` and other languages are rejected. | `docs/engines/kroko-fp32.md` |
| `kroko_onnx` | The documented Community models are English (`Kroko-EN-...`). Other `.data` models carry their own language. | `docs/engines/kroko-onnx.md` |

### Multilingual engines without a full list in the repo

| Engine | Model | What the repo says | Source |
|---|---|---|---|
| `faster_whisper`, `whisper_cpp`, `openai_whisper` with a multilingual model (`tiny`, `base`, `small`, `medium`, `large-*`) | Whisper | Multilingual; set `language` or leave it empty for auto-detection. The repo has no list; see OpenAI Whisper's language list. | `docs/engines/faster-whisper.md` |
| `parakeet` (NeMo), `transcribe_cpp`, `sherpa_onnx_parakeet` | `nvidia/parakeet-tdt-0.6b-v3` | Parakeet v3 supports 25 European languages (not named in the repo). The project's own German test results were poor, so run a per-language WER check first. | `docs/engines/transcribe-cpp.md` |
| Orukeet (sherpa-onnx final recognizer) | Parakeet TDT derivative | 25 European languages (not named in the repo). | `docs/engines/sherpa-onnx.md` |
| `sherpa_onnx_nemotron` | Nemotron 3.5 streaming 0.6B | Multilingual; a fixed language per stream (such as `de-DE`) or `auto` detection. No list. | `docs/engines/sherpa-onnx.md` |
| `funasr` | `iic/SenseVoiceSmall` (default) | Experimental multilingual; `language="auto"` supported. No list. | `docs/engines/funasr.md` |
| `omnilingual_asr` | Meta Omnilingual ASR (`omniASR_*`) | LLM model cards take Omnilingual language IDs such as `eng_Latn`; CTC cards ignore language. No list. Linux/WSL2 with Python 3.11 only. | `docs/engines/omnilingual-asr.md` |
| `granite_speech` | IBM Granite Speech | Two-letter codes are mapped for common languages: English, German, Spanish, French, Japanese, Korean, Portuguese, Russian, Chinese. The model's real coverage isn't stated. | `docs/engines/hf-transformers.md` |
| `cohere_transcribe` | Cohere Transcribe | Requires an explicit `language` (no auto-detection). No list. | `docs/engines/cohere.md` |
| `openai_api` | none | Placeholder, not wired yet. | `docs/transcription-engines.md` |
