# Benchmark conversation

Speak these five lines, in order, once for every model combination you test. They cover a greeting, proper nouns, a number, an open-ended request and a sign-off, so STT accuracy and reply length vary the same way in every run.

1. Hi, can you hear me clearly?
2. What is the capital of Australia?
3. How far is it from Sydney, in kilometers?
4. Give me one tip for staying focused while working.
5. Thanks, that's all for today.

## How to run one pass

1. Pick the models in the UI and wait until the loading pill disappears.
2. Press **Clear** so the conversation history is empty: every pass starts from the same state.
3. Press **Mic** and speak line 1. Wait until the reply has finished playing before speaking the next line.
4. Repeat for lines 2–5.

## What gets logged

Every turn appends to `logs/turns.jsonl` (machine-readable) and `logs/turns.log` (readable). Each record holds the selected models (`stt_final`, `stt_realtime`, `tts_model`), the text STT heard (`transcript`, shown as `heard:`), the reply, and every latency span. Compare `heard:` with the lines above to judge STT accuracy.

Summarize per model combination:

```
cd server; ../.venv/bin/python -m voicebot.latency_report --by stt_final,stt_realtime,tts_model --source voice
```
