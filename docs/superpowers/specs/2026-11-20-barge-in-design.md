# Local Voice Bot v3 — Barge-in

- **Date:** 2026-11-20
- **Status:** Draft, awaiting review
- **Branch:** `feat/v3-barge-In` (from `feat/voice-bot-v3-RAG` at `4abe1f3`)
- **Builds on:** [v1 design](2026-11-18-local-voice-bot-design.md), [v2 design](2026-11-18-v2-model-switching-design.md) and [v3 RAG design](2026-11-19-v3-rag-design.md). v2 §2 (Linux VM, browser audio) still applies. This spec lifts the "no barge-in" non-goal of v1 §2 and v2 §2.

## 1. Goal

Let the user interrupt the bot by speaking over it, and measure how fast the interruption takes effect. The interrupting speech becomes the next turn. Two trigger modes can be switched in the UI so their cost and false-trigger rate can be compared from the turn logs.

### What the user asked for

- Barge-in by voice.
- Two trigger modes, selectable from a UI setting (`off`, voice onset, first words).
- After a barge-in, history keeps only the words the user actually heard.

### Decisions made during design (2026-11-20)

| Topic | Decision | Reason |
|---|---|---|
| Mechanism | Use RealtimeSTT as it is. No vendored patch | After the first `text()` call the recorder sets `continuous_listening = True` (`RealtimeSTT/RealtimeSTT/core/lifecycle.py`, `wait_for_recorded_audio`). From then on it starts recording on voice even when no `text()` call is waiting, fires `on_vad_start` / `on_recording_start` (`core/recording.py`, around line 306), and queues the finished audio in `recorded_audio_queue`. The next `text()` takes it from there. So the mic only has to stay unmuted during a turn |
| Trigger | Both modes, chosen at runtime: `onset` (VAD onset plus a minimum speech time) and `words` (first realtime partial that is not noise) | `onset` is fast; `words` is robust. Measuring both is the point |
| Echo protection | Browser echo cancellation (already on in `web/src/audio/mic.ts`) plus `barge_in_min_ms`. No server-side echo gate | A home-made DSP gate is outside the vendored repos' resources. False triggers are counted so the rate is visible. Headphones are recommended for clean measurements |
| History after an interrupt | Keep only the words the browser revealed before the cut | The browser reveals each word when its audio plays (`web/src/words.ts`), so revealed text equals heard text. No duplication of the word-schedule logic in Python |
| Default mode | `off` | Existing runs and timings are unchanged unless barge-in is chosen |
| Stop button | Kept. It uses the same history cut as a barge-in | Both interrupts behave the same way |

### Success criteria

1. With mode `onset` or `words`, speaking over the bot stops its audio in the browser, and the speech is answered as the next turn.
2. With mode `off`, behaviour and timings are exactly as before (mic muted during the turn).
3. After any interrupt (barge-in or Stop), the assistant message stored in history is the text the user heard. If no audio played, no exchange is stored.
4. Every interrupted turn's `logs/turns.jsonl` record has `barge_detect_ms`, `barge_cut_ms`, `barge_in_mode`, `barge_in_min_ms` and `barge_false_triggers`, and `latency_report` can group by `barge_in_mode`.
5. The `words` option is disabled in the UI when no realtime STT model is loaded.

### Non-goals

- A server-side echo gate or any DSP beyond the browser's echo cancellation.
- Barge-in in server audio mode beyond the basic stop (no heard-text cut there; see §5).
- Resuming the interrupted reply.

## 2. Server flow

All changes are in `server/voicebot/session.py`, plus the STT callback wiring in `server/voicebot/stt.py`.

1. **Mic during a turn.** When `barge_in_mode != "off"`, `_run_turn` does not call `audio.mute_input(True)`. With `off`, the mute stays as today.
2. **Arming.** The mode and `barge_in_min_ms` are read once at turn start. The turn is armed from its start (through `thinking` and `speaking`) until its `finally` block. Speaking during `thinking` also cancels the reply: the LLM stops and the new utterance becomes the next turn. A typed turn (`send_text`) is armed only while the mic is listening.
3. **Trigger, `onset` mode.**
   - `on_recording_start` during an armed turn marks `barge_onset` and starts a `barge_in_min_ms` timer.
   - `on_turn_detection_start` before the timer runs out cancels it (the speech already ended) and increments `barge_false_triggers`.
   - When the timer runs out, the barge-in fires.
   - These are public RealtimeSTT callbacks. The existing timing callbacks in `stt.timing_callbacks` stay; the session chains its own handlers after them.
4. **Trigger, `words` mode.**
   - `on_recording_start` marks `barge_onset` as above (no timer).
   - The first `on_partial` text that `is_noise` does not reject fires the barge-in.
   - A recording that stops (`on_recording_stop`) before any such partial increments `barge_false_triggers` once. Counting each rejected partial would count one cough many times.
   - In `onset` mode, speech resuming (`on_turn_detection_stop` while a recording is active) after a cancelled timer re-arms the onset, because RealtimeSTT does not fire `on_recording_start` again inside the same recording.
5. **Fire.** Runs the existing `_stop_speaking` path (sets `stop_event`, `audio.cut()`, `player.stop(immediate=True)`, `stream.stop()` on a helper thread). It also marks `barge_in` on the turn timer and emits `{"type": "barge_in", "turn_id"}`. Only the first trigger in a turn fires; `stop_event` guards it.
6. **Next turn.** Nothing extra. The listen loop returns to `recorder.text()`, which picks up the queued (or still running) recording. A cancelled onset (a cough) also leaves a recording; its transcript is filtered by `is_noise` as today.
7. **Disarming.** Mic off, `clear`, a model switch or the end of the turn disarms. A pending onset timer is cancelled.

## 3. Protocol and browser

### New and changed messages

| Direction | Message | Fields | Notes |
|---|---|---|---|
| server → browser | `barge_in` | `turn_id` | New. Sent when a barge-in fires |
| browser → server | `playback_cut` | `turn_id`, `heard` | New. Sent after a `barge_in` event or after the user presses Stop |
| server → browser | `assistant_done` | adds `heard` (string or `null`) | `text` stays the full reply. `heard` is set on interrupted turns once the cut is known, else `null` |
| browser → server | `update_settings` | adds `barge_in_mode` (`"off"`, `"onset"`, `"words"`), `barge_in_min_ms` (int, 0–1000) | Bad values return `error{code:"bad_settings"}` and keep the old settings, as today |
| server → browser | `config` | adds `barge_in_mode`, `barge_in_min_ms`, `barge_words_available` (bool) | `barge_words_available` is true when a realtime STT model is loaded. It is resent after an STT switch |

### Browser behaviour

- On `barge_in` for the current turn: `player.flush()`, drop the scheduled word reveals, then send `playback_cut` with `heard` = the reply text revealed so far.
- On Stop: send `stop_speaking`, `player.flush()` (as today), then send `playback_cut` the same way.
- The transcript shows the full reply with the unheard tail greyed out, and the existing "interrupted" tag.
- Settings (Playground tab settings panel): a **Barge-in** select (`Off`, `Voice onset`, `First words`) and a **Min speech (ms)** number field. `First words` is disabled when `barge_words_available` is false. If the server reports `words` while it is unavailable (the realtime model was unloaded), the server falls back to `onset` and says so in `config`.

### History cut (server)

For an interrupted turn, before `conversation.add_exchange`:

1. Wait up to `ack_timeout` for `playback_cut` with the matching `turn_id`.
2. Ack received and `heard` not empty: store `heard` as the assistant reply.
3. Ack received and `heard` empty: the interrupt came before any audio was revealed. Store no exchange.
4. No ack (timeout, disconnect, server audio mode): store the full reply, as today.
5. A `playback_cut` with a stale or unknown `turn_id` is ignored, like a late `playback_started`.

`assistant_done` is emitted after this step so it can carry `heard`.

## 4. Metrics

New marks on the turn timer (server clock):

| Mark | When |
|---|---|
| `barge_onset` | `on_recording_start` during an armed turn |
| `barge_in` | The barge-in fires |
| `playback_cut` | The browser's `playback_cut` arrives |

New spans:

| Span | From → to | Meaning |
|---|---|---|
| `barge_detect_ms` | `barge_onset` → `barge_in` | What the trigger mode costs: the min-ms timer, or the wait for the first partial |
| `barge_cut_ms` | `barge_onset` → `playback_cut` | What the user experiences: detection, the round trip and the browser flush |

New record fields: `barge_in_mode`, `barge_in_min_ms`, `barge_false_triggers` (int, per turn).

`latency_report` keeps excluding interrupted turns from the reply-latency summaries. It adds a separate summary of the `barge_*` spans over interrupted turns, groupable by `barge_in_mode`.

## 5. Edge cases

- **Barge-in during the apology** after an LLM error: armed like any other reply.
- **Mic off mid-turn:** disarms. The turn finishes normally.
- **Model switch or `clear` during a turn:** existing guards apply; barge-in is disarmed.
- **Two triggers in one turn:** only the first fires.
- **`onset` with a realtime model loaded:** works; `on_recording_start` comes before any partial.
- **Server audio mode (Windows v1):** the barge-in fires and stops playback through the same path. There is no browser, so no `playback_cut` arrives and history keeps the full reply. `barge_cut_ms` is `null`.
- **Browser disconnects mid-turn:** no ack; the full reply is stored, as in rule 4 of §3.

## 6. Testing

All tests run without a GPU, Docker, network or models, with the existing fake recorder, TTS and backend.

**Server**

- `test_session.py`:
  - Mute stays off during a turn when the mode is `onset` or `words`, and stays on with `off`.
  - `onset`: fires after `barge_in_min_ms`; is cancelled by `on_turn_detection_start` before it, counting a false trigger.
  - `words`: fires on a partial that is not noise; noise partials are counted and ignored.
  - Only one fire per turn; disarmed after mic off and after the turn ends.
  - `barge_in` event emitted with the right `turn_id`.
  - History cut: ack with text stores `heard`; ack with empty text stores nothing; no ack stores the full reply; a stale `turn_id` is ignored. The Stop button follows the same rules.
  - `update_settings` rejects bad `barge_in_mode` and out-of-range `barge_in_min_ms`.
  - `config` reports `barge_words_available`.
- `test_timing.py`: the new marks and spans.
- `test_latency_report.py`: the `barge_*` summary and grouping by `barge_in_mode`.

**Web**

- `state.test.ts`: `barge_in` handling, `assistant_done.heard`, the new `config` fields.
- `words.test.ts`: the revealed text up to the cut.
- A settings test: the select sends `update_settings`, and `First words` is disabled when `barge_words_available` is false.

**Manual** (the user, through the SSH tunnel)

- Speak over the bot in each mode, with headphones and with speakers.
- Check that audio stops, the speech is answered next, history shows only the heard words, and `turns.jsonl` has `barge_cut_ms` and `barge_false_triggers`.
