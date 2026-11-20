# Call UI page (`/call`) — design

Date: 2026-11-20. Branch: `feat/call-UI-integration` (from `main`).

## 1. Goal

Add a second web page, `/call`, that looks and behaves like the call screen the TelcoAIAgent
frontend had (its `components/app/call/*` and `agent-session-block.tsx`, deleted in Telco commit
`a1d8317`; read them at `a1d8317^`). The current playground at `/` stays exactly as it is.

`/call` has:

- no settings, model pickers, prompt or RAG tabs: it uses whatever the server is set to;
- the call UI: avatar, status, waveform, and the mic / End call / speaker controls;
- a collapsible right pane with the conversation transcript and a message box.

It shows no latency readout; latency stays on `/`.

## 2. Decisions (agreed 2026-11-20)

| Topic | Decision |
|---|---|
| Base branch | `main` |
| Call start | A start screen with one "Start call" button; End call returns to it |
| Latency on `/call` | None |
| Styling | Tailwind CSS v4 + `motion` + `lucide-react`, as Telco used |
| Page split | Two Vite pages; Tailwind loads only on `/call` |
| Telco card overlay (packages, confirm codes) | Dropped: Telco-specific |
| Telco avatar SVG | Not reused (it is branded "Horizon Realty agent"); draw a neutral one |
| Implementation skills | `frontend-design` for visual direction, `ui-ux-pro-max` for accessibility, touch targets and motion checks |

## 3. Architecture

### 3.1 Two Vite pages

- `web/index.html` → `src/main.tsx` → `App` (unchanged playground).
- `web/call.html` → `src/call/main.tsx` → `CallApp`. Only this entry imports
  `src/call/call.css`, which holds `@import "tailwindcss"` and the call theme tokens. Tailwind's
  preflight therefore never reaches `/`.
- `vite.config.ts`: `build.rollupOptions.input = { main: "index.html", call: "call.html" }`,
  the `@tailwindcss/vite` plugin, and a dev-server rewrite of `/call` to `/call.html`.

### 3.2 Shared session hook

Move the session logic out of `App.tsx` into `web/src/session/useVoiceSession.ts`:
the `VoiceSocket`, the reducer, the player (created on `config`, recreated on a sample-rate
change), the mic, barge-in `cutPlayback`, `playback_started` acks, and the word-timed
`assistant_segment` / `assistant_words` display. The hook returns `state`, `dispatch` and the
actions `toggleMic`, `send`, `stop`, `clear`, `selectModel`, `updateSettings`, `updateRag`.

`App.tsx` keeps its JSX and calls the hook. Behaviour on `/` does not change; all existing
tests stay green. `CallApp` uses the same hook, so no session logic is duplicated.

### 3.3 Audio additions

- `Player` gains `setMuted(muted: boolean)` (a `GainNode` between the worklet and the
  destination) and `level(): number` (an `AnalyserNode` RMS, 0–1). Muting sets gain to 0 and
  keeps playback running, so word timing, `playback_started` and barge-in still work.
- `startMic` returns `level(): number` too (an `AnalyserNode` on the source).

### 3.4 Routing on the server

`server/voicebot/web.py`: register `GET /call` returning `FileResponse(WEB_DIST / "call.html")`
before the `StaticFiles` mount at `/`, only when the file exists. Test in `server/tests/test_web.py`.

### 3.5 Single client

The server takes one WebSocket at a time. `/` and `/call` cannot be open together; the second
shows the existing `session_busy` error. No server change.

## 4. Components (`web/src/call/`)

| File | Port of (Telco) | What it does |
|---|---|---|
| `CallApp.tsx` | `view-controller` + `agent-session-block` | Start screen ↔ call screen; owns `conversationOpen` and `speakerMuted` |
| `StartScreen.tsx` | `welcome-view` | "Start call" button. The click resumes audio and turns the mic on |
| `CallStage.tsx` | stage part of `agent-session-block` | Dark stage, soft radial glow, avatar, "Voice agent" title, status line, waveform |
| `CallAvatar.tsx` | `call-avatar.tsx` | Three ripple rings driven by `player.level()` while speaking; pulse when speaker muted; a halo driven by `mic.level()` while listening; a spinning arc while thinking |
| `CallWaveform.tsx` | `call-waveform.tsx` | 23 bars; agent colour while speaking, user colour while listening, idle otherwise |
| `CallControls.tsx` | `call-controls.tsx` | Mic (toggles listening), End call, Speaker (`setMuted`) |
| `ConversationButton.tsx` | `call-chat.tsx` (button) | Toggles the pane; `aria-expanded`, `aria-controls` |
| `ConversationPane.tsx` | `conversation-pane.tsx` | Right column that opens to 400px on wide screens and covers the call on narrow ones; inert when closed; Escape closes it and returns focus to the button |
| `CallTranscript.tsx` | `agent-chat-transcript` | User and assistant messages, live partial, interrupted/unheard marks, from `state.messages` and `state.partial` |
| `CallComposer.tsx` | `call-chat.tsx` (composer) | Rounded input + send; disabled while disconnected |
| `status.ts` | `status.ts` | `Status` → "Listening…", "Thinking…", "Speaking…", "Connected"; "Connecting…" while the socket is down |

Levels are read in leaf components with `requestAnimationFrame`, so per-frame updates do not
re-render the rest of the screen. `prefers-reduced-motion` stops ring scaling and the spinning arc.

### 4.1 Flows

- **Start call:** resume the player, `toggleMic()` on. Mic permission denied → stay on the
  call screen with the mic off and show the error under the status; typing still works.
- **End call:** `stop()` (cuts playback), mic off, `clear()`, close the pane, back to the start
  screen. The socket stays open.
- **Socket closed:** status shows "Connecting…"; `VoiceSocket` keeps reconnecting as it does now.
- **`session_busy`:** the server's error lands in `state.error` as today. The start screen shows
  `state.error`, and the Start button is disabled while `state.connected` is false.

## 5. Testing

Vitest, no GPU, network or models:

- `useVoiceSession`: the moved logic keeps the existing `state.test.ts` / `ws.test.ts` cases green.
- `status.ts`: every `Status` value and the disconnected case.
- `Player.setMuted`: gain goes to 0 and back; `schedule` callbacks still fire (mocked AudioContext).
- `CallApp`: start → call → end with a mocked socket; pane toggle, Escape, focus return.

Then `npm run test && npm run typecheck && npm run build`; the build emits `dist/call.html`.
Server: `pytest` for `GET /call`.

Manual (user, through the SSH tunnel): open `/call`, start a call, talk, barge in, type a
message, mute the speaker, open and close the pane at wide and phone widths, end the call.

## 6. Out of scope

Settings of any kind on `/call`, latency display on `/call`, Telco cards, LiveKit, and
multi-client support on the server.
