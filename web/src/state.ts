import type { CatalogEntry, Defaults, Metrics, Role, ServerConfig, ServerEvent, Status } from "./protocol";

export interface ChatMessage { id: number; role: "user" | "assistant"; text: string; streaming: boolean; interrupted: boolean }

export interface AppState {
  connected: boolean;
  listening: boolean;
  status: Status;
  config: ServerConfig | null;
  settings: Defaults | null;
  messages: ChatMessage[];
  metrics: Metrics | null;
  turnId: number | null;
  error: string | null;
  catalog: CatalogEntry[];
  /** The role being switched, and its target slug, while a model loads. */
  switching: Role | null;
  switchingSlug: string | null;
  /** Live realtime-STT text while the user speaks. */
  partial: string | null;
}

export type Action =
  | { kind: "server"; event: ServerEvent }
  | { kind: "connected"; value: boolean }
  | { kind: "listening"; value: boolean }
  | { kind: "settings"; patch: Partial<Defaults> }
  | { kind: "error"; message: string }
  | { kind: "dismiss_error" }
  | { kind: "clear" };

export const initialState: AppState = {
  connected: false,
  listening: false,
  status: "idle",
  config: null,
  settings: null,
  messages: [],
  metrics: null,
  turnId: null,
  error: null,
  catalog: [],
  switching: null,
  switchingSlug: null,
  partial: null,
};

const CONFIG_KEY: Record<Role, "stt_final" | "stt_realtime" | "tts"> = { stt_final: "stt_final", stt_realtime: "stt_realtime", tts: "tts" };

function nextId(messages: ChatMessage[]): number {
  return messages.length ? messages[messages.length - 1].id + 1 : 1;
}

function addMessage(messages: ChatMessage[], m: Omit<ChatMessage, "id">): ChatMessage[] {
  return [...messages, { ...m, id: nextId(messages) }];
}

function streamingAssistant(messages: ChatMessage[]): ChatMessage | undefined {
  const last = messages[messages.length - 1];
  return last && last.role === "assistant" && last.streaming ? last : undefined;
}

function applyServer(state: AppState, e: ServerEvent): AppState {
  switch (e.type) {
    case "config": {
      const { type: _type, ...config } = e;
      // Keep local edits if the server reconnects; take the server defaults only the first time.
      return { ...state, config, settings: state.settings ?? { ...config.defaults } };
    }
    case "status":
      return { ...state, status: e.state };
    case "user_transcript":
      return {
        ...state,
        partial: null,
        turnId: e.turn_id,
        messages: addMessage(state.messages, { role: "user", text: e.text, streaming: false, interrupted: false }),
      };
    case "assistant_delta": {
      const cur = streamingAssistant(state.messages);
      if (cur) {
        return { ...state, messages: state.messages.map(m => (m === cur ? { ...m, text: m.text + e.text } : m)) };
      }
      return { ...state, messages: addMessage(state.messages, { role: "assistant", text: e.text, streaming: true, interrupted: false }) };
    }
    case "assistant_done": {
      const cur = streamingAssistant(state.messages);
      if (cur) {
        const done = { ...cur, text: e.text || cur.text, streaming: false, interrupted: e.interrupted };
        return { ...state, messages: state.messages.map(m => (m === cur ? done : m)) };
      }
      if (!e.text) return state;
      return { ...state, messages: addMessage(state.messages, { role: "assistant", text: e.text, streaming: false, interrupted: e.interrupted }) };
    }
    case "metrics": {
      const { type: _type, ...metrics } = e;
      return { ...state, metrics };
    }
    case "error":
      return { ...state, error: e.message };
    case "catalog":
      return { ...state, catalog: e.entries };
    case "model_status": {
      if (e.state === "loading") return { ...state, switching: e.role, switchingSlug: e.slug };
      const cleared = { ...state, switching: null, switchingSlug: null };
      if (e.state === "failed") return { ...cleared, error: e.message ?? `Could not load ${e.slug}` };
      // STT switches send no new config: record the active slug here.
      return state.config ? { ...cleared, config: { ...state.config, [CONFIG_KEY[e.role]]: e.slug } } : cleared;
    }
    case "user_partial":
      return { ...state, partial: e.text };
  }
}

export function reducer(state: AppState, action: Action): AppState {
  switch (action.kind) {
    case "server":
      return applyServer(state, action.event);
    case "connected":
      return action.value ? { ...state, connected: true } : { ...state, connected: false, listening: false, status: "idle" };
    case "listening":
      return { ...state, listening: action.value };
    case "settings":
      return state.settings ? { ...state, settings: { ...state.settings, ...action.patch } } : state;
    case "error":
      return { ...state, error: action.message };
    case "dismiss_error":
      return { ...state, error: null };
    case "clear":
      return { ...state, messages: [], turnId: null };
  }
}
