import type { CatalogEntry, Defaults, Metrics, RagCollection, RagDocument, RagEmbedder, RagSettings, RagSource, Role, ServerConfig, ServerEvent, Status, WordAlignState } from "./protocol";

export interface ChatMessage { id: number; role: "user" | "assistant"; text: string; streaming: boolean; interrupted: boolean; sources?: RagSource[];
  /** The part of an interrupted reply that was never heard. */
  unheard?: string }

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
  rag: { settings: RagSettings; embedders: RagEmbedder[]; collection: RagCollection | null;
    documents: RagDocument[]; indexing: { done: number; total: number } | null; error: string | null } | null;
  tab: "playground" | "prompt" | "rag";
  /** Forced word alignment, as the server last reported it. */
  wordAlign: { state: WordAlignState; message: string | null };
}

export type Action =
  | { kind: "server"; event: ServerEvent }
  | { kind: "connected"; value: boolean }
  | { kind: "listening"; value: boolean }
  | { kind: "settings"; patch: Partial<Defaults> }
  | { kind: "error"; message: string }
  | { kind: "dismiss_error" }
  /** Playback reached a sentence of this turn's reply: show it. */
  | { kind: "spoken"; turn_id: number; text: string }
  | { kind: "tab"; tab: AppState["tab"] }
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
  rag: null,
  tab: "playground",
  wordAlign: { state: "off", message: null },
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
      // Alignment state comes from the server; a config only tells whether it is on (not a load error).
      const wordAlign = state.wordAlign.state === "error" || state.wordAlign.state === "loading"
        ? state.wordAlign
        : { state: config.defaults.word_align ? "ready" as const : "off" as const, message: null };
      return { ...state, config, wordAlign, settings: state.settings ?? { ...config.defaults } };
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
    case "assistant_delta":
      // The reply text appears sentence by sentence as it is spoken ("spoken"); a delta only opens the bubble.
      if (streamingAssistant(state.messages)) return state;
      return { ...state, messages: addMessage(state.messages, { role: "assistant", text: "", streaming: true, interrupted: false }) };
    case "word_align":
      return { ...state, wordAlign: { state: e.state, message: e.message ?? null } };
    case "assistant_segment":
    case "assistant_words":
    case "barge_in":
      return state; // App schedules the words on the player and dispatches "spoken" as each one sounds
    case "assistant_done": {
      const cur = streamingAssistant(state.messages);
      if (cur) {
        // Interrupted: keep only the sentences that were heard.
        const text = e.interrupted ? cur.text : e.text || cur.text;
        const done: ChatMessage = { ...cur, text, streaming: false, interrupted: e.interrupted };
        if (e.interrupted) {
          const n = text.split(/\s+/).filter(Boolean).length;
          const unheard = e.text.split(/\s+/).filter(Boolean).slice(n).join(" ");
          if (unheard) done.unheard = unheard;
        }
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
    case "rag_config": {
      const prev = state.rag;
      return { ...state, rag: { documents: [], indexing: null, ...prev, settings: e.settings, embedders: e.embedders,
        collection: e.collection, error: e.error ?? null } };
    }
    case "rag_status": {
      const prev = state.rag;
      if (!prev) return state; // rag_config always arrives first on connect
      return { ...state, rag: { ...prev, documents: e.documents, indexing: e.indexing } };
    }
    case "rag_sources": {
      if (e.turn_id !== state.turnId) return state;
      const last = state.messages[state.messages.length - 1];
      if (last && last.role === "assistant") {
        return { ...state, messages: state.messages.map(m => (m === last ? { ...m, sources: e.sources } : m)) };
      }
      // Sources can arrive before the first delta: open the bubble the same way assistant_delta does.
      return { ...state, messages: addMessage(state.messages, { role: "assistant", text: "", streaming: true, interrupted: false, sources: e.sources }) };
    }
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
    case "spoken": {
      const cur = streamingAssistant(state.messages);
      if (!cur || state.turnId !== action.turn_id) return state;
      const text = cur.text ? `${cur.text} ${action.text.trim()}` : action.text.trim();
      return { ...state, messages: state.messages.map(m => (m === cur ? { ...m, text } : m)) };
    }
    case "tab":
      return { ...state, tab: action.tab };
    case "clear":
      return { ...state, messages: [], turnId: null };
  }
}
