import { ChatTranscript } from "./components/ChatTranscript";
import { Composer } from "./components/Composer";
import { SettingsPanel } from "./components/SettingsPanel";
import { StatusBar } from "./components/StatusBar";
import { RagPanel } from "./components/RagPanel";
import { SystemPromptPanel } from "./components/SystemPromptPanel";
import { Tabs } from "./components/Tabs";
import { deleteDocument, ragLocked, reindexAll, uploadDocuments } from "./rag";
import { useVoiceSession } from "./session/useVoiceSession";
import type { AppState } from "./state";

const TAB_KEY = "voicebot.tab";

function initState(base: AppState): AppState {
  try {
    const t = window.localStorage.getItem(TAB_KEY);
    if (t === "playground" || t === "prompt" || t === "rag") return { ...base, tab: t };
  } catch { /* storage may be blocked */ }
  return base;
}

export default function App() {
  const session = useVoiceSession(initState);
  const { state, dispatch } = session;
  const onSettingsChange = session.updateSettings;
  const onToggleMic = session.toggleMic;
  const onSend = session.send;
  const onStop = session.stop;
  const onClear = session.clear;
  const onSelectModel = session.selectModel;
  const onUpdateRag = session.updateRag;

  const reportRagError = (err: unknown) =>
    dispatch({ kind: "error", message: err instanceof Error ? err.message : String(err) });

  const onTab =(tab: AppState["tab"]) => {
    dispatch({ kind: "tab", tab });
    try { window.localStorage.setItem(TAB_KEY, tab); } catch { /* storage may be blocked */ }
  };

  const ragDisabled = ragLocked(state.status, state.connected);

  return (
    <div className="shell">
      <Tabs tab={state.tab} onChange={onTab} />
      <div className="tabpanel" role="tabpanel" id="panel-playground" aria-labelledby="tab-playground"
        hidden={state.tab !== "playground"}>
        <div className="app">
          <SettingsPanel config={state.config} settings={state.settings} onChange={onSettingsChange}
            catalog={state.catalog} switching={state.switching} status={state.status} onSelectModel={onSelectModel}
            wordAlign={state.wordAlign} diarization={state.diarization} />
          <main className="main">
            <ChatTranscript messages={state.messages} partial={state.partial} />
            <div className="dock">
              <StatusBar config={state.config} switchingSlug={state.switchingSlug} connected={state.connected} status={state.status} metrics={state.metrics} error={state.error}
                onDismissError={() => dispatch({ kind: "dismiss_error" })} />
              <Composer status={state.status} listening={state.listening} connected={state.connected}
                onToggleMic={() => void onToggleMic()} onSend={onSend} onStop={onStop} onClear={onClear} />
            </div>
          </main>
        </div>
      </div>
      <div className="tabpanel scroll" role="tabpanel" id="panel-prompt" aria-labelledby="tab-prompt"
        hidden={state.tab !== "prompt"}>
        <SystemPromptPanel settings={state.settings} defaults={state.config?.defaults ?? null}
          disabled={!state.connected} onChange={onSettingsChange} />
      </div>
      <div className="tabpanel scroll" role="tabpanel" id="panel-rag" aria-labelledby="tab-rag"
        hidden={state.tab !== "rag"}>
        <RagPanel rag={state.rag} disabled={ragDisabled} onUpdate={onUpdateRag}
          error={state.error} onDismissError={() => dispatch({ kind: "dismiss_error" })}
          onUpload={files => uploadDocuments(files)} onDelete={id => void deleteDocument(id).catch(reportRagError)}
          onReindex={() => void reindexAll().catch(reportRagError)} />
      </div>
    </div>
  );
}
