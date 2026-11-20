/** What the call page shows, apart from the session itself. */
export interface CallView { inCall: boolean; paneOpen: boolean; speakerMuted: boolean }

export type CallEvent =
  | { kind: "start" }
  | { kind: "end" }
  | { kind: "pane"; open: boolean }
  | { kind: "speaker"; muted: boolean };

export const initialCall: CallView = { inCall: false, paneOpen: false, speakerMuted: false };

export function callReducer(v: CallView, e: CallEvent): CallView {
  switch (e.kind) {
    case "start": return { ...v, inCall: true };
    // A new call starts fresh: pane closed, speaker on.
    case "end": return initialCall;
    case "pane": return { ...v, paneOpen: e.open };
    case "speaker": return { ...v, speakerMuted: e.muted };
  }
}
