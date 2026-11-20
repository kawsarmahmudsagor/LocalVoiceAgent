import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { SettingsPanel } from "./SettingsPanel";
import type { Defaults, ServerConfig } from "../protocol";

const baseDefaults: Defaults = { system_prompt: "", temperature: 0.7, top_p: 0.9, max_tokens: 256, voice: "v", rag_template: "" };
const noop = () => {};

function render(configPatch: Partial<ServerConfig>, defaultsPatch: Partial<Defaults>): string {
  const defaults = { ...baseDefaults, ...defaultsPatch };
  const config: ServerConfig = {
    llm_backend: "llamacpp", audio_io: "browser", model_label: "m", voices: ["v"], output_format: null,
    defaults, ...configPatch,
  };
  return renderToStaticMarkup(
    <SettingsPanel config={config} settings={defaults} onChange={noop} catalog={[]} switching={null}
      status="idle" onSelectModel={noop} wordAlign={{ state: "off" } as never}
      diarization={{ state: "off", message: null }} />);
}

describe("SettingsPanel diarization", () => {
  it("shows the controls and disables them when unavailable", () => {
    const html = render({ diarization_available: false }, { diarization: false, diar_mode: "ultra_low_latency", diar_wait_ms: 600 });
    expect(html).toContain("Speaker diarization"); expect(html).toContain('id="diar-mode"');
    expect(html).toContain("Max wait (ms)"); expect(html).toMatch(/<select id="diar-mode"[^>]*disabled/);
  });
  it("hides the controls for an older server", () => {
    expect(render({}, {})).not.toContain("Speaker diarization");
  });
});

describe("SettingsPanel barge-in", () => {
  it("shows the barge-in controls and disables First words without a realtime model", () => {
    const html = render({ barge_words_available: false }, { barge_in_mode: "onset", barge_in_min_ms: 200 });
    expect(html).toContain('id="barge-in"');
    expect(html).toMatch(/<option value="words" disabled="">First words<\/option>/);
    expect(html).toContain("Min speech (ms)");
  });
  it("hides the barge-in controls for an older server", () => {
    expect(render({}, {})).not.toContain('id="barge-in"');
  });
});
