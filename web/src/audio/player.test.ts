import { afterEach, describe, it, expect, vi } from "vitest";

vi.mock("./player-worklet.ts?worker&url", () => ({ default: "player-worklet.js" }));

let gain: { gain: { value: number } };
let node: { port: { onmessage: ((e: MessageEvent) => void) | null; postMessage: () => void } };
let loaded: Promise<void>;

class FakeAudioContext {
  sampleRate = 48000;
  state = "running";
  destination = {};
  audioWorklet = { addModule: () => (loaded = Promise.resolve()) };
  createGain() { gain = { gain: { value: 1 }, connect: () => {} } as never; return gain; }
  createAnalyser() {
    return { fftSize: 0, connect: () => {}, getFloatTimeDomainData: (a: Float32Array) => a.fill(0.25) };
  }
  resume() { return Promise.resolve(); }
  close() { return Promise.resolve(); }
}
class FakeNode {
  port = { onmessage: null, postMessage: () => {} };
  constructor() { node = this as never; }
  connect() {}
}

afterEach(() => vi.unstubAllGlobals());

const { createPlayer } = await import("./player");

function make() {
  vi.stubGlobal("AudioContext", FakeAudioContext);
  vi.stubGlobal("AudioWorkletNode", FakeNode);
  return createPlayer({ sample_rate: 24000, channels: 1, encoding: "pcm_s16le" }, () => {});
}

describe("player speaker mute", () => {
  it("mutes before the worklet loads, and unmutes", async () => {
    const p = make();
    p.setMuted(true);
    await loaded;
    expect(gain.gain.value).toBe(0);
    p.setMuted(false);
    expect(gain.gain.value).toBe(1);
  });
  it("keeps word timing while muted", async () => {
    const p = make();
    await loaded;
    p.setMuted(true);
    const cb = vi.fn();
    p.schedule(0.5, cb);
    node.port.onmessage!({ data: { type: "reached", ids: [1] } } as MessageEvent);
    expect(cb).toHaveBeenCalledOnce();
  });
  it("reports the reply level", async () => {
    const p = make();
    expect(p.level()).toBe(0);
    await loaded;
    expect(p.level()).toBeCloseTo(0.25);
  });
});
