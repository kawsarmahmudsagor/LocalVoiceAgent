/** Root-mean-square level of an audio block, clamped to 0–1. */
export function rmsLevel(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sum = 0;
  for (const s of samples) sum += s * s;
  return Math.min(1, Math.sqrt(sum / samples.length));
}

/** An AnalyserNode's current level, reading into a reused buffer. */
export function analyserLevel(analyser: AnalyserNode, buf: Float32Array<ArrayBuffer>): number {
  analyser.getFloatTimeDomainData(buf);
  return rmsLevel(buf);
}
