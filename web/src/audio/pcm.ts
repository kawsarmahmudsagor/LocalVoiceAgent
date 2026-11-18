// PCM helpers shared by the mic worklet, the player and tests.

/** Float32 [-1, 1] → Int16, clamped. */
export function floatTo16BitPCM(f: Float32Array): Int16Array {
  const out = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const s = Math.max(-1, Math.min(1, f[i]));
    out[i] = Math.round(s < 0 ? s * 32768 : s * 32767);
  }
  return out;
}

/** Linear-interpolation resampler. Returns a copy when the rates are equal. */
export function resample(input: Float32Array, inRate: number, outRate: number): Float32Array {
  if (inRate === outRate) return new Float32Array(input);
  const outLen = Math.round((input.length * outRate) / inRate);
  const out = new Float32Array(outLen);
  const ratio = inRate / outRate;
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

/** s16le bytes → Float32 [-1, 1). */
export function pcm16ToFloat(buf: ArrayBuffer): Float32Array {
  const ints = new Int16Array(buf, 0, Math.floor(buf.byteLength / 2));
  const out = new Float32Array(ints.length);
  for (let i = 0; i < ints.length; i++) out[i] = ints[i] / 32768;
  return out;
}
