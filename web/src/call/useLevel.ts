import { useEffect, useState } from "react";

/**
 * A voice level, re-read every animation frame while `active` (0 otherwise). Read it in a leaf
 * component so the per-frame updates do not re-render the rest of the call screen.
 */
export function useLevel(read: () => number, active: boolean): number {
  const [level, setLevel] = useState(0);
  useEffect(() => {
    if (!active) {
      setLevel(0);
      return;
    }
    let frame = 0;
    const tick = () => {
      setLevel(read());
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [read, active]);
  return level;
}
