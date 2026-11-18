/** Runs at most one async action at a time; calls made while one is running are ignored. */
export function createExclusive(): (action: () => Promise<void>) => Promise<void> {
  let running = false;
  return async action => {
    if (running) return;
    running = true;
    try {
      await action();
    } finally {
      running = false;
    }
  };
}
