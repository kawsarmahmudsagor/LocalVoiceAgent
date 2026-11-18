import type { ClientCommand, ServerEvent } from "./protocol";

/** Reconnect backoff: 0.5 s doubling, capped at 8 s. */
export function nextDelay(attempt: number): number {
  return Math.min(500 * 2 ** attempt, 8000);
}

export function socketUrl(): string {
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
}

export interface SocketHandlers {
  onEvent(e: ServerEvent): void;
  onAudio(buf: ArrayBuffer): void;
  onOpen(): void;
  onClose(): void;
}

/** One WebSocket to the server: JSON events + binary PCM, reconnecting until close(). */
export class VoiceSocket {
  private ws: WebSocket | null = null;
  private attempt = 0;
  private closed = false;
  private retryTimer: number | undefined;
  private readonly url: string;
  private readonly handlers: SocketHandlers;

  constructor(url: string, handlers: SocketHandlers) {
    this.url = url;
    this.handlers = handlers;
    this.connect();
  }

  private connect(): void {
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      this.handlers.onOpen();
    };
    ws.onmessage = (e: MessageEvent) => {
      if (typeof e.data === "string") {
        try {
          this.handlers.onEvent(JSON.parse(e.data) as ServerEvent);
        } catch {
          // ignore malformed frames
        }
      } else {
        this.handlers.onAudio(e.data as ArrayBuffer);
      }
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      this.handlers.onClose();
      if (!this.closed) this.retryTimer = window.setTimeout(() => this.connect(), nextDelay(this.attempt++));
    };
  }

  private get openSocket(): WebSocket | null {
    return this.ws && this.ws.readyState === WebSocket.OPEN ? this.ws : null;
  }

  send(cmd: ClientCommand): void {
    this.openSocket?.send(JSON.stringify(cmd));
  }

  sendAudio(pcm: Int16Array): void {
    this.openSocket?.send(pcm);
  }

  close(): void {
    this.closed = true;
    window.clearTimeout(this.retryTimer);
    this.ws?.close();
    this.ws = null;
  }
}
