const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export interface ProgressMessage {
  type: string;
  chapter: number;
  total: number;
  stage: string;
  debate_round: number;
  message: string;
}

export function createProgressSocket(
  projectId: number,
  onMessage: (msg: ProgressMessage) => void,
  onClose?: () => void,
  onError?: (err: Event) => void
): WebSocket {
  const ws = new WebSocket(`${WS_BASE}/api/ws/projects/${projectId}/progress`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      // ignore parse errors
    }
  };

  ws.onclose = () => {
    onClose?.();
  };

  ws.onerror = (err) => {
    onError?.(err);
  };

  return ws;
}
