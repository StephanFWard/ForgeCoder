/** Minimal JSON HTTP + SSE client for the local Forge server. */
export class HttpError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = 'HttpError';
  }
}

export interface SseEvent {
  type: string;
  [key: string]: unknown;
}

export class HttpClient {
  constructor(private readonly baseUrl: string) {}

  get base(): string {
    return this.baseUrl.replace(/\/+$/, '');
  }

  async json<T>(path: string, body?: unknown, signal?: AbortSignal, headers?: Record<string, string>): Promise<T> {
    const resp = await this.fetch(path, body, signal);
    if (!resp.ok) {
      throw new HttpError(`${resp.status} ${resp.statusText}`, resp.status);
    }
    return (await resp.json()) as T;
  }

  async fetch(path: string, body?: unknown, signal?: AbortSignal, extraHeaders?: Record<string, string>): Promise<Response> {
    const headers: Record<string, string> = { Accept: 'application/json', ...(extraHeaders ?? {}) };
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json';
    }
    return fetch(`${this.base}${path}`, {
      method: 'POST',
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  }

  /** Read a Server-Sent-Events stream and dispatch each parsed event. */
  async stream(path: string, body: unknown, onEvent: (ev: SseEvent) => void, signal?: AbortSignal): Promise<void> {
    const resp = await this.fetch(path, body, signal);
    if (!resp.ok || !resp.body) {
      throw new HttpError(`${resp.status} ${resp.statusText}`, resp.status);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) {
          continue;
        }
        const payload = trimmed.slice(5).trim();
        if (payload === '[DONE]') {
          return;
        }
        try {
          onEvent(JSON.parse(payload) as SseEvent);
        } catch {
          // ignore malformed frames
        }
      }
    }
  }
}