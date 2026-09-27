import type {
  EvaluateResult,
  Health,
  Limits,
  ModelInfo,
  Question,
  SceneSummary,
  SceneTrace,
  State,
} from './types.js';

/**
 * A JEF server returned an error.
 *
 * Carries the server's own code so a caller can branch on `invalid_question`
 * versus `too_many_questions` without parsing prose.
 */
export class JefError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(`${code} (${status}): ${message}`);
    this.name = 'JefError';
    this.status = status;
    this.code = code;
  }
}

export interface JefClientOptions {
  baseUrl?: string;
  /** Milliseconds. Defaults to 30s: a cold CPU encode is not instant. */
  timeoutMs?: number;
  headers?: Record<string, string>;
  fetch?: typeof globalThis.fetch;
}

/**
 * Client for `/v1/systemone` and `/v1/scenes`.
 *
 * @example
 * ```ts
 * const jef = new JefClient({ baseUrl: 'http://jef.internal:8080' });
 *
 * const { answers } = await jef.evaluate(alert, {
 *   team: choice('應由哪個團隊處理？', { soc: '監控事件', infra: '基礎設施' }),
 *   severity: score('評估嚴重度', ['低', '中', '高'] as const),
 * });
 *
 * answers.team.choice;       // 'soc' | 'infra'
 * answers.severity.score;    // continuous, may land between levels
 * ```
 */
export class JefClient {
  readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly headers: Record<string, string>;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(options: JefClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? 'http://localhost:8080').replace(/\/+$/, '');
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.headers = { 'content-type': 'application/json', ...options.headers };
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  /** Evaluate typed questions against one state. */
  async evaluate<Q extends Record<string, Question>>(
    state: State,
    questions: Q,
    init?: { signal?: AbortSignal },
  ): Promise<EvaluateResult<Q>> {
    return this.post<EvaluateResult<Q>>('/v1/systemone', { state, questions }, init);
  }

  /**
   * Run a scene and return its decision trace.
   *
   * The trace is the product, not the verdict: it carries the evidence, the
   * questions, the permitted answers, where the probability mass fell and which
   * gate fired — which is what an audit needs and what hand-written playbook
   * branching cannot produce.
   */
  async runScene(
    scene: string,
    state: State,
    init?: { signal?: AbortSignal },
  ): Promise<SceneTrace> {
    return this.post<SceneTrace>(
      `/v1/scenes/${encodeURIComponent(scene)}:evaluate`,
      { state },
      init,
    );
  }

  async scenes(): Promise<SceneSummary[]> {
    const body = await this.get<{ data: SceneSummary[] }>('/v1/scenes');
    return body.data;
  }

  async models(): Promise<ModelInfo[]> {
    const body = await this.get<{ data: ModelInfo[] }>('/v1/models');
    return body.data;
  }

  async limits(): Promise<Limits> {
    return this.get<Limits>('/v1/limits');
  }

  async health(): Promise<Health> {
    return this.get<Health>('/healthz');
  }

  // -- transport ------------------------------------------------------------ //

  private async post<T>(
    path: string,
    body: unknown,
    init?: { signal?: AbortSignal },
  ): Promise<T> {
    return this.request<T>(path, {
      method: 'POST',
      body: JSON.stringify(body),
      ...(init?.signal ? { signal: init.signal } : {}),
    });
  }

  private async get<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: 'GET' });
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    // A caller-supplied signal and the timeout must both be able to abort, so
    // they are combined rather than one silently winning.
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;

    const response = await this.fetchImpl(`${this.baseUrl}${path}`, {
      ...init,
      headers: this.headers,
      signal,
    });

    if (!response.ok) {
      let code = 'http_error';
      let message = await response.text();
      try {
        const parsed = JSON.parse(message) as { error?: { code?: string; message?: string } };
        code = parsed.error?.code ?? code;
        message = parsed.error?.message ?? message;
      } catch {
        // Not a JEF error envelope; keep the raw body as the message.
      }
      throw new JefError(response.status, code, message.slice(0, 500));
    }

    return (await response.json()) as T;
  }
}
