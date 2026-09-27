import { describe, expect, it, vi } from 'vitest';

import { JefClient, JefError, boolean, choice, noul, score } from '../src/index.js';
import type { EvaluateResult } from '../src/types.js';

/** A fetch stand-in that records requests and replays canned responses. */
function stubFetch(responses: Array<{ status?: number; body: unknown }>) {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  let index = 0;
  const impl = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} });
    const next = responses[Math.min(index++, responses.length - 1)]!;
    return new Response(JSON.stringify(next.body), {
      status: next.status ?? 200,
      headers: { 'content-type': 'application/json' },
    });
  });
  return { impl: impl as unknown as typeof globalThis.fetch, calls };
}

describe('question builders', () => {
  it('keeps option order, because order decides the distribution index', () => {
    const q = choice('誰處理？', { soc: '監控', appsec: '程式碼', infra: '網路' });
    expect(Object.keys(q.criteria)).toEqual(['soc', 'appsec', 'infra']);
  });

  it('rejects a single-option choice', () => {
    expect(() => choice('誰處理？', { only: 'one' })).toThrow(/at least two options/);
  });

  it('keeps score levels ordered and distinct', () => {
    expect(score('嚴重度', ['低', '中', '高'] as const).criteria).toEqual(['低', '中', '高']);
    expect(() => score('嚴重度', ['低'] as const)).toThrow(/at least two/);
    expect(() => score('嚴重度', ['低', '低'] as const)).toThrow(/distinct/);
  });

  it('omits empty noul criteria rather than sending nulls', () => {
    expect(noul('是否緊急？')).not.toHaveProperty('criteria');
    expect(noul('是否緊急？', { true: '是' }).criteria).toEqual({ true: '是' });
  });

  it('spells the same primitive both ways', () => {
    expect(noul('是否緊急？').type).toBe('noul');
    expect(boolean('是否緊急？').type).toBe('boolean');
  });
});

describe('JefClient', () => {
  const answerBody: EvaluateResult = {
    model: 'jef/test',
    answers: {
      team: {
        type: 'choice',
        choice: 'soc',
        probabilities: { soc: 0.7, infra: 0.3 },
        confidence: 0.4,
      },
    },
    usage: { inputTokens: 42, outputTokens: 0, totalTokens: 42 },
  };

  it('posts the contract shape', async () => {
    const { impl, calls } = stubFetch([{ body: answerBody }]);
    const jef = new JefClient({ baseUrl: 'http://jef.test/', fetch: impl });

    const result = await jef.evaluate('付款失敗', {
      team: choice('誰處理？', { soc: '監控', infra: '網路' }),
    });

    expect(calls[0]!.url).toBe('http://jef.test/v1/systemone');
    const sent = JSON.parse(String(calls[0]!.init.body));
    expect(sent.state).toBe('付款失敗');
    expect(sent.questions.team.criteria).toEqual({ soc: '監控', infra: '網路' });
    expect(result.answers.team.choice).toBe('soc');
    // A System One model generates nothing.
    expect(result.usage.outputTokens).toBe(0);
  });

  it('accepts object and array states without turning an array into a batch', async () => {
    const { impl, calls } = stubFetch([{ body: answerBody }, { body: answerBody }]);
    const jef = new JefClient({ fetch: impl });

    await jef.evaluate({ alert: 'payout failed' }, { q: noul('緊急？') });
    await jef.evaluate(['甲', '乙', '丙'], { q: noul('緊急？') });

    expect(JSON.parse(String(calls[0]!.init.body)).state).toEqual({ alert: 'payout failed' });
    expect(JSON.parse(String(calls[1]!.init.body)).state).toEqual(['甲', '乙', '丙']);
  });

  it("surfaces the server's own error code rather than prose", async () => {
    const { impl } = stubFetch([
      { status: 422, body: { error: { code: 'invalid_question', message: 'unknown type' } } },
    ]);
    const jef = new JefClient({ fetch: impl });

    await expect(jef.evaluate('x', { q: noul('緊急？') })).rejects.toMatchObject({
      name: 'JefError',
      status: 422,
      code: 'invalid_question',
    });
  });

  it('keeps a non-envelope error body usable', async () => {
    const { impl } = stubFetch([{ status: 502, body: 'upstream exploded' }]);
    const jef = new JefClient({ fetch: impl });

    const error = await jef.evaluate('x', { q: noul('緊急？') }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(JefError);
    expect((error as JefError).status).toBe(502);
  });

  it('runs a scene and returns the trace intact', async () => {
    const trace = {
      scene: 'incident-triage',
      version: 1,
      model: 'jef/test',
      verdict: 'decided',
      action: 'escalate_human',
      human_review: true,
      action_params: { queue: 'analyst_review' },
      layers: [],
      layers_skipped: ['L3'],
      state_encodes: 1,
      questions_asked: 4,
      calibrated: false,
    };
    const { impl, calls } = stubFetch([{ body: trace }]);
    const jef = new JefClient({ fetch: impl });

    const result = await jef.runScene('incident-triage', '告警內容');
    expect(calls[0]!.url).toContain('/v1/scenes/incident-triage:evaluate');
    // The whole playbook costs one state read.
    expect(result.state_encodes).toBe(1);
    expect(result.layers_skipped).toEqual(['L3']);
  });

  it('escapes scene ids in the path', async () => {
    const { impl, calls } = stubFetch([{ body: { scene: 'x' } }]);
    await new JefClient({ fetch: impl }).runScene('a/b c', 'state').catch(() => undefined);
    expect(calls[0]!.url).toContain('a%2Fb%20c:evaluate');
  });

  it('honours a caller-supplied abort signal alongside its own timeout', async () => {
    const controller = new AbortController();
    const impl = vi.fn(async (_url: unknown, init?: RequestInit) => {
      expect(init?.signal).toBeDefined();
      expect(init?.signal?.aborted).toBe(true);
      return new Response(JSON.stringify(answerBody), { status: 200 });
    }) as unknown as typeof globalThis.fetch;

    controller.abort();
    await new JefClient({ fetch: impl }).evaluate('x', { q: noul('緊急？') }, {
      signal: controller.signal,
    });
  });

  it('strips trailing slashes from the base url', () => {
    expect(new JefClient({ baseUrl: 'http://jef.test///' }).baseUrl).toBe('http://jef.test');
  });
});
