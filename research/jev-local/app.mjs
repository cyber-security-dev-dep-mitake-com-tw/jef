// app.mjs — the request handler, separate from listening so it can be tested
// without a port and without a backend that leaves the machine.
import { createHash } from 'node:crypto';
import { gunzipSync, zstdDecompressSync } from 'node:zlib';
import * as core from './jev-core.mjs';

export const MODELS_NATIVE = [
  { id: 'jev-latest', name: 'Jev (latest)', description: 'TypeSafe System One decision model. Points to jev-1.13.0.', released: '2026-09-15' },
  { id: 'jev-preview', name: 'Jev (preview)', description: 'Most recent release. Points to jev-1.14.0.', released: '2026-10-14' },
];
export const MODELS_GATEWAY = [{
  id: 'typesafe-ai/jev', object: 'model', owned_by: 'typesafe-ai', name: 'Jev', type: 'evaluation',
  description: 'TypeSafe AI System One decision model (local reproduction). Accepts shared state and typed questions, returning choices, scores, and boolean probabilities.',
  context_window: 32000, max_tokens: 0, supported_specifications: ['v4'], modalities: { input: ['text'], output: ['text'] },
  pricing: { input: '0.000000042', output: '0' },
}];

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let size = 0;
    req.on('data', (c) => { size += c.length; if (size > core.LIMITS.maxBodyBytes) { reject(new core.JevError(413, 'payload_too_large', 'Body is over 32 MiB')); req.destroy(); } chunks.push(c); });
    req.on('end', () => resolve(Buffer.concat(chunks))); req.on('error', reject);
  });
}

/**
 * Decode one Content-Encoding layer and parse JSON.
 *
 * Both decompressors are capped at the same limit as a plain body. gzip used to
 * be uncapped: a 32 MiB gzip body expands to gigabytes, so the 32 MiB body
 * limit protected nothing once the client compressed. A bomb now stops at
 * maxBodyBytes of output and is answered with 413.
 */
export function decodeBody(headers, buf) {
  const enc = (headers['content-encoding'] || '').toLowerCase();
  if (enc.includes(',')) throw new core.JevError(415, 'unsupported_encoding', 'Only one Content-Encoding layer is supported');
  const cap = { maxOutputLength: core.LIMITS.maxBodyBytes };
  const tooLarge = () => new core.JevError(413, 'payload_too_large', 'Decompressed body is over 32 MiB');
  const isCap = (e) => e?.code === 'ERR_BUFFER_TOO_LARGE' || /buffer|output|length/i.test(String(e?.message));
  let bytes = buf;
  if (enc === 'gzip' || enc === 'x-gzip') {
    try { bytes = gunzipSync(buf, cap); } catch (e) { if (isCap(e)) throw tooLarge(); throw new core.JevError(400, 'malformed_body', 'Malformed gzip body'); }
  } else if (enc === 'zstd') {
    try { bytes = zstdDecompressSync(buf, cap); } catch (e) { if (isCap(e)) throw tooLarge(); throw new core.JevError(400, 'malformed_body', 'Malformed zstd body'); }
  } else if (enc) throw new core.JevError(415, 'unsupported_encoding', `Unsupported Content-Encoding ${enc}`);
  if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) throw new core.JevError(400, 'bom', 'UTF-8 BOM is not allowed');
  let text; try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); } catch { throw new core.JevError(400, 'invalid_utf8', 'Body is not valid UTF-8'); }
  try { return JSON.parse(text); } catch { throw new core.JevError(400, 'invalid_json', 'Body is not valid JSON'); }
}

/**
 * Who is asking, for cache isolation only. The digest of the Authorization
 * header: two callers with different tokens never share a cache entry, and
 * the token itself is never written anywhere.
 */
export function tenantOf(headers) {
  const auth = headers.authorization || '';
  return auth ? 't-' + createHash('sha256').update(auth).digest('hex').slice(0, 16) : 'anonymous';
}

const send = (res, status, obj, extra = {}) => { res.writeHead(status, { 'content-type': 'application/json', ...extra }); res.end(JSON.stringify(obj)); };

export function createHandler(backend) {
  const flags = { 'x-jev-backend': backend.name, ...(backend.testOnly ? { 'x-jev-test-only': 'true' } : {}) };

  async function evaluate(state, questions, tenant) {
    const r = await backend.infer({ state, questions, tenant });
    const raw = r.answers ?? r; const meta = r.meta ?? {};
    const answers = Object.fromEntries(Object.entries(questions).map(([id, q]) => [id, core.buildNativeAnswer(q, raw[id])]));
    return { answers, meta };
  }

  return async function handler(req, res) {
    const url = new URL(req.url, 'http://localhost');
    try {
      if (req.method === 'GET' && url.pathname === '/v1/models')
        return send(res, 200, url.searchParams.get('gateway') ? { object: 'list', data: MODELS_GATEWAY } : { models: MODELS_NATIVE });
      if (req.method === 'GET' && url.pathname === '/healthz')
        return send(res, 200, { ok: true, backend: backend.name, test_only: backend.testOnly, third_party: backend.thirdParty });
      if (req.method !== 'POST') return send(res, 404, { error: { code: 'not_found', message: 'Unknown route' } });

      const longContext = /^true$/i.test(req.headers['typesafe-danger-zone-long-context'] || '');
      const body = decodeBody(req.headers, await readBody(req));
      const tenant = tenantOf(req.headers);
      const started = Date.now();

      if (url.pathname === '/v1/systemone') {
        const modelId = core.validateNativeRequest(body, { longContext });
        const { answers, meta } = await evaluate(body.state, body.questions, tenant);
        const u = core.usageFor(body.state, body.questions);
        return send(res, 200, { model: modelId, answers, usage: u }, { ...flags, 'x-jev-latency-ms': String(Date.now() - started), ...(meta.cost_usd != null ? { 'x-jev-backend-cost-usd': String(meta.cost_usd) } : {}) });
      }

      if (url.pathname === '/v1/decision-model') {
        const spec = req.headers['ai-decision-model-specification-version'];
        if (spec && spec !== '4') return send(res, 400, { error: { code: 'unsupported_spec', message: `decision model specification v${spec} is not supported` } });
        const gatewayModelId = req.headers['ai-model-id'] || 'typesafe-ai/jev';
        if (!core.MODEL_IDS[gatewayModelId]) return send(res, 404, { error: { code: 'model_not_found', message: `Unknown model ${gatewayModelId}` } });
        const state = core.validateGatewayRequest(body, { longContext });
        const nativeQs = core.gatewayQuestionsToNative(body.questions);
        const { answers, meta } = await evaluate(state, nativeQs, tenant);
        const u = core.usageFor(state, nativeQs);
        const out = {
          answers: Object.fromEntries(Object.entries(answers).map(([id, a]) => [id, core.toGatewayAnswer(a)])),
          model: core.MODEL_IDS[gatewayModelId],
          rounding: core.ROUNDING,
          usage: { inputTokens: u.input_tokens, outputTokens: u.output_tokens },
          warnings: backend.testOnly ? ['backend=heuristic is TEST-ONLY: deterministic and semantically meaningless'] : [],
          providerMetadata: { gateway: { provider: 'typesafe-ai', backend: backend.name, ...meta }, typesafe: { confidence: Object.fromEntries(Object.entries(answers).filter(([, a]) => a.confidence != null).map(([id, a]) => [id, a.confidence])) } },
        };
        return send(res, 200, out, { ...flags, 'x-jev-latency-ms': String(Date.now() - started) });
      }
      return send(res, 404, { error: { code: 'not_found', message: 'Unknown route' } });
    } catch (e) {
      if (e instanceof core.JevError) return send(res, e.status, e.toJSON());
      console.error(e);
      return send(res, 500, { error: { code: 'internal', message: 'internal error' } });
    }
  };
}
