// server.mjs — local, Jev-compatible decision-model server.
//   POST /v1/systemone        TypeSafe-native API  (docs.typesafe.ai/api)
//   POST /v1/decision-model   Vercel AI Gateway dialect used by `ai` experimental_decide via @ai-sdk/gateway
//   GET  /v1/models           model list (TypeSafe shape); ?gateway=1 for the gateway shape
// Env: PORT (4141), JEV_BACKEND=heuristic|claude (heuristic), JEV_CLAUDE_MODEL
import http from 'node:http';
import { gunzipSync, zstdDecompressSync } from 'node:zlib';
import * as core from './jev-core.mjs';

const PORT = Number(process.env.PORT || 4141);
const BACKEND = process.env.JEV_BACKEND || 'heuristic';
const backend = await import(`./backends/${BACKEND}.mjs`);

const MODELS_NATIVE = [
  { id: 'jev-latest', name: 'Jev (latest)', description: 'TypeSafe System One decision model. Points to jev-1.13.0.', released: '2026-09-15' },
  { id: 'jev-preview', name: 'Jev (preview)', description: 'Most recent release. Points to jev-1.14.0.', released: '2026-10-14' },
];
const MODELS_GATEWAY = [{
  id: 'typesafe-ai/jev', object: 'model', owned_by: 'typesafe-ai', name: 'Jev', type: 'evaluation',
  description: 'TypeSafe AI System One decision model (local reproduction). Accepts shared state and typed questions, returning choices, scores, and boolean probabilities.',
  context_window: 32000, max_tokens: 0, supported_specifications: ['v4'], modalities: { input: ['text'], output: ['text'] },
  pricing: { input: '0.000000042', output: '0' },
}];

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let size = 0; let overLimit = false;
    req.on('data', (c) => {
      if (overLimit) return;
      size += c.length;
      if (size > core.LIMITS.maxBodyBytes) { overLimit = true; chunks.length = 0; reject(new core.JevError(413, 'payload_too_large', 'Body is over 32 MiB')); req.resume(); return; }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks))); req.on('error', reject);
  });
}

function decodeBody(req, buf) {
  const enc = (req.headers['content-encoding'] || '').toLowerCase();
  if (enc.includes(',')) throw new core.JevError(415, 'unsupported_encoding', 'Only one Content-Encoding layer is supported');
  let bytes = buf;
  if (enc === 'gzip' || enc === 'x-gzip') { try { bytes = gunzipSync(buf, { maxOutputLength: core.LIMITS.maxBodyBytes }); } catch { throw new core.JevError(400, 'malformed_body', 'Malformed gzip body'); } }
  else if (enc === 'zstd') { try { bytes = zstdDecompressSync(buf, { maxOutputLength: core.LIMITS.maxBodyBytes }); } catch { throw new core.JevError(400, 'malformed_body', 'Malformed zstd body'); } }
  else if (enc) throw new core.JevError(415, 'unsupported_encoding', `Unsupported Content-Encoding ${enc}`);
  if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) throw new core.JevError(400, 'bom', 'UTF-8 BOM is not allowed');
  let text; try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); } catch { throw new core.JevError(400, 'invalid_utf8', 'Body is not valid UTF-8'); }
  try { return JSON.parse(text); } catch { throw new core.JevError(400, 'invalid_json', 'Body is not valid JSON'); }
}

const send = (res, status, obj, extra = {}) => { res.writeHead(status, { 'content-type': 'application/json', ...extra }); res.end(JSON.stringify(obj)); };

async function evaluate(state, questions) {
  const r = await backend.infer({ state, questions });
  const raw = r.answers ?? r; const meta = r.meta ?? {};
  const answers = Object.fromEntries(Object.entries(questions).map(([id, q]) => [id, core.buildNativeAnswer(q, raw[id])]));
  return { answers, meta };
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    if (req.method === 'GET' && url.pathname === '/v1/models')
      return send(res, 200, url.searchParams.get('gateway') ? { object: 'list', data: MODELS_GATEWAY } : { models: MODELS_NATIVE });
    if (req.method === 'GET' && url.pathname === '/healthz') return send(res, 200, { ok: true, backend: BACKEND });
    if (req.method !== 'POST') return send(res, 404, { error: { code: 'not_found', message: 'Unknown route' } });

    const longContext = /^true$/i.test(req.headers['typesafe-danger-zone-long-context'] || '');
    const body = decodeBody(req, await readBody(req));
    const started = Date.now();

    if (url.pathname === '/v1/systemone') {
      const modelId = core.validateNativeRequest(body, { longContext });
      const { answers, meta } = await evaluate(body.state, body.questions);
      const u = core.usageFor(body.state, body.questions);
      return send(res, 200, { model: modelId, answers, usage: u }, { 'x-jev-backend': BACKEND, 'x-jev-latency-ms': String(Date.now() - started), ...(meta.cost_usd != null ? { 'x-jev-backend-cost-usd': String(meta.cost_usd) } : {}) });
    }

    if (url.pathname === '/v1/decision-model') {
      const spec = req.headers['ai-decision-model-specification-version'];
      if (spec && spec !== '4') return send(res, 400, { error: { code: 'unsupported_spec', message: `decision model specification v${spec} is not supported` } });
      const gatewayModelId = req.headers['ai-model-id'] || 'typesafe-ai/jev';
      if (!core.MODEL_IDS[gatewayModelId]) return send(res, 404, { error: { code: 'model_not_found', message: `Unknown model ${gatewayModelId}` } });
      const state = core.validateGatewayRequest(body, { longContext });
      const nativeQs = core.gatewayQuestionsToNative(body.questions);
      const { answers, meta } = await evaluate(state, nativeQs);
      const u = core.usageFor(state, nativeQs);
      const out = {
        answers: Object.fromEntries(Object.entries(answers).map(([id, a]) => [id, core.toGatewayAnswer(a)])),
        model: core.MODEL_IDS[gatewayModelId],
        rounding: core.ROUNDING,
        usage: { inputTokens: u.input_tokens, outputTokens: u.output_tokens },
        warnings: [],
        providerMetadata: { gateway: { provider: 'typesafe-ai', backend: BACKEND, ...meta } , typesafe: { confidence: Object.fromEntries(Object.entries(answers).filter(([, a]) => a.confidence != null).map(([id, a]) => [id, a.confidence])) } },
      };
      return send(res, 200, out, { 'x-jev-backend': BACKEND, 'x-jev-latency-ms': String(Date.now() - started) });
    }
    return send(res, 404, { error: { code: 'not_found', message: 'Unknown route' } });
  } catch (e) {
    if (e instanceof core.JevError) return send(res, e.status, e.toJSON());
    console.error(e);
    return send(res, 500, { error: { code: 'internal', message: String(e?.message || e) } });
  }
});

server.listen(PORT, '127.0.0.1', () => console.log(`jev-local listening on http://127.0.0.1:${PORT}  backend=${BACKEND}`));
