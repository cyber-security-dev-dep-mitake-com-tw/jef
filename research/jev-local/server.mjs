// server.mjs — local, Jev-compatible decision-model server.
//   POST /v1/systemone        TypeSafe-native API  (docs.typesafe.ai/api)
//   POST /v1/decision-model   Vercel AI Gateway dialect used by `ai` experimental_decide via @ai-sdk/gateway
//   GET  /v1/models           model list (TypeSafe shape); ?gateway=1 for the gateway shape
//   GET  /healthz             backend name, test_only, third_party
// Env: PORT (4141), JEV_BACKEND=heuristic|claude (heuristic), JEV_ALLOW_THIRD_PARTY=1 (required for claude),
//      JEV_CLAUDE_MODEL, JEV_REDACT
import http from 'node:http';
import { createHandler } from './app.mjs';
import { BackendConfigError, resolveBackend } from './backends/index.mjs';

const PORT = Number(process.env.PORT || 4141);

let backend;
try {
  backend = await resolveBackend(process.env.JEV_BACKEND);
} catch (e) {
  if (e instanceof BackendConfigError) { console.error(`jev-local: ${e.message}`); process.exit(2); }
  throw e;
}

if (backend.testOnly) console.warn('jev-local: backend=heuristic is TEST-ONLY. Its answers are deterministic and carry no meaning.');
if (backend.thirdParty) console.warn('jev-local: backend=claude sends request state to Anthropic. JEV_ALLOW_THIRD_PARTY=1 was set.');

http.createServer(createHandler(backend)).listen(PORT, '127.0.0.1', () =>
  console.log(`jev-local listening on http://127.0.0.1:${PORT}  backend=${backend.name}`),
);
