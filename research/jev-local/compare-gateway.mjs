// compare-gateway.mjs — side-by-side fidelity check of the local Jev reproduction
// against the real `typesafe-ai/jev` on Vercel AI Gateway.
//
//   (a) real   : ai `experimental_decide` with the DEFAULT gateway provider
//                (string model id), authenticated by AI_GATEWAY_API_KEY
//   (b) local  : the same client pointed at http://127.0.0.1:4141/v1 (heuristic backend)
//   (c) claude : optional, http://127.0.0.1:4142/v1 (JEV_BACKEND=claude), used when it answers /healthz
//
// Usage:
//   AI_GATEWAY_API_KEY=... node compare-gateway.mjs            # or put the key in ./.env.local
//   node compare-gateway.mjs --skip-claude                     # ignore port 4142 even if it is up
//   node compare-gateway.mjs --local-only                      # no gateway call (no agreement figures)
// Env: JEV_LOCAL_BASE (http://127.0.0.1:4141/v1), JEV_CLAUDE_BASE (http://127.0.0.1:4142/v1)
//
// Outputs:
//   - one table per question per case: both distributions, |Δp| per option, agreement
//   - a summary of agreement per target and question type, mean total-variation distance
//   - real-gateway-sample.json: raw gateway response body + headers for every case
//     (result.response.body / result.response.headers), plus the field-name diff against
//     the local server's raw body, so rounding/usage/warnings/providerMetadata can be matched exactly.

import { experimental_decide as decide, createGateway } from 'ai';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const VERSIONS = { ai: require('ai/package.json').version, gateway: require('@ai-sdk/gateway/package.json').version };

const args = new Set(process.argv.slice(2));
const LOCAL_BASE = process.env.JEV_LOCAL_BASE || 'http://127.0.0.1:4141/v1';
const CLAUDE_BASE = process.env.JEV_CLAUDE_BASE || 'http://127.0.0.1:4142/v1';
const SAMPLE_PATH = new URL('./real-gateway-sample.json', import.meta.url);
const REAL_MODEL_ID = 'typesafe-ai/jev';

// ---- key handling: env var, else ./.env.local (never hardcoded) --------------
if (!process.env.AI_GATEWAY_API_KEY) {
  const envFile = new URL('./.env.local', import.meta.url);
  if (existsSync(envFile)) {
    for (const line of readFileSync(envFile, 'utf8').split('\n')) {
      const m = line.match(/^\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
    }
  }
}
const haveKey = Boolean(process.env.AI_GATEWAY_API_KEY);
const localOnly = args.has('--local-only') || !haveKey;
if (!haveKey && !args.has('--local-only')) {
  console.error('AI_GATEWAY_API_KEY is not set. Export it, or write AI_GATEWAY_API_KEY=... to jev-repro/.env.local.');
  console.error('Running in --local-only mode (no real gateway call, no agreement figures).\n');
}

// ---- question set: the same four questions from test-decide.mjs, over several states ----
const QUESTIONS = {
  refunded:    { type: 'boolean', instructions: 'Was a refund issued?' },                                   // vercel.com snippet
  is_urgent:   { type: 'boolean', instructions: 'Does this convey urgency?', criteria: { true: 'Explicitly time-sensitive', false: 'No urgency expressed' } },
  department:  { type: 'choice',  instructions: 'Which team should handle this?', criteria: { billing: 'Payments, invoicing, refunds', technical: 'Bugs, outages, integrations', sales: 'Pricing, upgrades, new accounts' } },
  frustration: { type: 'score',   instructions: 'How frustrated is the customer?', criteria: ['Calm', 'Frustrated', 'Very angry'] },
};
const CASES = [
  { id: 'docs-payouts',   state: 'Help! My payouts have been failing for 3 days. The support agent issued a full refund to the customer.' },
  { id: 'calm-sales',     state: 'Hi, no rush at all. When you get a chance, could you tell me whether the Pro plan includes SSO? Thanks!' },
  { id: 'outage-angry',   state: 'The webhook integration returns 500 since your deploy this morning and our checkout is DOWN. Fix this NOW.' },
  { id: 'double-charge',  state: 'I was charged twice for invoice #4411. Please refund the duplicate charge.' },
  { id: 'resolved-thanks', state: 'Thanks for the quick fix yesterday, everything works again. We already received the refund for the downtime.' },
  { id: 'json-state',     state: { channel: 'chat', subject: 'Upgrade quote', body: 'We want to move 40 seats to Enterprise next quarter. What would that cost?' } },
];

// ---- targets -------------------------------------------------------------------
async function up(base) { try { const r = await fetch(base.replace(/\/v1$/, '') + '/healthz', { signal: AbortSignal.timeout(1500) }); return r.ok ? await r.json() : null; } catch { return null; } }

const targets = [];
if (!localOnly) targets.push({ name: 'real', model: REAL_MODEL_ID });   // string id → default gateway provider + AI_GATEWAY_API_KEY
const localHealth = await up(LOCAL_BASE);
if (!localHealth) { console.error(`Local server is not answering at ${LOCAL_BASE}. Start it with \`npm start\` (heuristic backend).`); process.exit(1); }
targets.push({ name: 'local', model: createGateway({ baseURL: LOCAL_BASE, apiKey: 'local-dev' }).decisionModel(REAL_MODEL_ID), backend: localHealth.backend });
if (!args.has('--skip-claude')) {
  const h = await up(CLAUDE_BASE);
  if (h) targets.push({ name: 'claude', model: createGateway({ baseURL: CLAUDE_BASE, apiKey: 'local-dev' }).decisionModel(REAL_MODEL_ID), backend: h.backend });
}
console.log(`targets: ${targets.map((t) => t.name + (t.backend ? `(${t.backend})` : '')).join(', ')}   ai@${VERSIONS.ai} @ai-sdk/gateway@${VERSIONS.gateway}\n`);

// ---- helpers ------------------------------------------------------------------
function distribution(question, answer) {
  if (answer.type === 'boolean') return { true: answer.probability, false: 1 - answer.probability };
  if (answer.type === 'choice') return answer.probabilities ?? Object.fromEntries(Object.keys(question.criteria).map((k) => [k, k === answer.choice ? 1 : 0]));
  if (answer.type === 'score') return answer.probabilities ?? null;
  return null;
}
function pick(question, answer) {
  if (answer.type === 'boolean') return String(answer.probability >= 0.5);
  if (answer.type === 'choice') return answer.choice;
  if (answer.type === 'score') {
    const d = distribution(question, answer);
    if (d) return Object.entries(d).sort((a, b) => b[1] - a[1])[0][0];
    return String(Math.round(answer.score));
  }
  return '?';
}
function optionsOf(question) {
  if (question.type === 'boolean') return ['true', 'false'];
  if (question.type === 'choice') return Object.keys(question.criteria);
  return question.criteria.map((_, i) => String(i));
}
const fmt = (x, w = 7) => (x == null ? '-' : x.toFixed(4)).padStart(w);
const pad = (s, w) => String(s).padEnd(w);
const tvd = (a, b, keys) => 0.5 * keys.reduce((s, k) => s + Math.abs((a?.[k] ?? 0) - (b?.[k] ?? 0)), 0);

// Field-name paths of a JSON value; answer ids and array indexes are collapsed to "*".
function keyPaths(v, prefix = '', out = new Set()) {
  if (Array.isArray(v)) { if (v.length === 0) out.add(prefix + '[]'); for (const x of v) keyPaths(x, prefix + '[*]', out); return out; }
  if (v && typeof v === 'object') {
    for (const [k, x] of Object.entries(v)) {
      const key = /^answers$/.test(prefix.split('.').pop() ?? '') ? '*' : k;
      const p = prefix ? `${prefix}.${key}` : key;
      out.add(`${p}:${Array.isArray(x) ? 'array' : x === null ? 'null' : typeof x}`);
      keyPaths(x, p, out);
    }
    return out;
  }
  return out;
}
const HOP_HEADERS = new Set(['date', 'content-length', 'content-type', 'connection', 'keep-alive', 'transfer-encoding', 'content-encoding', 'vary', 'server', 'alt-svc', 'cf-ray', 'age', 'via']);

// ---- run ----------------------------------------------------------------------
const runs = [];           // { case, target, result, ms, error }
for (const c of CASES) {
  for (const t of targets) {
    const started = performance.now();
    try {
      const result = await decide({ model: t.model, state: c.state, questions: QUESTIONS });
      runs.push({ case: c.id, target: t.name, result, ms: Math.round(performance.now() - started) });
    } catch (error) {
      runs.push({ case: c.id, target: t.name, error, ms: Math.round(performance.now() - started) });
      console.error(`[${t.name}] ${c.id}: ${error?.name ?? 'Error'}: ${error?.message ?? error}`);
      if (t.name === 'real' && /auth|401|api key|token/i.test(String(error?.message))) { console.error('Gateway authentication failed; check AI_GATEWAY_API_KEY.'); process.exit(1); }
    }
  }
}

const byCase = (id, name) => runs.find((r) => r.case === id && r.target === name);
const ref = targets[0].name;                 // 'real' when available, else 'local'
const others = targets.slice(1).map((t) => t.name);

// ---- per-question tables ------------------------------------------------------
const stats = {};  // target → type → { n, agree, tvdSum, maxAbs }
const bump = (t, type, agree, d, m) => { const s = ((stats[t] ??= {})[type] ??= { n: 0, agree: 0, tvdSum: 0, maxAbs: 0, scoreAbsSum: 0 }); s.n++; s.agree += agree ? 1 : 0; s.tvdSum += d; s.maxAbs = Math.max(s.maxAbs, m); };

for (const c of CASES) {
  const refRun = byCase(c.id, ref);
  console.log(`=== case ${c.id} ===`);
  console.log(`state: ${typeof c.state === 'string' ? c.state : JSON.stringify(c.state)}`);
  console.log(`latency ms: ${targets.map((t) => `${t.name}=${byCase(c.id, t.name)?.ms ?? '-'}`).join('  ')}`);
  for (const [qid, q] of Object.entries(QUESTIONS)) {
    const opts = optionsOf(q);
    const cols = targets.map((t) => { const r = byCase(c.id, t.name); const a = r?.result?.answers?.[qid]; return { name: t.name, answer: a, dist: a ? distribution(q, a) : null }; });
    const refCol = cols[0];
    const w = Math.max(10, ...opts.map((o) => o.length + 2));
    const header = [pad(`${qid} (${q.type})`, w), ...cols.map((col) => col.name.padStart(8)), ...cols.slice(1).map((col) => `|Δ ${col.name}|`.padStart(10))].join(' ');
    console.log(header);
    for (const o of opts) {
      console.log([pad('  ' + o, w), ...cols.map((col) => fmt(col.dist?.[o], 8)), ...cols.slice(1).map((col) => fmt(col.dist && refCol.dist ? Math.abs(col.dist[o] - refCol.dist[o]) : null, 10))].join(' '));
    }
    const picks = cols.map((col) => (col.answer ? pick(q, col.answer) : '?'));
    const extra = q.type === 'score' ? `   score: ${cols.map((col) => `${col.name}=${col.answer?.score ?? '-'}`).join(' ')}` : '';
    const agreeStr = cols.slice(1).map((col, i) => {
      if (!col.answer || !refCol.answer) return `${col.name}: n/a`;
      const agree = picks[i + 1] === picks[0];
      const d = tvd(refCol.dist, col.dist, opts);
      const m = Math.max(...opts.map((o) => Math.abs((refCol.dist?.[o] ?? 0) - (col.dist?.[o] ?? 0))));
      if (ref === 'real') bump(col.name, q.type, agree, d, m);
      return `${col.name}: ${agree ? 'AGREE' : 'DIFFER'} (tvd ${d.toFixed(4)})`;
    }).join('   ');
    console.log(`  chosen: ${cols.map((col, i) => `${col.name}=${picks[i]}`).join(' ')}${extra}`);
    console.log(`  vs ${ref}: ${agreeStr}\n`);
  }
}

// ---- summary ------------------------------------------------------------------
if (ref === 'real') {
  console.log('=== agreement with real typesafe-ai/jev (chosen option / argmax score level / P(true)>=0.5) ===');
  console.log([pad('target', 8), pad('type', 8), 'agree'.padStart(8), 'mean tvd'.padStart(10), 'max |Δp|'.padStart(10)].join(' '));
  for (const t of others) for (const type of ['boolean', 'choice', 'score']) {
    const s = stats[t]?.[type]; if (!s) continue;
    console.log([pad(t, 8), pad(type, 8), `${s.agree}/${s.n}`.padStart(8), (s.tvdSum / s.n).toFixed(4).padStart(10), s.maxAbs.toFixed(4).padStart(10)].join(' '));
  }
  for (const t of others) {
    const all = Object.values(stats[t] ?? {}); const n = all.reduce((a, s) => a + s.n, 0); const ag = all.reduce((a, s) => a + s.agree, 0); const tv = all.reduce((a, s) => a + s.tvdSum, 0);
    console.log(`${t}: overall ${ag}/${n} agree (${n ? (100 * ag / n).toFixed(0) : 0}%), mean tvd ${(n ? tv / n : 0).toFixed(4)}`);
  }
  console.log();
}

// ---- raw sample + field diff ---------------------------------------------------
const realRuns = runs.filter((r) => r.target === 'real' && r.result);
const localRun = runs.find((r) => r.target === 'local' && r.result);
if (realRuns.length) {
  const realBody = realRuns[0].result.response.body;
  const localBody = localRun?.result.response.body;
  const realPaths = keyPaths(realBody), localPaths = localBody ? keyPaths(localBody) : new Set();
  const fieldDiff = {
    inRealNotLocal: [...realPaths].filter((p) => !localPaths.has(p)).sort(),
    inLocalNotReal: [...localPaths].filter((p) => !realPaths.has(p)).sort(),
  };
  const hdr = (h) => Object.keys(h ?? {}).filter((k) => !HOP_HEADERS.has(k.toLowerCase())).map((k) => k.toLowerCase()).sort();
  const headerDiff = { real: hdr(realRuns[0].result.response.headers), local: hdr(localRun?.result.response.headers) };

  const sample = {
    capturedAt: new Date().toISOString(),
    client: { ai: VERSIONS.ai, '@ai-sdk/gateway': VERSIONS.gateway, call: 'experimental_decide, default gateway provider' },
    modelId: REAL_MODEL_ID,
    questions: QUESTIONS,
    calls: realRuns.map((r) => ({
      case: r.case,
      state: CASES.find((c) => c.id === r.case).state,
      latencyMs: r.ms,
      response: { modelId: r.result.response.modelId, headers: r.result.response.headers, body: r.result.response.body },
      result: { answers: r.result.answers, usage: r.result.usage, rounding: r.result.rounding, warnings: r.result.warnings, providerMetadata: r.result.providerMetadata },
    })),
    localSample: localBody ? { base: LOCAL_BASE, headers: localRun.result.response.headers, body: localBody } : null,
    fieldDiff,
    headerDiff,
  };
  writeFileSync(SAMPLE_PATH, JSON.stringify(sample, null, 2) + '\n');
  console.log(`=== raw real response (case ${realRuns[0].case}) ===`);
  console.log(JSON.stringify(realBody, null, 2));
  console.log('\n=== body field paths: real vs local ===');
  console.log(`in real, missing locally : ${fieldDiff.inRealNotLocal.length ? '\n  ' + fieldDiff.inRealNotLocal.join('\n  ') : '(none)'}`);
  console.log(`in local, absent in real : ${fieldDiff.inLocalNotReal.length ? '\n  ' + fieldDiff.inLocalNotReal.join('\n  ') : '(none)'}`);
  console.log(`\nresponse headers (non hop-by-hop)\n  real : ${headerDiff.real.join(', ')}\n  local: ${headerDiff.local.join(', ')}`);
  console.log(`\nsaved ${realRuns.length} raw gateway responses to real-gateway-sample.json`);
} else if (!localOnly) {
  console.error('No successful real gateway call; real-gateway-sample.json not written.');
  process.exitCode = 1;
}
