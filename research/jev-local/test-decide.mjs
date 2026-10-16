// test-decide.mjs — end-to-end: the real AI SDK `experimental_decide` + @ai-sdk/gateway
// client, pointed at the local Jev-compatible server, plus the native TypeSafe endpoint.
import { experimental_decide as decide, createGateway } from 'ai';

const BASE = process.env.JEV_BASE || 'http://127.0.0.1:4141/v1';
const gateway = createGateway({ baseURL: BASE, apiKey: 'local-dev' });
const model = gateway.decisionModel('typesafe-ai/jev');

const state = 'Help! My payouts have been failing for 3 days. The support agent issued a full refund to the customer.';

const result = await decide({
  model,
  state,
  questions: {
    refunded: { type: 'boolean', instructions: 'Was a refund issued?' },                       // vercel.com model-page snippet
    is_urgent: { type: 'boolean', instructions: 'Does this convey urgency?', criteria: { true: 'Explicitly time-sensitive', false: 'No urgency expressed' } },
    department: { type: 'choice', instructions: 'Which team should handle this?', criteria: { billing: 'Payments, invoicing, refunds', technical: 'Bugs, outages, integrations', sales: 'Pricing, upgrades, new accounts' } },
    frustration: { type: 'score', instructions: 'How frustrated is the customer?', criteria: ['Calm', 'Frustrated', 'Very angry'] },
  },
});

console.log('--- experimental_decide via gateway dialect ---');
console.log(JSON.stringify({ modelId: result.response.modelId, answers: result.answers, usage: result.usage, rounding: result.rounding, providerMetadata: result.providerMetadata }, null, 2));

console.log('--- native /v1/systemone ---');
const r = await fetch(`${BASE}/systemone`, { method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer local-dev' }, body: JSON.stringify({
  model: 'jev-latest', state,
  questions: { is_urgent: { type: 'noul', instructions: 'Does this convey urgency?' }, department: { type: 'choice', instructions: 'Which team should handle this?', criteria: { billing: 'Payments, invoicing, refunds', technical: 'Bugs, outages, integrations', sales: 'Pricing, upgrades, new accounts' } }, frustration: { type: 'score', instructions: 'How frustrated is the customer?', criteria: ['Calm', 'Frustrated', 'Very angry'] } },
}) });
console.log(r.status, JSON.stringify(await r.json(), null, 2));

console.log('--- contract checks: errors ---');
import assert from 'node:assert/strict';
const EXPECT = { 'unknown question type': [422, 'invalid_field'], 'missing model': [422, 'missing_field'], 'too many score levels': [422, 'invalid_field'], 'context too long': [400, 'max_tokens_exceeded'], 'context too long but long-context header': [200, null] };
for (const [name, body, headers] of [
  ['unknown question type', { model: 'jev-latest', state: 'x', questions: { q: { type: 'essay', instructions: 'write' } } }],
  ['missing model', { state: 'x', questions: { q: { type: 'noul', instructions: 'ok?' } } }],
  ['too many score levels', { model: 'jev-latest', state: 'x', questions: { q: { type: 'score', instructions: 'rate', criteria: Array.from({ length: 11 }, (_, i) => `L${i}`) } } }],
  ['context too long', { model: 'jev-latest', state: 'a'.repeat(140_000), questions: { q: { type: 'noul', instructions: 'ok?' } } }],
  ['context too long but long-context header', { model: 'jev-latest', state: 'a'.repeat(140_000), questions: { q: { type: 'noul', instructions: 'ok?' } } }, { 'TypeSafe-Danger-Zone-Long-Context': 'true' }],
]) {
  const rr = await fetch(`${BASE}/systemone`, { method: 'POST', headers: { 'content-type': 'application/json', ...(headers || {}) }, body: JSON.stringify(body) });
  const j = await rr.json();
  console.log(`${rr.status}  ${name}  ${j.error ? j.error.code : 'ok:' + Object.keys(j.answers).join(',')}`);
  const [status, code] = EXPECT[name];
  assert.equal(rr.status, status, `${name}: status`);
  if (code) assert.equal(j.error?.code, code, `${name}: error code`); else assert.ok(j.answers?.q, `${name}: answer q`);
}
// answer-shape assertions for the decide() result above
for (const [id, a] of Object.entries(result.answers)) {
  if (a.type === 'boolean') assert.ok(a.probability >= 0 && a.probability <= 1, `${id} probability range`);
  if (a.type === 'choice') { assert.ok(Object.hasOwn(a.probabilities, a.choice), `${id} choice in distribution`); assert.ok(Math.abs(Object.values(a.probabilities).reduce((s, x) => s + x, 0) - 1) < 1e-3, `${id} sums to 1`); }
  if (a.type === 'score') assert.ok(a.score >= 0 && a.score <= 2, `${id} score range`);
}
console.log('all contract assertions passed');
