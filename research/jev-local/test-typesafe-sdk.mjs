// test-typesafe-sdk.mjs — the official @typesafe-ai/sdk client pointed at local Jev.
// Default compression is zstd, so this also exercises the server's zstd path.
import { TypeSafeClient, choice, noul, score } from '@typesafe-ai/sdk';

const BASE = process.env.JEV_BASE || 'http://127.0.0.1:4141/v1';
const baseURL = BASE.replace(/\/v1$/, '');
const ticket = 'Our API integration started returning 500 errors on every request about 20 minutes ago, and we can\'t process any customer orders until this is fixed.';
const questions = {
  department: choice('Which team should handle this', { billing: 'Payment or subscription issues', technical: 'Bugs or integration problems', sales: 'Pricing or account questions' }),
  frustration: score('How frustrated the customer appears', ['Calm, just stating facts', 'Frustrated but civil', 'Very angry, strong language']),
  is_urgent: noul('The message conveys urgency or time-sensitivity'),
};
for (const compression of ['zstd', false]) {
  const client = new TypeSafeClient({ apiKey: 'local-dev', baseURL, defaultModel: 'jev-latest', compression, timeout: 180000 });
  const padded = compression ? ticket + ' ' + 'Context: '.repeat(150) : ticket; // >1 KiB so the SDK actually compresses
  const response = await client.systemOne({ state: padded, questions });
  console.log(`--- official SDK, compression=${compression} ---`);
  console.log('model:', response.model, '| department:', response.answers.department.choice, response.answers.department.confidence, '| frustration:', response.answers.frustration.score, '| is_urgent:', response.answers.is_urgent.noul, '| usage:', JSON.stringify(response.usage));
}
const models = await new TypeSafeClient({ apiKey: 'local-dev', baseURL }).models.list?.().catch(() => null);
if (models) console.log('models.list:', JSON.stringify(models).slice(0, 200));
