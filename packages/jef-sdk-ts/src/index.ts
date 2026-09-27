/**
 * `@jef-ai/sdk` — client for JEF, an open System One decision engine.
 *
 * Evaluate typed questions against a shared state and get back probability
 * distributions with calibrated confidence. No text generation, no parsing.
 *
 * @example
 * ```ts
 * import { JefClient, choice, noul, score } from '@jef-ai/sdk';
 *
 * const jef = new JefClient({ baseUrl: 'http://localhost:8080' });
 *
 * const { answers, usage } = await jef.evaluate(
 *   '客戶回報：連續三天付款失敗，已影響營收。錯誤碼 502。',
 *   {
 *     urgent: noul('這則訊息是否表達時間緊迫？'),
 *     team: choice('應由哪個團隊處理？', {
 *       billing: '付款、發票、退款',
 *       infra: '基礎設施、網路、主機層',
 *     }),
 *     severity: score('評估此事件的嚴重度', ['資訊', '低', '中', '高', '危急'] as const),
 *   },
 * );
 *
 * answers.team.choice;   // 'billing' | 'infra'
 * usage.outputTokens;    // 0 — nothing is ever generated
 * ```
 */

export { JefClient, JefError } from './client.js';
export type { JefClientOptions } from './client.js';
export { boolean, choice, noul, score } from './questions.js';
export type * from './types.js';

export const VERSION = '0.1.0';
