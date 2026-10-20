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
 *   'EDR 於 02:14 偵測主機 fs-core-02 執行 vssadmin delete shadows /all，五分鐘內 3,200 個檔案被改為 .lockbit。',
 *   {
 *     urgent: noul('這則告警是否需要立即處理？'),
 *     team: choice('應由哪個團隊處理？', {
 *       soc: '資安監控：惡意程式、勒索軟體、端點與網路入侵',
 *       infra: '基礎設施：主機、備份、憑證與可用性',
 *     }),
 *     severity: score('評估此告警的嚴重度', ['資訊', '低', '中', '高', '危急'] as const),
 *   },
 * );
 *
 * answers.team.choice;   // 'soc' | 'infra'
 * usage.outputTokens;    // 0 — nothing is ever generated
 * ```
 */

export { JefClient, JefError } from './client.js';
export type { JefClientOptions } from './client.js';
export { boolean, choice, noul, score } from './questions.js';
export type * from './types.js';

export { VERSION } from './version.js';
