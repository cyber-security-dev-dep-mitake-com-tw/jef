// register.mjs — makes the plain string form `decide({ model: 'typesafe-ai/jev' })`
// resolve to the local server, so snippets written for Vercel AI Gateway run unchanged:
//   node --import ./register.mjs snippet.mjs
import { createGateway } from 'ai';
globalThis.AI_SDK_DEFAULT_PROVIDER = createGateway({ baseURL: process.env.JEV_BASE || 'http://127.0.0.1:4141/v1', apiKey: 'local-dev' });
