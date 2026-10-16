// backends/claude.mjs — Jev's judgment engine on this Mac, using the local `claude`
// CLI (your Anthropic account). No Vercel traffic. One CLI call per request evaluates
// every question in parallel against the shared state, like Jev. Optional
// self-consistency sampling (JEV_SAMPLES>1) averages several independent judgments
// for better-calibrated probabilities. Results are cached on disk by content hash.
//
// Env: JEV_CLAUDE_MODEL (claude-sonnet-5-5; claude-haiku-5-5 is ~3x cheaper), JEV_SAMPLES (1), JEV_CACHE_DIR (.jev-cache),
//      JEV_CONCURRENCY (2), JEV_CLAUDE_TIMEOUT_MS (120000)

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const MODEL = process.env.JEV_CLAUDE_MODEL || 'claude-sonnet-5-5';
const SAMPLES = Math.max(1, Number(process.env.JEV_SAMPLES || 1));
const CACHE_DIR = process.env.JEV_CACHE_DIR || new URL('../.jev-cache/', import.meta.url).pathname;
const CONCURRENCY = Math.max(1, Number(process.env.JEV_CONCURRENCY || 2));
const TIMEOUT_MS = Number(process.env.JEV_CLAUDE_TIMEOUT_MS || 120000);

const SYSTEM = [
  'You are Jev, a System One decision model. You never write prose; you return calibrated probabilities as JSON.',
  'Read the STATE literally. Evaluate each QUESTION independently, using only its instructions and criteria as written and the state as given.',
  process.env.JEV_PROMPT === 'v1'
    ? 'Be calibrated: use probabilities near 0.5 when the state does not determine the answer. Do not reason at length.'
    : 'Calibration rules: be decisive when the state gives clear evidence (probabilities near 1 or 0); hedge toward the middle only when the state is genuinely ambiguous or silent; spread probability across the options that remain plausible.',
  'A choice question needs a probability for every listed option, summing to 1. A score question needs one probability per level in order, summing to 1; the levels are ordered from lowest to highest. A noul question needs a single number, P(yes).',
  'If an instruction refers to a field in backticks, look it up in the state or in the instructions object.',
  'Output exactly one JSON object, no code fences, no commentary: {"answers": {"<id>": <answer>}}.',
].join(' ');

function prompt({ state, questions }) {
  const spec = Object.entries(questions).map(([id, q]) => {
    if (q.type === 'noul') return { id, type: 'noul', instructions: q.instructions, ...(q.criteria ? { criteria: q.criteria } : {}), answer_format: 'number in [0,1] = P(yes)' };
    if (q.type === 'choice') return { id, type: 'choice', instructions: q.instructions, options: q.criteria, answer_format: `object with exactly these keys ${JSON.stringify(Object.keys(q.criteria))} mapping to probabilities summing to 1` };
    return { id, type: 'score', instructions: q.instructions, levels: q.criteria.map((d, i) => ({ level: i, description: d })), answer_format: `array of ${q.criteria.length} probabilities, one per level in order, summing to 1` };
  });
  return `${SYSTEM}\n\nSTATE:\n${typeof state === 'string' ? state : JSON.stringify(state)}\n\nQUESTIONS:\n${JSON.stringify(spec)}`;
}

function runClaude(text) {
  return new Promise((resolve, reject) => {
    const p = spawn('claude', ['-p', '--model', MODEL, '--output-format', 'json', '--no-session-persistence'], { stdio: ['pipe', 'pipe', 'pipe'] });
    let out = '', err = '';
    const timer = setTimeout(() => { p.kill('SIGKILL'); reject(new Error(`claude timed out after ${TIMEOUT_MS} ms`)); }, TIMEOUT_MS);
    p.stdout.on('data', (d) => (out += d)); p.stderr.on('data', (d) => (err += d));
    p.on('error', (e) => { clearTimeout(timer); reject(e); });
    p.on('close', (code) => { clearTimeout(timer); code === 0 ? resolve(out) : reject(new Error(`claude exited ${code}: ${err.slice(0, 300)}`)); });
    p.stdin.end(text);
  });
}

function extractAnswers(reply) {
  const start = reply.indexOf('{'); const end = reply.lastIndexOf('}');
  if (start < 0 || end < 0) throw new Error('no JSON object in model reply');
  const obj = JSON.parse(reply.slice(start, end + 1));
  if (!obj || typeof obj.answers !== 'object') throw new Error('reply lacks answers');
  return obj.answers;
}

/** One judgment pass → raw vectors per question, or throws. */
async function judge(state, questions) {
  let lastErr;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const envelope = JSON.parse(await runClaude(prompt({ state, questions })));
      const reply = typeof envelope.result === 'string' ? envelope.result : JSON.stringify(envelope.result ?? '');
      const answers = extractAnswers(reply);
      const out = {};
      for (const [id, q] of Object.entries(questions)) {
        const a = answers[id];
        if (q.type === 'noul') { if (typeof a !== 'number' || !(a >= 0 && a <= 1)) throw new Error(`bad noul for ${id}`); out[id] = a; }
        else if (q.type === 'choice') { const keys = Object.keys(q.criteria); const v = keys.map((k) => (a && typeof a[k] === 'number' && a[k] >= 0 ? a[k] : NaN)); if (v.some(Number.isNaN) || v.reduce((a, b) => a + b, 0) <= 0) throw new Error(`bad choice for ${id}`); out[id] = v; }
        else { const v = Array.isArray(a) ? a.map((x) => (typeof x === 'number' && x >= 0 ? x : NaN)) : []; if (v.length !== q.criteria.length || v.some(Number.isNaN) || v.reduce((a, b) => a + b, 0) <= 0) throw new Error(`bad score for ${id}`); out[id] = v; }
      }
      return { out, cost: envelope.total_cost_usd ?? 0, ms: envelope.duration_api_ms ?? 0 };
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

// simple concurrency gate so bursts of requests don't spawn unbounded CLI processes
let active = 0; const waiters = [];
const acquire = () => new Promise((r) => (active < CONCURRENCY ? (active++, r()) : waiters.push(r)));
const release = () => { active--; const w = waiters.shift(); if (w) { active++; w(); } };

export async function infer({ state, questions }) {
  const key = createHash('sha256').update(JSON.stringify({ MODEL, SAMPLES, state, questions })).digest('hex');
  const cacheFile = join(CACHE_DIR, key + '.json');
  try { return { ...JSON.parse(await readFile(cacheFile, 'utf8')), meta: { cached: true, model: MODEL } }; } catch {}

  await acquire();
  try {
    const runs = [];
    for (let i = 0; i < SAMPLES; i++) runs.push(await judge(state, questions));
    const answers = {};
    for (const [id, q] of Object.entries(questions)) {
      if (q.type === 'noul') answers[id] = runs.reduce((s, r) => s + r.out[id], 0) / runs.length;
      else answers[id] = runs[0].out[id].map((_, j) => runs.reduce((s, r) => s + r.out[id][j], 0) / runs.length);
    }
    const result = { answers };
    await mkdir(CACHE_DIR, { recursive: true }).then(() => writeFile(cacheFile, JSON.stringify(result))).catch(() => {});
    return { ...result, meta: { cached: false, model: MODEL, samples: SAMPLES, cost_usd: runs.reduce((s, r) => s + r.cost, 0), duration_ms: runs.reduce((s, r) => s + r.ms, 0) } };
  } finally { release(); }
}
