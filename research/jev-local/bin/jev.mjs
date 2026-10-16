#!/usr/bin/env node
// bin/jev.mjs — local Jev on this Mac.
//   jev serve [--port 4141] [--backend claude|heuristic] [--model claude-haiku-5-5] [--samples N]
//   jev ask "<state>" --noul id="question" --choice id="question" opt1="desc" opt2="desc" --score id="question" "L0" "L1" "L2"
//   jev models
// `ask` talks to a running server (JEV_BASE, default http://127.0.0.1:4141/v1) and prints the native TypeSafe response.
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const cmd = args.shift();
const BASE = process.env.JEV_BASE || 'http://127.0.0.1:4141/v1';

function flag(name, def) { const i = args.indexOf(name); if (i < 0) return def; const v = args[i + 1]; args.splice(i, 2); return v; }

if (cmd === 'serve') {
  const env = { ...process.env, PORT: flag('--port', process.env.PORT || '4141'), JEV_BACKEND: flag('--backend', process.env.JEV_BACKEND || 'claude') };
  const model = flag('--model'); if (model) env.JEV_CLAUDE_MODEL = model;
  const samples = flag('--samples'); if (samples) env.JEV_SAMPLES = samples;
  const p = spawn(process.execPath, [join(here, '..', 'server.mjs')], { stdio: 'inherit', env });
  p.on('exit', (c) => process.exit(c ?? 0));
} else if (cmd === 'models') {
  const r = await fetch(`${BASE}/models`); console.log(JSON.stringify(await r.json(), null, 2)); process.exit(r.ok ? 0 : 1);
} else if (cmd === 'ask') {
  const state = args.shift(); if (!state) usage();
  const questions = {};
  while (args.length) {
    const t = args.shift();
    const [id, instructions] = (args.shift() || '').split(/=(.*)/s);
    if (!id || instructions === undefined) usage();
    if (t === '--noul') questions[id] = { type: 'noul', instructions };
    else if (t === '--choice') { const criteria = {}; while (args.length && !args[0].startsWith('--')) { const [k, d] = args.shift().split(/=(.*)/s); criteria[k] = d ?? null; } questions[id] = { type: 'choice', instructions, criteria }; }
    else if (t === '--score') { const criteria = []; while (args.length && !args[0].startsWith('--')) criteria.push(args.shift()); questions[id] = { type: 'score', instructions, criteria }; }
    else usage();
  }
  const r = await fetch(`${BASE}/systemone`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: process.env.JEV_MODEL || 'jev-latest', state, questions }) });
  const j = await r.json(); console.log(JSON.stringify(j, null, 2)); process.exit(r.ok ? 0 : 1);
} else usage();

function usage() {
  console.error(`usage:\n  jev serve [--port 4141] [--backend claude|heuristic] [--model claude-haiku-5-5] [--samples N]\n  jev ask "<state>" --noul id="question" | --choice id="question" opt="desc" ... | --score id="question" "L0" "L1" ...\n  jev models`);
  process.exit(2);
}
