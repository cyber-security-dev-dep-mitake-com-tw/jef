// bench-fidelity.mjs — measures the local reproduction against Jev's documented
// answers (fixtures-docs.json: request + the real jev-1.13.0 response as published
// at docs.typesafe.ai). Usage: node bench-fidelity.mjs [baseURL ...]
//   default bases: http://127.0.0.1:4141/v1 (heuristic) and http://127.0.0.1:4142/v1 (claude) if reachable
import { readFileSync, writeFileSync } from 'node:fs';

const fixtures = JSON.parse(readFileSync(new URL('./fixtures-docs.json', import.meta.url)));
const bases = process.argv.slice(2).length ? process.argv.slice(2) : ['http://127.0.0.1:4141/v1', 'http://127.0.0.1:4142/v1'];

const l1 = (a, b) => Object.keys({ ...a, ...b }).reduce((s, k) => s + Math.abs((a[k] ?? 0) - (b[k] ?? 0)), 0);
const argmax = (o) => Object.entries(o).sort((x, y) => y[1] - x[1])[0][0];

async function alive(base) { try { const r = await fetch(`${base}/healthz`.replace('/v1/healthz', '/healthz')); return r.ok ? (await r.json()).backend : null; } catch { return null; } }

const report = {};
for (const base of bases) {
  const backend = await alive(base);
  if (!backend) { console.log(`skip ${base} (not running)`); continue; }
  const rows = []; let agree = 0, total = 0, pErr = 0, pN = 0, missing = 0;
  for (const f of fixtures) {
    const r = await fetch(`${base}/systemone`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'jev-latest', ...f.request }) });
    const got = await r.json();
    for (const [id, exp] of Object.entries(f.expected.answers)) {
      const a = got.answers?.[id]; if (!a) { total++; missing++; rows.push({ fixture: f.source, id, agree: false, error: got.error?.message || 'no answer' }); continue; }
      let row;
      if (exp.type === 'noul') { const d = Math.abs(a.noul - exp.noul); row = { kind: 'noul', expected: exp.noul, got: a.noul, abs_err: +d.toFixed(3), agree: (a.noul >= 0.5) === (exp.noul >= 0.5) }; pErr += d; pN++; }
      else if (exp.type === 'choice') { row = { kind: 'choice', expected: exp.choice, got: a.choice, agree: a.choice === exp.choice, l1: +l1(a.probabilities, exp.probabilities).toFixed(3), exp_conf: exp.confidence, got_conf: a.confidence }; pErr += row.l1 / 2; pN++; }
      else { row = { kind: 'score', expected: exp.score, got: a.score, abs_err: +Math.abs(a.score - exp.score).toFixed(3), agree: argmax(a.probabilities) === argmax(exp.probabilities), l1: +l1(a.probabilities, exp.probabilities).toFixed(3), exp_conf: exp.confidence, got_conf: a.confidence }; pErr += row.l1 / 2; pN++; }
      total++; if (row.agree) agree++;
      rows.push({ fixture: f.source, id, ...row });
    }
  }
  report[backend] = { base, questions: total, answered: total - missing, missing, complete: missing === 0, decision_agreement: +(agree / total).toFixed(3), mean_prob_error: pN ? +(pErr / pN).toFixed(3) : null, rows };
  console.log(`\n=== backend=${backend} (${base}) ===`);
  console.table(rows.map(({ fixture, id, kind, expected, got, agree, abs_err, l1 }) => ({ fixture, id, kind, expected: JSON.stringify(expected), got: JSON.stringify(got), agree, abs_err, l1 })));
  console.log(`decision agreement ${agree}/${total} = ${(100 * agree / total).toFixed(0)}%   mean probability error ${pN ? (pErr / pN).toFixed(3) : 'n/a'}${missing ? `   INCOMPLETE: ${missing} question(s) unanswered, counted as disagreements` : ''}`);
}
if (Object.keys(report).length === 0) { console.error('no backend answered; fidelity.json left unchanged'); process.exit(1); }
writeFileSync(new URL('./fidelity.json', import.meta.url), JSON.stringify(report, null, 2));
console.log('\nwrote fidelity.json');
if (Object.values(report).some((r) => !r.complete)) process.exit(1);
