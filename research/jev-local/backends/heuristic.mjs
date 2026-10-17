// backends/heuristic.mjs — deterministic, offline stand-in for Jev's inference. TEST-ONLY.
// Lexical overlap between (state + instructions) and each criterion, plus a few
// cue-word heuristics, turned into a softmax distribution. Useful for contract
// tests and demos without any model; NOT an accuracy clone of Jev.

const STOP = new Set('a an the is are was were be been of to in on for and or not no yes it its this that with by as at from do does did has have had i my me we our you your they their'.split(' '));
const YES = ['ye', 'true', 'urgent', 'immediately', 'asap', 'fail', 'broken', 'help', 'refund', 'issu', 'angry', 'furiou', 'now', 'critical', 'outage', 'down', '!'];
const NO = ['no', 'false', 'fine', 'thank', 'resolv', 'calm', 'whenever', 'later', 'not'];

const text = (v) => (typeof v === 'string' ? v : JSON.stringify(v ?? '')).toLowerCase();
const stem = (t) => t.replace(/(ing|ed|es|s)$/,'');
const toks = (s) => text(s).replace(/[^a-z0-9!']+/g, ' ').split(' ').filter((t) => t && !STOP.has(t)).map(stem);
const softmax = (xs, t = 1) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp((x - m) / t)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };

function overlap(a, b) {
  const A = new Set(a); let n = 0; for (const t of b) if (A.has(t)) n++;
  return n / Math.sqrt((a.length || 1) * (b.length || 1));
}

export async function infer({ state, questions }) {
  const stateToks = toks(state);
  const out = {};
  for (const [id, q] of Object.entries(questions)) {
    const ctx = stateToks.concat(toks(q.instructions));
    if (q.type === 'noul') {
      const yes = ctx.filter((t) => YES.includes(t)).length + (q.criteria?.true ? overlap(stateToks, toks(q.criteria.true)) * 4 : 0);
      const no = ctx.filter((t) => NO.includes(t)).length + (q.criteria?.false ? overlap(stateToks, toks(q.criteria.false)) * 4 : 0);
      out[id] = 1 / (1 + Math.exp(-(yes - no) * 1.2));
    } else if (q.type === 'choice') {
      const scores = Object.entries(q.criteria).map(([k, d]) => 3 * overlap(ctx, toks(k)) + 6 * overlap(stateToks, toks(d ?? '')) + 0.01);
      out[id] = softmax(scores, 0.35);
    } else if (q.type === 'score') {
      const scores = q.criteria.map((d) => 6 * overlap(stateToks, toks(d)) + 0.01);
      // cue words push toward the top of ordered rubrics (urgency/anger style scales)
      const cue = ctx.filter((t) => YES.includes(t)).length;
      const boosted = scores.map((s, i) => s + (cue * i) / (q.criteria.length - 1) * 0.6);
      out[id] = softmax(boosted, 0.35);
    }
  }
  return out;
}
