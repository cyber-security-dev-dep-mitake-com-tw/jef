# @jef-ai/sdk

TypeScript client for [JEF](https://github.com/cyber-security-dev-dep-mitake-com-tw/jef),
an open System One decision engine. Evaluate typed questions against a shared
state and get back probability distributions with calibrated confidence. No text
generation, no parsing, no prompt wrangling.

```bash
npm install @jef-ai/sdk@next   # release candidates publish to the `next` tag
```

## Questions

```ts
import { JefClient, choice, noul, score } from '@jef-ai/sdk';

const jef = new JefClient({ baseUrl: 'http://localhost:8080' });

const { answers, usage } = await jef.evaluate(
  'EDR 於 02:14 偵測主機 fs-core-02 執行 vssadmin delete shadows /all，五分鐘內 3,200 個檔案被改為 .lockbit。',
  {
    urgent: noul('這則告警是否需要立即處理？'),
    team: choice('應由哪個團隊處理？', {
      soc: '資安監控：惡意程式、勒索軟體、端點與網路入侵',
      infra: '基礎設施：主機、備份、憑證與可用性',
      identity: '身分與存取：帳號、權限、登入與 MFA',
    }),
    severity: score('評估此告警的嚴重度', ['資訊', '低', '中', '高', '危急'] as const),
  },
);

answers.team.choice;        // 'soc' | 'infra' | 'identity' — narrowed from your keys
answers.team.confidence;    // how peaked the distribution is, not P(correct)
answers.severity.score;     // continuous 0–4, may land between levels
answers.urgent.noul;        // P(true)
usage.outputTokens;         // 0 — nothing is ever generated
```

Every question in a request is evaluated independently against the same state,
which is read exactly once. Adding a question costs one short encode, not
another pass over the evidence.

### `confidence` is not correctness

`confidence` is `(n × peak − 1) / (n − 1)`: how *peaked* the distribution is.
A model can be decisive and wrong. Scene gates read `p_correct` instead — the
calibrated mapping of that number onto observed correctness — and `p_correct` is
`null` when the server has no calibration data to support one, which is not the
same as a low probability.

## Scenes

A scene is layers of typed questions with confidence gates between them. The
state is encoded once for the whole playbook, so twenty gates cost one state
read rather than twenty.

```ts
const trace = await jef.runScene('incident-triage', alert);

trace.action;          // 'assign' | 'escalate_human' | 'close' | ...
trace.human_review;    // whether a person must see this
trace.state_encodes;   // 1
trace.layers_skipped;  // layers never reached — "not asked", not "inconclusive"
```

The trace is the product, not the verdict. It records the evidence, each
question, the answers that were permitted, where the probability mass fell, and
which gate fired and why — which is what an audit actually needs, and what
hand-written playbook branching cannot produce.

**An uncalibrated server cannot automate.** Without calibration `p_correct` is
unavailable and the conformal prediction set excludes nothing, so no automating
gate fires and every path ends at a human. That is the design working.

## Both yes/no spellings

TypeSafe's native API spells the primitive `noul` and answers with `{ noul }`;
Vercel's AI SDK spells it `boolean` and answers with `{ probability }`. JEF
accepts both and replies in whichever you used, so code ported from
`experimental_evaluate` reads unchanged:

```ts
import { boolean } from '@jef-ai/sdk';

const { answers } = await jef.evaluate(transcript, {
  refunded: boolean('是否已退款給客戶？', {
    true: '已確認退款',
    false: '未退款或遭拒',
  }),
});

answers.refunded.probability;  // not `.noul`
```

## Errors

Failures throw `JefError` carrying the server's own code, so you can branch
without parsing prose:

```ts
import { JefError } from '@jef-ai/sdk';

try {
  await jef.evaluate(state, questions);
} catch (error) {
  if (error instanceof JefError && error.code === 'too_many_questions') {
    // split the batch
  }
}
```

## Notes

- Requires Node 20+ (uses `fetch`, `AbortSignal.any` and `AbortSignal.timeout`).
- Pass `fetch` to inject your own instrumented implementation.
- `score` levels are ordinal and must be lowest-first: the answer is the
  expectation over that ordering, so reversing them inverts the scale silently.

## License

Apache-2.0
