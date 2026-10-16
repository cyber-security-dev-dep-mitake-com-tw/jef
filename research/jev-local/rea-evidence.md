# REA evidence for the Jev wire protocol

Produced with `rea-agents@6.5.0 analyze-javascript-application` (static parse, no execution) on the npm packages installed in this folder, selecting `dist/index.js` in each. REA was run through a shadow Command Line Tools toolchain (`scripts/make-shadow-clt.sh`) because this host's `/Library/Developer/CommandLineTools/usr/include/swift/module.modulemap` duplicates `bridging.modulemap`, which breaks the Swift helper REA compiles for macOS process-ownership checks. Coordinates below are REA `identity.source_range` values in `dist/index.js`; the module rows are the original TypeScript files REA recovered from the shipped source maps. The complete Evidence JSON (20 MB and 114 MB) stays in the session scratchpad; the ids let it be re-exported.

## @ai-sdk/gateway@4.0.114

- Evidence ID: `ev_269409875df75d3db0d2b534c1b9f6b14a4387ee4dbb83c0c0e544bd2c206da0` (confidence: derived)
- Root artifact SHA-256: `2eb567fa0ca5e06e400094dc59cc56dc62256c23f3fb1533208fc192ef8cfe9e`
- `dist/index.js` SHA-256: `639f6c74038c6c629cc9181a2ac32a5396c185e686d77b73ed8acf799e6c4f73`
- Semantic nodes retained: 10557; parsed AST nodes: 17177; parse failures: 0

### Functions and bindings

| Node | Kind | dist/index.js range |
|---|---|---|
| `GatewayDecisionModel` | function | 1822:27 – 1894:1 |
| `doDecide` | function | 1840:1 – 1884:2 |
| `doEvaluate` | function | 1837:1 – 1839:2 |
| `getUrl` | function | 1323:1 – 1325:2 |
| `getUrl` | function | 2005:1 – 2007:2 |
| `getModelConfigHeaders` | function | 1683:1 – 1688:2 |
| `getModelConfigHeaders` | function | 1428:1 – 1433:2 |
| `toGatewayDecisionState` | function | 1947:0 – 1960:1 |
| `createDecisionModel` | function | 3271:29 – 3279:2 |
| `gatewayDecisionAnswerSchema` | binding | 1895:6 – 1895:33 |
| `gatewayDecisionResponseSchema` | binding | 1933:6 – 1933:35 |
| `gatewayDecisionWarningSchema` | binding | 1912:6 – 1912:34 |
| `gatewayDecisionProviderOptionsSchema` | binding | 1767:6 – 1767:42 |
| `stateParts` | binding | 1841:8 – 1841:18 |

### Source-map recovered modules (original TypeScript files)

- `src/gateway-decision-model-settings.ts`
- `src/gateway-decision-model.ts`

## ai@7.0.141

- Evidence ID: `ev_4a0f3bd9f9d5c3b67fbd712c1f94b913e7583435ca58c9633e84101c151dd4cb` (confidence: derived)
- Root artifact SHA-256: `73075d859066f092d512f46c62cd7470e347e4c40cf71dcd8ff75ddbc619e0f3`
- `dist/index.js` SHA-256: `01ab99c439a46358e668eab814bb57a7ff8531c43760b9c2f60344fdba2614af`
- Semantic nodes retained: 58025; parsed AST nodes: 97455; parse failures: 0

### Functions and bindings

| Node | Kind | dist/index.js range |
|---|---|---|
| `decide` | function | 13001:0 – 13123:1 |
| `validateDecisionAnswers` | function | 12958:0 – 12993:1 |
| `validateDecisionInput` | function | 12910:0 – 12940:1 |
| `resolveDecisionModel` | function | 1111:0 – 1141:1 |
| `prepareDecisionState` | function | 12834:0 – 12859:1 |
| `validateDistribution` | function | 12953:0 – 12957:1 |
| `roundingError` | function | 12959:1 – 12963:2 |
| `invalidAnswer` | function | 12941:0 – 12946:1 |
| `isProbability` | function | 12947:0 – 12949:1 |
| `experimental_evaluate` | binding | 13127:6 – 13127:27 |
| `refusedQuestionIds` | binding | 13070:10 – 13070:28 |
| `probabilityError` | binding | 12964:7 – 12964:23 |
| `scoreError` | binding | 12965:7 – 12965:17 |

### Source-map recovered modules (original TypeScript files)

- `src/decide/decide-events.ts`
- `src/decide/decide.ts`
- `src/decide/decision-provider.ts`
- `src/decide/decision-result.ts`
- `src/decide/decision-state.ts`
- `src/decide/index.ts`
- `src/decide/restricted-telemetry-dispatcher.ts`
- `src/decide/validate-decision.ts`
- `src/test/decision-mock-model-v4.ts`

## What the evidence establishes

- The gateway client implements the decision model as a class with `doDecide`, a URL getter, and a model-config-headers getter, and defines response/answer/warning schemas as separate bindings. The literal endpoint path, header names, and answer type strings are not retained as semantic nodes by REA's projection, so those exact strings were confirmed by reading the ranges above in `dist/index.js` (observed, not inferred).
- The AI SDK core validates inputs and answers in `validateDecisionInput` / `validateDecisionAnswers` and resolves string model ids through `resolveDecisionModel` using the default gateway provider; the local `register.mjs` shim relies on that path.
- Limitations carried from REA: static parse only; no proof of runtime execution; 60 (gateway) and 402 (ai) inventoried files outside the selected scope.
