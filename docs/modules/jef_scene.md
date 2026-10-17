# `jef_scene` — scene gates

Shipped inside the `jef` distribution (`pip install jef`); it is not a
separate package on PyPI.

Layered typed questions with calibrated confidence gates — the decision layer
[Jev](https://docs.typesafe.ai) leaves to the caller, and which every open
reimplementation therefore also omits.

```bash
pip install jef              # this module and the other three
pip install 'jef[torch]'     # + the real backbone
```

Most users want [`jef`](https://pypi.org/project/jef/) instead, which bundles
this with the engine, the SDK and a CLI.

```yaml
scene: incident-triage
fallthrough: escalate_human

layers:
  - id: L1-noise-filter
    questions:
      is_false_positive:
        type: noul
        instructions: 這則告警是否為已知的良性樣態（誤報）？
    gates:
      - when: 'is_false_positive.probability > 0.90'
        then: close
        because: 高度確信為已知良性樣態
      - else: true
        then: next

  - id: L2-routing
    questions:
      owner:
        type: choice
        instructions: 應由哪一個團隊處理？
        criteria:
          soc: 一般資安監控事件
          infra: 基礎設施、網路、主機層
    gates:
      # p_correct, not confidence: a gate should threshold on the chance of
      # being right, not on how decisive the model sounded.
      - when: 'owner.p_correct is not None and owner.p_correct >= 0.95'
        then: assign
      - else: true
        then: escalate_human

actions:
  close: {}
  assign: { params: { queue: team_inbox } }
  escalate_human: { human_review: true }
```

```python
from jef_core import Engine
from jef_scene import SceneEngine, load_scene

engine = SceneEngine(Engine("jhu-clsp/mmBERT-base"))
scene = load_scene("scenes/incident-triage.zh-tw.yaml")
engine.compile(scene)  # bad gates fail here, not during an incident

trace = engine.run(scene, alert)
trace.action  # 'assign' | 'escalate_human' | ...
trace.state_encodes  # 1, for the whole playbook
trace.layers_skipped  # never reached — not the same as inconclusive
```

## Three things it does deliberately

**One state read per scene.** Ten layers and thirty questions still encode the
state once. A twenty-gate playbook costs one state read, not twenty.

**Gate conditions are a whitelist, not `eval`.** Conditions come from YAML, and
YAML in a SOAR deployment comes from wherever playbooks come from. The AST is
walked before execution and anything outside a fixed set — calls, imports,
attribute access beyond the answer fields — is a load-time error.

**An uncalibrated deployment cannot automate.** Without calibration `p_correct`
is unavailable and the conformal prediction set excludes nothing, so no
automating gate can fire and every path ends at a human. That is the design
working, not a limitation.

## Decision traces

Every run emits the evidence, each question, the answers that were permitted,
where the probability mass fell, and which gate fired and why — which is what an
audit needs and what hand-written playbook branching cannot produce.

## License

Apache-2.0


## Policy verdict

A gate selecting an action is a proposal, not an authorization. After the
deciding gate fires, `jef_scene.policy.authorize` produces a deterministic
verdict that the trace carries as `policy_verdict` (`allow` or
`review_required`), `action_tier` and `reason_codes`. When the verdict is
`review_required`, `human_review` is true whatever the action declared, and
nothing downstream may flip it back.

Two rules. Actions are tiered T0 to T4, explicitly via `tier:` on the action or
inferred from the name (`close` is T2, `isolate`/`disable`/`block` T3, anything
else T1); only T0 and T1 may run without a human. And an automating action
must rest on questions whose calibration is `validated`: every question the
firing gate read must have `calibration_status == "validated"` (a correctness
map and a conformal quantile at the scene's alpha, both fitted on enough
data). `missing` and `insufficient` both give `review_required`, as does a
gate that read nothing (an `else` gate or the fallthrough).

Each question's trace carries `calibration_status`, and a gate may read it
(`owner.calibration_status == "validated"`), which is how the shipped scenes
make the requirement explicit rather than relying on the policy alone.
`jef validate` warns when an automating gate reads neither `p_correct` nor
`calibration_status`.

Reason codes are stable strings: `AUTO_CLOSE_PROHIBITED`,
`DISRUPTIVE_ACTION_REQUIRES_APPROVAL`, `NO_EVIDENCE_QUESTION`,
`UNVALIDATED_CALIBRATION:<question>`, `HUMAN_REVIEW_ACTION`.

`jef validate` reports an error for any gate that automates to a T2+ action,
so a scene learns this at lint time rather than from a trace after deployment.
The trace also carries `proposed_action`, `executed_action` (always `null`
here: the engine executes nothing) and `execution_status`
(`not_dispatched`), so a reader never mistakes a proposal for an outcome.
