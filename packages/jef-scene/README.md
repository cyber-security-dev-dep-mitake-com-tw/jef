# jef-scene

Layered typed questions with calibrated confidence gates — the decision layer
[Jev](https://docs.typesafe.ai) leaves to the caller, and which every open
reimplementation therefore also omits.

```bash
pip install jef-scene
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
