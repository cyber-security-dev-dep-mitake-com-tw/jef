"""D3 at the scene level: a deep playbook costs one state read.

This is the property that makes scenes worth having. If it breaks, a twenty-gate
playbook costs twenty encodes and the entire performance argument for putting
decision composition inside JEF -- rather than calling an evaluate endpoint per
gate -- evaporates.
"""

from __future__ import annotations

from jef_scene import SceneEngine, load_scene_text

DEEP = """
scene: deep
fallthrough: done
layers:
{layers}
actions:
  done: {{}}
"""

LAYER = """  - id: L{i}
    questions:
      q{i}a: {{type: noul, instructions: 問題 {i}a？}}
      q{i}b: {{type: noul, instructions: 問題 {i}b？}}
      q{i}c: {{type: choice, instructions: 問題 {i}c？, criteria: {{a: 甲, b: 乙, c: 丙}}}}
    gates:
      - when: 'q{i}a.probability > 1.5'
        then: done
      - else: true
        then: next
"""


def _deep_scene(n: int):
    return load_scene_text(DEEP.format(layers="".join(LAYER.format(i=i) for i in range(n))))


def test_ten_layers_thirty_questions_encode_the_state_once(scene_engine: SceneEngine) -> None:
    scene = _deep_scene(10)
    trace = scene_engine.run(scene, "一則需要深度分流的告警內容" * 40)
    assert trace.questions_asked == 30
    assert trace.state_encodes == 1, (
        f"D3 violated: 30 questions across 10 layers caused {trace.state_encodes} state encodes"
    )


def test_depth_does_not_multiply_state_reads(scene_engine: SceneEngine) -> None:
    shallow = scene_engine.run(_deep_scene(1), "告警內容")
    deep = scene_engine.run(_deep_scene(12), "告警內容")
    assert shallow.state_encodes == deep.state_encodes == 1
    assert deep.questions_asked == 12 * shallow.questions_asked


def test_layers_after_the_decision_are_never_asked(scene_engine: SceneEngine) -> None:
    """Skipped layers must cost nothing, and the trace must say they were skipped."""
    scene = load_scene_text("""
scene: early
layers:
  - id: L1
    questions:
      a: {type: noul, instructions: 甲？}
    gates:
      - else: true
        then: stop
  - id: L2
    questions:
      b: {type: noul, instructions: 乙？}
      c: {type: noul, instructions: 丙？}
    gates:
      - else: true
        then: stop
actions:
  stop: {}
""")
    trace = scene_engine.run(scene, "x")
    assert trace.action == "stop"
    assert trace.questions_asked == 1, "a decided layer must not evaluate later layers"
    assert trace.layers_skipped == ["L2"]
    assert [lt.id for lt in trace.layers] == ["L1"]


def test_every_layer_sees_the_same_state(scene_engine: SceneEngine) -> None:
    """The same question asked in two layers must get the same answer.

    Ids differ because a scene requires unique question ids -- that is what lets
    a later gate reference an earlier layer's answer without qualification.
    """
    scene = load_scene_text("""
scene: same
fallthrough: done
layers:
  - id: L1
    questions:
      first_ask: {type: noul, instructions: 是否緊急？}
    gates:
      - when: 'first_ask.probability > 1.5'
        then: done
      - else: true
        then: next
  - id: L2
    questions:
      second_ask: {type: noul, instructions: 是否緊急？}
    gates:
      - when: 'second_ask.probability > 1.5'
        then: done
      - else: true
        then: next
actions:
  done: {}
""")
    trace = scene_engine.run(scene, "付款服務連續三天失敗")
    first = trace.layers[0].questions[0].answer
    second = trace.layers[1].questions[0].answer
    assert first == second


def test_a_later_gate_can_read_an_earlier_layers_answer(scene_engine: SceneEngine) -> None:
    """Combining across layers is how a real playbook reasons.

    Without this a scene can only branch on the questions in the layer it is
    already in, which forces every correlated decision into one flat layer.
    """
    scene = load_scene_text("""
scene: crosslayer
fallthrough: missed
layers:
  - id: L1
    questions:
      remote: {type: noul, instructions: 是否可遠端觸發？}
    gates:
      - else: true
        then: next
  - id: L2
    questions:
      severe: {type: noul, instructions: 是否嚴重？}
    gates:
      # `remote` comes from L1; `severe` from this layer.
      - when: 'remote.probability >= 0.0 and severe.probability >= 0.0'
        then: combined
      - else: true
        then: missed
actions:
  combined: {}
  missed: {}
""")
    assert scene_engine.run(scene, "x").action == "combined"


def test_duplicate_question_ids_across_layers_are_rejected() -> None:
    """Ambiguity is refused at load time rather than resolved by guessing."""
    import pytest
    from jef_core.errors import SceneError

    with pytest.raises(SceneError, match="unique across a scene"):
        load_scene_text("""
scene: dup
fallthrough: done
layers:
  - id: L1
    questions:
      q: {type: noul, instructions: 甲？}
    gates: [{else: true, then: next}]
  - id: L2
    questions:
      q: {type: noul, instructions: 乙？}
    gates: [{else: true, then: done}]
actions:
  done: {}
""")
