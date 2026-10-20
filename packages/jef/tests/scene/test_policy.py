"""The policy layer: a gate proposes, the policy decides whether a human is needed.

POL-001 from the v2 review is the headline: an uncalibrated deployment with a
model that is 99.9% sure an alert is a false positive must not get `close`
authorised. The rest pin the tiering, the evidence rule, and that nothing in
the scene or the model can flip `review_required` back to `allow`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jef_core import Engine
from jef_core.calibration import Calibrator
from jef_core.types import Answer, ChoiceAnswer, NoulAnswer
from jef_scene import SceneEngine, load_scene, load_scene_text
from jef_scene.policy import (
    MAX_AUTO_TIER,
    authorize,
    inferred_tier,
    referenced_questions,
    tier_of,
)
from jef_scene.schema import Action


class ScriptedEngine(Engine):
    """Returns fixed answers, so a test can hand the gate any probability it likes."""

    def __init__(self, answers: dict[str, Answer], calibrator: Calibrator | None = None) -> None:
        super().__init__(calibrator=calibrator)
        self._scripted = answers

    def answer(self, shared, questions):  # type: ignore[override]
        return {qid: self._scripted[qid] for qid in questions}


def sure_false_positive(p: float = 0.999) -> dict[str, Answer]:
    return {"is_fp": NoulAnswer(noul=p, confidence=abs(2 * p - 1))}


TRIAGE = """
scene: t
fallthrough: review
layers:
  - id: L1
    questions:
      is_fp:
        type: noul
        instructions: 是否為誤報？
    gates:
      - when: 'is_fp.probability > 0.90'
        then: {target}
      - else: true
        then: review
actions:
  close: {{}}
  request_context: {{}}
  nuke: {{ tier: 3 }}
  gentle_close: {{ tier: 1 }}
  review: {{ human_review: true }}
"""


def scene_for(target: str):
    return load_scene_text(TRIAGE.format(target=target))


# --------------------------------------------------------------------------- #
# POL-001
# --------------------------------------------------------------------------- #


def test_pol_001_uncalibrated_auto_close_is_not_authorised() -> None:
    """The exact scenario from the shipped scene: raw 0.999, no calibration, `close`."""
    engine = SceneEngine(ScriptedEngine(sure_false_positive()))
    trace = engine.run(scene_for("close"), "IDS 回報連接埠掃描")
    # The gate fired, and the trace says so -- but as a withheld selection,
    # not as a recommendation: nothing measured supports "close".
    assert trace.withheld_action == "close", "the policy must not hide what the gate selected"
    assert trace.layers[0].gates[0].fired and trace.layers[0].gates[0].withheld
    assert trace.action == trace.proposed_action == "review"
    assert trace.verdict == "withheld" and trace.advisory_only is True
    assert trace.advisory_reasons == ["UNVALIDATED_GATE:is_fp:missing"]
    assert trace.policy_verdict == "review_required"
    assert trace.human_review is True
    assert trace.executed_action is None
    assert trace.execution_status == "not_dispatched"


def test_pol_001_a_fabricated_certainty_changes_nothing() -> None:
    """0.99999 is still a model output nobody calibrated."""
    engine = SceneEngine(ScriptedEngine(sure_false_positive(0.99999)))
    trace = engine.run(scene_for("close"), "x")
    assert trace.proposed_action == "review"
    assert trace.policy_verdict == "review_required"
    assert trace.human_review is True


def test_close_stays_review_required_even_when_calibrated(calibrated_engine: Engine) -> None:
    """Tier beats calibration: a wrong close is not undone by a better model."""
    engine = SceneEngine(ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator))
    trace = engine.run(scene_for("close"), "x")
    assert trace.policy_verdict == "review_required"
    assert trace.reason_codes == ["AUTO_CLOSE_PROHIBITED"], trace.reason_codes


# --------------------------------------------------------------------------- #
# Tier 1 needs calibrated evidence; with it, automation is allowed
# --------------------------------------------------------------------------- #


def test_tier_one_without_calibration_needs_a_human() -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive()))
    trace = engine.run(scene_for("request_context"), "x")
    assert trace.withheld_action == "request_context"
    assert trace.action == "review"
    assert trace.policy_verdict == "review_required"
    assert trace.advisory_reasons == ["UNVALIDATED_GATE:is_fp:missing"]
    assert trace.human_review is True


def test_tier_one_with_calibrated_evidence_is_allowed(calibrated_engine: Engine) -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator))
    trace = engine.run(scene_for("request_context"), "x")
    assert trace.policy_verdict == "allow"
    assert trace.reason_codes == []
    assert trace.human_review is False
    assert trace.action_tier == 1


def test_explicit_tier_can_raise_but_never_lower(calibrated_engine: Engine) -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator))
    harmless_name_dangerous_tier = engine.run(scene_for("nuke"), "x")
    assert harmless_name_dangerous_tier.policy_verdict == "review_required"
    assert "DISRUPTIVE_ACTION_REQUIRES_APPROVAL" in harmless_name_dangerous_tier.reason_codes
    # Declaring a close action T1 in the YAML does not make it automatable:
    # auto-close stays prohibited, with calibrated evidence and a sure answer.
    declared_safe = engine.run(scene_for("gentle_close"), "x")
    assert declared_safe.policy_verdict == "review_required"
    assert declared_safe.action_tier == 2
    assert "AUTO_CLOSE_PROHIBITED" in declared_safe.reason_codes


# --------------------------------------------------------------------------- #
# Evidence rule: else gates and fallthrough cannot automate
# --------------------------------------------------------------------------- #

ELSE_AUTOMATES = """
scene: t
layers:
  - id: L1
    questions:
      is_fp:
        type: noul
        instructions: 是否為誤報？
    gates:
      - when: 'is_fp.probability > 0.99999'
        then: review
      - else: true
        then: enrich
actions:
  enrich: {}
  review: {human_review: true}
"""


def test_an_else_gate_cannot_automate(calibrated_engine: Engine) -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive(0.5), calibrated_engine.calibrator))
    trace = engine.run(load_scene_text(ELSE_AUTOMATES), "x")
    assert trace.action == "enrich"
    assert trace.policy_verdict == "review_required"
    assert trace.reason_codes == ["NO_EVIDENCE_QUESTION"]


def test_human_review_actions_are_marked_as_such() -> None:
    engine = SceneEngine(ScriptedEngine(sure_false_positive(0.1)))
    trace = engine.run(scene_for("close"), "x")  # 0.1 fails the gate -> else -> review
    assert trace.action == "review"
    assert trace.policy_verdict == "review_required"
    assert trace.reason_codes == ["HUMAN_REVIEW_ACTION"]
    assert trace.human_review is True


def test_fallthrough_is_review_required(scene_engine: SceneEngine) -> None:
    scene = load_scene_text(
        TRIAGE.format(target="close").replace("      - else: true\n        then: review\n", "")
    )
    trace = SceneEngine(ScriptedEngine(sure_false_positive(0.1))).run(scene, "x")
    assert trace.verdict == "fallthrough"
    assert trace.action == "review"
    assert trace.policy_verdict == "review_required"


# --------------------------------------------------------------------------- #
# The pure function and the tiering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "tier"),
    [
        ("close", 2),
        ("auto_close_and_suppress", 2),
        ("suppress_alert", 2),
        ("isolate_host", 3),
        ("disable_user", 3),
        ("block_ip", 3),
        ("quarantine_mailbox", 3),
        ("assign", 1),
        ("request_context", 1),
        ("page_oncall", 1),
        ("patch_now", 1),
        ("tag", 1),
    ],
)
def test_inferred_tiers(name: str, tier: int) -> None:
    assert inferred_tier(name) == tier


def test_tier_of_is_the_higher_of_declared_and_inferred() -> None:
    assert tier_of("close", Action(tier=1)) == 2, "a close action cannot be declared T1"
    assert tier_of("suppress_alert", Action(tier=0)) == 2
    assert tier_of("isolate_host", Action(tier=1)) == 3
    assert tier_of("enrich", Action(tier=4)) == 4
    assert tier_of("close", Action()) == 2
    assert MAX_AUTO_TIER == 1


def test_referenced_questions_reads_only_known_names() -> None:
    names = referenced_questions("a.p_correct is not None and b.score > 2 and c", {"a", "b"})
    assert names == {"a", "b"}
    assert referenced_questions(None, {"a"}) == set()


def test_authorize_never_consults_the_probability() -> None:
    """A view with a huge probability but no p_correct is still unvalidated."""

    class View:
        p_correct = None
        calibration_status = "missing"
        probability = 0.99999

    verdict = authorize("assign", Action(), referenced={"q"}, views={"q": View()})  # type: ignore[dict-item]
    assert verdict.verdict == "review_required"
    assert verdict.reason_codes == ["UNVALIDATED_CALIBRATION:q"]
    assert verdict.human_review is True


def test_choice_views_work_too(calibrated_engine: Engine) -> None:
    answers: dict[str, Answer] = {
        "owner": ChoiceAnswer(
            choice="soc",
            probabilities={
                "soc": 0.9,
                "appsec": 0.025,
                "infra": 0.025,
                "identity": 0.025,
                "fraud": 0.025,
            },
            confidence=0.875,
        )
    }
    scene = load_scene_text(
        """
scene: r
fallthrough: review
layers:
  - id: L1
    questions:
      owner:
        type: choice
        instructions: 誰處理？
        criteria: {soc: a, appsec: b, infra: c, identity: d, fraud: e}
    gates:
      - when: 'owner.p_correct is not None and owner.choice == "soc"'
        then: assign
actions:
  assign: {}
  review: {human_review: true}
"""
    )
    trace = SceneEngine(ScriptedEngine(answers, calibrated_engine.calibrator)).run(scene, "x")
    assert trace.action == "assign"
    assert trace.policy_verdict == "allow"


# --------------------------------------------------------------------------- #
# The shipped scenes
# --------------------------------------------------------------------------- #


def _repo_root() -> Path:
    return next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())


@pytest.mark.parametrize("name", ["incident-triage.zh-tw.yaml", "vuln-triage.zh-tw.yaml"])
def test_shipped_scenes_never_automate_a_state_changing_action(name: str) -> None:
    scene = load_scene(_repo_root() / "scenes" / name)
    for action_name, action in scene.actions.items():
        if not action.human_review:
            assert tier_of(action_name, action) <= MAX_AUTO_TIER, (
                f"{name}: {action_name} automates a T{tier_of(action_name, action)} action"
            )


def test_incident_triage_declares_no_state_changing_action() -> None:
    """Closing comes back only with a validated task and a measured threshold.

    Until then the scene has no T2+ action at all, so no gate, rename or
    calibration file can make it recommend one.
    """
    scene = load_scene(_repo_root() / "scenes" / "incident-triage.zh-tw.yaml")
    for name, action in scene.actions.items():
        assert tier_of(name, action) <= MAX_AUTO_TIER, f"{name} is T{tier_of(name, action)}"
    assert scene.unvalidated_fallback == "escalate_human"
    assert scene.allowed_when_unvalidated == {"escalate_human", "queue_for_review"}
    assert scene.advisory.observe_only == ["is_false_positive"]


#: The answers the K8S Lab run (2026-10-20) reported, as far as they were
#: recorded, plus the extremes. Probabilities of the noul questions are P(true).
LAB_LIKE = [
    pytest.param(0.92, 0.9, "infra", 1, 0.1, id="win-patch-0.92"),
    pytest.param(0.85, 0.9, "fraud", 1, 0.9, id="ransomware-0.85"),
    pytest.param(0.99, 0.9, "fraud", 0, 0.99, id="ransomware-0.99"),
    pytest.param(0.49, 0.2, "fraud", 4, 0.9, id="c2-0.49"),
    pytest.param(0.01, 0.01, "soc", 4, 0.01, id="nothing-sure"),
]


def _triage_answers(fp: float, ev: float, owner: str, sev: int, urgent: float) -> dict:
    from jef_core.types import ScoreAnswer

    owners = ["soc", "appsec", "infra", "identity", "fraud"]
    probs = {k: (0.96 if k == owner else 0.01) for k in owners}
    legend = ["資訊", "低", "中", "高", "危急"]
    sev_probs = {k: (0.96 if i == sev else 0.01) for i, k in enumerate(legend)}
    return {
        "is_false_positive": NoulAnswer(noul=fp, confidence=abs(2 * fp - 1)),
        "has_evidence": NoulAnswer(noul=ev, confidence=abs(2 * ev - 1)),
        "owner": ChoiceAnswer(choice=owner, probabilities=probs, confidence=0.95),
        "severity": ScoreAnswer(
            score=float(sev), legend=legend, probabilities=sev_probs, confidence=0.95
        ),
        "urgent": NoulAnswer(noul=urgent, confidence=abs(2 * urgent - 1)),
    }


@pytest.mark.parametrize(("fp", "ev", "owner", "sev", "urgent"), LAB_LIKE)
def test_shipped_triage_never_recommends_more_than_escalation_when_uncalibrated(
    fp: float, ev: float, owner: str, sev: int, urgent: float
) -> None:
    """Whatever the model says, an uncalibrated run ends at a person, undispatched."""
    scene = load_scene(_repo_root() / "scenes" / "incident-triage.zh-tw.yaml")
    answers = _triage_answers(fp, ev, owner, sev, urgent)
    trace = SceneEngine(ScriptedEngine(answers)).run(scene, "x")
    assert trace.proposed_action in {"escalate_human", "queue_for_review"}
    assert trace.policy_verdict == "review_required" and trace.human_review
    assert trace.executed_action is None and trace.execution_status == "not_dispatched"


def test_trace_dict_carries_the_policy_fields(scene_engine: SceneEngine, minimal_scene) -> None:
    d = scene_engine.run(minimal_scene, "x").to_dict()
    for key in (
        "proposed_action",
        "policy_verdict",
        "action_tier",
        "reason_codes",
        "executed_action",
        "execution_status",
    ):
        assert key in d
    assert d["executed_action"] is None
    assert d["execution_status"] == "not_dispatched"


# --------------------------------------------------------------------------- #
# Calibration status travels with every answer
# --------------------------------------------------------------------------- #


def test_question_trace_carries_calibration_status(calibrated_engine: Engine) -> None:
    uncal = SceneEngine(ScriptedEngine(sure_false_positive())).run(scene_for("close"), "x")
    assert uncal.layers[0].questions[0].calibration_status == "missing"
    cal = SceneEngine(ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator)).run(
        scene_for("close"), "x"
    )
    assert cal.layers[0].questions[0].calibration_status == "validated"


def test_insufficient_bucket_is_held_for_review() -> None:
    """A bucket that exists but could not fit a correctness map is not validated."""
    import numpy as np

    rng = np.random.default_rng(1)
    cal = Calibrator()
    logits = [rng.normal(0, 1, size=2) for _ in range(60)]  # below MIN_CORRECTNESS_SAMPLES
    cal.fit_bucket("noul", 2, logits, [int(rng.integers(2)) for _ in range(60)])
    assert cal.status("noul", 2) == "insufficient"
    trace = SceneEngine(ScriptedEngine(sure_false_positive(), cal)).run(
        scene_for("request_context"), "x"
    )
    assert trace.layers[0].questions[0].calibration_status == "insufficient"
    assert trace.policy_verdict == "review_required"
    assert trace.advisory_reasons == ["UNVALIDATED_GATE:is_fp:insufficient"]


STATUS_GATE = """
scene: t
fallthrough: review
layers:
  - id: L1
    questions:
      is_fp:
        type: noul
        instructions: 是否為誤報？
    gates:
      - when: 'is_fp.calibration_status == "validated" and is_fp.probability > 0.90'
        then: request_context
      - else: true
        then: review
actions:
  request_context: {}
  review: {human_review: true}
"""


def test_a_gate_can_read_calibration_status(calibrated_engine: Engine) -> None:
    """The scene author can make the requirement explicit instead of relying on the policy."""
    scene = load_scene_text(STATUS_GATE)
    uncal = SceneEngine(ScriptedEngine(sure_false_positive())).run(scene, "x")
    assert uncal.action == "review", "the gate must not fire on a missing bucket"
    cal = SceneEngine(ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator)).run(
        scene, "x"
    )
    assert cal.action == "request_context"
    assert cal.policy_verdict == "allow"


# --------------------------------------------------------------------------- #
# Task identity reaches the policy
# --------------------------------------------------------------------------- #

TASKED = """
scene: t
fallthrough: review
layers:
  - id: L1
    questions:
      is_fp:
        type: noul
        task_id: soar_zh.false_positive.v1
        instructions: 是否為誤報？
    gates:
      - when: 'is_fp.calibration_status == "validated" and is_fp.probability > 0.90'
        then: request_context
      - else: true
        then: review
actions:
  request_context: {}
  review: {human_review: true}
"""


def test_ml_001_legacy_calibration_never_automates(calibrated_engine: Engine) -> None:
    """A schema-1 file is legacy: shown, never acted on, whatever its numbers say."""
    v1 = {
        k: v
        for k, v in calibrated_engine.calibrator.to_dict().items()
        if k not in ("calibration_schema", "artifact")
    }
    for b in v1["buckets"].values():
        b.pop("task_id", None)
        b.pop("metrics", None)
    legacy = Calibrator.from_dict(v1)
    assert legacy.legacy and legacy.is_fitted()

    trace = SceneEngine(ScriptedEngine(sure_false_positive(), legacy)).run(
        scene_for("request_context"), "x"
    )
    q = trace.layers[0].questions[0]
    assert q.calibration_status == "legacy"
    assert q.p_correct is not None, "the number is still shown for analysis"
    assert trace.policy_verdict == "review_required"
    assert trace.withheld_action == "request_context"
    assert trace.advisory_reasons == ["UNVALIDATED_GATE:is_fp:legacy"]
    assert trace.calibrated is True, "is_fitted stays true; status is what changed"


def test_a_tasked_question_needs_its_own_task_bucket(calibrated_engine: Engine) -> None:
    scene = load_scene_text(TASKED)
    # Shape-scoped buckets only: the tasked question finds nothing.
    shape_only = SceneEngine(
        ScriptedEngine(sure_false_positive(), calibrated_engine.calibrator)
    ).run(scene, "x")
    assert shape_only.layers[0].questions[0].task_id == "soar_zh.false_positive.v1"
    assert shape_only.layers[0].questions[0].calibration_status == "missing"
    assert shape_only.action == "review"

    import numpy as np

    rng = np.random.default_rng(3)
    tasked = Calibrator()
    logits, labels = [], []
    for _ in range(400):
        y = int(rng.integers(2))
        lg = rng.normal(0, 0.5, size=2)
        lg[y] += 2.5
        logits.append(lg)
        labels.append(y)
    tasked.fit_bucket("noul", 2, logits, labels, task_id="soar_zh.false_positive.v1")
    with_task = SceneEngine(ScriptedEngine(sure_false_positive(), tasked)).run(scene, "x")
    assert with_task.layers[0].questions[0].calibration_status == "validated"
    assert with_task.action == "request_context"
    assert with_task.policy_verdict == "allow"


def test_shipped_scenes_name_a_task_for_every_question() -> None:
    for name in ("incident-triage.zh-tw.yaml", "vuln-triage.zh-tw.yaml"):
        scene = load_scene(_repo_root() / "scenes" / name)
        for layer in scene.layers:
            for qid, q in layer.questions.items():
                assert getattr(q, "task_id", None), f"{name}: {qid} has no task_id"


# --------------------------------------------------------------------------- #
# Advisory guard: a gate on unvalidated evidence recommends nothing it cannot
# stand behind (incident-triage validation-first PR-01)
# --------------------------------------------------------------------------- #

ADVISORY = """
scene: t
fallthrough: escalate
advisory:
  unvalidated_fallback: escalate
  allowed_when_unvalidated: [escalate, backlog]
layers:
  - id: L1
    questions:
      is_fp:
        type: noul
        task_id: soar_zh.false_positive.v1
        instructions: 是否為誤報？
    gates:
      - when: 'is_fp.probability > 0.80'
        then: {target}
      - else: true
        then: backlog
actions:
  close_pending_review: {{ human_review: true, tier: 2 }}
  normal_maintenance: {{ human_review: true }}
  escalate: {{ human_review: true }}
  backlog: {{ human_review: true }}
"""


@pytest.mark.parametrize("p", [0.85, 0.92, 0.99, 0.99999])
@pytest.mark.parametrize("target", ["close_pending_review", "normal_maintenance"])
def test_unvalidated_gate_never_recommends_its_selection(p: float, target: str) -> None:
    """The Lab's 0.92 patch alert, and a ransomware alert at any probability.

    ``normal_maintenance`` is a close action renamed so no tier pattern
    matches it: the allowlist is by key, so the rename does not get through.
    """
    scene = load_scene_text(ADVISORY.format(target=target))
    trace = SceneEngine(ScriptedEngine(sure_false_positive(p))).run(scene, "x")
    assert trace.withheld_action == target
    assert trace.proposed_action == trace.action == "escalate"
    assert trace.advisory_only and trace.verdict == "withheld"
    assert trace.advisory_reasons == ["UNVALIDATED_GATE:is_fp:missing"]
    assert trace.layers[0].outcome == "escalate"
    assert trace.policy_verdict == "review_required" and trace.human_review
    assert trace.executed_action is None and trace.execution_status == "not_dispatched"


def test_an_allowlisted_action_is_not_withheld() -> None:
    scene = load_scene_text(ADVISORY.format(target="backlog"))
    trace = SceneEngine(ScriptedEngine(sure_false_positive())).run(scene, "x")
    assert trace.action == "backlog" and not trace.advisory_only
    assert trace.withheld_action is None and trace.advisory_reasons == []


def test_an_else_gate_is_not_withheld() -> None:
    scene = load_scene_text(ADVISORY.format(target="close_pending_review"))
    trace = SceneEngine(ScriptedEngine(sure_false_positive(0.1))).run(scene, "x")
    assert trace.action == "backlog" and trace.verdict == "decided"
    assert not trace.advisory_only


def test_validated_close_is_proposed_and_still_needs_a_human(calibrated_engine: Engine) -> None:
    """Validated calibration lets the gate recommend; it never lets close run alone."""
    import numpy as np

    rng = np.random.default_rng(3)
    tasked = Calibrator()
    logits, labels = [], []
    for _ in range(400):
        y = int(rng.integers(2))
        lg = rng.normal(0, 0.5, size=2)
        lg[y] += 2.5
        logits.append(lg)
        labels.append(y)
    tasked.fit_bucket("noul", 2, logits, labels, task_id="soar_zh.false_positive.v1")
    scene = load_scene_text(ADVISORY.format(target="close_pending_review"))
    trace = SceneEngine(ScriptedEngine(sure_false_positive(), tasked)).run(scene, "x")
    assert trace.action == "close_pending_review" and not trace.advisory_only
    assert trace.policy_verdict == "review_required" and trace.human_review
    assert trace.action_tier == 2


def test_no_safe_fallback_proposes_nothing() -> None:
    """A fallthrough that would itself automate is not a safe place to land."""
    scene = load_scene_text("""
scene: t
fallthrough: done
layers:
  - id: L1
    questions:
      is_fp: {type: noul, instructions: 是否為誤報？}
    gates:
      - when: 'is_fp.probability > 0.5'
        then: close
      - else: true
        then: done
actions:
  close: {}
  done: {}
""")
    assert scene.unvalidated_fallback is None
    trace = SceneEngine(ScriptedEngine(sure_false_positive())).run(scene, "x")
    assert trace.action is None and trace.proposed_action is None
    assert trace.withheld_action == "close"
    assert trace.advisory_reasons == ["UNVALIDATED_GATE:is_fp:missing", "ADVISORY_NO_FALLBACK"]
    assert trace.human_review is True and trace.policy_verdict == "review_required"


@pytest.mark.parametrize(
    "advisory",
    [
        "{unvalidated_fallback: close}",  # T2 by name
        "{unvalidated_fallback: auto}",  # human_review false
        "{allowed_when_unvalidated: [hidden_close]}",  # declared tier 2
        "{unvalidated_fallback: nowhere}",  # undeclared
    ],
)
def test_advisory_names_must_be_safe_actions(advisory: str) -> None:
    from jef_core.errors import SceneError

    with pytest.raises(SceneError, match="advisory"):
        load_scene_text(f"""
scene: t
fallthrough: review
advisory: {advisory}
layers:
  - id: L1
    questions:
      a: {{type: noul, instructions: 甲？}}
    gates:
      - else: true
        then: review
actions:
  close: {{human_review: true}}
  auto: {{}}
  hidden_close: {{human_review: true, tier: 2}}
  review: {{human_review: true}}
""")
