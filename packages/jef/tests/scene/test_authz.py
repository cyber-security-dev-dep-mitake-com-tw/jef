"""Proposal, grant, receipt: the model proposes, a person authorises, the connector reports.

POL-003 to POL-007 from the v2 review. Every refusal is checked with the
executor's own dispatch counter: a refused action never reached the transport.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from jef_core import Engine
from jef_scene import SceneEngine, load_scene_text
from jef_scene.authz import (
    POLICY_VERSION,
    AuthorizationGrant,
    GrantError,
    PolicyDecision,
    ProposedDecision,
)
from jef_scene.executor import Executor, TransportResult

SECRET = b"approval-service-secret"
T0 = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


class FakeTransport:
    """Records calls; behaviour scripted per test."""

    def __init__(self, behaviour: str = "ok") -> None:
        self.behaviour = behaviour
        self.calls: list[dict[str, Any]] = []

    def execute(
        self, action: str, asset_id: str | None, params: Mapping[str, Any], *, idempotency_key: str
    ) -> TransportResult:
        self.calls.append({"action": action, "asset": asset_id, "key": idempotency_key})
        if self.behaviour == "timeout":
            raise TimeoutError("EDR API did not answer in 30s")
        if self.behaviour == "error":
            raise RuntimeError("EDR returned 500")
        if self.behaviour == "refused":
            return TransportResult(ok=False, detail="host not managed")
        return TransportResult(ok=True, detail="isolated", external_ref="edr-123")


def proposal(
    action: str = "disable_domain_admin",
    *,
    tier: int = 3,
    verdict: str = "review_required",
    reasons: tuple[str, ...] = ("DISRUPTIVE_ACTION_REQUIRES_APPROVAL",),
    asset: str | None = "host-01",
) -> ProposedDecision:
    return ProposedDecision(
        case_id="case-1",
        action=action,
        action_tier=tier,
        policy_verdict=verdict,
        reason_codes=reasons,
        scene="incident-triage",
        scene_version=1,
        model_digest="sha256:abc",
        trace_id="t-1",
        asset_id=asset,
    )


def grant(**overrides: Any) -> AuthorizationGrant:
    kwargs: dict[str, Any] = dict(
        secret=SECRET,
        case_id="case-1",
        action="disable_domain_admin",
        asset_id="host-01",
        approvers=("analyst-anon-01",),
        approver_role="soc-lead",
        issuer="approval-service",
        ttl_seconds=900,
        now=T0,
    )
    kwargs.update(overrides)
    return AuthorizationGrant.issue(**kwargs)


def executor(transport: FakeTransport | None = None, **kw: Any) -> Executor:
    return Executor(
        "fake-edr",
        transport or FakeTransport(),
        grant_secret=SECRET,
        clock=lambda: T0 + timedelta(seconds=1),
        **kw,
    )


# --------------------------------------------------------------------------- #
# POL-003: confidence is not authorization
# --------------------------------------------------------------------------- #


def test_pol_003_disruptive_action_without_grant_is_not_dispatched() -> None:
    """p_correct 0.99 and an `allow`-looking verdict still need a grant at T3."""
    ex = executor()
    p = proposal(verdict="allow", reasons=())  # a bug or a tampered trace says allow
    receipt = ex.dispatch(p, idempotency_key="k1")
    assert receipt.status == "not_dispatched"
    assert "NO_GRANT" in receipt.reason_codes
    assert ex.dispatch_count == 0
    assert ex.transport.calls == []  # type: ignore[attr-defined]


def test_review_required_t1_also_needs_a_grant() -> None:
    ex = executor()
    p = proposal("request_context", tier=1, reasons=("UNVALIDATED_CALIBRATION:q",))
    receipt = ex.dispatch(p, idempotency_key="k")
    assert receipt.status == "not_dispatched"
    assert "UNVALIDATED_CALIBRATION:q" in receipt.reason_codes
    assert "NO_GRANT" in receipt.reason_codes
    assert ex.dispatch_count == 0


def test_allowed_t1_runs_without_a_grant() -> None:
    ex = executor()
    p = proposal("request_context", tier=1, verdict="allow", reasons=())
    receipt = ex.dispatch(p, idempotency_key="k")
    assert receipt.status == "success"
    assert receipt.executed
    assert ex.dispatch_count == 1


def test_valid_grant_dispatches_once() -> None:
    ex = executor()
    receipt = ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    assert receipt.status == "success"
    assert receipt.external_ref == "edr-123"
    assert receipt.grant_nonce
    assert receipt.policy_version == POLICY_VERSION
    assert [e.event for e in ex.audit] == [
        "proposed",
        "decided",
        "granted",
        "dispatched",
        "receipt",
    ]


# --------------------------------------------------------------------------- #
# POL-004: scope and expiry
# --------------------------------------------------------------------------- #


def test_pol_004_grant_for_another_asset_is_rejected() -> None:
    ex = executor()
    receipt = ex.dispatch(proposal(asset="host-02"), idempotency_key="k", grant=grant())
    assert receipt.status == "not_dispatched"
    assert receipt.reason_codes == ("GRANT_ASSET_MISMATCH",)
    assert ex.dispatch_count == 0


def test_pol_004_expired_grant_is_rejected() -> None:
    ex = Executor(
        "fake", FakeTransport(), grant_secret=SECRET, clock=lambda: T0 + timedelta(hours=1)
    )
    receipt = ex.dispatch(proposal(), idempotency_key="k", grant=grant(ttl_seconds=600))
    assert receipt.status == "not_dispatched"
    assert receipt.reason_codes == ("GRANT_EXPIRED",)
    assert ex.dispatch_count == 0


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"case_id": "case-9"}, "GRANT_CASE_MISMATCH"),
        ({"action": "isolate_host"}, "GRANT_ACTION_MISMATCH"),
    ],
)
def test_pol_004_grant_scope_is_exact(override: dict[str, Any], code: str) -> None:
    ex = executor()
    receipt = ex.dispatch(proposal(), idempotency_key="k", grant=grant(**override))
    assert receipt.reason_codes == (code,)
    assert ex.dispatch_count == 0


def test_a_proposal_without_an_asset_cannot_be_granted() -> None:
    ex = executor()
    receipt = ex.dispatch(proposal(asset=None), idempotency_key="k", grant=grant())
    assert receipt.reason_codes == ("GRANT_ASSET_MISMATCH",)


def test_t4_needs_two_distinct_approvers() -> None:
    ex = executor()
    p = proposal("disable_break_glass", tier=4)
    one = grant(action="disable_break_glass", approvers=("a",))
    same_twice = grant(action="disable_break_glass", approvers=("a", "a"))
    two = grant(action="disable_break_glass", approvers=("a", "b"))
    assert ex.dispatch(p, idempotency_key="k1", grant=one).reason_codes == (
        "DUAL_APPROVAL_REQUIRED",
    )
    assert ex.dispatch(p, idempotency_key="k2", grant=same_twice).reason_codes == (
        "DUAL_APPROVAL_REQUIRED",
    )
    assert ex.dispatch(p, idempotency_key="k3", grant=two).status == "success"


# --------------------------------------------------------------------------- #
# POL-005: idempotency
# --------------------------------------------------------------------------- #


def test_pol_005_same_key_ten_times_executes_once() -> None:
    ex = executor()
    receipts = [ex.dispatch(proposal(), idempotency_key="same", grant=grant()) for _ in range(10)]
    assert ex.dispatch_count == 1
    assert len({id(r) for r in receipts}) == 1, "the stored receipt is returned, not a new one"
    assert receipts[0].status == "success"


def test_a_refusal_is_also_idempotent() -> None:
    """A retried refused step must not get a second chance with a fresh evaluation."""
    ex = executor()
    first = ex.dispatch(proposal(), idempotency_key="k")
    second = ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    assert first is second
    assert second.status == "not_dispatched"


# --------------------------------------------------------------------------- #
# POL-006: approved: true is not a grant
# --------------------------------------------------------------------------- #


def test_pol_006_forged_approval_in_json_is_refused() -> None:
    ex = executor()
    for forged in (
        {"approved": True},
        {"approved": True, "approver": "ciso", "case_id": "case-1"},
        {**grant().to_dict(), "signature": "0" * 64},
        {**grant().to_dict(), "asset_id": "host-02"},  # edited after signing
        "approved",
        True,
    ):
        receipt = ex.dispatch(proposal(), idempotency_key=f"k-{id(forged)}", grant=forged)
        assert receipt.status == "not_dispatched", forged
        assert receipt.reason_codes == ("INVALID_GRANT",), forged
    assert ex.dispatch_count == 0


def test_grant_parse_rejects_bad_signatures_and_accepts_good_ones() -> None:
    g = grant()
    assert AuthorizationGrant.parse(g.to_dict(), secret=SECRET) == g
    with pytest.raises(GrantError, match="signature"):
        AuthorizationGrant.parse(g.to_dict(), secret=b"other")
    with pytest.raises(GrantError, match="missing"):
        AuthorizationGrant.parse({"approved": True}, secret=SECRET)
    assert not g.verify(b"")


def test_executor_without_a_secret_refuses_every_grant() -> None:
    ex = Executor("fake", FakeTransport(), grant_secret=None, clock=lambda: T0)
    receipt = ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    assert receipt.reason_codes == ("NO_GRANT_SECRET",)
    assert ex.dispatch_count == 0


def test_issue_refuses_nonsense() -> None:
    with pytest.raises(GrantError):
        grant(secret=b"")
    with pytest.raises(GrantError):
        grant(approvers=())
    with pytest.raises(GrantError):
        grant(ttl_seconds=0)


# --------------------------------------------------------------------------- #
# POL-007: a timeout is unknown, never success
# --------------------------------------------------------------------------- #


def test_pol_007_timeout_is_unknown() -> None:
    ex = executor(FakeTransport("timeout"))
    receipt = ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    assert receipt.status == "unknown"
    assert not receipt.executed
    assert "OUTCOME_UNOBSERVED" in receipt.reason_codes
    assert ex.dispatch_count == 1, "the call was made; only its outcome is unknown"


def test_transport_error_is_failed_and_refusal_is_failed() -> None:
    assert (
        executor(FakeTransport("error"))
        .dispatch(proposal(), idempotency_key="k", grant=grant())
        .status
        == "failed"
    )
    assert (
        executor(FakeTransport("refused"))
        .dispatch(proposal(), idempotency_key="k", grant=grant())
        .status
        == "failed"
    )


# --------------------------------------------------------------------------- #
# dry run, allowlist, proposal from a trace
# --------------------------------------------------------------------------- #


def test_dry_run_checks_everything_and_sends_nothing() -> None:
    ex = executor()
    assert ex.dispatch(proposal(), idempotency_key="k1", dry_run=True).status == "not_dispatched"
    receipt = ex.dispatch(proposal(), idempotency_key="k2", grant=grant(), dry_run=True)
    assert receipt.status == "dry_run"
    assert ex.dispatch_count == 0
    assert ex.transport.calls == []  # type: ignore[attr-defined]


def test_allowlist_restricts_what_a_connector_performs() -> None:
    ex = executor(allowed_actions=frozenset({"isolate_host"}))
    receipt = ex.dispatch(proposal(), idempotency_key="k", grant=grant())
    assert receipt.reason_codes == ("ACTION_NOT_ALLOWLISTED",)
    assert ex.dispatch_count == 0


def test_proposal_from_a_real_trace(scene_engine: SceneEngine) -> None:
    # The gate's `close` is withheld on an uncalibrated run; the advisory
    # fallback gives the trace something safe to propose and seal.
    scene = load_scene_text("""
scene: t
advisory: {unvalidated_fallback: review}
layers:
  - id: L1
    questions:
      flag: {type: noul, instructions: 是否為誤報？}
    gates:
      - when: 'flag.probability > 0.5'
        then: close
      - else: true
        then: review
actions:
  close: {}
  review: {human_review: true}
""")
    trace = scene_engine.run(scene, "x")
    assert trace.withheld_action == "close" and trace.action == "review"
    p = ProposedDecision.from_trace(trace, case_id="case-7", asset_id="host-9")
    assert p is not None
    assert p.action == trace.action
    assert p.policy_verdict == trace.policy_verdict
    assert p.reason_codes == tuple(trace.reason_codes)
    assert p.scene == "t" and p.scene_version == 1
    assert p.evidence_refs == ("L1.flag",)
    assert p.trace_id and p.model_digest
    assert not p.automatable, "an uncalibrated run is never automatable"
    decision = PolicyDecision.for_proposal(p)
    assert decision.verdict == "review_required"
    assert decision.policy_version == POLICY_VERSION


def test_proposal_from_a_trace_that_decided_nothing() -> None:
    scene = load_scene_text(
        """
scene: t
fallthrough: review
layers:
  - id: L1
    questions:
      flag: {type: noul, instructions: x}
    gates:
      - when: 'flag.probability > 2'
        then: review
actions:
  review: {human_review: true}
"""
    )
    trace = SceneEngine(Engine()).run(scene, "x")
    assert trace.verdict == "fallthrough" and trace.action == "review"
    assert ProposedDecision.from_trace(trace, case_id="c") is not None


# --------------------------------------------------------------------------- #
# Sealed proposals: a connector cannot edit what the scene proposed
# --------------------------------------------------------------------------- #

SEAL = b"server-proposal-secret"


def test_a_sealed_proposal_round_trips() -> None:
    p = proposal()
    token = p.seal(SEAL, now=T0)
    assert ProposedDecision.unseal(token, SEAL, now=T0 + timedelta(minutes=5)) == p


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d["proposal"].update(action_tier=1),
        lambda d: d["proposal"].update(policy_verdict="allow"),
        lambda d: d["proposal"].update(asset_id="host-02"),
        lambda d: d["proposal"].update(action="tag"),
    ],
)
def test_editing_a_sealed_proposal_breaks_it(edit) -> None:
    import base64
    import json as _json

    from jef_scene.authz import ProposalError

    token = proposal().seal(SEAL, now=T0)
    encoded, mac = token.split(".")
    body = _json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    edit(body)
    forged = base64.urlsafe_b64encode(_json.dumps(body).encode()).decode().rstrip("=") + "." + mac
    with pytest.raises(ProposalError, match="does not verify"):
        ProposedDecision.unseal(forged, SEAL, now=T0)


def test_sealed_proposals_expire_and_need_the_secret() -> None:
    from jef_scene.authz import ProposalError

    token = proposal().seal(SEAL, now=T0)
    with pytest.raises(ProposalError, match="expired"):
        ProposedDecision.unseal(token, SEAL, now=T0 + timedelta(hours=2))
    with pytest.raises(ProposalError):
        ProposedDecision.unseal(token, b"other", now=T0)
    with pytest.raises(ProposalError):
        ProposedDecision.unseal("not-a-token", SEAL)
    with pytest.raises(ProposalError):
        proposal().seal(b"")


# --------------------------------------------------------------------------- #
# Two-phase: authorize, the platform acts, complete
# --------------------------------------------------------------------------- #


def test_two_phase_authorize_then_complete() -> None:
    ex = executor()
    first = ex.authorize(proposal(), idempotency_key="k", grant=grant())
    assert first.status == "authorized"
    assert first.grant_nonce
    assert ex.dispatch_count == 0, "authorizing performs nothing"
    done = ex.complete("k", "success", detail="isolated", external_ref="edr-9")
    assert done.status == "success" and done.external_ref == "edr-9"
    assert ex.receipt("k") is done
    assert ex.dispatch_count == 1


def test_pol_005_two_phase_is_idempotent() -> None:
    ex = executor()
    receipts = [ex.authorize(proposal(), idempotency_key="same", grant=grant()) for _ in range(10)]
    assert len({id(r) for r in receipts}) == 1
    ex.complete("same", "success")
    assert ex.authorize(proposal(), idempotency_key="same", grant=grant()).status == "success"
    assert ex.complete("same", "success").status == "success"
    assert ex.dispatch_count == 1


def test_pol_007_two_phase_timeout_is_unknown() -> None:
    ex = executor()
    ex.authorize(proposal(), idempotency_key="k", grant=grant())
    done = ex.complete("k", "timeout")
    assert done.status == "unknown"
    assert "OUTCOME_UNOBSERVED" in done.reason_codes


def test_complete_refuses_what_was_never_authorized() -> None:
    from jef_scene.executor import OutcomeConflict

    ex = executor()
    with pytest.raises(OutcomeConflict, match="never authorized"):
        ex.complete("nope", "success")
    refused = ex.authorize(proposal(), idempotency_key="r")  # no grant
    assert refused.status == "not_dispatched"
    with pytest.raises(OutcomeConflict, match="not_dispatched"):
        ex.complete("r", "success")
    ex.authorize(proposal(), idempotency_key="k", grant=grant())
    ex.complete("k", "failed")
    with pytest.raises(OutcomeConflict, match="cannot become success"):
        ex.complete("k", "success")
    with pytest.raises(OutcomeConflict, match="outcome must be"):
        ex.complete("k", "done")


def test_two_phase_dry_run_records_nothing() -> None:
    ex = executor()
    assert (
        ex.authorize(proposal(), idempotency_key="k", grant=grant(), dry_run=True).status
        == "dry_run"
    )
    assert ex.receipt("k") is None
    assert ex.authorize(proposal(), idempotency_key="k", grant=grant()).status == "authorized"
