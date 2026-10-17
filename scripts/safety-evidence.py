#!/usr/bin/env python3
"""Generate the Safety Evidence Pack for one commit (plan v2, NEXT-A / §6.4).

    uv run python scripts/safety-evidence.py [--out DIR]

Runs every automated test that proves one of the safety cases S01-S16, once,
and writes to ``evidence/safety/mainline-integration-<sha>/`` (or ``--out``):

* ``summary.md`` and ``summary.json``: each case, its tests, pass/fail.
* ``junit.xml``: the raw pytest result.
* ``environment.json``: git SHA and whether the tree was dirty, Python,
  platform, uv.lock SHA-256, and the shipped model manifest's calibration
  facts, so the pack can be tied to exactly what was tested.

S14 (egress) is a property of the lab network, not of this code; it is listed
as manual with the command that produces its evidence. The exit status is
non-zero if any automated case failed, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRV = "packages/jef-server/tests"
SCN = "packages/jef/tests/scene"

CASES: dict[str, tuple[str, list[str]]] = {
    "S01": (
        "No API key: 401, no inference",
        [
            f"{SRV}/test_security.py::test_api_002_protected_endpoints_need_a_key",
            f"{SRV}/test_security.py::test_wrong_key_is_forbidden_right_key_is_accepted",
        ],
    ),
    "S02": (
        "Oversized body, deep JSON, oversized state: rejected before inference",
        [
            f"{SRV}/test_security.py::test_api_001_oversized_body_is_413_before_the_model_runs",
            f"{SRV}/test_security.py::test_api_001_oversized_state_is_413_before_the_model_runs",
            f"{SRV}/test_security.py::test_api_001_deep_json_is_422_before_the_model_runs",
            f"{SRV}/test_security.py::test_chunked_and_compressed_bodies_are_refused",
        ],
    ),
    "S03": (
        "Rate limit: 429 with Retry-After",
        [
            f"{SRV}/test_security.py::test_rate_limit_returns_429_with_retry_after",
        ],
    ),
    "S04": (
        "No signed grant for a disruptive action: not dispatched",
        [
            f"{SCN}/test_authz.py::test_pol_003_disruptive_action_without_grant_is_not_dispatched",
            f"{SRV}/test_decisions.py::test_an_unauthorized_proposal_needs_a_grant",
        ],
    ),
    "S05": (
        "Forged proposal token or grant: refused, no side effect",
        [
            f"{SCN}/test_authz.py::test_pol_006_forged_approval_in_json_is_refused",
            f"{SCN}/test_authz.py::test_editing_a_sealed_proposal_breaks_it",
            f"{SRV}/test_decisions.py::test_pol_006_forged_grant_and_forged_token_are_refused",
        ],
    ),
    "S06": (
        "Grant for host A used on host B: GRANT_ASSET_MISMATCH",
        [
            f"{SCN}/test_authz.py::test_pol_004_grant_for_another_asset_is_rejected",
            f"{SCN}/test_store.py::test_a_grant_for_one_asset_never_moves_to_another",
        ],
    ),
    "S07": (
        "Same grant, another idempotency key: GRANT_ALREADY_USED",
        [
            f"{SCN}/test_store.py::test_one_grant_cannot_authorise_two_commands",
        ],
    ),
    "S08": (
        "Same key retried concurrently: one authorisation, one side effect",
        [
            f"{SCN}/test_authz.py::test_pol_005_same_key_ten_times_executes_once",
            f"{SCN}/test_store.py::test_ten_workers_authorise_the_same_key_exactly_once",
        ],
    ),
    "S09": (
        "Platform acted but did not answer: unknown, never retried as new",
        [
            f"{SCN}/test_authz.py::test_pol_007_timeout_is_unknown",
            f"{SCN}/test_authz.py::test_pol_007_two_phase_timeout_is_unknown",
            f"{SCN}/test_store.py::test_timeout_is_unknown_and_stays_unknown",
        ],
    ),
    "S10": (
        "Restart: receipts and spent nonces survive",
        [
            f"{SCN}/test_store.py::test_receipts_and_spent_nonces_survive_a_restart",
            f"{SCN}/test_store.py::test_crash_between_nonce_and_receipt_recovers_with_the_same_key_only",
            f"{SRV}/test_decisions_persistent.py::test_a_decision_survives_a_restart",
        ],
    ),
    "S11": (
        "Store unwritable, locked or down: fail closed, no memory fallback",
        [
            f"{SCN}/test_store.py::test_unavailable_store_is_an_error_not_a_fallback",
            f"{SCN}/test_store.py::test_an_unwritable_database_fails_closed",
            f"{SCN}/test_store.py::test_a_locked_database_fails_closed_instead_of_waiting_forever",
            f"{SRV}/test_decisions_persistent.py::test_a_dead_store_answers_503_and_authorises_nothing",
            f"{SRV}/test_decisions_persistent.py::test_a_bad_dsn_is_a_startup_error",
        ],
    ),
    "S12": (
        "Audit row edited, or whole chain recomputed: detected",
        [
            f"{SCN}/test_store.py::test_a_tampered_audit_row_breaks_the_chain",
            f"{SCN}/test_store.py::test_an_exported_head_catches_a_recomputed_chain",
        ],
    ),
    "S13": (
        "Prompt injection in the alert: never authority",
        [
            f"{SRV}/test_injection.py",
        ],
    ),
    "S15": (
        "Schema-1 / unvalidated calibration: automation_ready false",
        [
            f"{SRV}/test_readiness.py::test_legacy_schema_1_artifact_never_reaches_ready",
            f"{SRV}/test_readiness.py::test_default_deployment_is_live_but_ready_for_nothing",
            f"{SRV}/test_readiness.py::test_a_bucket_for_another_task_does_not_count",
            f"{SRV}/test_readiness.py::test_incompatible_artifact_blocks",
            f"{SRV}/test_decisions_persistent.py::test_memory_store_is_reported_as_not_persistent",
        ],
    ),
    "S16": (
        "Separate processes authorise one key: no double spend",
        [
            f"{SCN}/test_store.py::test_four_processes_authorise_the_same_key_once",
        ],
    ),
    "P01": (
        "JEF_PRODUCTION refuses every insecure configuration",
        [
            f"{SRV}/test_production_profile.py",
        ],
    ),
}
MANUAL = {
    "S14": (
        "No egress from the lab segment",
        "bash deploy/lab/egress_check.sh  (inside JEF-LAB-CORE; must end in PASS)",
    ),
}


def _sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(*args: str) -> str:
    cmd = ["git", *args]  # fixed program, arguments from this file only
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def environment() -> dict[str, object]:
    manifest = ROOT / "models" / "jef-v0" / "manifest.json"
    cal = json.loads(manifest.read_text())["calibration"] if manifest.is_file() else None
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        "shipped_model_calibration": cal,
    }


def _attribute(junit: Path) -> dict[str, list[dict[str, str]]]:
    by_node: dict[str, list[dict[str, str]]] = {}
    # Our own pytest just wrote this file a moment ago; it is not untrusted input.
    for tc in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314
        cls, name = tc.get("classname", ""), tc.get("name", "")
        outcome = "passed"
        if tc.find("failure") is not None or tc.find("error") is not None:
            outcome = "failed"
        elif tc.find("skipped") is not None:
            outcome = "skipped"
        file = cls.replace(".", "/") + ".py"
        by_node.setdefault(file, []).append(
            {"test": name.split("[")[0], "id": name, "outcome": outcome}
        )
    return by_node


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    env = environment()
    out = (
        args.out
        or ROOT / "evidence" / "safety" / f"mainline-integration-{str(env['git_sha'])[:12]}"
    )
    out.mkdir(parents=True, exist_ok=True)

    nodes = sorted({n for _, ns in CASES.values() for n in ns})
    junit = out / "junit.xml"
    run = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
            *nodes,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (out / "pytest.log").write_text(run.stdout + run.stderr)
    results = _attribute(junit)

    summary: dict[str, object] = {"environment": env, "cases": {}}
    lines = [
        f"# Safety Evidence Pack — `{str(env['git_sha'])[:12]}`",
        "",
        f"Generated {env['generated_at']} on {env['platform']} (Python {env['python']}). "
        f"Tree dirty: **{env['git_dirty']}**. uv.lock `{env['uv_lock_sha256']}`.",
        "",
        "| Case | What it proves | Tests | Result |",
        "|---|---|---|---|",
    ]
    failed = 0
    for cid, (what, ns) in CASES.items():
        outcomes: list[dict[str, str]] = []
        for n in ns:
            file, _, test = n.partition("::")
            rows = results.get(file, [])
            outcomes += [r for r in rows if not test or r["test"] == test]
        ok = bool(outcomes) and all(o["outcome"] == "passed" for o in outcomes)
        failed += 0 if ok else 1
        summary["cases"][cid] = {
            "what": what,
            "tests": outcomes,
            "result": "pass" if ok else "FAIL",
        }  # type: ignore[index]
        lines.append(f"| {cid} | {what} | {len(outcomes)} | {'pass' if ok else '**FAIL**'} |")
    for cid, (what, how) in MANUAL.items():
        summary["cases"][cid] = {"what": what, "result": "manual", "how": how}  # type: ignore[index]
        lines.append(f"| {cid} | {what} | manual | `{how}` |")
    cal = env.get("shipped_model_calibration") or {}
    lines += [
        "",
        "## What this pack does not show",
        "",
        "- Detection quality. That needs a locked Tier A gold set (`jef-bench evaluate`).",
        f"- That the shipped model is calibrated for SOC tasks: its calibration is schema "
        f"{cal.get('schema')} with {len(cal.get('tasks') or [])} task-scoped buckets, so "
        "`automation_ready` stays false by design.",
        "- Egress isolation (S14), which is a property of the lab network and is checked there.",
        "",
        f"Automated cases failed: **{failed}**.",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}  (automated cases failed: {failed})")
    return 1 if failed or run.returncode not in (0,) else 0


if __name__ == "__main__":
    raise SystemExit(main())
