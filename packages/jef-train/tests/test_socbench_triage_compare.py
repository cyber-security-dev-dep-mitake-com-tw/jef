"""Head-to-head: refuses unfair inputs, pairs and clusters, and claims nothing without Tier A.

Reports come from ``jef-bench triage evaluate`` with the hashing backbone; a
second "system" is the same report with chosen predictions rewritten, so the
expected direction of every difference is known.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jef_train.socbench import cli, triage_compare
from jef_train.socbench.triage import load_pack, pack_manifest
from jef_train.socbench.triage_compare import CompareError, compare
from jef_train.socbench.triage_synth import triage_lock

ROOT = Path(__file__).resolve().parents[3]
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"


def evaluate(pack: Path, out: Path, mode: str = "questions") -> Path:
    assert cli.main(
        ["triage", "evaluate", "--mode", mode, "--pack", str(pack), "--out", str(out)]
    ) in (0, 1)
    return out / "results.json"


def oracle(report: Path, pack_dir: Path, out: Path) -> Path:
    """The same report, but every prediction is the gold answer (P(FP) to match)."""
    pack = load_pack(pack_dir)
    data = json.loads(report.read_text())
    for row in data["rows"]:
        e = pack.expectations[row["case_id"]]["expected"]
        if not e:
            continue
        for qid, task in [("is_false_positive", "false_positive"), ("has_evidence", "evidence_sufficiency"),
                          ("owner", "owner"), ("severity", "severity"), ("urgent", "urgency")]:  # fmt: skip
            gold = e[task]
            if gold == "unknown":
                continue
            q = row["questions"][qid]
            q["prediction"] = gold[0] if isinstance(gold, list) else gold
            if qid == "is_false_positive":
                q["p_true"] = 0.99 if gold else 0.01
    data["provenance"]["model"]["name"] = "oracle"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(data, ensure_ascii=False))
    return out / "results.json"


@pytest.fixture(scope="module")
def dev_reports(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("cmp")
    base = evaluate(DEV, tmp / "hashing")
    return base, oracle(base, DEV, tmp / "oracle")


def dev_lock() -> dict[str, Any]:
    return json.loads((DEV / "lock.json").read_text())


def test_the_oracle_beats_the_reference_and_it_is_still_not_a_claim(
    dev_reports: tuple[Path, Path],
) -> None:
    result = compare(load_pack(DEV), dev_lock(), list(dev_reports), n_boot=200)
    assert result["evidence_class"] == "NOT A CLAIM"
    assert any("Tier A" in r for r in result["not_a_claim_because"])
    assert result["verdicts"] is None
    owner = result["tasks"]["owner"]
    assert owner["systems"]["oracle"]["macro_f1"] == 1.0
    assert owner["paired_vs_reference"]["oracle"]["better_than_reference_by"] > 0
    sev = result["tasks"]["severity"]
    assert (
        sev["systems"]["oracle"]["mae"] == 0 and sev["systems"]["oracle"]["critical_to_low"] == []
    )
    assert (
        result["ranking"]["oracle"]["recall_at_20pct"]
        >= result["ranking"][result["reference"]]["recall_at_20pct"]
    )


def test_support_is_counted_in_families_not_cases(dev_reports: tuple[Path, Path]) -> None:
    result = compare(load_pack(DEV), dev_lock(), list(dev_reports), n_boot=50)
    for t in result["tasks"].values():
        assert t["families"] <= t["cases"]
        assert t["support"] == (
            "ok" if t["families"] >= triage_compare.MIN_FAMILIES else "insufficient"
        )


def test_unanswered_questions_shrink_the_pair_and_are_listed(
    dev_reports: tuple[Path, Path], tmp_path: Path
) -> None:
    base, orc = dev_reports
    data = json.loads(orc.read_text())
    test_ids = set(dev_lock()["splits"]["test"])
    dropped = next(
        r["case_id"]
        for r in data["rows"]
        if r["case_id"] in test_ids
        and load_pack(DEV).expectations[r["case_id"]]["expected"]["owner"] != "unknown"
    )
    for r in data["rows"]:
        if r["case_id"] == dropped:
            r["questions"]["owner"] = {"status": "missing_answer"}
    (tmp_path / "results.json").write_text(json.dumps(data, ensure_ascii=False))
    full = compare(load_pack(DEV), dev_lock(), [base, orc], n_boot=50)["tasks"]["owner"]
    part = compare(load_pack(DEV), dev_lock(), [base, tmp_path / "results.json"], n_boot=50)[
        "tasks"
    ]["owner"]
    assert part["cases"] == full["cases"] - 1
    assert dropped in part["systems"]["oracle"]["unanswered"]


def test_reports_on_other_bytes_or_another_mode_are_refused(
    dev_reports: tuple[Path, Path], tmp_path: Path
) -> None:
    base, orc = dev_reports
    data = json.loads(orc.read_text())
    data["provenance"]["pack"]["files"]["cases.jsonl"] = "sha256:" + "0" * 64
    (tmp_path / "other.json").write_text(json.dumps(data))
    with pytest.raises(CompareError, match="pack's bytes"):
        compare(load_pack(DEV), dev_lock(), [base, tmp_path / "other.json"])
    data = json.loads(orc.read_text())
    data["provenance"]["scene"]["sha256"] = "sha256:" + "1" * 64
    (tmp_path / "scene.json").write_text(json.dumps(data))
    with pytest.raises(CompareError, match="different scenes"):
        compare(load_pack(DEV), dev_lock(), [base, tmp_path / "scene.json"])
    scene_report = evaluate(DEV, tmp_path / "scene", mode="scene")
    with pytest.raises(CompareError, match="questions-mode"):
        compare(load_pack(DEV), dev_lock(), [base, scene_report])
    with pytest.raises(CompareError, match="at least two"):
        compare(load_pack(DEV), dev_lock(), [base])
    with pytest.raises(CompareError, match="--reference 2"):
        compare(load_pack(DEV), dev_lock(), [base, orc], reference=2)


def test_with_tier_a_labels_and_enough_families_the_verdicts_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dev = load_pack(DEV)
    pack_dir = tmp_path / "tier-a"
    pack_dir.mkdir()
    cases = [{**c, "origin": "reviewer-sourced"} for c in dev.cases]
    exps = [
        {**dev.expectations[c["case_id"]], "label_origin": "double-human-reviewed",
         "label_status": "confirmed", "reviewers": ["r1", "r2"]}
        for c in dev.cases
    ]  # fmt: skip
    (pack_dir / "cases.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    )
    (pack_dir / "expectations.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exps)
    )
    (pack_dir / "manifest.json").write_text(json.dumps(pack_manifest(pack_dir)))
    lock = triage_lock(cases, exps)
    base = evaluate(pack_dir, tmp_path / "base")
    orc = oracle(base, pack_dir, tmp_path / "oracle")
    monkeypatch.setattr(triage_compare, "MIN_FAMILIES", 2)
    result = compare(load_pack(pack_dir), lock, [base, orc], n_boot=200)
    assert result["evidence_class"] == "PRE-REGISTERED COMPARISON", result["not_a_claim_because"]
    assert result["verdicts"]["owner"]["oracle"] in {
        "superior_to_reference",
        "non_inferior_to_reference",
    }
    # A copy of the lock that no longer matches the bytes is never a claim.
    stale = copy.deepcopy(lock)
    stale["cases_digest"] = "sha256:" + "2" * 64
    assert (
        compare(load_pack(pack_dir), stale, [base, orc], n_boot=20)["evidence_class"]
        == "NOT A CLAIM"
    )


def test_cli(
    dev_reports: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base, orc = dev_reports
    out = tmp_path / "cmp.json"
    rc = cli.main(["triage", "compare", "--pack", str(DEV), "--lock", str(DEV / "lock.json"),
                   "--report", str(base), "--report", str(orc), "--n-boot", "50", "--out", str(out)])  # fmt: skip
    assert rc == 0 and "NOT A CLAIM" in capsys.readouterr().out
    assert json.loads(out.read_text())["preregistration"].endswith(
        "head-to-head-preregistration.md"
    )
