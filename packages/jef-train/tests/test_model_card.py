"""§C.4: the model card is generated from artifacts and cannot drift from them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from jef_core.calibration import Bucket, Calibrator, bucket_key
from jef_train import model_card

ROOT = Path(__file__).resolve().parents[3]
SHIPPED = ROOT / "models" / "jef-v0"
TASK = "soc.ndr.technique.v1"


def candidate(tmp_path: Path) -> Path:
    """A schema-2, task-scoped candidate built from the shipped head and report."""
    d = tmp_path / "jef-soc-candidate"
    d.mkdir()
    for name in ("head.npz", "report.json"):
        shutil.copy(SHIPPED / name, d / name)
    cal = Calibrator(backbone="jhu-clsp/mmBERT-base", head="bilinear-r64")
    cal.buckets[bucket_key("choice", 6, TASK)] = Bucket(
        temperature=1.1,
        conformal={"0.1000": 0.4},
        correctness_map=[[0.0, 0.4], [1.0, 0.95]],
        n_samples=320,
        task_id=TASK,
    )
    cal.buckets[bucket_key("choice", 5, "soc.alert.verdict.v1")] = Bucket(
        temperature=1.0,
        conformal={},
        correctness_map=[],
        n_samples=40,
        task_id="soc.alert.verdict.v1",
    )
    cal.save(d / "calibration.json")
    return d


def test_the_shipped_card_is_current() -> None:
    assert model_card.main([str(SHIPPED), "--check"]) == 0


def test_shipped_card_states_legacy_and_unevaluated() -> None:
    card = model_card.build_card(SHIPPED)
    assert "Schema **1** (legacy" in card
    assert "**not possible**" in card
    assert "**Not evaluated.**" in card
    assert "| `head.npz` |" in card and "| yes |" in card


def test_a_task_scoped_candidate_lists_only_its_validated_tasks(tmp_path: Path) -> None:
    card = model_card.build_card(candidate(tmp_path))
    assert (
        "Schema **2**" in card
        and "legacy" not in card.split("## Training data")[0].split("Schema")[1][:40]
    )
    assert f"possible for: `{TASK}`" in card
    assert "soc.alert.verdict.v1" in card and "insufficient" in card, (
        "the thin task is shown, not listed"
    )
    assert "possible for: `soc.alert.verdict.v1`" not in card


def test_a_tampered_artifact_is_called_out(tmp_path: Path) -> None:
    d = tmp_path / "m"
    shutil.copytree(SHIPPED, d)
    (d / "report.json").write_text((d / "report.json").read_text().replace("0.7", "0.9", 1))
    card = model_card.build_card(d)
    assert "| `report.json` |" in card and "**NO**" in card
    assert "report.json does not match its manifest" in card


def test_eval_and_corpus_sections_come_from_the_files(tmp_path: Path) -> None:
    ev = {
        "lock": {
            "split": "test",
            "n": 312,
            "cases_digest": "sha256:" + "a" * 64,
            "gold_digest": "sha256:" + "b" * 64,
        },
        "comparison": [
            {
                "scorer": "jef",
                "task": TASK,
                "macro_f1": 0.61,
                "pr_auc": 0.7,
                "ece": 0.05,
                "mistaken_closures": 0,
                "failure_rate": 0.0,
            }
        ],
    }
    cm = {
        "builder": "jef_train.socbench.corpus_export",
        "lock": {"cases_digest": "sha256:" + "c" * 64, "seed": "s", "ratios": {}},
        "min_trust_tier": "A",
        "excluded_by_tier": {"D": 900},
        "splits": {"train": 200},
        "trust_tiers_by_split": {},
        "label_origins": {},
    }
    card = model_card.build_card(candidate(tmp_path), corpus_manifest=cm, eval_report=ev)
    assert "n=312" in card and "| jef | soc.ndr.technique.v1 | 0.610 |" in card
    assert "Minimum trust tier trained on: **A**" in card and "{'D': 900}" in card


def test_check_fails_on_a_stale_card(tmp_path: Path, capsys) -> None:
    d = candidate(tmp_path)
    assert model_card.main([str(d), "--check"]) == 1, "no card yet"
    assert model_card.main([str(d)]) == 0
    assert model_card.main([str(d), "--check"]) == 0
    (d / "report.json").write_text(json.dumps({"splits": {"train": 1}}))
    assert model_card.main([str(d), "--check"]) == 1
    assert "stale" in capsys.readouterr().err
