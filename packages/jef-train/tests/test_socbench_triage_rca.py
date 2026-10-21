"""The RCA probes run as a library on any engine (here the hashing backbone)."""

from __future__ import annotations

from pathlib import Path

from jef_core import Engine
from jef_scene import load_scene
from jef_train.socbench.triage import load_pack
from jef_train.socbench.triage_eval import scene_questions
from jef_train.socbench.triage_rca import NEUTRAL, label_prior, rca_probes, rotations, summary

ROOT = Path(__file__).resolve().parents[3]
SCENE = load_scene(ROOT / "scenes" / "incident-triage.zh-tw.yaml")
DEV = ROOT / "benchmark" / "incident-triage" / "synthetic" / "dev-v1"
CANARY = ROOT / "benchmark" / "incident-triage" / "canary" / "six-alerts-v1"


def test_rotations_cover_every_order_once() -> None:
    rots = rotations(scene_questions(SCENE)["owner"])
    firsts = [next(iter(r["criteria"])) for r in rots]
    assert sorted(firsts) == sorted(rots[0]["criteria"]) and len(set(firsts)) == len(rots)


def test_label_prior_probes_the_four_content_free_states() -> None:
    r5 = label_prior(Engine(), SCENE)
    assert list(r5["probes"]) == NEUTRAL
    assert 0.0 <= r5["mean_p_fraud"] <= 1.0
    assert r5["prior_towards_fraud"] == (r5["mean_p_fraud"] > 0.30)


def test_all_probes_run_on_any_engine() -> None:
    probes = rca_probes(Engine(), SCENE, load_pack(DEV), load_pack(CANARY))
    assert set(probes["verdict"]) == {
        "R1_fraud_bias_present",
        "R2_order_sensitive",
        "R3_language_dependent",
        "R4_length_driven",
        "R5_prior_towards_fraud",
    }
    assert probes["R3"]["pairs"] > 0 and len(probes["L_canary_live_vs_lab"]) == 6
    assert summary(probes)["verdict"] == probes["verdict"]
