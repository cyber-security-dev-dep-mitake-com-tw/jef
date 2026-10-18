"""Cost-aware training: the expected false-negative term and hard-positive retraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jef_core.backends import load_backbone
from jef_core.head import BilinearHead
from jef_train.cache import build_feature_cache
from jef_train.fit import _cost_groups
from jef_train.schema import Sample, write_samples
from jef_train.train import TrainConfig, hard_positive_multipliers, logits_for

KEYS = ["botnet", "unknown", "benign"]


def sample(state: str, label: str, group: str = "g") -> Sample:
    return Sample(
        state=state,
        kind="choice",
        instructions="Which behaviour?",
        option_keys=list(KEYS),
        option_labels=["botnet traffic", "cannot tell", "benign traffic"],
        label=KEYS.index(label),
        source="unit",
        group=group,
        task="unit.ndr.v1",
    )


def corpus() -> list[Sample]:
    # The same-looking DNS flows carry both labels, as CTU-13 scenario 1 does.
    out = []
    for i in range(40):
        out.append(sample(f"udp dport 53 flows {i % 5}", "botnet" if i % 2 else "benign", f"g{i}"))
        out.append(sample(f"tcp dport 6667 irc {i}", "botnet", f"h{i}"))
        out.append(sample(f"tcp dport 443 browse {i}", "benign", f"k{i}"))
    return out


def test_cost_groups_come_from_the_samples_option_keys(tmp_path: Path) -> None:
    write_samples(corpus(), tmp_path / "train.jsonl")
    groups = _cost_groups(tmp_path, "botnet,c2", "benign,normal_automation")
    assert groups == (("unit.ndr.v1|choice:3", (0,), (2,)),)
    assert _cost_groups(tmp_path, "", "benign") == ()
    with pytest.raises(SystemExit, match="both attack and closing"):
        _cost_groups(tmp_path, "botnet,benign", "benign")


def test_hard_positives_are_training_attacks_the_head_mostly_closed() -> None:
    cache = build_feature_cache(corpus(), load_backbone("hashing"))
    head = BilinearHead.init_identity(cache.dim, 8)
    groups = (("unit.ndr.v1|choice:3", (0,), (2,)),)
    attacks = int(sum(cache.labels == 0))
    everything, n_all = hard_positive_multipliers(head, cache, groups, weight=3.0, threshold=0.0)
    assert n_all == attacks and set(np.unique(everything)) == {1.0, 3.0}
    assert all(everything[i] == 3.0 for i in range(len(cache)) if cache.labels[i] == 0)
    nothing, n_none = hard_positive_multipliers(head, cache, groups, weight=3.0, threshold=1.01)
    assert n_none == 0 and set(np.unique(nothing)) == {1.0}


def test_fn_cost_moves_probability_off_closing_for_attacks() -> None:
    pytest.importorskip("torch")
    from jef_train.train import train_head

    cache = build_feature_cache(corpus(), load_backbone("hashing"))
    groups = (("unit.ndr.v1|choice:3", (0,), (2,)),)

    def closing_mass_on_attacks(cfg: TrainConfig) -> float:
        head = train_head(cache, None, cfg)
        masses = []
        for i, z in enumerate(logits_for(head, cache)):
            if cache.labels[i] == 0:
                p = np.exp(z - z.max())
                masses.append(float((p / p.sum())[2]))
        return float(np.mean(masses))

    plain = closing_mass_on_attacks(TrainConfig(rank=8, epochs=30, seed=7))
    costly = closing_mass_on_attacks(
        TrainConfig(rank=8, epochs=30, seed=7, fn_cost=4.0, cost_groups=groups)
    )
    assert costly < plain, (plain, costly)


def test_off_by_default_trains_exactly_as_before() -> None:
    pytest.importorskip("torch")
    from jef_train.train import train_head

    cache = build_feature_cache(corpus(), load_backbone("hashing"))
    a = train_head(cache, None, TrainConfig(rank=8, epochs=5, seed=3))
    b = train_head(
        cache,
        None,
        TrainConfig(rank=8, epochs=5, seed=3, cost_groups=(("x|choice:3", (0,), (2,)),)),
    )
    np.testing.assert_array_equal(a.u, b.u)
    assert a.scale == b.scale
