"""Sample validation and the train/calibration/test split."""

from __future__ import annotations

from collections import Counter

import pytest
from jef_train.schema import Sample, read_samples, split_samples, write_samples


def choice(label: int = 0, source: str = "s") -> Sample:
    return Sample(
        state="付款服務連續三天失敗",
        kind="choice",
        instructions="誰處理？",
        option_keys=["billing", "infra", "appsec"],
        option_labels=["付款", "網路", "漏洞"],
        label=label,
        source=source,
    )


def noul(label: int = 1, source: str = "n") -> Sample:
    return Sample(
        state="付款服務失敗",
        kind="noul",
        instructions="是否緊急？",
        option_keys=["false", "true"],
        option_labels=[None, None],
        label=label,
        source=source,
    )


def test_bucket_matches_the_engine_s_calibration_key() -> None:
    assert choice().bucket == "choice:3"
    assert noul().bucket == "noul:2"


def test_rejects_fewer_than_two_options() -> None:
    with pytest.raises(ValueError, match=">= 2 options"):
        Sample(
            state="x",
            kind="choice",
            instructions="q",
            option_keys=["only"],
            option_labels=[None],
            label=0,
            source="s",
        )


def test_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="align"):
        Sample(
            state="x",
            kind="choice",
            instructions="q",
            option_keys=["a", "b"],
            option_labels=[None],
            label=0,
            source="s",
        )


def test_rejects_out_of_range_label() -> None:
    with pytest.raises(ValueError, match="out of range"):
        Sample(
            state="x",
            kind="choice",
            instructions="q",
            option_keys=["a", "b"],
            option_labels=[None, None],
            label=5,
            source="s",
        )


def test_noul_option_order_is_fixed() -> None:
    # Index 1 must always be P(true) -- the engine depends on it.
    with pytest.raises(ValueError, match="option_keys"):
        Sample(
            state="x",
            kind="noul",
            instructions="q",
            option_keys=["true", "false"],
            option_labels=[None, None],
            label=0,
            source="s",
        )


def test_to_question_round_trips_through_the_engine() -> None:
    from jef_core import Engine

    engine = Engine()
    for sample in (choice(), noul()):
        answers = engine.evaluate(sample.state, {"q": sample.to_question()}).answers
        assert "q" in answers


def test_to_question_omits_empty_noul_criteria() -> None:
    assert "criteria" not in noul().to_question()


def test_jsonl_round_trip(tmp_path) -> None:
    samples = [choice(), noul()]
    path = tmp_path / "s.jsonl"
    assert write_samples(samples, path) == 2
    back = list(read_samples(path))
    assert [s.state for s in back] == [s.state for s in samples]
    assert back[0].option_labels == samples[0].option_labels


def test_split_is_three_way_and_disjoint() -> None:
    samples = [choice(i % 3, "a") for i in range(300)] + [noul(i % 2, "b") for i in range(200)]
    tr, cal, te = split_samples(samples)
    assert len(tr) + len(cal) + len(te) == len(samples)
    # Calibration must never overlap training: fitting temperature on data the
    # head saw produces a calibrator that looks excellent and generalises not at
    # all, which is the exact failure this project claims to fix.
    assert not ({id(s) for s in tr} & {id(s) for s in cal})
    assert not ({id(s) for s in cal} & {id(s) for s in te})


def test_split_is_stratified_across_sources_and_buckets() -> None:
    samples = [choice(i % 3, "a") for i in range(300)] + [noul(i % 2, "b") for i in range(200)]
    for part in split_samples(samples):
        sources = Counter(s.source for s in part)
        assert set(sources) == {"a", "b"}, "every split needs every generator"


def test_split_is_deterministic_for_a_seed() -> None:
    samples = [choice(i % 3) for i in range(100)]
    a = split_samples(samples, seed=7)[0]
    b = split_samples(samples, seed=7)[0]
    assert [s.label for s in a] == [s.label for s in b]


def test_split_rejects_impossible_ratios() -> None:
    with pytest.raises(ValueError, match="sum to"):
        split_samples([choice()], train=0.8, calibration=0.3)
