"""Sample validation and the train/calibration/test split."""

from __future__ import annotations

import pytest
from jef_train.schema import (
    Sample,
    assert_no_group_leakage,
    read_samples,
    split_samples,
    write_samples,
)


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


def _corpus(n_groups: int = 120) -> list[Sample]:
    """A corpus shaped like the real one: several questions per piece of evidence."""
    out: list[Sample] = []
    for g in range(n_groups):
        state = f"漏洞描述編號 {g}：一段足夠長的說明文字。"
        out.append(
            Sample(
                state=state,
                kind="choice",
                instructions="誰處理？",
                option_keys=["billing", "infra", "appsec"],
                option_labels=[None, None, None],
                label=g % 3,
                source="a",
                group=f"evidence:{g}",
            )
        )
        out.append(
            Sample(
                state=state,
                kind="noul",
                instructions="是否緊急？",
                option_keys=["false", "true"],
                option_labels=[None, None],
                label=g % 2,
                source="b",
                group=f"evidence:{g}",
            )
        )
    return out


def test_split_is_three_way_and_covers_everything() -> None:
    samples = _corpus()
    tr, cal, te = split_samples(samples)
    assert len(tr) + len(cal) + len(te) == len(samples)
    assert tr and cal and te


def test_evidence_never_straddles_a_split() -> None:
    """The property the whole pipeline's credibility rests on.

    One CVE yields a severity question and an attack-vector question over the
    same description. Splitting per sample puts that text in train and in test
    at once; the first run of this pipeline scored 1.000 on SOAR routing for
    exactly that reason.
    """
    tr, cal, te = split_samples(_corpus())
    assert_no_group_leakage(tr, cal, te)

    groups = [{s.group_key for s in part} for part in (tr, cal, te)]
    assert not (groups[0] & groups[1])
    assert not (groups[1] & groups[2])
    assert not (groups[0] & groups[2])


def test_state_text_never_straddles_a_split() -> None:
    """The consequence that actually matters: no test state was seen in training."""
    tr, cal, te = split_samples(_corpus())
    assert not ({s.state for s in tr} & {s.state for s in te})
    assert not ({s.state for s in tr} & {s.state for s in cal})


def test_leakage_assertion_catches_a_bad_split() -> None:
    shared = Sample(
        state="x" * 100,
        kind="noul",
        instructions="甲？",
        option_keys=["false", "true"],
        option_labels=[None, None],
        label=0,
        source="s",
        group="g1",
    )
    with pytest.raises(ValueError, match="both sides of the split"):
        assert_no_group_leakage([shared], [shared])


def test_ungrouped_samples_still_split_by_state() -> None:
    """A builder that forgets to set a group must not silently leak."""
    a = Sample(
        state="同一段文字",
        kind="noul",
        instructions="甲？",
        option_keys=["false", "true"],
        option_labels=[None, None],
        label=0,
        source="s",
    )
    b = Sample(
        state="同一段文字",
        kind="choice",
        instructions="乙？",
        option_keys=["x", "y"],
        option_labels=[None, None],
        label=0,
        source="s",
    )
    assert a.group_key == b.group_key


def test_uids_are_stable_and_distinct() -> None:
    samples = _corpus(5)
    assert len({s.uid for s in samples}) == len(samples)
    # Same content, same uid -- this is what lets a feature cache be sliced.
    again = _corpus(5)
    assert [s.uid for s in samples] == [s.uid for s in again]


def test_split_is_stratified_across_sources() -> None:
    for part in split_samples(_corpus()):
        assert {s.source for s in part} == {"a", "b"}, "every split needs every generator"


def test_split_is_deterministic_for_a_seed() -> None:
    samples = _corpus()
    a = split_samples(samples, seed=7)[0]
    b = split_samples(samples, seed=7)[0]
    assert [s.uid for s in a] == [s.uid for s in b]


def test_split_rejects_impossible_ratios() -> None:
    with pytest.raises(ValueError, match="sum to"):
        split_samples(_corpus(4), train=0.8, calibration=0.3)
