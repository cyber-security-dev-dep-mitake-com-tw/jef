"""A trained head and its calibration belong to one backbone, at one revision.

The dimension check cannot see this: two 768-wide encoders produce vectors of
the same shape and entirely different meaning. So the head records the weights
it was fitted on, and the engine refuses any other pairing.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from jef_core import Calibrator, Engine
from jef_core.backends import HashingBackbone, resolve_backbone_spec
from jef_core.backends.pins import DEFAULT_MODEL_ID, PINNED_REVISIONS
from jef_core.errors import ArtifactMismatchError
from jef_core.head import BilinearHead

REV_A = "a" * 40
REV_B = "b" * 40
URGENT = {"type": "noul", "instructions": "是否緊急？"}


def _backbone(name: str = "org/model", revision: str | None = REV_A) -> HashingBackbone:
    """A hashing backbone wearing a real model's identity."""
    bb = HashingBackbone()
    bb.name = name
    bb.revision = revision  # type: ignore[attr-defined]
    return bb


def _head(backbone: str | None, revision: str | None = None, dim: int = 256) -> BilinearHead:
    head = BilinearHead.init_identity(dim, 8)
    head.backbone, head.backbone_revision = backbone, revision
    return head


# --------------------------------------------------------------------------- #
# spec resolution
# --------------------------------------------------------------------------- #


def test_bare_default_id_resolves_to_the_pinned_commit() -> None:
    assert resolve_backbone_spec(DEFAULT_MODEL_ID) == (
        DEFAULT_MODEL_ID,
        PINNED_REVISIONS[DEFAULT_MODEL_ID],
    )


def test_inline_revision_wins_and_unknown_ids_stay_unpinned() -> None:
    assert resolve_backbone_spec(f"{DEFAULT_MODEL_ID}@{REV_B}") == (DEFAULT_MODEL_ID, REV_B)
    assert resolve_backbone_spec("org/other") == ("org/other", None)
    assert resolve_backbone_spec("org/other", REV_A) == ("org/other", REV_A)


def test_conflicting_or_empty_revisions_are_refused() -> None:
    with pytest.raises(ValueError, match="also passed"):
        resolve_backbone_spec(f"org/m@{REV_A}", REV_B)
    with pytest.raises(ValueError, match="not a valid backbone spec"):
        resolve_backbone_spec("org/m@")


# --------------------------------------------------------------------------- #
# the head records its backbone
# --------------------------------------------------------------------------- #


def test_head_identity_round_trips(tmp_path) -> None:
    path = tmp_path / "head.npz"
    _head("org/model", REV_A).save(path)
    back = BilinearHead.load(path)
    assert (back.backbone, back.backbone_revision) == ("org/model", REV_A)
    # Strings only: the file must still load with allow_pickle=False.
    assert np.load(path, allow_pickle=False)["backbone"].dtype.kind == "U"


def test_a_head_saved_without_identity_loads_as_legacy(tmp_path) -> None:
    path = tmp_path / "head.npz"
    _head(None).save(path)
    assert set(np.load(path).files) == {"u", "v", "bias", "scale", "name"}
    back = BilinearHead.load(path)
    assert back.backbone is None and back.backbone_revision is None


# --------------------------------------------------------------------------- #
# the engine refuses a head trained on other weights
# --------------------------------------------------------------------------- #


def test_matching_backbone_and_revision_is_accepted() -> None:
    engine = Engine(_backbone(), _head("org/model", REV_A))
    assert engine.evaluate("付款失敗", {"u": URGENT}).answers["u"]


def test_same_dimension_other_model_is_refused() -> None:
    with pytest.raises(ArtifactMismatchError, match="trained on backbone 'org/other'"):
        Engine(_backbone(), _head("org/other", REV_A))


def test_same_model_other_revision_is_refused() -> None:
    with pytest.raises(ArtifactMismatchError, match=REV_B):
        Engine(_backbone(), _head("org/model", REV_B))


def test_a_head_trained_on_a_real_model_is_refused_on_the_test_stub() -> None:
    with pytest.raises(ArtifactMismatchError, match="'hashing'"):
        Engine("hashing", _head(DEFAULT_MODEL_ID, PINNED_REVISIONS[DEFAULT_MODEL_ID]))


def test_unverifiable_revision_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="jef.engine"):
        Engine(_backbone(revision=None), _head("org/model", REV_A))
    assert "cannot be verified" in caplog.text


def test_legacy_head_warns_unless_its_calibration_names_the_backbone(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="jef.engine"):
        Engine(_backbone(), _head(None))
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "does not record its backbone" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="jef.engine"):
        Engine(_backbone(), _head(None), Calibrator(backbone="org/model"))
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "its calibration names org/model" in caplog.text


# --------------------------------------------------------------------------- #
# calibration for another backbone is dropped, not applied
# --------------------------------------------------------------------------- #


def _fitted(backbone: str, revision: str = "") -> Calibrator:
    rng = np.random.default_rng(0)
    logits, labels = [], []
    for _ in range(400):
        y = int(rng.integers(2))
        lg = rng.normal(0, 0.5, size=2)
        lg[y] += 2.0
        logits.append(lg)
        labels.append(y)
    c = Calibrator(backbone=backbone, backbone_revision=revision)
    c.fit_bucket("noul", 2, logits, labels, task_id="t.v1")
    return c


@pytest.mark.parametrize(
    ("backbone", "revision"),
    [("org/other", REV_A), ("org/model", REV_B)],
    ids=["other-model", "other-revision"],
)
def test_calibration_for_another_backbone_is_incompatible(backbone, revision, caplog) -> None:
    calibrator = _fitted(backbone, revision)
    with caplog.at_level(logging.ERROR, logger="jef.engine"):
        engine = Engine(_backbone(), None, calibrator)
    assert calibrator.incompatible
    assert calibrator.status("noul", 2, task_id="t.v1") == "incompatible"
    assert "calibration is for another backbone" in caplog.text
    question = {"u": {**URGENT, "task_id": "t.v1"}}
    raw = Engine(_backbone(), None, Calibrator()).evaluate("付款失敗", question)
    dropped = engine.evaluate("付款失敗", question)
    assert dropped.answers["u"] == raw.answers["u"], "foreign temperatures must not be applied"


def test_calibration_for_this_backbone_stays_validated() -> None:
    calibrator = _fitted("org/model", REV_A)
    engine = Engine(_backbone(), None, calibrator)
    assert not calibrator.incompatible
    assert calibrator.status("noul", 2, task_id="t.v1") == "validated"
    # The contrast that makes the test above meaningful: applied, it moves the answer.
    question = {"u": {**URGENT, "task_id": "t.v1"}}
    raw = Engine(_backbone(), None, Calibrator()).evaluate("付款失敗", question)
    assert engine.evaluate("付款失敗", question).answers["u"] != raw.answers["u"]


def test_calibration_revision_round_trips(tmp_path) -> None:
    path = tmp_path / "c.json"
    _fitted("org/model", REV_A).save(path)
    back = Calibrator.load(path)
    assert (back.backbone, back.backbone_revision) == ("org/model", REV_A)
