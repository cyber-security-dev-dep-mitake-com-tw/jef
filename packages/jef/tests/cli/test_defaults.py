"""What the CLI runs when the user has not said.

The behaviour that matters here is a product claim, not an implementation
detail: `jef ask` must not quietly answer from an uncalibrated stub, and must
not pretend it can when the torch extra is absent.
"""

from __future__ import annotations

import pytest
from jef_cli import defaults


@pytest.fixture
def with_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(defaults, "torch_available", lambda: True)


@pytest.fixture
def without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(defaults, "torch_available", lambda: False)


@pytest.fixture
def weights_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        defaults,
        "_published_weights",
        lambda: (defaults.DEFAULT_HEAD, defaults.DEFAULT_CALIBRATION),
    )


@pytest.fixture
def weights_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(defaults, "_published_weights", lambda: (None, None))


def test_auto_gives_the_real_model_and_the_published_weights(
    with_torch: None, weights_reachable: None
) -> None:
    backbone, head, calibration = defaults.resolve_model("auto")
    assert backbone == defaults.DEFAULT_BACKBONE
    assert head == defaults.DEFAULT_HEAD
    assert calibration == defaults.DEFAULT_CALIBRATION


def test_unreachable_weights_keep_the_real_backbone(
    with_torch: None, weights_unreachable: None
) -> None:
    # Degrade, don't explode: a failed optional download leaves a semantic
    # backbone with a zero-shot head, not a traceback on the first command.
    assert defaults.resolve_model("auto") == (defaults.DEFAULT_BACKBONE, None, None)


def test_a_hub_failure_is_reported_not_raised(
    with_torch: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_: object) -> object:
        raise OSError("hub unreachable")

    monkeypatch.setattr("jef_core.artifacts.resolve_artifact", boom)
    with caplog.at_level("WARNING"):
        backbone, head, calibration = defaults.resolve_model("auto")

    assert (backbone, head, calibration) == (defaults.DEFAULT_BACKBONE, None, None)
    assert "hub unreachable" in caplog.text


def test_auto_without_torch_falls_back_to_the_stub(without_torch: None) -> None:
    # Falling back rather than raising: `jef validate` and the scene linter need
    # no backbone, and failing here would break them for a user who never
    # intended to run inference.
    backbone, head, calibration = defaults.resolve_model("auto")
    assert backbone == "hashing"
    assert head is None
    assert calibration is None


def test_the_fallback_says_what_to_install(
    without_torch: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        defaults.resolve_model("auto")
    assert 'pip install "jef[torch]"' in caplog.text


def test_auto_does_not_warn_when_the_real_engine_is_available(
    with_torch: None, weights_reachable: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        defaults.resolve_model("auto")
    assert caplog.text == ""


@pytest.mark.parametrize("backbone", ["hashing", "jhu-clsp/mmBERT-base", "some/other-model"])
def test_an_explicit_backbone_is_never_rewritten(backbone: str, with_torch: None) -> None:
    assert defaults.resolve_model(backbone) == (backbone, None, None)


def test_an_explicit_backbone_keeps_weights_unset(without_torch: None) -> None:
    # The stub has no head by design; asking for it explicitly must not drag the
    # published weights in behind the user's back.
    assert defaults.resolve_model("hashing") == ("hashing", None, None)


def test_explicit_weights_win_over_the_published_ones(
    with_torch: None, weights_reachable: None
) -> None:
    backbone, head, calibration = defaults.resolve_model(
        "auto", head="mine.npz", calibration="mine.json"
    )
    assert backbone == defaults.DEFAULT_BACKBONE
    assert (head, calibration) == ("mine.npz", "mine.json")


def test_one_explicit_weight_does_not_suppress_the_other(
    with_torch: None, weights_reachable: None
) -> None:
    _, head, calibration = defaults.resolve_model("auto", head="mine.npz")
    assert head == "mine.npz"
    assert calibration == defaults.DEFAULT_CALIBRATION


def test_the_published_weights_point_at_the_repo_that_exists() -> None:
    # The namespace was wrong for two release candidates; pin it here so a
    # rename cannot silently reintroduce a 404 default.
    assert defaults.DEFAULT_HEAD.startswith("hf://dennislee928tw/jef-v0/")
    assert defaults.DEFAULT_CALIBRATION.startswith("hf://dennislee928tw/jef-v0/")
