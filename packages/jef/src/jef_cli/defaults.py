"""What `jef` runs when the user has not said what to run.

A first command has to either work or refuse. Until now it did neither: the CLI
defaulted to the `hashing` backbone with no head and no calibration, so
`pip install jef && jef ask ...` returned well-formed answers carrying no
signal. It printed a warning, but a warning next to a confident-looking number
is not the same as not producing the number -- and "calibrated confidence" is
the entire pitch, so an uncalibrated default contradicts the product.

Getting the real thing took four undocumented steps: the `torch` extra, the
backbone id, and two `hf://` paths. None of them are discoverable from `--help`
at the moment you need them.

So `--backbone auto` (now the default) resolves the whole set: the real
backbone and the trained head and calibrator from the Hub, downloaded once into
the usual Hub cache. When the `torch` extra is missing the engine cannot run at
all, so it falls back to `hashing` and says plainly what happened and what to
install -- a stub the user was told about, rather than one they were not.

The library keeps its own default. `Jef()` is still `hashing` with no weights:
that is what makes the contract tests run in milliseconds without reaching the
network, and a library caller is in a position to say what it wants. This
module is about the command line, where the user is not.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

log = logging.getLogger("jef.cli")

__all__ = [
    "AUTO",
    "DEFAULT_BACKBONE",
    "DEFAULT_CALIBRATION",
    "DEFAULT_HEAD",
    "resolve_model",
    "torch_available",
]

AUTO = "auto"

#: Decision D2. The multilingual backbone, because zh-TW is a first-class
#: requirement rather than a later port.
DEFAULT_BACKBONE = "jhu-clsp/mmBERT-base"

#: Pinned by repo, not by revision: `jef-v0` is versioned by release tag and the
#: resolver accepts `@revision` when a run has to be reproducible.
DEFAULT_HEAD = "hf://dennislee928tw/jef-v0/head.npz"
DEFAULT_CALIBRATION = "hf://dennislee928tw/jef-v0/calibration.json"


def torch_available() -> bool:
    """Whether the `torch` extra is installed.

    `find_spec` rather than an import: importing torch costs seconds and pulls
    CUDA probing with it, and this runs on every invocation just to pick a
    default.
    """
    return importlib.util.find_spec("torch") is not None


def resolve_model(
    backbone: str,
    head: str | Path | None = None,
    calibration: str | Path | None = None,
) -> tuple[str, str | Path | None, str | Path | None]:
    """Expand `auto` into a backbone, a head and a calibrator.

    Anything the user passed explicitly wins -- `--backbone auto --head mine.npz`
    means the real backbone with their head, not the published one.
    """
    if backbone != AUTO:
        return backbone, head, calibration

    if not torch_available():
        # Not an error: `jef validate`, `jef scene --help` and the whole scene
        # linter work without a backbone, and refusing here would break them to
        # punish a user who may not need inference at all.
        log.warning(
            "running the 'hashing' test backbone: answers will be well-formed "
            "but meaningless. The real engine needs the torch extra -- "
            'install it with: pip install "jef[torch]"'
        )
        return "hashing", head, calibration

    if head is not None or calibration is not None:
        # The user named at least one artifact, so they are steering. Fill only
        # the gap and let their own path fail loudly if it is wrong.
        return (
            DEFAULT_BACKBONE,
            DEFAULT_HEAD if head is None else head,
            DEFAULT_CALIBRATION if calibration is None else calibration,
        )

    return (DEFAULT_BACKBONE, *_published_weights())


def _published_weights() -> tuple[str | None, str | None]:
    """The published head and calibrator, or `(None, None)` if unreachable.

    A default has to degrade, not explode. The weights live on the Hub, so this
    can fail for reasons that are nobody's fault -- offline, Hub down, or the
    release that publishes them has not run yet. The honest outcome is the real
    backbone with a zero-shot head: still semantic, merely uncalibrated, and
    `Jef` already warns about exactly that. Turning a missing optional download
    into a traceback on someone's first command would be the worse trade.
    """
    from jef_core.artifacts import resolve_artifact

    try:
        resolve_artifact(DEFAULT_HEAD)
        resolve_artifact(DEFAULT_CALIBRATION)
    except Exception as exc:  # any failure here means "no weights", never a crash
        log.warning(
            "could not fetch the published weights (%s: %s). Falling back to the "
            "zero-shot head on the real backbone: answers are meaningful but "
            "uncalibrated. Pass --head/--calibration to use a local copy.",
            type(exc).__name__,
            exc,
        )
        return None, None

    return DEFAULT_HEAD, DEFAULT_CALIBRATION
