from __future__ import annotations

import numpy as np
import pytest
from jef_core import Engine
from jef_core.calibration import Calibrator
from jef_scene import SceneEngine, load_scene_text

MINIMAL = """
scene: t
layers:
  - id: L1
    questions:
      flag:
        type: noul
        instructions: 是否為誤報？
    gates:
      - when: 'flag.probability > 0.5'
        then: close
      - else: true
        then: review
actions:
  close: {}
  review: {human_review: true}
"""


@pytest.fixture
def minimal_scene():
    return load_scene_text(MINIMAL)


@pytest.fixture
def scene_engine() -> SceneEngine:
    return SceneEngine(Engine())


@pytest.fixture
def calibrated_engine() -> Engine:
    """An engine whose calibrator is fitted, so gates can actually automate."""
    rng = np.random.default_rng(5)
    calibrator = Calibrator(backbone="hashing", head="zeroshot")
    for kind, width in (("noul", 2), ("choice", 5), ("score", 5)):
        logits, labels = [], []
        for _ in range(400):
            y = int(rng.integers(width))
            lg = rng.normal(0, 0.5, size=width)
            lg[y] += 2.5
            logits.append(lg)
            labels.append(y)
        calibrator.fit_bucket(kind, width, logits, labels)
    return Engine(calibrator=calibrator)
