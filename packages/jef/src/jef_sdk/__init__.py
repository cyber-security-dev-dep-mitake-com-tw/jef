"""JEF Python SDK.

Two entry points with the same shape, so moving a workload between them is a
change of constructor rather than a change of code:

    from jef_sdk import Jef, JefClient, choice, noul, score

    jef = Jef("jhu-clsp/mmBERT-base", calibration="models/jef-v0/calibration.json")
    jef = JefClient("http://jef.internal:8080")

    result = jef.evaluate(alert, {"team": choice("誰處理？", soc="監控", infra="網路")})
    trace  = jef.run_scene("incident-triage", alert)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .embedded import Jef
from .questions import boolean, choice, noul, score
from .remote import JefClient, JefHTTPError

try:
    __version__ = _dist_version("jef")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "Jef",
    "JefClient",
    "JefHTTPError",
    "__version__",
    "boolean",
    "choice",
    "noul",
    "score",
]
