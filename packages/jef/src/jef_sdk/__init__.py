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

from .embedded import Jef
from .questions import boolean, choice, noul, score
from .remote import JefClient, JefHTTPError

__version__ = "0.1.0"

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
