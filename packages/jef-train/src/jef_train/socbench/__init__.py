"""SOCBench data layer: the source manifest, artifact pins, fetch and check.

Upgrade-plan item 08. Nothing here produces a label. The layer knows where
each dataset comes from, which of them a program may download at all, and the
SHA-256 of what it last downloaded, so a case built later can cite the exact
bytes it was derived from (`source_hash` in ``schema/case.schema.json``).

No dataset is committed. Downloads land in an ignored directory
(``JEF_SOCBENCH_DATA``, default ``.cache/socbench``); only the digests live in
``benchmark/socbench/manifests/``.
"""

from .adapters import ADAPTERS, Adapter
from .manifest import (
    FETCHABLE_ACCESS,
    RESTRICTED_ACCESS,
    Artifact,
    Pins,
    SocbenchRepo,
    Source,
)
from .pipeline import AccessRestricted, ArtifactStatus, check_source, fetch_source

__all__ = [
    "ADAPTERS",
    "FETCHABLE_ACCESS",
    "RESTRICTED_ACCESS",
    "AccessRestricted",
    "Adapter",
    "Artifact",
    "ArtifactStatus",
    "Pins",
    "SocbenchRepo",
    "Source",
    "check_source",
    "fetch_source",
]
