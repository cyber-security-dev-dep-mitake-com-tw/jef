"""Backbone implementations.

``hashing`` has no heavy dependencies and is deterministic -- it exercises every
code path in CI without downloading a model, but its vectors carry no semantics,
so it must never be used to produce published numbers.

``mmbert`` is the real backbone (decision D2: ``jhu-clsp/mmBERT-base``, chosen
over ModernBERT-large because zh-TW is a first-class requirement).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..backbone import Backbone

__all__ = [
    "DEFAULT_MODEL_ID",
    "HashingBackbone",
    "backbone_revision",
    "load_backbone",
    "resolve_backbone_spec",
]

from .hashing import HashingBackbone
from .pins import DEFAULT_MODEL_ID, resolve_backbone_spec


def load_backbone(spec: str = "hashing", **kwargs: object) -> Backbone:
    """Load a backbone by spec.

    Args:
        spec: ``"hashing"`` for the dependency-free deterministic stub, or a
            HuggingFace model id (e.g. ``"jhu-clsp/mmBERT-base"``) for the real
            encoder, which requires the ``torch`` extra. ``owner/model@revision``
            loads that exact commit; a bare id uses the pinned revision if
            there is one (see :mod:`.pins`).
    """
    if spec == "hashing":
        return HashingBackbone(**kwargs)  # type: ignore[arg-type]

    from .mmbert import MmBertBackbone  # imported lazily: torch is an extra

    return MmBertBackbone(model_id=spec, **kwargs)  # type: ignore[arg-type]


def backbone_revision(backbone: object) -> str | None:
    """The weights revision a backbone loaded, or ``None`` if it has none.

    Read with ``getattr`` rather than declared on :class:`Backbone`: the
    protocol is runtime-checkable, and adding a member would stop every
    existing third-party backbone from satisfying it.
    """
    value = getattr(backbone, "revision", None)
    return str(value) if value else None
