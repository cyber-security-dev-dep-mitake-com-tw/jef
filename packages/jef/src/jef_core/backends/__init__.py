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

__all__ = ["HashingBackbone", "load_backbone"]

from .hashing import HashingBackbone


def load_backbone(spec: str = "hashing", **kwargs: object) -> Backbone:
    """Load a backbone by spec.

    Args:
        spec: ``"hashing"`` for the dependency-free deterministic stub, or a
            HuggingFace model id (e.g. ``"jhu-clsp/mmBERT-base"``) for the real
            encoder, which requires the ``torch`` extra.
    """
    if spec == "hashing":
        return HashingBackbone(**kwargs)  # type: ignore[arg-type]

    from .mmbert import MmBertBackbone  # imported lazily: torch is an extra

    return MmBertBackbone(model_id=spec, **kwargs)  # type: ignore[arg-type]
