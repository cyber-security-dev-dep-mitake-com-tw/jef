"""Backbone protocol and the shared-state encoding contract.

Decision D3: the state is encoded **once** and reused by every question in a
request, and by every layer of a scene. This mirrors Jev's KV-cache broadcast.
Degrading to one full forward pass per question would make a 20-gate scene 20x
slower and destroy the entire performance claim, so :class:`Backbone` exposes an
``encode_count`` that tests assert against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["Backbone", "InstrumentedBackbone", "StateEncoding"]


@dataclass(frozen=True)
class StateEncoding:
    """Token-level hidden states for one state, computed once and reused.

    Attributes:
        hidden: ``(T, D)`` contextual token vectors.
        mask: ``(T,)`` 1 for real tokens, 0 for padding.
        n_tokens: number of real tokens (for usage accounting).
        backbone: name of the backbone that produced this, to catch mismatches.
    """

    hidden: NDArray[np.float32]
    mask: NDArray[np.float32]
    n_tokens: int
    backbone: str
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hidden.ndim != 2:
            raise ValueError(f"hidden must be (T, D), got {self.hidden.shape}")
        if self.mask.shape != (self.hidden.shape[0],):
            raise ValueError("mask must be (T,) matching hidden's first axis")

    @property
    def dim(self) -> int:
        return int(self.hidden.shape[1])


@runtime_checkable
class Backbone(Protocol):
    """A frozen text encoder.

    Frozen is not incidental: decision D1 (CPU-only training) depends on being
    able to precompute and cache these outputs for the whole training set, then
    train only the small head on top.
    """

    name: str
    dim: int

    def encode_state(self, text: str) -> StateEncoding:
        """Encode the state, retaining token-level hidden states."""
        ...

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode short question/option texts to ``(N, D)`` pooled vectors."""
        ...

    def count_tokens(self, text: str) -> int:
        """Token count, used only for usage reporting."""
        ...


class InstrumentedBackbone:
    """Wraps a backbone and counts ``encode_state`` calls.

    The D3 guarantee is only meaningful if it is enforced, so the engine always
    wraps its backbone in this and the test suite asserts ``encode_count == 1``
    for a full multi-question request and for a full multi-layer scene run.
    """

    def __init__(self, inner: Backbone) -> None:
        self._inner = inner
        self.encode_count = 0
        self.query_encode_count = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dim(self) -> int:
        return self._inner.dim

    def reset_counters(self) -> None:
        self.encode_count = 0
        self.query_encode_count = 0

    def encode_state(self, text: str) -> StateEncoding:
        self.encode_count += 1
        return self._inner.encode_state(text)

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        self.query_encode_count += 1
        return self._inner.encode_queries(texts)

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)
