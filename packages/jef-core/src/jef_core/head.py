"""Decision heads: score N options against one shared state encoding.

This is the mechanism that reproduces Jev's semantics on an encoder (D3):

    state ──► frozen backbone ──► H (T, D)        [computed ONCE per request/scene]
                                    │
    question_i + option_j ──────────┴──► query q_ij (D,)
                                          │
                              attention-pool H with q_ij  ──►  c_ij (D,)
                                          │
                                    score(c_ij, q_ij) ──► logit_ij
                                          │
                                    softmax over j ──► probabilities

Every question attends over the *same* H independently, so questions cannot
contaminate each other (no context rot) and the marginal cost of one more
question is a short query encode plus a tiny matmul -- not another full pass
over the state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .backbone import StateEncoding
from .types import NormalizedQuestion

__all__ = ["BilinearHead", "DecisionHead", "ZeroShotHead", "build_query_texts"]

_KIND_HINT = {
    "choice": "選項",
    "score": "等級",
    "noul": "判斷",
}


def build_query_texts(q: NormalizedQuestion) -> list[str]:
    """Render one query string per option.

    The option's own description (when supplied) matters more than its key, so
    it is placed last where it is least likely to be truncated.
    """
    texts: list[str] = []
    hint = _KIND_HINT[q.kind]
    for key, label in zip(q.option_keys, q.option_labels, strict=True):
        parts = [q.instructions, f"{hint}: {key}"]
        if label:
            parts.append(label)
        texts.append(" | ".join(parts))
    return texts


def attention_pool(
    hidden: NDArray[np.float32],
    mask: NDArray[np.float32],
    queries: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Pool ``hidden`` once per query vector.

    Public because it is the contract between training and inference: it uses no
    trainable parameters, so jef-train precomputes its output for the whole
    corpus and never runs the backbone again. Changing this function invalidates
    every cached feature and every trained head.

    Args:
        hidden: ``(T, D)`` shared state tokens.
        mask: ``(T,)`` real-token mask.
        queries: ``(N, D)`` option query vectors.

    Returns:
        ``(N, D)`` context vectors, each a mask-aware softmax-weighted mean of
        the state tokens most relevant to that option.
    """
    d = hidden.shape[1]
    scores = (queries @ hidden.T) / np.sqrt(d, dtype=np.float64)  # (N, T)
    scores = np.where(mask[None, :] > 0, scores, -np.inf)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True).clip(min=1e-12)
    return (weights @ hidden).astype(np.float32)


def _l2(x: NDArray[np.float32]) -> NDArray[np.float32]:
    n = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-9)
    return (x / n).astype(np.float32)


@runtime_checkable
class DecisionHead(Protocol):
    """Turns (shared state, question, option queries) into per-option logits."""

    name: str

    def logits(
        self,
        state: StateEncoding,
        question: NormalizedQuestion,
        queries: NDArray[np.float32],
    ) -> NDArray[np.float64]: ...


class ZeroShotHead:
    """Untrained baseline: cosine similarity between pooled context and query.

    This is what P0 ships so the contract is exercisable before any weights
    exist. It is a genuine baseline, not a placeholder -- P1's trained head is
    measured against it on the same independent benchmarks.
    """

    def __init__(self, scale: float = 10.0) -> None:
        self.name = "zeroshot"
        self.scale = scale

    def logits(
        self,
        state: StateEncoding,
        question: NormalizedQuestion,
        queries: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        ctx = _l2(attention_pool(state.hidden, state.mask, queries))
        q = _l2(queries)
        cos = np.einsum("nd,nd->n", ctx, q).astype(np.float64)
        return cos * self.scale


class BilinearHead:
    """Trained low-rank bilinear head -- the only thing JEF actually trains.

    ``logit_j = (c_j @ U) · (q_j @ V) * scale + bias``

    With the backbone frozen (D1), ``U`` and ``V`` are a few million parameters,
    so training runs in minutes on the 16 vCPU target once the backbone outputs
    for the training set have been cached.
    """

    def __init__(
        self,
        u: NDArray[np.float32],
        v: NDArray[np.float32],
        bias: float = 0.0,
        scale: float = 1.0,
        name: str = "bilinear",
    ) -> None:
        if u.shape != v.shape:
            raise ValueError(f"U and V must share shape, got {u.shape} vs {v.shape}")
        self.u = u.astype(np.float32)
        self.v = v.astype(np.float32)
        self.bias = float(bias)
        self.scale = float(scale)
        self.name = name

    @property
    def dim(self) -> int:
        return int(self.u.shape[0])

    @property
    def rank(self) -> int:
        return int(self.u.shape[1])

    def logits(
        self,
        state: StateEncoding,
        question: NormalizedQuestion,
        queries: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        if state.dim != self.dim:
            raise ValueError(
                f"head expects dim {self.dim}, state encoding has {state.dim} "
                f"(backbone {state.backbone!r} -- head/backbone mismatch)"
            )
        ctx = attention_pool(state.hidden, state.mask, queries)
        a = _l2(ctx @ self.u)
        b = _l2(queries @ self.v)
        return (np.einsum("nr,nr->n", a, b).astype(np.float64) * self.scale) + self.bias

    # -- persistence ------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            u=self.u,
            v=self.v,
            bias=np.float32(self.bias),
            scale=np.float32(self.scale),
            name=np.array(self.name),
        )

    @classmethod
    def load(cls, path: str | Path) -> BilinearHead:
        z = np.load(Path(path), allow_pickle=False)
        return cls(
            u=z["u"],
            v=z["v"],
            bias=float(z["bias"]),
            scale=float(z["scale"]),
            name=str(z["name"]) if "name" in z else "bilinear",
        )

    @classmethod
    def init_identity(cls, dim: int, rank: int, seed: int = 0) -> BilinearHead:
        """Initialise near the zero-shot solution so training starts sane."""
        rng = np.random.default_rng(seed)
        base = np.zeros((dim, rank), dtype=np.float32)
        k = min(dim, rank)
        base[np.arange(k), np.arange(k)] = 1.0
        noise = rng.normal(0.0, 0.01, size=(dim, rank)).astype(np.float32)
        return cls(u=base + noise, v=base + noise.copy(), bias=0.0, scale=10.0)
