"""Precompute the frozen backbone's contribution once, then train on that.

This is what makes CPU-only training feasible (D1), and it works because of a
specific property of the head: attention pooling uses the *raw* query vectors,
and the trainable factors ``U`` and ``V`` act only on the pooled result.

    ctx_j = attention_pool(H, mask, q_j)        <- no trainable parameters
    logit_j = cos(ctx_j @ U, q_j @ V) * s + b   <- trainable

So ``ctx`` and ``q`` are constants for a frozen backbone. Caching them instead of
the token-level hidden states is the difference between ~5 GB and ~170 MB for
this corpus, and it removes the backbone from the training loop entirely -- an
epoch becomes a few small matmuls rather than 5,000 transformer forward passes.

The tradeoff is explicit: pooling is fixed, not learned. Making it learnable
would require the full ``H`` per sample and would put the 16 vCPU target out of
reach. If that changes, this module is what has to change with it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from jef_core.backbone import Backbone
from jef_core.head import attention_pool, build_query_texts
from jef_core.types import NormalizedQuestion
from numpy.typing import NDArray

from .schema import Sample

log = logging.getLogger("jef.train.cache")

__all__ = ["FeatureCache", "build_feature_cache"]


class FeatureCache:
    """Flattened per-option features for a list of samples.

    Stored flat with an offsets array rather than as a ragged list, so a whole
    bucket can be gathered with one slice and fed to the optimiser as a batch.
    """

    def __init__(
        self,
        ctx: NDArray[np.float32],
        queries: NDArray[np.float32],
        offsets: NDArray[np.int64],
        labels: NDArray[np.int64],
        n_options: NDArray[np.int64],
        kinds: list[str],
        sources: list[str],
        backbone: str,
    ) -> None:
        self.ctx = ctx
        self.queries = queries
        self.offsets = offsets
        self.labels = labels
        self.n_options = n_options
        self.kinds = kinds
        self.sources = sources
        self.backbone = backbone

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def dim(self) -> int:
        return int(self.ctx.shape[1])

    def sample(self, i: int) -> tuple[NDArray[np.float32], NDArray[np.float32], int]:
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.ctx[lo:hi], self.queries[lo:hi], int(self.labels[i])

    def bucket_of(self, i: int) -> str:
        return f"{self.kinds[i]}:{int(self.n_options[i])}"

    def buckets(self) -> dict[str, list[int]]:
        """Sample indices grouped by calibration bucket.

        Grouping is by (kind, n_options) rather than n_options alone because the
        calibrator buckets the same way -- a 2-option noul and a 2-option choice
        have different sharpness profiles even at identical width.
        """
        out: dict[str, list[int]] = {}
        for i in range(len(self)):
            out.setdefault(self.bucket_of(i), []).append(i)
        return out

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            ctx=self.ctx,
            queries=self.queries,
            offsets=self.offsets,
            labels=self.labels,
            n_options=self.n_options,
            kinds=np.array(self.kinds),
            sources=np.array(self.sources),
            backbone=np.array(self.backbone),
        )

    @classmethod
    def load(cls, path: str | Path) -> FeatureCache:
        z = np.load(Path(path), allow_pickle=False)
        return cls(
            ctx=z["ctx"],
            queries=z["queries"],
            offsets=z["offsets"],
            labels=z["labels"],
            n_options=z["n_options"],
            kinds=[str(k) for k in z["kinds"]],
            sources=[str(s) for s in z["sources"]],
            backbone=str(z["backbone"]),
        )


def build_feature_cache(
    samples: list[Sample],
    backbone: Backbone,
    *,
    log_every: int = 200,
) -> FeatureCache:
    """Run the frozen backbone over every sample once."""
    ctx_rows: list[NDArray[np.float32]] = []
    query_rows: list[NDArray[np.float32]] = []
    offsets: list[int] = [0]
    labels: list[int] = []
    n_options: list[int] = []
    kinds: list[str] = []
    sources: list[str] = []

    for i, s in enumerate(samples):
        question = NormalizedQuestion(
            id=s.source,
            kind=s.kind,
            instructions=s.instructions,
            option_keys=s.option_keys,
            option_labels=s.option_labels,
        )
        encoding = backbone.encode_state(s.state)
        queries = backbone.encode_queries(build_query_texts(question))
        ctx = attention_pool(encoding.hidden, encoding.mask, queries)

        ctx_rows.append(ctx)
        query_rows.append(queries)
        offsets.append(offsets[-1] + ctx.shape[0])
        labels.append(s.label)
        n_options.append(s.n_options)
        kinds.append(s.kind)
        sources.append(s.source)

        if log_every and (i + 1) % log_every == 0:
            log.info("encoded %d/%d", i + 1, len(samples))

    return FeatureCache(
        ctx=np.vstack(ctx_rows).astype(np.float32),
        queries=np.vstack(query_rows).astype(np.float32),
        offsets=np.array(offsets, dtype=np.int64),
        labels=np.array(labels, dtype=np.int64),
        n_options=np.array(n_options, dtype=np.int64),
        kinds=kinds,
        sources=sources,
        backbone=backbone.name,
    )
