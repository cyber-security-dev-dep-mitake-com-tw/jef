"""Train the bilinear decision head on cached backbone features.

The backbone never appears in this loop. :mod:`jef_train.cache` already ran it
once over the corpus, so an epoch here is a handful of small matmuls over a few
hundred megabytes -- which is why the 16 vCPU, no-GPU target is a reasonable
place to train rather than an act of optimism.

Samples are batched by calibration bucket because options-per-question varies
across sources (14 ATT&CK tactics, 5 severity levels, 2 for yes/no) and a single
padded tensor would either waste work or need masking for no benefit.

Classes are weighted by inverse frequency within their bucket. The corpus is
genuinely skewed -- CVE attack vectors are ~74% `network`, only a fifth of
alerts are known-benign -- so unweighted training buys accuracy by learning the
prior and ignoring the evidence, which is precisely the behaviour a confidence
number is supposed to expose rather than hide.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
from jef_core.head import BilinearHead

from .cache import FeatureCache

log = logging.getLogger("jef.train")

__all__ = ["TrainConfig", "logits_for", "train_head"]


@dataclass(frozen=True)
class TrainConfig:
    rank: int = 64
    epochs: int = 40
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    class_weighting: bool = True
    seed: int = 1337
    #: Stop when validation loss has not improved for this many epochs.
    patience: int = 8
    threads: int | None = None


def _class_weights(cache: FeatureCache) -> np.ndarray:
    """Per-sample weight = inverse frequency of its label within its bucket."""
    weights = np.ones(len(cache), dtype=np.float32)
    for _bucket, idx in cache.buckets().items():
        counts = Counter(int(cache.labels[i]) for i in idx)
        mean = len(idx) / len(counts)
        for i in idx:
            weights[i] = mean / counts[int(cache.labels[i])]
    return weights


def _batches(
    cache: FeatureCache, batch_size: int, rng: np.random.Generator
) -> list[tuple[str, list[int]]]:
    out: list[tuple[str, list[int]]] = []
    for bucket, idx in cache.buckets().items():
        shuffled = list(rng.permutation(idx))
        for start in range(0, len(shuffled), batch_size):
            out.append((bucket, [int(i) for i in shuffled[start : start + batch_size]]))
    rng.shuffle(out)
    return out


def _stack(cache: FeatureCache, indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather a same-width batch into (B, n, D) tensors."""
    ctx = np.stack([cache.ctx[cache.offsets[i] : cache.offsets[i + 1]] for i in indices])
    queries = np.stack([cache.queries[cache.offsets[i] : cache.offsets[i + 1]] for i in indices])
    labels = np.array([cache.labels[i] for i in indices], dtype=np.int64)
    return ctx, queries, labels


def train_head(
    cache: FeatureCache,
    val_cache: FeatureCache | None = None,
    config: TrainConfig | None = None,
) -> BilinearHead:
    """Fit U, V, scale and bias. Returns the trained head."""
    import torch
    import torch.nn.functional as F

    cfg = config or TrainConfig()
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    dim = cache.dim
    init = BilinearHead.init_identity(dim=dim, rank=cfg.rank, seed=cfg.seed)
    u = torch.nn.Parameter(torch.from_numpy(init.u.copy()))
    v = torch.nn.Parameter(torch.from_numpy(init.v.copy()))
    scale = torch.nn.Parameter(torch.tensor(float(init.scale)))
    bias = torch.nn.Parameter(torch.tensor(float(init.bias)))

    optimiser = torch.optim.AdamW([u, v, scale, bias], lr=cfg.lr, weight_decay=cfg.weight_decay)
    sample_weights = (
        _class_weights(cache) if cfg.class_weighting else np.ones(len(cache), np.float32)
    )

    def forward(ctx: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        a = F.normalize(ctx @ u, dim=-1)
        b = F.normalize(queries @ v, dim=-1)
        return (a * b).sum(-1) * scale + bias

    def evaluate(target: FeatureCache) -> tuple[float, float]:
        total_loss, correct, seen = 0.0, 0, 0
        with torch.no_grad():
            for _bucket, idx in _batches(target, 256, np.random.default_rng(0)):
                ctx_np, q_np, y_np = _stack(target, idx)
                logits = forward(torch.from_numpy(ctx_np), torch.from_numpy(q_np))
                y = torch.from_numpy(y_np)
                total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
                correct += int((logits.argmax(-1) == y).sum())
                seen += len(idx)
        return total_loss / max(seen, 1), correct / max(seen, 1)

    best_loss = float("inf")
    best_state: tuple[np.ndarray, np.ndarray, float, float] | None = None
    stale = 0

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss, seen = 0.0, 0
        for _bucket, idx in _batches(cache, cfg.batch_size, rng):
            ctx_np, q_np, y_np = _stack(cache, idx)
            logits = forward(torch.from_numpy(ctx_np), torch.from_numpy(q_np))
            per_sample = F.cross_entropy(logits, torch.from_numpy(y_np), reduction="none")
            w = torch.from_numpy(sample_weights[idx])
            loss = (per_sample * w).sum() / w.sum()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach()) * len(idx)
            seen += len(idx)

        train_loss = epoch_loss / max(seen, 1)
        if val_cache is not None:
            val_loss, val_acc = evaluate(val_cache)
            log.info(
                "epoch %02d  train_loss %.4f  val_loss %.4f  val_acc %.3f",
                epoch,
                train_loss,
                val_loss,
                val_acc,
            )
            monitored = val_loss
        else:
            log.info("epoch %02d  train_loss %.4f", epoch, train_loss)
            monitored = train_loss

        if monitored < best_loss - 1e-5:
            best_loss = monitored
            best_state = (
                u.detach().numpy().copy(),
                v.detach().numpy().copy(),
                float(scale),
                float(bias),
            )
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                # Keep the best weights, not the last ones: without this, early
                # stopping still ships whatever the final overfitting epoch left.
                log.info("early stop at epoch %d (best %.4f)", epoch, best_loss)
                break

    if best_state is None:  # pragma: no cover -- only if epochs == 0
        best_state = (u.detach().numpy(), v.detach().numpy(), float(scale), float(bias))

    bu, bv, bscale, bbias = best_state
    return BilinearHead(u=bu, v=bv, bias=bbias, scale=bscale, name=f"bilinear-r{cfg.rank}")


def logits_for(head: BilinearHead, cache: FeatureCache) -> list[np.ndarray]:
    """Replay a trained head over cached features, one logit vector per sample.

    Used to fit calibration and to evaluate, both of which need the model's raw
    scores rather than its answers.
    """
    out: list[np.ndarray] = []
    for i in range(len(cache)):
        ctx, queries, _ = cache.sample(i)
        a = ctx @ head.u
        a = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(min=1e-9)
        b = queries @ head.v
        b = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(min=1e-9)
        out.append(np.einsum("nr,nr->n", a, b).astype(np.float64) * head.scale + head.bias)
    return out
