"""Deterministic, dependency-free backbone for tests and contract conformance.

This produces *stable* vectors from character n-grams, not *meaningful* ones. It
exists so the whole engine -- normalization, shared-state encoding, the head,
calibration, gates, the HTTP contract -- can be tested in milliseconds without
downloading 300M parameters.

Never publish accuracy or calibration numbers from this backbone.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from ..backbone import StateEncoding

__all__ = ["HashingBackbone"]

_TOKEN_CHARS = 4  # chars per pseudo-token; CJK-friendly (no whitespace assumption)


def _pseudo_tokens(text: str, limit: int) -> list[str]:
    """Split into fixed-width character chunks.

    Whitespace splitting would collapse Chinese text into a handful of tokens,
    so we chunk by characters instead -- zh-TW is a first-class language here.
    """
    chunks = [text[i : i + _TOKEN_CHARS] for i in range(0, len(text), _TOKEN_CHARS)]
    return chunks[:limit] or [""]


def _hash_vec(token: str, dim: int) -> NDArray[np.float32]:
    """Map a token to a stable unit vector via SHA-256 expansion."""
    need = dim * 4
    buf = bytearray()
    counter = 0
    while len(buf) < need:
        buf += hashlib.sha256(f"{token}\x00{counter}".encode()).digest()
        counter += 1
    raw = np.frombuffer(bytes(buf[:need]), dtype=np.uint32).astype(np.float64)
    v = (raw / np.float64(2**32 - 1)) * 2.0 - 1.0
    norm = np.linalg.norm(v)
    if norm == 0:  # pragma: no cover -- unreachable for sha256 output
        return np.zeros(dim, dtype=np.float32)
    return (v / norm).astype(np.float32)


class HashingBackbone:
    """Stable character-n-gram hashing encoder."""

    def __init__(self, dim: int = 256, max_tokens: int = 2048) -> None:
        self.name = "hashing"
        self.dim = dim
        self.max_tokens = max_tokens

    def encode_state(self, text: str) -> StateEncoding:
        tokens = _pseudo_tokens(text, self.max_tokens)
        hidden = np.stack([_hash_vec(t, self.dim) for t in tokens])
        mask = np.ones(len(tokens), dtype=np.float32)
        return StateEncoding(hidden=hidden, mask=mask, n_tokens=len(tokens), backbone=self.name)

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = _pseudo_tokens(t, self.max_tokens)
            v = np.mean([_hash_vec(x, self.dim) for x in toks], axis=0)
            n = np.linalg.norm(v)
            out[i] = v / n if n > 0 else v
        return out

    def count_tokens(self, text: str) -> int:
        return len(_pseudo_tokens(text, self.max_tokens))
