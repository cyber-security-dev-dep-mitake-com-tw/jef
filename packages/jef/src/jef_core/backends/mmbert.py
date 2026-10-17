"""The real backbone: a frozen multilingual ModernBERT-family encoder.

Decision D2 -- ``jhu-clsp/mmBERT-base`` (ModernBERT architecture, 8192 context,
1833 languages including Chinese). It is chosen over ``ModernBERT-large`` for
exactly one reason: zh-TW is a first-class requirement, and ModernBERT-large is
English-only. ``jhu-clsp/mmBERT-small`` is the low-latency fallback tier.

Decision D1 -- the backbone is frozen and always runs under ``inference_mode``.
Nothing here is trainable; only the head on top is. That is what makes CPU-only
training feasible, because these outputs can be computed once and cached.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backbone import StateEncoding
from ..errors import BackendUnavailableError
from .pins import DEFAULT_MODEL_ID, resolve_backbone_spec

__all__ = ["DEFAULT_MODEL_ID", "MmBertBackbone"]

log = logging.getLogger("jef.backbone")


class MmBertBackbone:
    """Frozen HuggingFace encoder exposing token-level hidden states.

    ``revision`` is the Hub commit actually loaded, or ``None`` when it cannot
    be known (a local directory). A trained head records the revision it was
    fitted against, and the engine refuses to pair it with any other.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        device: str | None = None,
        max_state_tokens: int = 8192,
        max_query_tokens: int = 256,
        threads: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover -- depends on extras
            raise BackendUnavailableError(
                "the 'torch' extra is required for model backbones: pip install 'jef[torch]'"
            ) from exc

        self._torch = torch
        if threads:
            # The deployment target is a 16 vCPU Proxmox VM with no GPU, so
            # thread count is the main throughput lever.
            torch.set_num_threads(threads)

        model_id, revision = resolve_backbone_spec(model_id, revision)
        if revision is None and not Path(model_id).is_dir():
            log.warning(
                "backbone %s is not pinned to a revision: it follows the Hub's main "
                "branch, so an upstream push changes its vectors. Use %s@<commit>.",
                model_id,
                model_id,
            )

        self.name = model_id
        self.max_state_tokens = max_state_tokens
        self.max_query_tokens = max_query_tokens
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(model_id, revision=revision)
        # The commit the Hub resolved, so a branch name or `main` is recorded
        # as what it pointed at when these weights were read.
        self.revision: str | None = getattr(model.config, "_commit_hash", None) or revision
        model.eval()
        for p in model.parameters():  # frozen: D1
            p.requires_grad_(False)
        self._model = model.to(self.device)
        self.dim = int(model.config.hidden_size)

    def _forward(self, texts: list[str], max_len: int) -> tuple[Any, Any]:
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            out = self._model(**enc)
        return out.last_hidden_state, enc["attention_mask"]

    def encode_state(self, text: str) -> StateEncoding:
        hidden, mask = self._forward([text], self.max_state_tokens)
        h = hidden[0].to("cpu").float().numpy()
        m = mask[0].to("cpu").float().numpy()
        return StateEncoding(
            hidden=h.astype(np.float32),
            mask=m.astype(np.float32),
            n_tokens=int(m.sum()),
            backbone=self.name,
            meta={"device": self.device},
        )

    def encode_queries(self, texts: list[str]) -> NDArray[np.float32]:
        hidden, mask = self._forward(texts, self.max_query_tokens)
        m = mask.unsqueeze(-1).float()
        # Mean-pool over real tokens only.
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return pooled.to("cpu").float().numpy().astype(np.float32)

    def count_tokens(self, text: str) -> int:
        return len(
            self._tokenizer(text, truncation=True, max_length=self.max_state_tokens)["input_ids"]
        )
