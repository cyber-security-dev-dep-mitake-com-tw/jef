"""Decision heads and the shared-state pooling mechanism."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from jef_core.backbone import InstrumentedBackbone, StateEncoding
from jef_core.backends import load_backbone
from jef_core.head import BilinearHead, ZeroShotHead, build_query_texts
from jef_core.types import normalize

CHOICE = normalize(
    "t",
    {
        "type": "choice",
        "instructions": "誰處理？",
        "criteria": {"billing": "付款", "infra": "網路"},
    },
)


@pytest.fixture
def backbone():
    return load_backbone("hashing")


@pytest.fixture
def state(backbone) -> StateEncoding:
    return backbone.encode_state("付款服務連續三天失敗")


def test_query_text_includes_instructions_key_and_label() -> None:
    texts = build_query_texts(CHOICE)
    assert len(texts) == 2
    assert "誰處理？" in texts[0]
    assert "billing" in texts[0]
    assert "付款" in texts[0]


def test_query_text_tolerates_missing_labels() -> None:
    q = normalize("t", {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}})
    texts = build_query_texts(q)
    assert all(t for t in texts)


def test_zeroshot_returns_one_logit_per_option(backbone, state) -> None:
    q = backbone.encode_queries(build_query_texts(CHOICE))
    lg = ZeroShotHead().logits(state, CHOICE, q)
    assert lg.shape == (2,)
    assert np.all(np.isfinite(lg))


def test_pooling_respects_the_mask(backbone) -> None:
    """Padding tokens must not contribute to the pooled context."""
    real = backbone.encode_state("付款失敗")
    pad = np.zeros((3, real.dim), dtype=np.float32)
    padded = StateEncoding(
        hidden=np.vstack([real.hidden, pad]),
        mask=np.concatenate([real.mask, np.zeros(3, dtype=np.float32)]),
        n_tokens=real.n_tokens,
        backbone=real.backbone,
    )
    q = backbone.encode_queries(build_query_texts(CHOICE))
    head = ZeroShotHead()
    assert np.allclose(head.logits(real, CHOICE, q), head.logits(padded, CHOICE, q), atol=1e-5)


def test_bilinear_head_roundtrips(tmp_path, backbone, state) -> None:
    head = BilinearHead.init_identity(dim=backbone.dim, rank=32, seed=1)
    q = backbone.encode_queries(build_query_texts(CHOICE))
    before = head.logits(state, CHOICE, q)

    path = tmp_path / "head.npz"
    head.save(path)
    after = BilinearHead.load(path).logits(state, CHOICE, q)
    assert np.allclose(before, after)


def test_bilinear_head_rejects_dim_mismatch(state) -> None:
    head = BilinearHead.init_identity(dim=state.dim + 8, rank=16)
    q = np.zeros((2, head.dim), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatch"):
        head.logits(state, CHOICE, q)


def test_bilinear_requires_matching_factor_shapes() -> None:
    with pytest.raises(ValueError, match="share shape"):
        BilinearHead(u=np.zeros((8, 4), np.float32), v=np.zeros((8, 5), np.float32))


def test_state_encoding_validates_shapes() -> None:
    with pytest.raises(ValueError, match=r"\(T, D\)"):
        StateEncoding(
            hidden=np.zeros(4, np.float32), mask=np.ones(4, np.float32), n_tokens=4, backbone="x"
        )
    with pytest.raises(ValueError, match="mask must be"):
        StateEncoding(
            hidden=np.zeros((4, 2), np.float32),
            mask=np.ones(3, np.float32),
            n_tokens=4,
            backbone="x",
        )


def test_hashing_backbone_is_deterministic(backbone) -> None:
    a = backbone.encode_state("同樣的文字")
    b = backbone.encode_state("同樣的文字")
    assert np.array_equal(a.hidden, b.hidden)


def test_hashing_backbone_chunks_cjk_without_whitespace(backbone) -> None:
    """Whitespace splitting would collapse Chinese into one token -- it must not."""
    enc = backbone.encode_state("付款服務連續三天失敗無空白字元")
    assert enc.n_tokens > 1


def test_instrumented_backbone_counts_and_resets(backbone) -> None:
    inst = InstrumentedBackbone(backbone)
    inst.encode_state("a")
    inst.encode_state("b")
    inst.encode_queries(["q"])
    assert (inst.encode_count, inst.query_encode_count) == (2, 1)
    inst.reset_counters()
    assert inst.encode_count == 0


def test_missing_torch_extra_raises_actionable_error(monkeypatch) -> None:
    """A missing extra must name the extra, not surface a bare ImportError.

    This simulates the absent dependency rather than loading a real model: a
    unit test that downloads 300M parameters is not a unit test, and it would
    make CI depend on the Hugging Face CDN.
    """
    import builtins

    from jef_core.errors import BackendUnavailableError

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.delitem(sys.modules, "jef_core.backends.mmbert", raising=False)

    with pytest.raises(BackendUnavailableError, match="torch"):
        load_backbone("jhu-clsp/mmBERT-base")


def test_default_backbone_never_downloads_anything() -> None:
    """Engine() with no arguments must stay offline and instant."""
    from jef_core import Engine

    engine = Engine()
    assert engine.backbone.name == "hashing"
