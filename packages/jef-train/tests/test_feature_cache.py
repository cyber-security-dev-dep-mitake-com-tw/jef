"""build_feature_cache: the memos change how often the backbone runs, never the features."""

from __future__ import annotations

import numpy as np
from jef_core.backends import load_backbone
from jef_core.head import attention_pool, build_query_texts
from jef_core.types import NormalizedQuestion
from jef_train.cache import FeatureCache, build_feature_cache
from jef_train.schema import Sample


def sample(state: str, question: str, label: int = 0) -> Sample:
    return Sample(
        state=state,
        kind="choice",
        instructions=question,
        option_keys=["a", "b", "c"],
        option_labels=["first", "second", "third"],
        label=label,
        source="unit",
        group="g1",
        lang="en",
    )


class Counting:
    """Wraps a backbone and counts how often each encoder runs."""

    def __init__(self) -> None:
        self.inner = load_backbone("hashing")
        self.name = self.inner.name
        self.states = 0
        self.queries = 0

    def encode_state(self, text: str):  # type: ignore[no-untyped-def]
        self.states += 1
        return self.inner.encode_state(text)

    def encode_queries(self, texts: list[str]):  # type: ignore[no-untyped-def]
        self.queries += 1
        return self.inner.encode_queries(texts)


def test_memos_match_encoding_every_sample_alone() -> None:
    samples = [
        sample("flow: beacon every 60s", "Which behaviour?"),
        sample("flow: beacon every 60s", "Attack activity?", 1),  # same state, adjacent
        sample("alert: brute force", "Which behaviour?", 2),
        sample("alert: brute force", "Attack activity?"),
        sample("flow: beacon every 60s", "Which behaviour?"),  # not adjacent: re-encoded
    ]
    backbone = Counting()
    cache = build_feature_cache(samples, backbone)  # type: ignore[arg-type]
    assert backbone.states == 3, "adjacent samples with one state share one encode"
    assert backbone.queries == 2, "one query encode per distinct question"

    plain = load_backbone("hashing")
    for i, s in enumerate(samples):
        q = NormalizedQuestion(
            id=s.source,
            kind=s.kind,
            instructions=s.instructions,
            option_keys=s.option_keys,
            option_labels=s.option_labels,
        )
        enc = plain.encode_state(s.state)
        queries = plain.encode_queries(build_query_texts(q))
        ctx = attention_pool(enc.hidden, enc.mask, queries)
        lo, hi = cache.offsets[i], cache.offsets[i + 1]
        np.testing.assert_allclose(cache.ctx[lo:hi], ctx, rtol=0, atol=1e-6)
        np.testing.assert_allclose(cache.queries[lo:hi], queries, rtol=0, atol=1e-6)
    assert list(cache.labels) == [s.label for s in samples]


def test_buckets_use_the_samples_own_task(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An exported SOCBench sample is calibrated under its task, not its source."""
    from dataclasses import replace

    samples = [
        replace(
            sample("flow a", "Which behaviour?"),
            source="socbench.ctu-13",
            task="soc.ndr.technique.v1",
        ),
        sample("flow b", "Which behaviour?"),  # no task: derived from source
    ]
    cache = build_feature_cache(samples, load_backbone("hashing"))
    assert [cache.bucket_of(i) for i in range(2)] == [s.bucket for s in samples]
    assert cache.bucket_of(0) == "soc.ndr.technique.v1|choice:3"
    path = tmp_path / "c.npz"
    cache.save(path)
    again = FeatureCache.load(path)
    assert [again.bucket_of(i) for i in range(2)] == [s.bucket for s in samples]
    assert again.subset([samples[0].uid]).bucket_of(0) == "soc.ndr.technique.v1|choice:3"
