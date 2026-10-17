"""Which exact backbone weights a model id means.

A trained head is a function of one encoder's output vectors. ``from_pretrained``
on a bare model id follows the repo's ``main`` branch, so an upstream push would
change every vector underneath a published head: every request still succeeds,
and every answer is quietly computed against weights the head never saw. A
commit SHA cannot move, so the backbone the head was trained on is pinned here,
and a spec can name any other revision as ``owner/model@revision``.
"""

from __future__ import annotations

__all__ = ["DEFAULT_MODEL_ID", "PINNED_REVISIONS", "resolve_backbone_spec"]

#: Decision D2. Kept here rather than in ``mmbert`` so resolving a spec never
#: needs the torch extra.
DEFAULT_MODEL_ID = "jhu-clsp/mmBERT-base"

#: The revisions shipped heads were trained against. ``c5955035`` (2025-10-07,
#: "Add languages tag") has been upstream ``main`` since before ``models/jef-v0``
#: was fitted, so it is the commit that head saw.
#: That commit ships ``pytorch_model.bin`` only -- upstream deleted its
#: safetensors copy -- which is why loading also requires torch >= 2.6, where
#: ``torch.load`` defaults to ``weights_only=True``.
PINNED_REVISIONS: dict[str, str] = {
    DEFAULT_MODEL_ID: "c5955035435e2bf121cde7f3c8863ef52ff35d82",
}


def resolve_backbone_spec(spec: str, revision: str | None = None) -> tuple[str, str | None]:
    """``"owner/model[@revision]"`` -> ``(model_id, revision)``.

    An explicit revision wins; otherwise a pinned one; otherwise ``None``, which
    means the Hub's ``main`` and is reported by the loader as unpinned.
    """
    model_id, sep, inline = spec.partition("@")
    if sep:
        if not model_id or not inline:
            raise ValueError(
                f"{spec!r} is not a valid backbone spec; expected owner/model@revision"
            )
        if revision and revision != inline:
            raise ValueError(f"{spec!r} names revision {inline!r} but {revision!r} was also passed")
        return model_id, inline
    return model_id, revision or PINNED_REVISIONS.get(model_id)
