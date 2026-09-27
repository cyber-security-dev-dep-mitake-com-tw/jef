"""Export the frozen backbone to ONNX for the Go serving path.

    python -m jef_train.export_onnx --out models/backbone

The Go server exists so a deployment can be one static binary with a small
memory footprint instead of a Python process with torch in it. That only works
if the backbone is portable, which is what this produces -- along with the
tokenizer files, because a tokenizer mismatch is the quietest possible way for
two implementations to disagree: every request succeeds and every answer is
subtly wrong.

Exported with dynamic batch and sequence axes so one graph serves both the long
state encode and the short batched query encode.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger("jef.train.export")

__all__ = ["export", "main"]

DEFAULT_MODEL = "jhu-clsp/mmBERT-base"

#: Files the Go tokenizer needs. tokenizer.json is the whole tokenizer; the rest
#: are copied when present so the exported directory is self-describing.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "config.json",
)


def export(model_id: str, out: Path, *, opset: int = 17) -> dict[str, object]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Two sentences of different lengths, so the traced graph cannot bake in a
    # fixed sequence length or assume an all-ones attention mask.
    encoded = tokenizer(
        ["付款服務連續三天失敗，已影響營收。", "短句"],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    inputs = (encoded["input_ids"], encoded["attention_mask"])

    onnx_path = out / "backbone.onnx"
    log.info("exporting %s to %s (opset %d)", model_id, onnx_path, opset)
    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    from huggingface_hub import hf_hub_download

    copied: list[str] = []
    for name in _TOKENIZER_FILES:
        try:
            path = hf_hub_download(repo_id=model_id, filename=name)
        except Exception as exc:
            log.info("skipping %s: %s", name, type(exc).__name__)
            continue
        shutil.copy(path, out / name)
        copied.append(name)

    meta = {
        "model_id": model_id,
        "opset": opset,
        "hidden_size": int(model.config.hidden_size),
        "max_position_embeddings": int(getattr(model.config, "max_position_embeddings", 0)),
        "tokenizer_files": copied,
        "note": (
            "The Go server must use the tokenizer shipped here, not a "
            "reimplementation. A tokenizer mismatch makes every request succeed "
            "and every answer subtly wrong, which is the hardest kind of bug to "
            "notice in production."
        ),
    }
    (out / "backbone.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "wrote %s (%.0f MiB) + %d tokenizer file(s)",
        onnx_path,
        onnx_path.stat().st_size / 1024 / 1024,
        len(copied),
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the JEF backbone to ONNX")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="models/backbone")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    export(args.model, Path(args.out), opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
