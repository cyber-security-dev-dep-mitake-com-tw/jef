"""Emit cross-language parity fixtures.

JEF has two serving implementations: the Python one and the Go/ONNX one. They
can sit behind the same load balancer, so an answer that depends on which binary
took the request is not an answer. Parity is therefore tested, not assumed:
this writes the Python results, and the Go test suite asserts against them.

    python -m jef_train.golden --out packages/jef-go/testdata/golden.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from jef_core.backbone import StateEncoding
from jef_core.calibration import Calibrator, isotonic_fit
from jef_core.head import BilinearHead, ZeroShotHead, attention_pool
from jef_core.mathx import brier_score, confidence, score_expectation, softmax
from jef_core.types import NormalizedQuestion

__all__ = ["build_fixtures", "main"]

_LOGIT_CASES: list[list[float]] = [
    [1.0, 2.0, 3.0],
    [0.0, 0.0, 0.0],
    [10.0, -10.0],
    # The overflow case: without max-subtraction this is Inf/Inf -> NaN, and a
    # NaN reaching a gate condition is a silent wrong decision.
    [1000.0, -1000.0],
    [-1000.0, -1000.0, -1000.0],
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [3.2, 1.1, -0.4, 2.9, 0.0],
    [1e-8, -1e-8],
    [7.0] * 14,
]

_PROB_CASES: list[list[float]] = [
    [0.5, 0.3, 0.2],
    [1 / 3, 1 / 3, 1 / 3],
    [1.0, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.1, 0.2, 0.3, 0.4],
    [0.9, 0.1],
    [0.5, 0.5],
]


def build_head_fixtures(rng: np.random.Generator, head_path: Path) -> dict[str, Any]:
    """Pooling and head fixtures, plus the head.npz the Go tests load.

    The Go side reads the *same* .npz rather than a converted format: a
    conversion step is somewhere for the two implementations to drift, and the
    whole point of these fixtures is that they cannot.
    """
    dim, rank, tokens = 48, 16, 23
    hidden = rng.normal(0, 1, size=(tokens, dim)).astype(np.float32)
    mask = np.ones(tokens, dtype=np.float32)
    # Mask the tail, so the fixture proves padding is genuinely excluded rather
    # than merely present.
    mask[-5:] = 0.0

    cases: list[dict[str, Any]] = []
    head = BilinearHead.init_identity(dim=dim, rank=rank, seed=7)
    head.save(head_path)
    zero_shot = ZeroShotHead()

    for n_options in (2, 5, 14):
        queries = rng.normal(0, 1, size=(n_options, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=-1, keepdims=True)
        pooled = attention_pool(hidden, mask, queries)
        question = NormalizedQuestion(
            id="q",
            kind="choice" if n_options != 2 else "noul",
            instructions="fixture",
            option_keys=[str(i) for i in range(n_options)],
            option_labels=[None] * n_options,
        )
        encoding = StateEncoding(hidden=hidden, mask=mask, n_tokens=tokens, backbone="fixture")
        cases.append(
            {
                "n_options": n_options,
                "queries": queries.tolist(),
                "expected_pooled": pooled.tolist(),
                "expected_bilinear_logits": list(
                    map(float, head.logits(encoding, question, queries))
                ),
                "expected_zeroshot_logits": list(
                    map(float, zero_shot.logits(encoding, question, queries))
                ),
            }
        )

    return {
        "dim": dim,
        "rank": rank,
        "hidden": hidden.tolist(),
        "mask": mask.tolist(),
        "head_scale": float(head.scale),
        "head_bias": float(head.bias),
        "cases": cases,
    }


def build_fixtures(seed: int = 1337, head_path: Path | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    softmax_cases = [
        {
            "logits": logits,
            "temperature": t,
            "expected": list(softmax(np.array(logits), temperature=t)),
        }
        for logits in _LOGIT_CASES
        for t in (0.5, 1.0, 2.0, 7.3)
    ]

    confidence_cases = [{"probabilities": p, "expected": confidence(p)} for p in _PROB_CASES]
    score_cases = [
        {"probabilities": p, "expected": score_expectation(p)} for p in _PROB_CASES if len(p) >= 2
    ]
    brier_cases = [
        {
            "probabilities": p,
            "correct": i % len(p),
            "expected": brier_score(np.array(p), i % len(p)),
        }
        for i, p in enumerate(_PROB_CASES)
    ]

    # A calibrator fitted on synthetic-but-realistic logits, so the Go side can
    # check temperature, conformal sets and the correctness map end to end.
    calibrator = Calibrator(backbone="fixture", head="fixture")
    for kind, width in (("noul", 2), ("choice", 5), ("score", 5), ("choice", 14)):
        logits, labels = [], []
        for _ in range(600):
            y = int(rng.integers(width))
            lg = rng.normal(0.0, 0.6, size=width)
            lg[y if rng.random() < 0.75 else int(rng.integers(width))] += 2.2
            logits.append(lg)
            labels.append(y)
        calibrator.fit_bucket(kind, width, logits, labels)

    calibration_cases: list[dict[str, Any]] = []
    for kind, width in (("noul", 2), ("choice", 5), ("score", 5), ("choice", 14)):
        for _ in range(12):
            lg = rng.normal(0.0, 1.2, size=width)
            probs = calibrator.apply(lg, kind, width)
            conf = confidence(probs)
            calibration_cases.append(
                {
                    "kind": kind,
                    "n_options": width,
                    "logits": list(map(float, lg)),
                    "expected_temperature": calibrator.temperature(kind, width),
                    "expected_probabilities": list(map(float, probs)),
                    "expected_confidence": conf,
                    "expected_p_correct": calibrator.p_correct(conf, kind, width),
                    "expected_prediction_set_010": calibrator.prediction_set(
                        probs, kind, width, 0.10
                    ),
                }
            )

    isotonic_cases = []
    for n in (200, 800):
        # Correctness rises with confidence, which is the relationship the
        # isotonic map is supposed to recover.
        confidences = rng.uniform(0, 1, n)
        correct = (rng.uniform(0, 1, n) < confidences).astype(float)
        isotonic_cases.append(
            {
                "x": [float(v) for v in confidences],
                "y": [float(v) for v in correct],
                "expected": isotonic_fit(list(confidences), list(correct)),
            }
        )

    from jef_core.backends.hashing import HashingBackbone

    stub = HashingBackbone(dim=256)
    stub_texts = [
        "付款服務連續三天失敗，已影響營收。錯誤碼 502。",
        "short",
        "",
        "混合 mixed 文字 text 123",
    ]
    hashing_cases = [
        {
            "text": text,
            "n_tokens": stub.encode_state(text).n_tokens,
            "expected_state_first": stub.encode_state(text).hidden[0].tolist(),
            "expected_state_last": stub.encode_state(text).hidden[-1].tolist(),
            "expected_query": stub.encode_queries([text])[0].tolist(),
        }
        for text in stub_texts
    ]

    head_fixtures = (
        build_head_fixtures(np.random.default_rng(seed + 1), head_path)
        if head_path is not None
        else None
    )

    return {
        "note": (
            "Generated by jef_train.golden. The Go implementation must reproduce "
            "these to within 1e-9; a divergence means the same alert can get "
            "different verdicts from different binaries."
        ),
        "softmax": softmax_cases,
        "confidence": confidence_cases,
        "score_expectation": score_cases,
        "brier": brier_cases,
        "calibrator": calibrator.to_dict(),
        "calibration": calibration_cases,
        "isotonic": isotonic_cases,
        "head": head_fixtures,
        "hashing_backbone": {"dim": 256, "cases": hashing_cases},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit Go parity fixtures")
    parser.add_argument("--out", default="packages/jef-go/testdata/golden.json")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    head_path = path.parent / "head.npz"
    path.write_text(json.dumps(build_fixtures(args.seed, head_path), indent=1), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB) and {head_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
