#!/usr/bin/env python3
"""Assert the Python and Go servers answer identically.

Run against two live servers. This is the check that keeps "either binary can
take the request" true: golden fixtures pin the maths, but only this exercises
both full HTTP stacks -- request parsing, state rendering, option ordering,
dialect echo and JSON shape -- against each other.

    python scripts/check-parity.py --python http://127.0.0.1:8080 --go http://127.0.0.1:8090
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx

#: One quantum of the six-decimal grid the servers report on, plus slack.
#:
#: Not looseness. The two runtimes compute the same formula over the same
#: float64s, but numpy sums with pairwise/SIMD accumulation and Go with a plain
#: loop, so a probability can differ in its last bit (~1e-16). That is invisible
#: until the value sits exactly on a rounding boundary, where a 1e-16 difference
#: becomes a full 1e-6 step in the *reported* number. Tightening this below one
#: quantum would make the check fail on float representation rather than on any
#: behavioural divergence.
#:
#: The distributions themselves are asserted to 1e-9 by the Go golden-fixture
#: tests, which compare before rounding.
TOLERANCE = 1.5e-6

QUESTIONS: dict[str, Any] = {
    "urgent": {"type": "noul", "instructions": "這則訊息是否表達時間緊迫？"},
    "urgent_boolean": {"type": "boolean", "instructions": "這則訊息是否表達時間緊迫？"},
    "team": {
        "type": "choice",
        "instructions": "應由哪個團隊處理？",
        "criteria": {
            "billing": "付款、發票、退款",
            "infra": "基礎設施、網路、主機層",
            "appsec": "應用程式漏洞與程式碼相關",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "評估此事件的嚴重度",
        "criteria": ["資訊", "低", "中", "高", "危急"],
    },
    "refunded": {
        "type": "boolean",
        "instructions": "是否已退款給客戶？",
        "criteria": {"true": "已確認退款", "false": "未退款或遭拒"},
    },
}

STATES: list[Any] = [
    "客戶回報：連續三天付款失敗，已影響營收。錯誤碼 502。",
    "CrowdStrike 於 02:14 偵測到 db-core-02 出現大量檔案加密行為。",
    {"alert": "payout failed", "count": 3, "host": "ap-trade-07"},
    ["事件一", "事件二", "事件三"],
    "a" * 5000,
    "混合 mixed 文字 text 123 !@#",
]


def compare(a: Any, b: Any, path: str, failures: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        # Key ORDER is compared, not just membership. JSON gives object key
        # order no meaning, but Go's encoding/json sorts map keys while Python
        # preserves insertion order, so the two servers emitted differently
        # shaped bodies for the same answer -- and this checker did not notice,
        # because it only compared values. Option order is also the index each
        # option occupies in the distribution, so it is worth holding onto.
        if list(a) != list(b):
            failures.append(f"{path}: key order {list(a)} vs {list(b)}")
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                failures.append(f"{path}.{key}: present in only one response")
                continue
            compare(a[key], b[key], f"{path}.{key}", failures)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            failures.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            compare(x, y, f"{path}[{i}]", failures)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > TOLERANCE:
            failures.append(f"{path}: {a} vs {b}")
    elif a != b:
        failures.append(f"{path}: {a!r} vs {b!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two JEF servers")
    parser.add_argument("--python", default="http://127.0.0.1:8080")
    parser.add_argument("--go", default="http://127.0.0.1:8090")
    args = parser.parse_args()

    failures: list[str] = []
    for i, state in enumerate(STATES):
        payload = {"state": state, "questions": QUESTIONS}
        left = httpx.post(f"{args.python}/v1/systemone", json=payload, timeout=60).json()
        right = httpx.post(f"{args.go}/v1/systemone", json=payload, timeout=60).json()
        # `model` and `warnings` legitimately differ (different runtimes);
        # `answers` and `usage.outputTokens` must not.
        compare(left["answers"], right["answers"], f"state[{i}].answers", failures)
        if left["usage"]["outputTokens"] != right["usage"]["outputTokens"]:
            failures.append(f"state[{i}].usage.outputTokens differs")

    if failures:
        print(f"PARITY FAILED: {len(failures)} difference(s)", file=sys.stderr)
        for f in failures[:40]:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"parity ok: {len(STATES)} states x {len(QUESTIONS)} questions, no divergence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
