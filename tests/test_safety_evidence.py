"""NEXT-V3-05: S14 comes from a real lab egress log, and the pack records where it ran."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "safety_evidence", ROOT / "scripts" / "safety-evidence.py"
)
assert spec and spec.loader
se = importlib.util.module_from_spec(spec)
sys.modules["safety_evidence"] = se
spec.loader.exec_module(se)

PASS_LOG = """ok    https://api.typesafe.ai blocked
ok    https://pypi.org blocked
ok    external DNS does not resolve
---
PASS: no egress from this segment
"""
FAIL_LOG = """LEAK  https://pypi.org reachable
ok    external DNS does not resolve
---
FAIL: 1 path(s) out of the lab; fix the vSwitch/port-group ACL before any replay
"""


def test_a_pass_log_passes_and_is_copied_with_its_digest(tmp_path: Path) -> None:
    log = tmp_path / "egress.log"
    log.write_text(PASS_LOG)
    r = se._egress(log, tmp_path)
    assert r["result"] == "pass" and r["leaks"] == []
    assert (tmp_path / "egress-check.log").read_text() == PASS_LOG
    assert str(r["log_sha256"]).startswith("sha256:")


def test_a_leak_fails_even_if_someone_appends_pass(tmp_path: Path) -> None:
    log = tmp_path / "egress.log"
    log.write_text(FAIL_LOG)
    assert se._egress(log, tmp_path)["result"] == "FAIL"
    log.write_text(FAIL_LOG + "PASS: no egress from this segment\n")
    r = se._egress(log, tmp_path)
    assert r["result"] == "FAIL" and r["leaks"] == ["LEAK  https://pypi.org reachable"]
