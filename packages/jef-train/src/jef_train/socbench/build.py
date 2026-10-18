"""``jef-bench build``: turn a fetched source into validated cases and gold.

    jef-bench build --source ait-ads            # uses the pinned files in the data dir
    jef-bench build --source ctu-13 --input path/to/CTU-13-Dataset.tar.bz2

Output goes to ``<data-dir>/derived/<source>/`` (ignored by git):
``cases.jsonl``, ``gold.jsonl`` and ``summary.json``. Every record is checked
against the JSON schemas before the files are written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jef_train.corpus.sources import sha256_of

from . import ait_ads, ctu13, ctu13_hosts, iot23
from .manifest import SocbenchRepo
from .pipeline import data_dir
from .records import validate_records, write_jsonl

__all__ = ["BUILDERS", "build_source"]

BUILDERS = ("ait-ads", "ctu-13", "iot-23")


def _default_input(source: str, base: Path | None) -> dict[str, Path]:
    d = data_dir(source, base)
    if source == "ait-ads":
        return {"archive": d / "ait_ads.zip", "labels": d / "labels.csv"}
    if source == "iot-23":
        return {"archive": d}  # the directory of pinned conn.log.labeled files
    return {"archive": d / "CTU-13-Dataset.tar.bz2"}


def build_source(
    repo: SocbenchRepo,
    source: str,
    *,
    base: Path | None = None,
    input_path: Path | None = None,
    labels_path: Path | None = None,
    out_dir: Path | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one source. Returns the summary (also written as summary.json)."""
    if source not in BUILDERS:
        raise ValueError(f"no case builder for {source!r}; have: {', '.join(BUILDERS)}")
    paths = _default_input(source, base)
    if input_path is not None:
        paths["archive"] = input_path
    if labels_path is not None:
        paths["labels"] = labels_path
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{source}: {name} not found at {path}. Fetch it first "
                f"(`jef-bench fetch --source {source} --pin`) or pass --input/--labels."
            )

    if paths["archive"].is_dir():
        # A digest over every input file's digest, in name order.
        parts = [
            f"{f.name}:{sha256_of(f)}" for f in sorted(paths["archive"].glob("*.conn.log.labeled"))
        ]
        digest = "sha256:" + hashlib.sha256("\n".join(parts).encode()).hexdigest()
    else:
        digest = (
            "sha256:" + sha256_of(paths["archive"])
            if paths["archive"].is_file()
            else ("sha256:" + "0" * 64)
        )
    opts = options or {}
    if source == "ait-ads":
        windows = ait_ads.read_phase_windows(paths["labels"].read_text(encoding="utf-8"))
        stats = ait_ads.ParseStats()
        alerts = ait_ads.iter_alerts(paths["archive"], windows, stats)
        cases, golds, summary = ait_ads.build_cases(alerts, windows, source_hash=digest, **opts)
        summary["parse"] = asdict(stats)
    elif source == "iot-23":
        cases, golds, summary = ctu13_hosts.build_host_windows(
            iot23.iter_flows(paths["archive"]),
            source_hash=digest,
            window_s=int(opts.get("window_s", 300)),
            max_cases_per_class=int(opts.get("max_cases_per_class", 200)),
            gold="behaviour",
            fusion=True,
            classify=iot23.classify,
            host_roles=iot23.host_roles(paths["archive"]),
            internal=iot23.internal,
            source_id=iot23.SOURCE_ID,
            case_tag="iot23f",
            family_prefix="iot23-",
        )
    elif opts.get("unit") in {"host-window", "host-fusion"}:
        host_opts = {
            k: v
            for k, v in opts.items()
            if k in {"window_s", "background_host_rate", "max_cases_per_class"}
        }
        if opts.get("unit") == "host-fusion":
            host_opts |= {"gold": "behaviour", "fusion": True}
        cases, golds, summary = ctu13_hosts.build_host_windows(
            ctu13.iter_flows(paths["archive"]), source_hash=digest, **host_opts
        )
    else:
        flow_opts = {k: v for k, v in opts.items() if k != "unit"}
        cases, golds, summary = ctu13.build_cases(
            ctu13.iter_flows(paths["archive"]), source_hash=digest, **flow_opts
        )

    validate_records(repo, "case", cases)
    validate_records(repo, "gold", golds)
    out = out_dir or (base or data_dir(source).parent) / "derived" / source
    summary["inputs"] = {k: str(v) for k, v in paths.items()}
    summary["source_hash"] = digest
    summary["written"] = {
        "cases": write_jsonl(out / "cases.jsonl", cases),
        "gold": write_jsonl(out / "gold.jsonl", golds),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    summary["out_dir"] = str(out)
    return summary
