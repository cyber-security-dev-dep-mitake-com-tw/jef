"""One adapter per source: how a human obtains it, and a peek at what arrived.

Deliberately thin. Item 08 is "stubs that only parse and hash": an adapter
here can tell you a zip has 11 members or a CSV has 400 rows, and nothing in
this module may emit a label. Items 09 to 11 give AIT-ADS, CTU-13 and BOTS v3
real parsers, each in its own PR with its own tests on label provenance.
"""

from __future__ import annotations

import csv
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ADAPTERS", "Adapter", "inspect_artifact"]


def _inspect_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    return {"kind": "zip", "members": len(names), "first": names[:5]}


def _inspect_tar(path: Path) -> dict[str, Any]:
    # Member listing only; nothing is extracted. A multi-gigabyte bz2 is
    # read sequentially, which is slow but bounded.
    with tarfile.open(path) as tf:
        count = sum(1 for _ in tf)
    return {"kind": "tar", "members": count}


def _inspect_csv(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return {"kind": "csv", "columns": header, "rows": rows}


def _inspect_text(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        lines = sum(1 for _ in fh)
    return {"kind": "text", "lines": lines}


def _inspect_opaque(path: Path) -> dict[str, Any]:
    return {"kind": "opaque", "bytes": path.stat().st_size}


_BY_SUFFIX: dict[str, Callable[[Path], dict[str, Any]]] = {
    ".zip": _inspect_zip,
    ".tgz": _inspect_tar,
    ".tar.gz": _inspect_tar,
    ".tar.bz2": _inspect_tar,
    ".csv": _inspect_csv,
    ".md": _inspect_text,
    ".txt": _inspect_text,
    ".json": _inspect_text,
}


def inspect_artifact(path: Path) -> dict[str, Any]:
    """Shape of a fetched file, by suffix. Never its labels."""
    name = path.name.lower()
    for suffix, fn in sorted(_BY_SUFFIX.items(), key=lambda kv: -len(kv[0])):
        if name.endswith(suffix):
            return fn(path)
    return _inspect_opaque(path)


@dataclass(frozen=True)
class Adapter:
    """What the pipeline knows about one source beyond its manifests."""

    source_id: str
    #: Printed when the source is restricted, or alongside a sample-only
    #: fetch. Written for the human who has to click the form.
    instructions: str
    #: Sources whose "data" is a reference implementation or context, not a
    #: corpus to download. ``fetch`` reports "nothing to fetch" for these.
    reference_only: bool = False


ADAPTERS: dict[str, Adapter] = {
    a.source_id: a
    for a in (
        Adapter(
            "ait-ads",
            "Open record on Zenodo (CC-BY-4.0). ait_ads.zip and labels.csv download directly; "
            "the per-source alert fields and the attack-phase windows in labels.csv are "
            "joined in item 09, never as per-alert verdicts.",
        ),
        Adapter(
            "ctu-13",
            "Public directory at mcfp.felk.cvut.cz (CC-BY). The full archive is ~1.9 GB. "
            "only-exes-of-ctu-13.zip holds the botnet binaries and is deliberately not an "
            "artifact here. Background flows become `unknown` in item 10.",
        ),
        Adapter(
            "bots-v3",
            "Public S3 tarball (CC0 claimed; verify on the release used), ~320 MB, pre-indexed "
            "Splunk buckets. The investigation questions and answers are not in the GitHub "
            "repository; the access status of the Q&A is recorded in the manifest note and "
            "cases are built from questions in item 11.",
        ),
        Adapter(
            "lanl-cyber1",
            "LANL asks for a short request form (email and intended use) before download at "
            "https://csr.lanl.gov/data/cyber1/ . CC0, ~12 GB compressed. Fill the form "
            "yourself, place auth/proc/flows/dns/redteam .txt.gz under the data directory, "
            "and pin them with `jef-bench pin --source lanl-cyber1 --local`. The pipeline "
            "will not submit the form for you.",
        ),
        Adapter(
            "otrf-security-datasets",
            "Public GitHub repository, pinned at tag 0.0.2 by commit. Each dataset inside "
            "carries its own licence; check it before any derived case leaves the ignored "
            "data directory. GitHub regenerates archives, so the archive SHA-256 may change "
            "while the commit does not; the manifest note says what to do then.",
        ),
        Adapter(
            "cicids2017",
            "The download page at cicresearch.ca/CICDataset/CIC-IDS-2017/ is a request form "
            "(name, organisation, intended use) rather than a direct link, verified "
            "2026-10-16. Fill it yourself and place MachineLearningCSV.zip or "
            "GeneratedLabelledFlows.zip under the data directory. Secondary source: the "
            "labelling problems in doi:10.1109/SPW53761.2021.00009 apply.",
        ),
        Adapter(
            "community-soc-benchmark",
            "A harness-design reference, not a corpus. Clone "
            "https://github.com/vicks00/soc-benchmark if you want to read it; nothing is "
            "fetched or cited as independent analyst labels.",
            reference_only=True,
        ),
        Adapter(
            "cac-cyber-dataset-1.0",
            "Registration or contact with the operator is required (portal "
            "https://www.cybersac.cn/newhome). Redistribution terms are unknown and "
            "per-record label quality is unverified. A human applies; nothing is fetched. "
            "Report zh-CN separately from zh-TW.",
        ),
        Adapter(
            "trec06c",
            "The TREC 2006 spam track corpora were served from plg.uwaterloo.ca; on "
            "2026-10-16 that URL redirected to a 404. Obtain trec06c.tgz from the track "
            "organisers or a mirror you have verified, place it under the data directory, "
            "and pin it locally. Spam/ham is not phishing/legitimate.",
        ),
        Adapter(
            "ts-bench",
            "Public GitHub repository (mtkresearch/TS-Bench); data/TSB400.csv pinned by commit. "
            "Licence per the repository. 400 zh-TW scam prompts with paired hard negatives; "
            "not enterprise telemetry.",
        ),
        Adapter(
            "chiphish",
            "The GitHub repository (joanyy/ChiPhish) carries a 20-sample public subset and is "
            "fetched here, pinned by commit. The full dataset on Zenodo (record 19558545, "
            "CC-BY-4.0) is access-restricted: request it through the form the record links "
            "to, then place and pin the files locally. Web pages, not mail or SIEM alerts.",
        ),
        Adapter(
            "twcert-cc",
            "Taiwan open government data (data.gov.tw dataset 163766): monthly threat-trend "
            "statistics as CSV/JSON. Refreshed monthly upstream, so the SHA-256 is recorded "
            "for the month fetched and a later `check` legitimately reports a change. "
            "Context and taxonomy only, never per-case labels.",
        ),
    )
}
