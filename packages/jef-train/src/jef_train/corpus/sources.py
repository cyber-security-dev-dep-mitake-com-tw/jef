"""Cached fetching of public security corpora.

Every source here is public and redistributable-by-reference. Nothing is derived
from Jev's outputs (D6): the labels come from the taxonomies themselves, which
makes them ground truth rather than another model's opinion, and keeps the
resulting weights clean to publish under Apache-2.0.

Downloads are cached on disk so a rebuild is reproducible and offline-capable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("jef.train.corpus")

__all__ = ["CACHE_DIR", "fetch", "fetch_json", "ATTACK_ENTERPRISE_URL", "NVD_API_URL", "CWE_CSV_URL"]

CACHE_DIR = Path(os.environ.get("JEF_CORPUS_CACHE", ".cache/corpus"))

ATTACK_ENTERPRISE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CWE_CSV_URL = "https://cwe.mitre.org/data/csv/1000.csv.zip"

_UA = "jef-corpus-builder/0.1 (+https://github.com/cyber-security-dev-dep-mitake-com-tw/jef)"


def _cache_path(url: str, suffix: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}{suffix}"


def fetch(url: str, *, suffix: str = ".bin", refresh: bool = False, timeout: int = 120) -> Path:
    """Download ``url`` into the cache and return the local path.

    A cached file is reused unless ``refresh`` is set, so corpus builds are
    reproducible and work behind an air gap once primed.
    """
    path = _cache_path(url, suffix)
    if path.exists() and not refresh:
        log.debug("cache hit %s -> %s", url, path)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(path)  # atomic: a killed download never leaves a valid-looking cache entry
    return path


def fetch_json(url: str, *, refresh: bool = False, timeout: int = 120) -> Any:
    path = fetch(url, suffix=".json", refresh=refresh, timeout=timeout)
    return json.loads(path.read_text(encoding="utf-8"))
