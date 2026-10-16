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
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("jef.train.corpus")

__all__ = [
    "ATTACK_ENTERPRISE_URL",
    "CACHE_DIR",
    "CWE_CSV_URL",
    "NVD_API_URL",
    "fetch",
    "fetch_json",
    "fetch_to",
    "sha256_of",
]

CACHE_DIR = Path(os.environ.get("JEF_CORPUS_CACHE", ".cache/corpus"))

ATTACK_ENTERPRISE_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
CWE_CSV_URL = "https://cwe.mitre.org/data/csv/1000.csv.zip"

_UA = "jef-corpus-builder/0.1 (+https://github.com/cyber-security-dev-dep-mitake-com-tw/jef)"


def _require_https(url: str) -> None:
    """Reject anything but https.

    Without this, a corpus URL sourced from config or an environment variable
    could use file: or ftp: and turn the builder into a local-file read. Every
    source this module ships is https, so the restriction costs nothing.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme != "https":
        raise ValueError(f"corpus sources must be https, got {scheme!r}: {url}")


def _cache_path(url: str, suffix: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}{suffix}"


def fetch_to(url: str, dest: Path, *, refresh: bool = False, timeout: int = 120) -> Path:
    """Download ``url`` to exactly ``dest`` and return it.

    The one download routine in this repository: https only, streamed in 1 MiB
    chunks, written to a ``.part`` sibling and renamed into place, so a killed
    download never leaves a valid-looking file behind. ``fetch`` (the
    URL-keyed corpus cache) and the SOCBench dataset adapters both sit on it;
    a second copy of this loop would be a second place to forget the scheme
    check.
    """
    _require_https(url)
    if dest.exists() and not refresh:
        log.debug("cache hit %s -> %s", url, dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching %s", url)
    # Scheme is validated above, so neither line can open file: or ftp:.
    req = urllib.request.Request(url, headers={"User-Agent": _UA})  # noqa: S310
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)  # atomic: a killed download never leaves a valid-looking cache entry
    return dest


def fetch(url: str, *, suffix: str = ".bin", refresh: bool = False, timeout: int = 120) -> Path:
    """Download ``url`` into the cache and return the local path.

    A cached file is reused unless ``refresh`` is set, so corpus builds are
    reproducible and work behind an air gap once primed.
    """
    return fetch_to(url, _cache_path(url, suffix), refresh=refresh, timeout=timeout)


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, streamed so a multi-gigabyte archive is fine."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str, *, refresh: bool = False, timeout: int = 120) -> Any:
    path = fetch(url, suffix=".json", refresh=refresh, timeout=timeout)
    return json.loads(path.read_text(encoding="utf-8"))
