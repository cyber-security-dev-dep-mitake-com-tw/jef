"""Fetch into the ignored data directory, hash, pin, and check.

The rules, in the order they are enforced:

1. A restricted source (``request-form``, ``registration-or-contact``) is never
   downloaded. ``fetch_source`` raises ``AccessRestricted`` with the adapter's
   instructions before any network call.
2. Every URL is https (``fetch_to`` refuses anything else).
3. What arrives is hashed. If the manifest pins a SHA-256, a different digest
   is a ``MISMATCH`` and the file is kept for inspection but reported as such;
   if the manifest pins nothing, the digest is recorded only with ``pin=True``.
4. ``check_source`` never touches the network: it compares the cache against
   the pins and reports ``missing`` / ``unpinned`` / ``ok`` / ``MISMATCH``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jef_train.corpus.sources import fetch_to, sha256_of

from .adapters import ADAPTERS, inspect_artifact
from .manifest import Artifact, Pins, SocbenchRepo, Source

__all__ = [
    "DATA_DIR",
    "AccessRestricted",
    "ArtifactStatus",
    "check_source",
    "data_dir",
    "fetch_source",
    "pin_local",
]

#: Where downloads land. ``.cache/`` is gitignored at the repository root.
DATA_DIR = Path(os.environ.get("JEF_SOCBENCH_DATA", ".cache/socbench"))


def data_dir(source_id: str, base: Path | None = None) -> Path:
    return (base or DATA_DIR) / source_id


class AccessRestricted(PermissionError):
    """The source's owner gates access; a human has to act."""

    def __init__(self, source: Source, instructions: str) -> None:
        super().__init__(f"{source.id} is {source.access}: not downloading. {instructions}")
        self.source = source
        self.instructions = instructions


@dataclass
class ArtifactStatus:
    name: str
    state: str  # ok | pinned | unpinned | missing | MISMATCH | skipped
    detail: str = ""
    path: Path | None = None

    @property
    def failed(self) -> bool:
        return self.state == "MISMATCH"


def _md5(path: Path) -> str:
    # Cross-check against a checksum the *source* published, nothing more.
    # Our own integrity pin is the SHA-256 next to it.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(artifact: Artifact, path: Path, *, pin: bool) -> ArtifactStatus:
    """Hash ``path`` and reconcile it with the manifest entry."""
    digest = sha256_of(path)
    size = path.stat().st_size
    notes: list[str] = []

    if artifact.size is not None and artifact.size != size:
        return ArtifactStatus(
            artifact.name, "MISMATCH", f"size {size} != manifest {artifact.size}", path
        )
    if artifact.upstream_checksum and artifact.upstream_checksum.startswith("md5:"):
        if _md5(path) != artifact.upstream_checksum[4:]:
            return ArtifactStatus(
                artifact.name, "MISMATCH", "md5 differs from the checksum upstream published", path
            )
        notes.append("upstream md5 agrees")

    if artifact.sha256 is None:
        if pin:
            artifact.sha256 = digest
            artifact.size = size
            artifact.recorded_at = datetime.now(UTC).date().isoformat()
            notes.append(f"pinned sha256 {digest[:12]}…")
            return ArtifactStatus(artifact.name, "pinned", "; ".join(notes), path)
        notes.append("no pin recorded (use --pin)")
        return ArtifactStatus(artifact.name, "unpinned", "; ".join(notes), path)

    if digest != artifact.sha256:
        return ArtifactStatus(
            artifact.name,
            "MISMATCH",
            f"sha256 {digest[:12]}… != pinned {artifact.sha256[:12]}…",
            path,
        )
    notes.append("sha256 matches pin")
    return ArtifactStatus(artifact.name, "ok", "; ".join(notes), path)


def fetch_source(
    repo: SocbenchRepo,
    source: Source,
    *,
    base: Path | None = None,
    refresh: bool = False,
    pin: bool = False,
    inspect: bool = False,
) -> tuple[list[ArtifactStatus], Pins | None]:
    """Download every artifact of ``source``, hash it, reconcile with the pins.

    Returns the per-artifact statuses and, when ``pin`` recorded anything, the
    updated ``Pins`` (already saved). Raises ``AccessRestricted`` for a gated
    source before any download.
    """
    adapter = ADAPTERS[source.id]
    if source.restricted:
        raise AccessRestricted(source, adapter.instructions)
    if adapter.reference_only:
        return [ArtifactStatus("-", "skipped", "reference-only source; nothing to fetch")], None

    pins = repo.load_pins(source.id)
    if not pins.artifacts:
        return [ArtifactStatus("-", "skipped", pins.note or "no artifacts listed")], None

    statuses: list[ArtifactStatus] = []
    changed = False
    target = data_dir(source.id, base)
    for artifact in pins.artifacts:
        dest = target / artifact.name
        fetch_to(artifact.url, dest, refresh=refresh)
        status = _compare(artifact, dest, pin=pin)
        if status.state == "pinned":
            changed = True
        if inspect and dest.is_file():
            status.detail = f"{status.detail}; {inspect_artifact(dest)}"
        statuses.append(status)

    if changed:
        repo.save_pins(pins)
        return statuses, pins
    return statuses, None


def check_source(
    repo: SocbenchRepo, source: Source, *, base: Path | None = None
) -> list[ArtifactStatus]:
    """Offline: compare the cached files against the pins. No network."""
    adapter = ADAPTERS[source.id]
    if adapter.reference_only or (source.restricted and not repo.pins_path(source.id).exists()):
        return [ArtifactStatus("-", "skipped", "nothing to check")]
    pins = repo.load_pins(source.id)
    if not pins.artifacts:
        return [ArtifactStatus("-", "skipped", pins.note or "no artifacts listed")]
    target = data_dir(source.id, base)
    out: list[ArtifactStatus] = []
    for artifact in pins.artifacts:
        path = target / artifact.name
        if not path.is_file():
            out.append(ArtifactStatus(artifact.name, "missing", f"not in {target}", path))
            continue
        out.append(_compare(artifact, path, pin=False))
    return out


def pin_local(
    repo: SocbenchRepo, source: Source, *, base: Path | None = None
) -> tuple[list[ArtifactStatus], Pins]:
    """Record digests for files a human placed by hand (restricted sources).

    The artifacts must already be listed in the manifest with a URL (the page
    the human obtained them from) and a null sha256. Nothing is downloaded.
    """
    pins = repo.load_pins(source.id)
    target = data_dir(source.id, base)
    out: list[ArtifactStatus] = []
    changed = False
    for artifact in pins.artifacts:
        path = target / artifact.name
        if not path.is_file():
            out.append(ArtifactStatus(artifact.name, "missing", f"place it at {path}", path))
            continue
        status = _compare(artifact, path, pin=True)
        changed = changed or status.state == "pinned"
        out.append(status)
    if changed:
        repo.save_pins(pins)
    return out, pins
