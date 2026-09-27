"""Resolve artifact locations, including `hf://` references.

A trained head is ~400KB, so it can reasonably live in three places: next to the
code, attached to a GitHub release, or on Hugging Face. The last one is where
people look for model weights, so paths like

    hf://dennislee928/jef-v0/head.npz
    hf://dennislee928/jef-v0@v0.2.0/calibration.json

resolve to a local file via the Hub cache. Everything else is treated as an
ordinary path, so nothing changes for a local file or a mounted volume.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .errors import BackendUnavailableError

log = logging.getLogger("jef.artifacts")

__all__ = ["HF_SCHEME", "is_hf_uri", "resolve_artifact"]

HF_SCHEME = "hf://"


def is_hf_uri(value: str | Path) -> bool:
    return str(value).startswith(HF_SCHEME)


def _parse(uri: str) -> tuple[str, str, str | None]:
    """`hf://owner/repo[@revision]/path` -> (repo_id, filename, revision)."""
    body = uri[len(HF_SCHEME) :]
    parts = body.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"{uri!r} is not a complete Hugging Face reference; "
            f"expected {HF_SCHEME}owner/repo/filename"
        )
    owner, repo = parts[0], parts[1]
    filename = "/".join(parts[2:])

    revision: str | None = None
    if "@" in repo:
        repo, revision = repo.split("@", 1)

    if not owner or not repo or not filename:
        raise ValueError(f"{uri!r} has an empty owner, repo or filename")
    return f"{owner}/{repo}", filename, revision


def resolve_artifact(location: str | Path) -> Path:
    """Return a local path for an artifact, downloading from the Hub if needed.

    Downloads land in the Hugging Face cache, so a container that mounts
    `HF_HOME` fetches once rather than on every restart -- the same cache the
    backbone already uses.
    """
    if not is_hf_uri(location):
        return Path(location)

    repo_id, filename, revision = _parse(str(location))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise BackendUnavailableError(
            f"{location} needs the Hugging Face Hub client: "
            "pip install huggingface_hub (or use a local path)"
        ) from exc

    log.info("resolving %s from the Hugging Face Hub", location)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
