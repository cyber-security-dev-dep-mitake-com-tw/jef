"""The two manifests: ``sources.yaml`` (provenance) and ``manifests/<id>.json`` (pins).

Both are validated against the JSON schemas in ``benchmark/socbench/schema/``
on load, so a hand edit that drops a required field fails here, loudly, and
not three steps later when an adapter indexes a missing key.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

__all__ = [
    "FETCHABLE_ACCESS",
    "RESTRICTED_ACCESS",
    "Artifact",
    "Pins",
    "SocbenchRepo",
    "Source",
]

#: Access values a program may act on. ``sample-public-full-restricted`` is
#: here because the *listed* artifacts are the public sample; the adapter's
#: instructions say how a human requests the rest.
FETCHABLE_ACCESS: frozenset[str] = frozenset(
    {"public", "public-download", "public-download-verify", "sample-public-full-restricted"}
)

#: Access values where downloading is a human's act (a form, a registration, a
#: contact). The pipeline refuses these outright rather than scraping around
#: them: the gate is the data owner's condition of use.
RESTRICTED_ACCESS: frozenset[str] = frozenset(
    {"request-form", "registration-or-contact", "application", "own-zone", "unverified"}
)


@dataclass(frozen=True)
class Source:
    """One row of ``sources.yaml``."""

    id: str
    name: str
    url: str
    language: str
    ground_truth_level: str
    realism: str
    hard_negatives: str
    jef_use: tuple[str, ...]
    access: str
    license: str
    priority: int | str
    caveat: str
    kind: str = "dataset"
    label_granularity: str = "unknown"
    trust_tier: str = "U"
    adapter: str = "implemented"
    access_verified: str | None = None
    links: dict[str, str] = field(default_factory=dict)

    @property
    def restricted(self) -> bool:
        return self.access in RESTRICTED_ACCESS

    @property
    def fetchable(self) -> bool:
        return self.wired and self.access in FETCHABLE_ACCESS

    @property
    def wired(self) -> bool:
        """An adapter and a pin manifest exist; catalogue rows are planning entries."""
        return self.adapter == "implemented"


@dataclass
class Artifact:
    """One file an adapter fetches, and what we know about its bytes."""

    name: str
    url: str
    sha256: str | None = None
    size: int | None = None
    upstream_checksum: str | None = None
    recorded_at: str | None = None
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if not (k in {"note", "recorded_at"} and v is None)}


@dataclass
class Pins:
    """``manifests/<source>.json``: the artifacts and their recorded digests."""

    source: str
    artifacts: list[Artifact]
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source}
        if self.note is not None:
            out["note"] = self.note
        out["artifacts"] = [a.to_json() for a in self.artifacts]
        return out


def _find_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "benchmark" / "socbench" / "sources" / "sources.yaml").is_file():
            return candidate / "benchmark" / "socbench"
    return None


class SocbenchRepo:
    """The ``benchmark/socbench/`` directory: where the manifests and schemas live.

    Located from ``JEF_SOCBENCH_ROOT`` if set, else by walking up from the
    working directory, else relative to this package (a source checkout). An
    installed wheel with no checkout and no env var has no manifests to read,
    and says so.
    """

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            env = os.environ.get("JEF_SOCBENCH_ROOT")
            if env:
                root = Path(env)
            else:
                root = _find_root(Path.cwd()) or _find_root(Path(__file__).resolve())
        if root is None or not (root / "sources" / "sources.yaml").is_file():
            raise FileNotFoundError(
                "benchmark/socbench/sources/sources.yaml not found; run from a checkout or set "
                "JEF_SOCBENCH_ROOT"
            )
        self.root = Path(root)
        self._validators: dict[str, Draft202012Validator] = {}

    # -- paths --------------------------------------------------------------

    @property
    def sources_path(self) -> Path:
        return self.root / "sources" / "sources.yaml"

    @property
    def schema_dir(self) -> Path:
        return self.root / "schema"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    def pins_path(self, source_id: str) -> Path:
        return self.manifests_dir / f"{source_id}.json"

    # -- schemas ------------------------------------------------------------

    def validator(self, name: str) -> Draft202012Validator:
        if name not in self._validators:
            schema = json.loads((self.schema_dir / f"{name}.schema.json").read_text("utf-8"))
            Draft202012Validator.check_schema(schema)
            self._validators[name] = Draft202012Validator(schema)
        return self._validators[name]

    def validate(self, name: str, instance: Any, *, where: str) -> None:
        errors = sorted(self.validator(name).iter_errors(instance), key=lambda e: list(e.path))
        if errors:
            lines = [f"{where} does not match {name}.schema.json:"]
            lines += [
                f"  {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
            ]
            raise ValueError("\n".join(lines))

    # -- sources ------------------------------------------------------------

    def load_sources(self, *, catalogue: bool = False) -> list[Source]:
        """The wired sources; with ``catalogue=True`` also the planning-only rows."""
        raw = yaml.safe_load(self.sources_path.read_text("utf-8"))
        # YAML reads an unquoted 2026-10-16 as a date object; the schema wants
        # the string, and so does anything that round-trips the manifest.
        if isinstance(raw, dict) and isinstance(raw.get("checked"), date):
            raw["checked"] = raw["checked"].isoformat()
        for row in raw.get("sources", []) if isinstance(raw, dict) else []:
            if isinstance(row, dict) and isinstance(row.get("access_verified"), date):
                row["access_verified"] = row["access_verified"].isoformat()
        self.validate("sources", raw, where=str(self.sources_path))
        sources: list[Source] = []
        seen: set[str] = set()
        for row in raw["sources"]:
            if row["id"] in seen:
                raise ValueError(f"duplicate source id {row['id']!r} in {self.sources_path}")
            seen.add(row["id"])
            links = {
                k: row[k] for k in ("portal", "reference", "full_dataset", "open_data") if k in row
            }
            sources.append(
                Source(
                    id=row["id"],
                    name=row["name"],
                    url=row["url"],
                    language=row["language"],
                    ground_truth_level=row["ground_truth_level"],
                    realism=row["realism"],
                    hard_negatives=row["hard_negatives"],
                    jef_use=tuple(row["jef_use"]),
                    access=row["access"],
                    license=str(row["license"]),
                    priority=row["priority"],
                    caveat=row["caveat"],
                    kind=row["kind"],
                    label_granularity=row["label_granularity"],
                    trust_tier=row["trust_tier"],
                    adapter=row["adapter"],
                    access_verified=row["access_verified"] or None,
                    links=links,
                )
            )
        return sources if catalogue else [s for s in sources if s.wired]

    def source(self, source_id: str) -> Source:
        for s in self.load_sources():
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    # -- pins ---------------------------------------------------------------

    def load_pins(self, source_id: str) -> Pins:
        path = self.pins_path(source_id)
        raw = json.loads(path.read_text("utf-8"))
        self.validate("artifacts", raw, where=str(path))
        if raw["source"] != source_id:
            raise ValueError(f"{path} says source={raw['source']!r}, expected {source_id!r}")
        return Pins(
            source=raw["source"],
            note=raw.get("note"),
            artifacts=[Artifact(**a) for a in raw["artifacts"]],
        )

    def save_pins(self, pins: Pins) -> Path:
        data = pins.to_json()
        self.validate("artifacts", data, where=f"pins for {pins.source}")
        path = self.pins_path(pins.source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
        return path
