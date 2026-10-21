"""The SOCBench data layer may fetch, hash and refuse. It may not label.

The manifests are the contract: every source has an adapter and a pin file,
restricted sources never reach the network, and a changed file is caught by
the digest. All tests run offline against temp directories; the one that
touches the repository's own manifests only reads them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from jef_train.socbench import (
    ADAPTERS,
    FETCHABLE_ACCESS,
    RESTRICTED_ACCESS,
    AccessRestricted,
    SocbenchRepo,
    check_source,
    cli,
    fetch_source,
    pipeline,
)
from jef_train.socbench.adapters import inspect_artifact

ROOT = Path(__file__).resolve().parents[3]
SOCBENCH = ROOT / "benchmark" / "socbench"


@pytest.fixture
def repo() -> SocbenchRepo:
    return SocbenchRepo(SOCBENCH)


# --------------------------------------------------------------------------- #
# The repository's own manifests
# --------------------------------------------------------------------------- #


def test_sources_yaml_validates_and_ids_are_unique(repo: SocbenchRepo) -> None:
    sources = repo.load_sources()
    assert len(sources) >= 12
    assert len({s.id for s in sources}) == len(sources)


def test_every_source_has_an_adapter_and_every_adapter_a_source(repo: SocbenchRepo) -> None:
    ids = {s.id for s in repo.load_sources()}
    assert ids == set(ADAPTERS), (
        f"sources without adapter: {ids - set(ADAPTERS)}; adapters without source: {set(ADAPTERS) - ids}"
    )


def test_every_source_has_a_valid_pin_manifest(repo: SocbenchRepo) -> None:
    for s in repo.load_sources():
        pins = repo.load_pins(s.id)  # validates against artifacts.schema.json
        assert pins.source == s.id
        for a in pins.artifacts:
            assert a.url.startswith("https://"), (s.id, a.name)
            assert "/" not in a.name and "\\" not in a.name, (s.id, a.name)


def test_no_stray_manifest(repo: SocbenchRepo) -> None:
    ids = {s.id for s in repo.load_sources()}
    on_disk = {p.stem for p in repo.manifests_dir.glob("*.json")}
    assert on_disk == ids


def test_access_values_are_partitioned() -> None:
    assert not FETCHABLE_ACCESS & RESTRICTED_ACCESS
    schema = json.loads((SOCBENCH / "schema" / "sources.schema.json").read_text("utf-8"))
    allowed = set(schema["$defs"]["source"]["properties"]["access"]["enum"])
    assert allowed == FETCHABLE_ACCESS | RESTRICTED_ACCESS


def test_catalogue_rows_are_never_wired(repo: SocbenchRepo) -> None:
    every = repo.load_sources(catalogue=True)
    wired = {s.id for s in repo.load_sources()}
    catalogue = [s for s in every if not s.wired]
    assert catalogue, "the v2 registry carries catalogue rows"
    for s in catalogue:
        assert s.id not in ADAPTERS, s.id
        assert not repo.pins_path(s.id).exists(), s.id
        assert not s.fetchable, s.id
    assert wired == {s.id for s in every if s.wired}


def test_registry_never_claims_human_tiers_or_five_task_gold(repo: SocbenchRepo) -> None:
    raw = yaml.safe_load(repo.sources_path.read_text("utf-8"))
    for row in raw["sources"]:
        assert row["five_task_gold"] is False, row["id"]
        assert row["trust_tier"] in {"C", "D", "E", "U", "none"}, row["id"]
        if row["kind"] in {"rules", "tool", "schema", "statistics", "reference"}:
            assert row["trust_tier"] == "none", row["id"]
        if row["kind"] == "own-telemetry":
            assert row["access"] == "own-zone" and row["adapter"] == "catalogue", row["id"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("five_task_gold", True),
        ("trust_tier", "A"),
        ("trust_tier", "B"),
        ("label_granularity", "per-alert-gold"),
        ("adapter", "maybe"),
    ],
)
def test_schema_rejects_an_upgraded_label(repo: SocbenchRepo, field: str, value: object) -> None:
    raw = yaml.safe_load(repo.sources_path.read_text("utf-8"))
    raw["checked"] = str(raw["checked"])
    for row in raw["sources"]:
        row["access_verified"] = str(row["access_verified"]) if row["access_verified"] else False
    raw["sources"][0][field] = value
    assert list(repo.validator("sources").iter_errors(raw))


def test_schema_rejects_a_tier_on_a_rule_library(repo: SocbenchRepo) -> None:
    raw = yaml.safe_load(repo.sources_path.read_text("utf-8"))
    raw["checked"] = str(raw["checked"])
    for row in raw["sources"]:
        row["access_verified"] = str(row["access_verified"]) if row["access_verified"] else False
    sigma = next(r for r in raw["sources"] if r["id"] == "sigma-rules")
    sigma["trust_tier"] = "C"
    assert list(repo.validator("sources").iter_errors(raw))


def test_fetching_a_catalogue_row_is_refused(repo: SocbenchRepo) -> None:
    with pytest.raises(SystemExit, match="catalogue-only"):
        cli._select(repo, ["loghub"], all_public=False)


def test_restricted_sources_are_the_ones_we_expect(repo: SocbenchRepo) -> None:
    restricted = {s.id for s in repo.load_sources() if s.restricted}
    assert restricted == {"lanl-cyber1", "cicids2017", "cac-cyber-dataset-1.0"}


def test_schemas_are_valid_and_gold_requires_two_reviewers(repo: SocbenchRepo) -> None:
    gold = repo.validator("gold")
    record = {
        "case_id": "socb-zh-TW-000001",
        "task_id": "soc.phishing.verdict.v1",
        "label_origin": "double-human-reviewed",
        "gold": {"verdict": "confirmed_benign", "confidence_of_gold": "high"},
        "annotators": ["analyst-anon-01"],
        "adjudication_status": "agreed",
    }
    assert list(gold.iter_errors(record)), "one annotator must not pass as double-reviewed"
    record["annotators"].append("analyst-anon-02")
    assert not list(gold.iter_errors(record))
    # A program-derived label never needs annotators.
    derived = {
        **record,
        "label_origin": "attack-phase-window",
        "adjudication_status": "not-applicable",
    }
    derived.pop("annotators")
    assert not list(gold.iter_errors(derived))


def test_case_schema_rejects_a_verdict(repo: SocbenchRepo) -> None:
    case = {
        "case_id": "socb-en-000001",
        "family_id": "f1",
        "source_dataset": "ait-ads",
        "source_hash": "sha256:" + "0" * 64,
        "telemetry_origin": "lab-reproduction",
        "language": "en",
        "task_id": "soc.alert.verdict.v1",
        "state": {"alert": "x"},
        "evidence": [{"id": "e1", "field": "alert", "value": "x"}],
    }
    assert not list(repo.validator("case").iter_errors(case))
    assert list(repo.validator("case").iter_errors({**case, "gold": {"verdict": "malicious"}}))


# --------------------------------------------------------------------------- #
# Fetch / pin / check against a scratch repo, with the network stubbed out
# --------------------------------------------------------------------------- #

MINI_SOURCES = """
version: 2
checked: 2026-10-16
sources:
  - id: ts-bench
    name: T
    url: https://example.invalid/t
    language: zh-TW
    ground_truth_level: prompts
    realism: x
    hard_negatives: x
    jef_use: [a]
    access: public-download
    license: see-repo
    priority: 1
    kind: dataset
    label_granularity: event
    trust_tier: C
    five_task_gold: false
    adapter: implemented
    access_verified: 2026-10-16
    caveat: c
  - id: lanl-cyber1
    name: L
    url: https://example.invalid/l
    language: n/a
    ground_truth_level: partial
    realism: x
    hard_negatives: x
    jef_use: [a]
    access: request-form
    license: CC0
    priority: 2
    kind: dataset
    label_granularity: event
    trust_tier: C
    five_task_gold: false
    adapter: implemented
    access_verified: 2026-10-16
    caveat: c
  - id: community-soc-benchmark
    name: R
    url: https://example.invalid/r
    language: en
    ground_truth_level: ref
    realism: x
    hard_negatives: x
    jef_use: [a]
    access: public
    license: see-repo
    priority: reference
    kind: dataset
    label_granularity: event
    trust_tier: C
    five_task_gold: false
    adapter: implemented
    access_verified: 2026-10-16
    caveat: c
"""

PAYLOAD = b"id,text\n1,hello\n2,world\n"


@pytest.fixture
def scratch(tmp_path: Path) -> tuple[SocbenchRepo, Path]:
    root = tmp_path / "socbench"
    (root / "sources").mkdir(parents=True)
    (root / "manifests").mkdir()
    shutil.copytree(SOCBENCH / "schema", root / "schema")
    (root / "sources" / "sources.yaml").write_text(MINI_SOURCES, "utf-8")
    for sid, arts in {
        "ts-bench": [
            {"name": "TSB400.csv", "url": "https://example.invalid/TSB400.csv", "sha256": None}
        ],
        "lanl-cyber1": [
            {"name": "redteam.txt.gz", "url": "https://example.invalid/", "sha256": None}
        ],
        "community-soc-benchmark": [],
    }.items():
        (root / "manifests" / f"{sid}.json").write_text(
            json.dumps({"source": sid, "artifacts": arts}), "utf-8"
        )
    return SocbenchRepo(root), tmp_path / "data"


@pytest.fixture
def fake_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fetch_to(url: str, dest: Path, *, refresh: bool = False, timeout: int = 120) -> Path:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or refresh:
            dest.write_bytes(PAYLOAD)
        return dest

    monkeypatch.setattr(pipeline, "fetch_to", fetch_to)
    return calls


def test_restricted_source_never_touches_the_network(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str]
) -> None:
    repo, data = scratch
    with pytest.raises(AccessRestricted) as exc:
        fetch_source(repo, repo.source("lanl-cyber1"), base=data)
    assert "request form" in str(exc.value).lower() or "form" in exc.value.instructions
    assert fake_network == []
    assert not (data / "lanl-cyber1").exists()


def test_reference_only_source_fetches_nothing(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str]
) -> None:
    repo, data = scratch
    statuses, pins = fetch_source(repo, repo.source("community-soc-benchmark"), base=data)
    assert [s.state for s in statuses] == ["skipped"]
    assert pins is None
    assert fake_network == []


def test_fetch_without_pin_records_nothing(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str]
) -> None:
    repo, data = scratch
    statuses, pins = fetch_source(repo, repo.source("ts-bench"), base=data)
    assert [s.state for s in statuses] == ["unpinned"]
    assert pins is None
    assert repo.load_pins("ts-bench").artifacts[0].sha256 is None
    assert fake_network == ["https://example.invalid/TSB400.csv"]


def test_fetch_with_pin_records_the_digest_and_check_passes(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str]
) -> None:
    repo, data = scratch
    statuses, pins = fetch_source(repo, repo.source("ts-bench"), base=data, pin=True)
    assert [s.state for s in statuses] == ["pinned"]
    assert pins is not None
    saved = repo.load_pins("ts-bench").artifacts[0]
    assert saved.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert saved.size == len(PAYLOAD)
    assert saved.recorded_at
    assert [s.state for s in check_source(repo, repo.source("ts-bench"), base=data)] == ["ok"]


def test_check_detects_a_changed_file(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str]
) -> None:
    repo, data = scratch
    fetch_source(repo, repo.source("ts-bench"), base=data, pin=True)
    (data / "ts-bench" / "TSB400.csv").write_bytes(PAYLOAD + b"3,tampered\n")
    (status,) = check_source(repo, repo.source("ts-bench"), base=data)
    assert status.state == "MISMATCH"
    assert status.failed
    # Same size, different bytes: the digest, not the length, must catch it.
    (data / "ts-bench" / "TSB400.csv").write_bytes(PAYLOAD[:-1] + b"!")
    (status,) = check_source(repo, repo.source("ts-bench"), base=data)
    assert status.state == "MISMATCH"
    assert "sha256" in status.detail


def test_a_pinned_fetch_that_differs_is_a_mismatch_not_a_repin(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str]
) -> None:
    repo, data = scratch
    fetch_source(repo, repo.source("ts-bench"), base=data, pin=True)
    before = repo.load_pins("ts-bench").artifacts[0].sha256
    (data / "ts-bench" / "TSB400.csv").write_bytes(b"different upstream")
    statuses, pins = fetch_source(repo, repo.source("ts-bench"), base=data, pin=True)
    assert statuses[0].state == "MISMATCH"
    assert pins is None
    assert repo.load_pins("ts-bench").artifacts[0].sha256 == before


def test_check_reports_missing_without_network(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str]
) -> None:
    repo, data = scratch
    (status,) = check_source(repo, repo.source("ts-bench"), base=data)
    assert status.state == "missing"
    assert fake_network == []


def test_pin_local_records_hand_placed_files(scratch: tuple[SocbenchRepo, Path]) -> None:
    repo, data = scratch
    target = data / "lanl-cyber1" / "redteam.txt.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"gz")
    statuses, _ = pipeline.pin_local(repo, repo.source("lanl-cyber1"), base=data)
    assert [s.state for s in statuses] == ["pinned"]
    assert repo.load_pins("lanl-cyber1").artifacts[0].sha256 == hashlib.sha256(b"gz").hexdigest()


def test_manifest_validation_rejects_http_and_paths(scratch: tuple[SocbenchRepo, Path]) -> None:
    repo, _ = scratch
    path = repo.pins_path("ts-bench")
    path.write_text(
        json.dumps(
            {
                "source": "ts-bench",
                "artifacts": [{"name": "../x", "url": "http://e/x", "sha256": None}],
            }
        )
    )
    with pytest.raises(ValueError, match=r"artifacts\.schema\.json"):
        repo.load_pins("ts-bench")


def test_sources_yaml_missing_field_fails_loudly(scratch: tuple[SocbenchRepo, Path]) -> None:
    repo, _ = scratch
    raw = yaml.safe_load(repo.sources_path.read_text("utf-8"))
    del raw["sources"][0]["caveat"]
    repo.sources_path.write_text(yaml.safe_dump(raw), "utf-8")
    with pytest.raises(ValueError, match="caveat"):
        repo.load_sources()


def test_inspect_reports_shape_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "x.csv"
    csv_path.write_bytes(PAYLOAD)
    assert inspect_artifact(csv_path) == {"kind": "csv", "columns": ["id", "text"], "rows": 2}
    import zipfile

    zpath = tmp_path / "x.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.txt", "a")
        zf.writestr("b.txt", "b")
    assert inspect_artifact(zpath)["members"] == 2
    blob = tmp_path / "x.bin"
    blob.write_bytes(b"\x00" * 10)
    assert inspect_artifact(blob) == {"kind": "opaque", "bytes": 10}


# --------------------------------------------------------------------------- #
# CLI exit codes
# --------------------------------------------------------------------------- #


def test_cli_sources_lists_everything(
    scratch: tuple[SocbenchRepo, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    repo, _ = scratch
    assert cli.main(["--root", str(repo.root), "sources"]) == 0
    out = capsys.readouterr().out
    assert "ts-bench" in out and "lanl-cyber1" in out and "HUMAN" in out


def test_cli_fetch_restricted_by_name_exits_2(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    repo, data = scratch
    rc = cli.main(
        ["--root", str(repo.root), "--data-dir", str(data), "fetch", "--source", "lanl-cyber1"]
    )
    assert rc == 2
    assert "HUMAN" in capsys.readouterr().out
    assert fake_network == []


def test_cli_all_public_skips_restricted_and_pins(
    scratch: tuple[SocbenchRepo, Path], fake_network: list[str]
) -> None:
    repo, data = scratch
    rc = cli.main(
        ["--root", str(repo.root), "--data-dir", str(data), "fetch", "--all-public", "--pin"]
    )
    assert rc == 0
    assert fake_network == ["https://example.invalid/TSB400.csv"]
    assert cli.main(["--root", str(repo.root), "--data-dir", str(data), "check"]) == 0
    (data / "ts-bench" / "TSB400.csv").write_bytes(b"x")
    assert cli.main(["--root", str(repo.root), "--data-dir", str(data), "check"]) == 1


def test_cli_unknown_source_is_an_error(scratch: tuple[SocbenchRepo, Path]) -> None:
    repo, _ = scratch
    with pytest.raises(SystemExit, match="unknown source"):
        cli.main(["--root", str(repo.root), "fetch", "--source", "nope"])


def test_cli_without_a_checkout_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JEF_SOCBENCH_ROOT", raising=False)
    assert cli.main(["--root", str(tmp_path / "nowhere"), "sources"]) == 2
