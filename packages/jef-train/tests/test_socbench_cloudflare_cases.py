"""Cloudflare events -> pseudonymous requests -> a private, unlabelled pack.

What must hold: no raw identifier or query value survives; one request is one
request however many rules it hit; detector outputs never become labels or
reach a scorer; twins do not inherit labels; one actor never spans splits;
nothing is written inside the checkout; aggregates over sampled data carry no
rate.
"""

from __future__ import annotations

import builtins
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from jef_scene import load_scene
from jef_train.socbench import cli
from jef_train.socbench.cloudflare_cases import (
    build_cases,
    group_incidents,
    public_report,
    write_private_pack,
)
from jef_train.socbench.cloudflare_mock import FakeZone, load_fixture_events
from jef_train.socbench.cloudflare_normalize import (
    PseudonymError,
    Pseudonymizer,
    normalize_events,
    query_keys,
    route_template,
    ua_family,
)
from jef_train.socbench.cloudflare_source import GraphQLClient, PrivatePathError, ingest
from jef_train.socbench.triage import load_pack, privacy_findings
from jef_train.socbench.triage_candidate import CandidateError, export_triage_corpus
from jef_train.socbench.triage_eval import score_questions
from jef_train.socbench.triage_synth import triage_lock, verify_triage_lock

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "benchmark" / "incident-triage" / "fixtures" / "cloudflare-synthetic"
SCENE = ROOT / "scenes" / "incident-triage.zh-tw.yaml"
KEY = b"k" * 40
RAW_SECRETS = [
    "203.0.113.",
    "198.51.100.",
    "192.0.2.",
    "abc123fixtureonly",
    "deadbeefcafe0011",
    "UNION",
    "3f2b9c1e-8d4a",
    "123456",
    "8f00",
    "sqlmap.org",
    "uptimerobot.com",
]
DETECTOR_WORDS = [
    "block",
    "managed_challenge",
    "firewallManaged",
    "ratelimit",
    "100001",
    "rl-login",
]


def events() -> list[dict[str, Any]]:
    return load_fixture_events(FIXTURE / "events.jsonl")


def context() -> list[dict[str, Any]]:
    return [json.loads(x) for x in (FIXTURE / "context.jsonl").read_text().splitlines()]


def built(**kw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pseud = Pseudonymizer(KEY)
    reqs = normalize_events(events(), pseud)
    return build_cases(reqs, zone_alias="site-a", pseud=pseud, context=context(), **kw)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #


def test_three_rules_on_one_request_are_one_request_with_three_detections() -> None:
    reqs = normalize_events(events(), Pseudonymizer(KEY))
    assert len(reqs) == len(events()) - 2  # the scanner's first request raised three events
    env = next(r for r in reqs if r.route == "/.env")
    assert [d["rule"] for d in env.detections] == ["100001", "100002", "100003"]


def test_without_ray_ids_the_same_client_second_and_path_still_fold() -> None:
    rows = [{k: v for k, v in e.items() if k != "rayName"} for e in events()]
    assert len(normalize_events(rows, Pseudonymizer(KEY))) == len(events()) - 2


def test_pseudonyms_are_stable_per_key_and_unrelated_across_keys() -> None:
    a, b = Pseudonymizer(KEY), Pseudonymizer(b"z" * 40)
    assert a("actor", "203.0.113.10") == a("actor", "203.0.113.10")
    assert a("actor", "203.0.113.10") != b("actor", "203.0.113.10")
    assert a("actor", "x") != a("req", "x")  # kinds are separate namespaces
    assert "k" * 10 not in repr(a)


def test_a_short_or_missing_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(PseudonymError, match="at least 32"):
        Pseudonymizer(b"short")
    monkeypatch.delenv("JEF_PSEUDONYM_KEY", raising=False)
    with pytest.raises(PseudonymError, match="not set"):
        Pseudonymizer.from_env()


@pytest.mark.parametrize(
    ("path", "template"),
    [
        ("/api/orders/123456", "/api/orders/{id}"),
        ("/api/users/3f2b9c1e-8d4a-4c1e-9b7a-1a2b3c4d5e6f/profile", "/api/users/{id}/profile"),
        ("/reset/eyJhbGciOiJIUzI1NiJ9abcdef0123", "/reset/{id}"),
        ("/u/someone%40example.com", "/u/{id}"),
        ("/files/deadbeef00/x", "/files/{id}/x"),
        ("/login", "/login"),
        ("", "/"),
    ],
)
def test_route_templates_keep_no_identifier(path: str, template: str) -> None:
    assert route_template(path) == template


def test_query_values_never_survive() -> None:
    keys = query_keys("?id=1%27%20OR%201%3D1&token=abc123fixtureonly&token=x&debug=")
    assert keys == ["debug", "id", "token"]
    assert "abc123" not in json.dumps(keys)


@pytest.mark.parametrize(
    ("ua", "family"),
    [
        ("sqlmap/1.8.4#stable (https://sqlmap.org)", "scanner:sqlmap"),
        (
            "Mozilla/5.0+(compatible; UptimeRobot/2.0; http://www.uptimerobot.com/)",
            "monitor:uptime",
        ),
        (
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/129.0 Safari/537.36",
            "browser:chrome",
        ),
        ("python-requests/2.32.3", "client:python"),
        ("", "none"),
        ("something-odd", "other"),
    ],
)
def test_user_agents_become_coarse_families(ua: str, family: str) -> None:
    assert ua_family(ua) == family


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #


def test_one_actor_session_is_one_incident_whatever_the_routes() -> None:
    reqs = normalize_events(events(), Pseudonymizer(KEY))
    incidents = group_incidents(reqs)
    scanner = [i for i in incidents if any(r.route == "/.env" for r in i.requests)]
    assert len(scanner) == 1 and len(scanner[0].requests) == 4


def test_a_long_gap_splits_an_actor_but_keeps_one_family() -> None:
    rows = events()
    later = [
        {**e, "datetime": "2026-10-19T03:00:00Z", "rayName": "late" + str(n)}
        for n, e in enumerate(rows)
        if e["clientIP"] == "198.51.100.23"
    ][:2]
    pseud = Pseudonymizer(KEY)
    cases, _, _ = build_cases(
        normalize_events(rows + later, pseud), zone_alias="site-a", pseud=pseud
    )
    actor = pseud("actor", "198.51.100.23")
    mine = [c for c in cases if actor in c["state"]]
    assert len(mine) == 2
    assert len({c["case_family"] for c in mine}) == 1


def test_every_expectation_is_unlabelled_and_every_state_hides_the_detector() -> None:
    cases, exps, weak = built()
    assert {e["label_status"] for e in exps} == {"unlabelled"}
    assert all(e["expected"] is None and e["label_origin"] == "none" for e in exps)
    for c in cases:
        assert not c["vendor_signals"]
        for word in DETECTOR_WORDS:
            assert word not in c["state"], (c["case_id"], word)
            assert word not in json.dumps(c["evidence"]), (c["case_id"], word)
    assert {w["trust_tier"] for w in weak} == {"D"}
    assert sum(w["actions"].get("block", 0) for w in weak) > 0


def test_vendor_variants_are_marked_and_share_the_family() -> None:
    cases, _, _ = built(with_vendor_signals=True)
    by_id = {c["case_id"]: c for c in cases}
    variants = [c for c in cases if c.get("variant_of")]
    assert variants
    for v in variants:
        assert v["vendor_signals"] is True and "Detector output (Cloudflare)" in v["state"]
        assert v["case_family"] == by_id[v["variant_of"]]["case_family"]


def test_approval_twins_drop_the_evidence_and_inherit_nothing() -> None:
    cases, exps, _ = built()
    twins = [c for c in cases if c.get("twin_of")]
    assert len(twins) == 2  # the DAST window and the uptime monitor
    by_id = {c["case_id"]: c for c in cases}
    exp_by = {e["case_id"]: e for e in exps}
    for t in twins:
        base = by_id[t["twin_of"]]
        assert "Approval or change evidence: none provided." in t["state"]
        assert "CHG-" in base["state"] or "MON-" in base["state"]
        assert t["case_family"] == base["case_family"]
        assert exp_by[t["case_id"]] is not exp_by[base["case_id"]]
        assert exp_by[t["case_id"]]["expected"] is None


def test_no_raw_value_reaches_the_pack(tmp_path: Path) -> None:
    cases, exps, weak = built(with_vendor_signals=True)
    out = write_private_pack(tmp_path / "pack", cases, exps, weak)
    blob = "".join(p.read_text() for p in out.iterdir())
    for secret in RAW_SECRETS:
        assert secret not in blob, secret
    for c in cases:
        assert privacy_findings(c["state"]) == [], c["case_id"]


def test_the_pack_loads_and_its_files_are_owner_only(tmp_path: Path) -> None:
    out = write_private_pack(tmp_path / "pack", *built())
    pack = load_pack(out)
    assert pack.manifest["evidence_class"].startswith("PRIVATE REAL TELEMETRY")
    assert "weak.jsonl" in pack.manifest["files"]
    for p in out.iterdir():
        assert p.stat().st_mode & 0o077 == 0, p


def test_a_pack_inside_the_checkout_is_refused() -> None:
    with pytest.raises(PrivatePathError):
        write_private_pack(ROOT / "out" / "cf-pack", *built())


def test_unlabelled_cases_are_never_scored_or_exported(tmp_path: Path) -> None:
    out = write_private_pack(tmp_path / "pack", *built())
    pack = load_pack(out)
    scene = load_scene(SCENE)
    rows = [{"case_id": c["case_id"], "questions": {}} for c in pack.cases]
    scored = score_questions(rows, pack.expectations, scene=scene)
    for task in scored["tasks"].values():
        assert task["scored"] == 0
        assert task["excluded"] == {"label_unlabelled": len(pack.cases)}
    exps = [pack.expectations[c["case_id"]] for c in pack.cases]
    lock = triage_lock(pack.cases, exps, training=[])
    with pytest.raises(CandidateError, match="no labels"):
        export_triage_corpus(pack, lock, scene, tmp_path / "corpus")


def test_weak_labels_are_never_read_by_load_score_or_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = write_private_pack(tmp_path / "pack", *built())
    real_open, real_read_text = builtins.open, Path.read_text

    def guarded_open(file: Any, *a: Any, **k: Any) -> Any:
        assert not str(file).endswith("weak.jsonl"), "weak.jsonl was opened"
        return real_open(file, *a, **k)

    def guarded_read_text(self: Path, *a: Any, **k: Any) -> str:
        assert self.name != "weak.jsonl", "weak.jsonl was read as text"
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    pack = load_pack(out)  # hashes the bytes for the manifest, never parses them
    scene = load_scene(SCENE)
    rows = [{"case_id": c["case_id"], "questions": {}} for c in pack.cases]
    score_questions(rows, pack.expectations, scene=scene)
    exps = [pack.expectations[c["case_id"]] for c in pack.cases]
    lock = triage_lock(pack.cases, exps, training=[])
    with pytest.raises(CandidateError):
        export_triage_corpus(pack, lock, scene, tmp_path / "corpus")


# --------------------------------------------------------------------------- #
# temporal holdout
# --------------------------------------------------------------------------- #


def test_temporal_holdout_puts_late_families_in_test_and_nothing_else() -> None:
    rows = events()
    for week in (1, 2, 3):
        rows += [
            {
                **e,
                "datetime": (datetime(2026, 10, 19, tzinfo=UTC) + timedelta(weeks=week, minutes=n))
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "clientIP": f"203.0.113.{200 + week * 10 + n}",
                "rayName": f"w{week}-{n}",
            }
            for n, e in enumerate(rows[:8])
        ]  # fmt: skip
    pseud = Pseudonymizer(KEY)
    cases, exps, _ = build_cases(normalize_events(rows, pseud), zone_alias="site-a", pseud=pseud)
    boundary = "2026-11-02"
    lock = triage_lock(cases, exps, training=[], holdout_after=boundary)
    split_of = {cid: s for s, ids in lock["splits"].items() for cid in ids}
    fam_splits: dict[str, set[str]] = {}
    for c in cases:
        fam_splits.setdefault(c["case_family"], set()).add(split_of[c["case_id"]])
        if c["observed_at"] >= "2026-11-02T00:00Z":
            assert split_of[c["case_id"]] == "test", c["case_id"]
    assert all(len(s) == 1 for s in fam_splits.values())
    early_only = {
        c["case_family"]
        for c in cases
        if all(
            o["observed_at"] < "2026-11-02T00:00Z"
            for o in cases
            if o["case_family"] == c["case_family"]
        )
    }
    assert all(fam_splits[f] != {"test"} for f in early_only)
    assert verify_triage_lock(lock, cases, exps) == []
    assert verify_triage_lock({**lock, "holdout_after": "2026-10-01"}, cases, exps)


# --------------------------------------------------------------------------- #
# public aggregates
# --------------------------------------------------------------------------- #


def test_sampled_aggregates_have_no_denominator_and_no_rate() -> None:
    cases, exps, weak = built(with_vendor_signals=True)
    report = public_report(cases, weak, exps)
    assert report["denominator"] == "denominator_unknown" and report["rates"] is None
    assert report["incidents"] == 6  # twins and variants are not counted twice
    assert report["security_events"] == len(events())
    assert privacy_findings(json.dumps(report)) == []


def test_unsampled_aggregates_still_carry_no_rate() -> None:
    rows = [{**e, "sampleInterval": 1} for e in events()]
    pseud = Pseudonymizer(KEY)
    cases, exps, weak = build_cases(normalize_events(rows, pseud), zone_alias="site-a", pseud=pseud)
    report = public_report(cases, weak, exps)
    assert report["denominator"] == "complete" and report["rates"] is None


# --------------------------------------------------------------------------- #
# the command line, end to end on the fixture
# --------------------------------------------------------------------------- #


def test_cli_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, zone = "cf-test-token-0123456789abcdef", "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
    monkeypatch.setenv("CFT", token)
    monkeypatch.setenv("CFZ", zone)
    monkeypatch.setenv("JEF_PSEUDONYM_KEY", KEY.decode())
    raw, norm, pack, report = (tmp_path / n for n in ("raw", "norm", "pack", "report.json"))
    steps = [
        ["ingest-cloudflare", "--token-env", "CFT", "--zone-id-env", "CFZ", "--zone-alias",
         "site-a", "--start", "2026-10-19T00:00:00Z", "--end", "2026-10-19T01:00:00Z",
         "--out-private", str(raw), "--mock-events", str(FIXTURE / "events.jsonl")],
        ["normalize-cloudflare", "--input", str(raw), "--output", str(norm)],
        ["build-cloudflare-cases", "--input", str(norm), "--out", str(pack), "--zone-alias",
         "site-a", "--context", str(FIXTURE / "context.jsonl"), "--with-vendor-signals"],
        ["cloudflare-report", "--pack", str(pack), "--out", str(report)],
    ]  # fmt: skip
    for step in steps:
        assert cli.main(["triage", *step]) == 0, step
    published = "".join(p.read_text() for d in (norm, pack) for p in d.iterdir())
    published += report.read_text()
    for secret in [*RAW_SECRETS, token, zone]:
        assert secret not in published, secret
    assert json.loads(report.read_text())["denominator"] == "denominator_unknown"


def test_fake_zone_round_trip_matches_the_fixture(tmp_path: Path) -> None:
    token, zone = "t" * 24, "z" * 32
    c = GraphQLClient(token, zone, transport=FakeZone(token, zone, events()), sleep=lambda _s: None)
    result = ingest(
        c,
        zone_alias="site-a",
        start=datetime(2026, 10, 19, tzinfo=UTC),
        end=datetime(2026, 10, 19, 1, tzinfo=UTC),
        out_dir=tmp_path / "raw",
    )
    assert result.events == len(events())
