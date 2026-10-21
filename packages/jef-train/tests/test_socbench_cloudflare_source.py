"""The Cloudflare reader against a recorded fake: errors, paging, resume, secrecy.

Nothing here reaches the network. The token and zone id are made-up strings
the tests then hunt for in every byte the reader wrote or raised.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pytest
from jef_train.socbench import cli
from jef_train.socbench.cloudflare_mock import FakeZone, load_fixture_events
from jef_train.socbench.cloudflare_source import (
    REQUIRED_FIELDS,
    CloudflareAuthError,
    CloudflareError,
    CloudflareQueryError,
    GraphQLClient,
    PrivatePathError,
    check_private_dir,
    fetch_window,
    ingest,
    resolve_fields,
    windows,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "benchmark" / "incident-triage" / "fixtures" / "cloudflare-synthetic"
TOKEN = "cf-test-token-0123456789abcdef"
ZONE = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
T0 = datetime(2026, 10, 19, tzinfo=UTC)


def events() -> list[dict[str, object]]:
    return load_fixture_events(FIXTURE / "events.jsonl")


def client(zone: FakeZone, **kw: object) -> GraphQLClient:
    return GraphQLClient(TOKEN, ZONE, transport=zone, sleep=lambda _s: None, **kw)  # type: ignore[arg-type]


def every_byte(path: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file())


@pytest.fixture
def private(tmp_path: Path) -> Path:
    return tmp_path / "private"


def test_windows_cover_the_span_half_open() -> None:
    spans = list(windows(T0, T0 + timedelta(minutes=150), timedelta(hours=1)))
    assert spans[0] == (T0, T0 + timedelta(hours=1))
    assert spans[-1] == (T0 + timedelta(hours=2), T0 + timedelta(minutes=150))
    assert all(a[1] == b[0] for a, b in pairwise(spans))


def test_ingest_writes_every_event_once(private: Path) -> None:
    zone = FakeZone(TOKEN, ZONE, events())
    result = ingest(
        client(zone), zone_alias="site-a", start=T0, end=T0 + timedelta(hours=1), out_dir=private
    )
    rows = [
        json.loads(line)
        for p in (private / "events").glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert result.events == len(rows) == len(events())
    assert not result.incomplete and result.dropped_fields == []
    assert result.sampled is True  # the log-only rows carry sampleInterval 10
    assert zone.requests[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_small_pages_neither_duplicate_nor_skip(private: Path) -> None:
    # limit 4 forces many pages, several of which end inside one second (the
    # largest second holds three events; a page of exactly three there is
    # ambiguous and reported incomplete).
    zone = FakeZone(TOKEN, ZONE, events())
    c = client(zone)
    fields, _ = resolve_fields(c, (T0, T0 + timedelta(hours=1)))
    rows, complete = fetch_window(c, fields, T0, T0 + timedelta(hours=1), limit=4)
    key = lambda r: (r["datetime"], r["rayName"], r["ruleId"])  # noqa: E731
    assert complete
    assert sorted(map(key, rows)) == sorted(map(key, events()))


def test_more_events_in_one_second_than_a_page_is_incomplete_not_truncated() -> None:
    # The scanner's first request raised three events in the same second.
    zone = FakeZone(TOKEN, ZONE, events())
    c = client(zone)
    fields, _ = resolve_fields(c, (T0, T0 + timedelta(hours=1)))
    start = T0 + timedelta(minutes=2, seconds=7)
    rows, complete = fetch_window(c, fields, start, start + timedelta(seconds=1), limit=1)
    assert not complete
    assert len(rows) == 1


def test_resume_skips_complete_windows(private: Path) -> None:
    zone = FakeZone(TOKEN, ZONE, events())
    kw = {"zone_alias": "site-a", "start": T0, "end": T0 + timedelta(hours=1), "out_dir": private}
    first = ingest(client(zone), step=timedelta(minutes=30), **kw)  # type: ignore[arg-type]
    second = ingest(client(zone), step=timedelta(minutes=30), **kw)  # type: ignore[arg-type]
    assert first.windows == 2 and second.windows == 0 and second.skipped == 2
    with pytest.raises(CloudflareError, match="belongs to zone alias"):
        ingest(client(zone), **{**kw, "zone_alias": "site-b"})  # type: ignore[arg-type]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_fail_fast(status: int, private: Path) -> None:
    zone = FakeZone(TOKEN, ZONE, events(), script=[(status, {"errors": []})] * 5)
    c = client(zone)
    with pytest.raises(CloudflareAuthError):
        ingest(c, zone_alias="site-a", start=T0, end=T0 + timedelta(hours=1), out_dir=private)
    assert c.attempts == 1


def test_rate_limit_and_server_errors_are_retried(private: Path) -> None:
    script = [(429, {}), (503, {}), (502, {})]
    zone = FakeZone(TOKEN, ZONE, events(), script=list(script))
    result = ingest(
        client(zone), zone_alias="site-a", start=T0, end=T0 + timedelta(hours=1), out_dir=private
    )
    assert result.events == len(events())
    assert result.failures == 3


def test_retry_budget_runs_out() -> None:
    zone = FakeZone(TOKEN, ZONE, events(), script=[(429, {})] * 10)
    with pytest.raises(CloudflareError, match="429 after 3 attempts"):
        client(zone, retries=2).query("query { viewer { zones { x } } }", {})


def test_graphql_errors_under_http_200_are_failures() -> None:
    reply = {"data": None, "errors": [{"message": f"zone {ZONE} is not authorised for {TOKEN}"}]}
    zone = FakeZone(TOKEN, ZONE, events(), script=[(200, reply)])
    with pytest.raises(CloudflareQueryError) as exc:
        client(zone).query("query { x }", {})
    assert TOKEN not in str(exc.value) and ZONE not in str(exc.value)


def test_a_zone_without_an_optional_field_drops_it_and_says_so(private: Path) -> None:
    zone = FakeZone(TOKEN, ZONE, events(), refuse=frozenset({"rayName", "sampleInterval"}))
    result = ingest(
        client(zone), zone_alias="site-a", start=T0, end=T0 + timedelta(hours=1), out_dir=private
    )
    assert sorted(result.dropped_fields) == ["rayName", "sampleInterval"]
    assert result.sampled is None  # unknown, not "not sampled"
    run = json.loads((private / "runs.jsonl").read_text().splitlines()[0])
    assert run["sampled"] == "unknown" and "rayName" not in run["fields"]


def test_a_zone_without_a_required_field_cannot_be_ingested(private: Path) -> None:
    zone = FakeZone(TOKEN, ZONE, events(), refuse=frozenset({"clientRequestPath"}))
    with pytest.raises(CloudflareQueryError, match="required field clientRequestPath"):
        ingest(
            client(zone),
            zone_alias="site-a",
            start=T0,
            end=T0 + timedelta(hours=1),
            out_dir=private,
        )
    assert "clientRequestPath" in REQUIRED_FIELDS


def test_the_token_and_zone_id_are_never_written(private: Path) -> None:
    zone = FakeZone(TOKEN, ZONE, events())
    c = client(zone)
    ingest(c, zone_alias="site-a", start=T0, end=T0 + timedelta(hours=1), out_dir=private)
    blob = every_byte(private)
    assert TOKEN.encode() not in blob and ZONE.encode() not in blob
    assert TOKEN not in repr(c) and ZONE not in repr(c)
    for p in private.rglob("*"):
        if p.is_file():
            assert p.stat().st_mode & 0o077 == 0, p  # owner-only


def test_a_wrong_zone_is_an_error_not_an_empty_result(private: Path) -> None:
    zone = FakeZone(TOKEN, "another-zone", events())
    with pytest.raises(CloudflareQueryError, match="no such zone"):
        ingest(
            client(zone),
            zone_alias="site-a",
            start=T0,
            end=T0 + timedelta(hours=1),
            out_dir=private,
        )


def test_output_inside_a_git_checkout_is_refused() -> None:
    with pytest.raises(PrivatePathError):
        check_private_dir(ROOT / "out" / "cloudflare")
    with pytest.raises(PrivatePathError):
        check_private_dir(ROOT)


def test_alias_must_not_be_the_zone_id(private: Path) -> None:
    zone = FakeZone(TOKEN, ZONE, events())
    with pytest.raises(CloudflareError, match="zone alias"):
        ingest(
            client(zone),
            zone_alias=ZONE.upper(),
            start=T0,
            end=T0 + timedelta(hours=1),
            out_dir=private,
        )


def test_cli_needs_the_token_in_the_environment(
    private: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CF_TEST_TOKEN", raising=False)
    monkeypatch.setenv("CF_TEST_ZONE", ZONE)
    rc = cli.main(
        [
            "triage", "ingest-cloudflare", "--token-env", "CF_TEST_TOKEN", "--zone-id-env",
            "CF_TEST_ZONE", "--zone-alias", "site-a", "--start", "2026-10-19T00:00:00Z",
            "--end", "2026-10-19T01:00:00Z", "--out-private", str(private),
        ]
    )  # fmt: skip
    assert rc == 2
    assert "CF_TEST_TOKEN" in capsys.readouterr().err


def test_cli_mock_run_end_to_end(
    private: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CF_TEST_TOKEN", TOKEN)
    monkeypatch.setenv("CF_TEST_ZONE", ZONE)
    rc = cli.main(
        [
            "triage", "ingest-cloudflare", "--token-env", "CF_TEST_TOKEN", "--zone-id-env",
            "CF_TEST_ZONE", "--zone-alias", "site-a", "--start", "2026-10-19T00:00:00Z",
            "--end", "2026-10-19T01:00:00Z", "--out-private", str(private),
            "--mock-events", str(FIXTURE / "events.jsonl"),
        ]
    )  # fmt: skip
    out = capsys.readouterr()
    assert rc == 0, out.err
    summary = json.loads(out.out)
    assert summary["events"] == len(events()) and summary["mock"] is True
    assert TOKEN not in out.out + out.err and ZONE not in out.out + out.err
