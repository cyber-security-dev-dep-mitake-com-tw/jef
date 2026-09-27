"""NVD CVE records -> typed decision samples.

A CVSS vector is structured ground truth attached to free-text prose, which is
exactly the shape this project needs: the description is the state, and the
vector supplies labels nobody had to guess.

  - ``score``  severity band from baseScore, using CVSS's own published cut-offs
  - ``choice`` attack vector (network / adjacent / local / physical)
  - ``noul``   privileges required, user interaction -- boolean by construction

Only records carrying a CVSS v3.x metric are used. v2 assigns different
semantics to the same field names, and quietly mixing the two would produce
labels that contradict each other.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from ..schema import Sample
from .sources import NVD_API_URL, fetch_json

log = logging.getLogger("jef.train.corpus.cve")

__all__ = ["ATTACK_VECTOR_ZH", "SEVERITY_ZH", "build_cve_samples", "severity_band"]

#: CVSS v3.1's own published bands, ordered low to high -- which is what `score`
#: requires, since the engine takes the expectation over this ordering.
SEVERITY_ZH: list[tuple[str, str, float]] = [
    ("無", "無影響（CVSS 0.0）", 0.0),
    ("低", "低度風險（CVSS 0.1–3.9）", 0.1),
    ("中", "中度風險（CVSS 4.0–6.9）", 4.0),
    ("高", "高度風險（CVSS 7.0–8.9）", 7.0),
    ("危急", "危急風險（CVSS 9.0–10.0）", 9.0),
]

ATTACK_VECTOR_ZH: dict[str, tuple[str, str]] = {
    "NETWORK": ("network", "可經由網路遠端觸發，攻擊者不需位於同一網段"),
    "ADJACENT_NETWORK": ("adjacent", "需與目標位於同一邏輯或實體鄰接網段"),
    "LOCAL": ("local", "需在本機上以既有存取權執行"),
    "PHYSICAL": ("physical", "需實體接觸目標裝置"),
}

#: NVD caps a publication-date query at 120 days.
_WINDOW_DAYS = 120


def severity_band(base_score: float) -> int:
    """Index into :data:`SEVERITY_ZH` for a CVSS base score."""
    band = 0
    for i, (_, _, lower) in enumerate(SEVERITY_ZH):
        if base_score >= lower:
            band = i
    return band


def _clean(text: str, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _english_description(cve: dict[str, Any]) -> str:
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            return _clean(d.get("value", ""))
    return ""


def _cvss_v3(cve: dict[str, Any]) -> dict[str, Any] | None:
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key) or []
        # Prefer NVD's own assessment over a vendor's self-report.
        primary = next((e for e in entries if e.get("type") == "Primary"), None)
        chosen = primary or (entries[0] if entries else None)
        if chosen:
            return chosen.get("cvssData")
    return None


def _windows(count: int, end: datetime) -> list[tuple[str, str]]:
    """Recent publication-date windows, newest first.

    Paging from ``startIndex=0`` returns the oldest CVEs in the database, which
    predate CVSS v3 entirely -- a 500-record page yielded 14 usable samples.
    Querying by publication date gets modern entries, which carry v3 metrics
    almost without exception.
    """
    fmt = "%Y-%m-%dT%H:%M:%S.000"
    out: list[tuple[str, str]] = []
    for i in range(count):
        stop = end - timedelta(days=_WINDOW_DAYS * i)
        start = stop - timedelta(days=_WINDOW_DAYS)
        out.append((start.strftime(fmt), stop.strftime(fmt)))
    return out


def _samples_for(cve: dict[str, Any], cvss: dict[str, Any], polarity: int) -> list[Sample]:
    """Every sample derivable from one CVE record."""
    severity_keys = [k for k, _, _ in SEVERITY_ZH]
    severity_labels = [f"{k}：{d}" for k, d, _ in SEVERITY_ZH]
    av_keys = [k for k, _ in ATTACK_VECTOR_ZH.values()]
    av_labels = [d for _, d in ATTACK_VECTOR_ZH.values()]

    cve_id = cve.get("id", "")
    state = f"漏洞編號：{cve_id}\n\n漏洞描述：\n{_english_description(cve)}"
    meta = {"cve_id": cve_id, "state_lang": "en", "cvss": cvss.get("vectorString", "")}
    # Severity, attack vector and the boolean facts all read the same
    # description. One CVE is one split unit.
    group = f"cve:{cve_id}"

    out: list[Sample] = [
        Sample(
            state=state,
            kind="score",
            instructions="依據漏洞描述評估其嚴重程度。",
            option_keys=severity_keys,
            option_labels=severity_labels,
            label=severity_band(float(cvss.get("baseScore", 0.0))),
            source="cve.severity",
            group=group,
            meta={**meta, "base_score": float(cvss.get("baseScore", 0.0))},
        )
    ]

    av = cvss.get("attackVector")
    if av in ATTACK_VECTOR_ZH:
        out.append(
            Sample(
                state=state,
                kind="choice",
                instructions="攻擊者要利用此漏洞，需要什麼樣的存取位置？",
                option_keys=av_keys,
                option_labels=av_labels,
                label=av_keys.index(ATTACK_VECTOR_ZH[av][0]),
                source="cve.attack_vector",
                group=group,
                meta=meta,
            )
        )

    # Two boolean facts the vector states outright. Alternating which one is
    # asked keeps both phrasings represented without doubling the sample count
    # on a single question.
    if polarity % 2 == 0:
        pr = cvss.get("privilegesRequired")
        if pr in ("NONE", "LOW", "HIGH"):
            out.append(
                Sample(
                    state=state,
                    kind="noul",
                    instructions="利用此漏洞是否需要事先取得任何權限？",
                    option_keys=["false", "true"],
                    option_labels=["不需任何權限即可利用", "需要既有帳號或權限"],
                    label=0 if pr == "NONE" else 1,
                    source="cve.privileges",
                    group=group,
                    meta=meta,
                )
            )
    else:
        ui = cvss.get("userInteraction")
        if ui in ("NONE", "REQUIRED"):
            out.append(
                Sample(
                    state=state,
                    kind="noul",
                    instructions="利用此漏洞是否需要使用者的操作配合？",
                    option_keys=["false", "true"],
                    option_labels=["無需使用者互動", "需使用者點擊或開啟等操作"],
                    label=1 if ui == "REQUIRED" else 0,
                    source="cve.user_interaction",
                    group=group,
                    meta=meta,
                )
            )
    return out


def build_cve_samples(
    *,
    max_records: int = 2000,
    page_size: int = 500,
    windows: int = 6,
    end_date: datetime | None = None,
    refresh: bool = False,
) -> list[Sample]:
    """Derive samples from recently published CVEs carrying CVSS v3.x."""
    samples: list[Sample] = []
    seen = 0
    usable = 0
    end = end_date or datetime.now(UTC)

    for pub_start, pub_end in _windows(windows, end):
        index = 0
        while seen < max_records:
            url = (
                f"{NVD_API_URL}?resultsPerPage={page_size}&startIndex={index}"
                f"&pubStartDate={pub_start}&pubEndDate={pub_end}"
            )
            page = fetch_json(url, refresh=refresh)
            vulns = page.get("vulnerabilities", [])
            if not vulns:
                break

            for entry in vulns:
                seen += 1
                cve = entry.get("cve", {})
                cvss = _cvss_v3(cve)
                if not cvss or len(_english_description(cve)) < 80:
                    continue
                samples.extend(_samples_for(cve, cvss, usable))
                usable += 1

            index += page_size
            if index >= page.get("totalResults", 0):
                break
        if seen >= max_records:
            break

    log.info(
        "cve: %d samples from %d usable records (%d seen, %.0f%% usable)",
        len(samples),
        usable,
        seen,
        100.0 * usable / max(seen, 1),
    )
    return samples
