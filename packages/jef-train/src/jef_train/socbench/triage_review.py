"""The reviewer queue for the five incident-triage tasks: sample, export blind, import.

This is how a pack gets labels a candidate may train on. Nothing here labels
anything: it decides *which* cases a person sees, shows them only what the
alert showed, checks what comes back, and records who said what.

``sample``
    A stratified batch over the case ``domain`` (web, edr, identity, ndr,
    routing, ...). A stratum with too few eligible cases is reported as a
    shortfall and left short -- never topped up with synthetic cases.
    Synthetic cases are not eligible unless asked for, and such a batch is
    marked a dry run. At most ``max_per_family`` cases per family, so one
    campaign does not fill a stratum; approval twins of a sampled case come
    along, vendor-signal variants do not (they are the same incident).
``export``
    One worksheet per reviewer: state and evidence only. No model output,
    no detector output (``detector_output`` evidence, vendor-signal
    variants, ``weak.jsonl``), no other reviewer's answers, no source label.
    The five questions, their options and their order come from the scene,
    so the worksheet cannot drift from what the model is asked. ``unknown``
    is always an option, and ``required_context`` is always asked.
``import``
    One answer file per reviewer, each a distinct handle. One reviewer gives
    ``single-human-reviewed`` / ``confirmed`` (Tier B). Two give
    ``double-human-reviewed`` only where they agree on every task, or where a
    third person -- a handle different from both -- adjudicated; any other
    case stays ``disputed`` and is counted, not scored. Answers are checked
    against the scene's options; a missing task is an error.
``check``
    Tier counts, per-task unknown rates, Cohen's kappa per task when two
    reviewers answered, minutes spent, and a cross-check sample: a 20 %
    second look at a single-reviewer batch is a drift check, not Tier A.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jef_core.types import normalize
from jef_scene import Scene

from .annotate import cohen_kappa
from .cloudflare_source import _private_dirs
from .triage import TASK_OF_QUESTION, Pack, pack_manifest
from .triage_eval import scene_questions

__all__ = [
    "DEFAULT_STRATA",
    "ReviewError",
    "check_review",
    "cross_check_sample",
    "export_worksheet",
    "import_answers",
    "sample_batch",
    "worksheet_questions",
]

#: The v5 plan's first pilot (§5.1), mapped onto the case ``domain`` field.
#: Normal-operations hard negatives have no domain of their own; they are
#: drawn from whichever domain they belong to.
DEFAULT_STRATA: dict[str, int] = {
    "web": 25,
    "edr": 20,
    "identity": 20,
    "ndr": 15,
    "routing": 10,
    "severity": 10,
}
TRUTH_OF_EVENT = ("malicious", "benign", "unknown")
TRUTH_OF_ALERT = ("true_positive", "false_positive", "unknown")
#: Evidence fields a reviewer must not see: the detector's own verdict.
HIDDEN_EVIDENCE = frozenset({"detector_output"})
#: Origins whose cases are private: the reviewed pack is refused inside a checkout.
PRIVATE_ORIGINS = frozenset({"own-telemetry", "reviewer-sourced"})
BATCH_VERSION = 1
#: Marks an expectation copied from its base case onto a vendor-signal variant.
VARIANT_NOTE = "labels of {base}: the same incident with the detector's output shown"


class ReviewError(ValueError):
    """A batch, worksheet or answer file that cannot be used as it is."""


def _rank(seed: str, case_id: str) -> str:
    return hashlib.sha256(f"{seed}\x00{case_id}".encode()).hexdigest()


def _refuse_inside_checkout(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    for p in [resolved, *resolved.parents]:
        if (p / ".git").exists():
            raise ReviewError(
                f"refusing to write reviewed private cases inside the git checkout at {p}"
            )
    return resolved


# --------------------------------------------------------------------------- #
# questions
# --------------------------------------------------------------------------- #


def worksheet_questions(scene: Scene) -> dict[str, dict[str, Any]]:
    """The five questions as a reviewer sees them, from the scene, in scene order."""
    out: dict[str, dict[str, Any]] = {}
    for qid, q in scene_questions(scene).items():
        if qid not in TASK_OF_QUESTION:
            continue
        nq = normalize(qid, q)
        labels = [str(x) if x is not None else "" for x in nq.option_labels]
        meanings = dict(zip(nq.option_keys, labels, strict=True))
        if nq.kind == "noul":
            options: list[Any] = [True, False, "unknown"]
            multi = False
        else:
            options = [*nq.option_keys, "unknown"]
            multi = True  # every acceptable owner / severity; the first is primary
        out[qid] = {
            "task": TASK_OF_QUESTION[qid],
            "task_id": nq.task_id,
            "instructions": nq.instructions,
            "options": options,
            "meanings": meanings,
            "multiple": multi,
        }
    missing = set(TASK_OF_QUESTION) - set(out)
    if missing:
        raise ReviewError(f"the scene does not ask {sorted(missing)}")
    return out


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #


def _all_cases(packs: Sequence[Pack]) -> dict[str, tuple[Pack, dict[str, Any]]]:
    out: dict[str, tuple[Pack, dict[str, Any]]] = {}
    for pack in packs:
        for c in pack.cases:
            if c["case_id"] in out:
                raise ReviewError(f"case {c['case_id']} is in two packs")
            out[c["case_id"]] = (pack, c)
    return out


def sample_batch(
    packs: Sequence[Pack],
    *,
    batch_id: str,
    strata: Mapping[str, int] = DEFAULT_STRATA,
    seed: str = "triage-review",
    max_per_family: int = 2,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Case ids for one batch (no text), with every shortfall stated."""
    every = _all_cases(packs)
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    for _, c in every.values():
        if c.get("twin_of") or c.get("variant_of"):
            continue  # follow their base case
        if c["origin"] == "synthetic" and not allow_synthetic:
            skipped["synthetic"] += 1
            continue
        eligible[str(c.get("domain"))].append(c)
    selected: list[str] = []
    shortfall: dict[str, int] = {}
    per_stratum: dict[str, int] = {}
    for stratum, want in strata.items():
        per_family: Counter[str] = Counter()
        got = 0
        for c in sorted(eligible.get(stratum, []), key=lambda c: _rank(seed, c["case_id"])):
            if got == want:
                break
            if per_family[c["case_family"]] >= max_per_family:
                continue
            per_family[c["case_family"]] += 1
            selected.append(c["case_id"])
            got += 1
        per_stratum[stratum] = got
        if got < want:
            shortfall[stratum] = want - got
    twins = sorted(cid for cid, (_, c) in every.items() if c.get("twin_of") in set(selected))
    origins = Counter(every[cid][1]["origin"] for cid in selected)
    return {
        "batch_version": BATCH_VERSION,
        "batch_id": batch_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "dry_run": allow_synthetic,
        "strata_requested": dict(strata),
        "strata_selected": per_stratum,
        "shortfall": shortfall,
        "shortfall_note": (
            "left short on purpose: a missing stratum is a data gap, never filled with synthetic cases"
            if shortfall
            else None
        ),
        "skipped": dict(skipped),
        "origins": dict(origins),
        "case_ids": sorted(selected) + twins,
        "packs": {p.path.name: p.manifest.get("files", {}) for p in packs},
    }


def cross_check_sample(
    batch: Mapping[str, Any], *, fraction: float = 0.2, seed: str = ""
) -> list[str]:
    """A share of a single-reviewer batch for a second look (a drift check, not Tier A)."""
    ids = sorted(batch["case_ids"], key=lambda cid: _rank(seed or batch["batch_id"], cid))
    return sorted(ids[: max(1, round(len(ids) * fraction))])


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def export_worksheet(
    packs: Sequence[Pack],
    batch: Mapping[str, Any],
    scene: Scene,
    *,
    reviewer: str,
    case_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Blind rows for one reviewer, in an order of their own."""
    if not reviewer or any(ch.isspace() for ch in reviewer):
        raise ReviewError("a reviewer handle is one word with no spaces")
    every = _all_cases(packs)
    questions = worksheet_questions(scene)
    ids = list(case_ids if case_ids is not None else batch["case_ids"])
    unknown = [cid for cid in ids if cid not in every]
    if unknown:
        raise ReviewError(f"cases not in the packs: {unknown}")
    rows = []
    for cid in sorted(ids, key=lambda cid: _rank(f"{batch['batch_id']}\x00{reviewer}", cid)):
        c = every[cid][1]
        if c.get("vendor_signals"):
            raise ReviewError(f"{cid} shows the detector's own output; review its base case")
        rows.append(
            {
                "batch_id": batch["batch_id"],
                "reviewer": reviewer,
                "case_id": cid,
                "state": c["state"],
                "evidence": [e for e in c["evidence"] if e["field"] not in HIDDEN_EVIDENCE],
                "questions": questions,
                "answers": {qid: None for qid in questions},
                "truth_of_event": None,
                "truth_of_alert": None,
                "required_context": [],
                "notes": "",
                "minutes_spent": None,
            }
        )
    return rows


def worksheet_html(rows: Sequence[Mapping[str, Any]]) -> str:
    """A self-contained offline form that saves the reviewer's answers as JSONL."""
    data = json.dumps(list(rows), ensure_ascii=False).replace("</", "<\\/")
    title = (
        html.escape(f"Triage review {rows[0]['batch_id']} — {rows[0]['reviewer']}") if rows else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fff; --fg:#1b1b1b; --muted:#666; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#eee; --muted:#aaa; --line:#333; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; margin:0 auto; max-width:860px; padding:16px; }}
section {{ border-top:1px solid var(--line); padding:12px 0; }}
.state {{ white-space:pre-wrap; }} .muted {{ color:var(--muted); font-size:13px; }}
fieldset {{ border:1px solid var(--line); margin:8px 0; }} label {{ margin-right:12px; display:inline-block; }}
</style></head><body>
<h1>{title}</h1>
<p class="muted">Answer from what the alert shows. Unknown is a valid answer. Nothing here shows a model's or a detector's opinion.</p>
<div id="cases"></div>
<button id="save">Save answers (.jsonl)</button>
<script>
const rows = {data};
const root = document.getElementById("cases");
const truths = {{truth_of_event: {json.dumps(list(TRUTH_OF_EVENT))}, truth_of_alert: {json.dumps(list(TRUTH_OF_ALERT))}}};
rows.forEach((r, i) => {{
  const s = document.createElement("section");
  s.innerHTML = `<h2>${{i + 1}}. <code>${{r.case_id}}</code></h2><p class="state"></p>`;
  s.querySelector(".state").textContent = r.state;
  Object.entries(r.questions).forEach(([qid, q]) => {{
    const f = document.createElement("fieldset");
    f.innerHTML = `<legend></legend>`;
    f.querySelector("legend").textContent = q.instructions;
    q.options.forEach(opt => {{
      const l = document.createElement("label");
      const inp = document.createElement("input");
      inp.type = q.multiple ? "checkbox" : "radio"; inp.name = `${{i}}-${{qid}}`; inp.value = JSON.stringify(opt);
      l.append(inp, " " + String(opt)); f.append(l);
    }});
    s.append(f);
  }});
  Object.entries(truths).forEach(([k, opts]) => {{
    const f = document.createElement("fieldset"); f.innerHTML = `<legend>${{k}}</legend>`;
    opts.forEach(o => {{ const l = document.createElement("label"); const inp = document.createElement("input");
      inp.type = "radio"; inp.name = `${{i}}-${{k}}`; inp.value = o; l.append(inp, " " + o); f.append(l); }});
    s.append(f);
  }});
  s.insertAdjacentHTML("beforeend", `<p>required_context (comma-separated) <input name="${{i}}-ctx" size="50"></p>
    <p>notes <input name="${{i}}-notes" size="60"> minutes <input name="${{i}}-min" size="4"></p>`);
  root.append(s);
}});
document.getElementById("save").onclick = () => {{
  const out = rows.map((r, i) => {{
    const pick = name => [...document.getElementsByName(name)].filter(x => x.checked).map(x => x.value);
    const a = {{}};
    Object.entries(r.questions).forEach(([qid, q]) => {{
      const v = pick(`${{i}}-${{qid}}`).map(x => JSON.parse(x));
      a[qid] = q.multiple ? (v.includes("unknown") || !v.length ? (v.length ? "unknown" : null) : v) : (v.length ? v[0] : null);
    }});
    const one = n => (pick(n)[0] ?? null);
    const val = n => document.getElementsByName(n)[0].value;
    return {{...r, answers: a, truth_of_event: one(`${{i}}-truth_of_event`), truth_of_alert: one(`${{i}}-truth_of_alert`),
      required_context: val(`${{i}}-ctx`).split(",").map(x => x.trim()).filter(Boolean),
      notes: val(`${{i}}-notes`), minutes_spent: Number(val(`${{i}}-min`)) || null}};
  }});
  const blob = new Blob(out.map(o => JSON.stringify(o) + "\\n"), {{type: "application/jsonl"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `${{rows[0].batch_id}}-${{rows[0].reviewer}}.jsonl`; a.click();
}};
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #


def _check_answer(
    row: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    cid = row.get("case_id")
    answers = row.get("answers") or {}
    out: dict[str, Any] = {}
    for qid, q in questions.items():
        value = answers.get(qid)
        if value is None:
            raise ReviewError(f"{row.get('reviewer')} {cid}: {qid} is not answered")
        allowed = q["options"]
        if q["multiple"]:
            if value != "unknown":
                if not isinstance(value, list) or not value or len(set(value)) != len(value):
                    raise ReviewError(f"{cid}: {qid} must be 'unknown' or a list of options")
                bad = [v for v in value if v not in allowed or v == "unknown"]
                if bad:
                    raise ReviewError(f"{cid}: {qid} has options the scene does not offer: {bad}")
        elif value not in allowed:
            raise ReviewError(f"{cid}: {qid}={value!r} is not one of {allowed}")
        out[q["task"]] = value
    for key, allowed_truth in (
        ("truth_of_event", TRUTH_OF_EVENT),
        ("truth_of_alert", TRUTH_OF_ALERT),
    ):
        if row.get(key) not in allowed_truth:
            raise ReviewError(f"{cid}: {key} must be one of {allowed_truth}")
    return out


def _primary(value: Any) -> Any:
    return value[0] if isinstance(value, list) else value


def import_answers(
    packs: Sequence[Pack],
    batch: Mapping[str, Any],
    scene: Scene,
    answer_files: Sequence[Sequence[Mapping[str, Any]]],
    *,
    adjudications: Sequence[Mapping[str, Any]] = (),
    label_version: str = "v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """``(cases, expectations, review)`` for the reviewed batch.

    Each element of ``answer_files`` is one reviewer's rows. Vendor-signal
    variants of a reviewed case get the same labels, because they are the
    same incident rendered differently; approval twins are reviewed on their
    own and never copy a label.
    """
    if not 1 <= len(answer_files) <= 2:
        raise ReviewError("one reviewer (Tier B) or two (Tier A candidates), not more")
    questions = worksheet_questions(scene)
    every = _all_cases(packs)
    batch_ids = set(batch["case_ids"])
    by_reviewer: dict[str, dict[str, dict[str, Any]]] = {}
    meta: dict[str, dict[str, Any]] = defaultdict(dict)
    for rows in answer_files:
        in_file = {r.get("reviewer") for r in rows}
        if len(in_file) != 1 or not next(iter(in_file)):
            raise ReviewError("each answer file holds exactly one reviewer's handle")
        handle = str(next(iter(in_file)))
        if handle in by_reviewer:
            raise ReviewError(f"{handle} answered twice; a second review needs a second person")
        got: dict[str, dict[str, Any]] = {}
        for r in rows:
            cid = str(r.get("case_id"))
            if r.get("batch_id") != batch["batch_id"]:
                raise ReviewError(f"{handle} {cid}: answer belongs to batch {r.get('batch_id')}")
            if cid not in batch_ids:
                raise ReviewError(f"{handle} {cid}: not in batch {batch['batch_id']}")
            got[cid] = {
                "labels": _check_answer(r, questions),
                "truth_of_event": r["truth_of_event"],
                "truth_of_alert": r["truth_of_alert"],
                "required_context": list(r.get("required_context") or []),
            }
            meta[cid][handle] = {
                "minutes_spent": r.get("minutes_spent"),
                "notes": r.get("notes", ""),
            }
        missing = batch_ids - set(got)
        if missing:
            raise ReviewError(f"{handle} did not answer {sorted(missing)}")
        by_reviewer[handle] = got
    handles = sorted(by_reviewer)
    decided = {str(d["case_id"]): d for d in adjudications}
    for d in adjudications:
        if d.get("adjudicator") in handles or not d.get("adjudicator"):
            raise ReviewError("an adjudicator must be a third person, not one of the reviewers")
    if adjudications and len(handles) < 2:
        raise ReviewError("adjudication needs two reviews to adjudicate between")

    expectations: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    for cid in sorted(batch_ids):
        answers = [by_reviewer[h][cid] for h in handles]
        if len(answers) == 1:
            a = answers[0]
            exp = {
                "label_origin": "single-human-reviewed",
                "label_status": "confirmed",
                "reviewers": handles,
                "truth_of_event": a["truth_of_event"],
                "truth_of_alert": a["truth_of_alert"],
                "expected": a["labels"],
                "required_context": a["required_context"],
            }
        else:
            a, b = answers
            agree = (
                a["labels"] == b["labels"]
                and a["truth_of_event"] == b["truth_of_event"]
                and a["truth_of_alert"] == b["truth_of_alert"]
            )
            final = a if agree else None
            reviewers = handles
            if not agree and cid in decided:
                d = decided[cid]
                final = {
                    "labels": _check_answer({**d, "reviewer": d["adjudicator"]}, questions),
                    "truth_of_event": d["truth_of_event"],
                    "truth_of_alert": d["truth_of_alert"],
                    "required_context": list(d.get("required_context") or []),
                }
                reviewers = [*handles, str(d["adjudicator"])]
            # A disputed case keeps no one's answer as "expected": it is
            # counted, not scored, and the answers stay in the answer files.
            exp = {
                "label_origin": "double-human-reviewed",
                "label_status": "confirmed" if final else "disputed",
                "reviewers": reviewers,
                "truth_of_event": final["truth_of_event"] if final else "unknown",
                "truth_of_alert": final["truth_of_alert"] if final else "unknown",
                "expected": final["labels"] if final else dict.fromkeys(a["labels"], "unknown"),
                "required_context": sorted(set(a["required_context"]) | set(b["required_context"])),
            }
            if not final:
                exp["note"] = "reviewers disagree; counted, not scored, until adjudicated"
        exp = {"case_id": cid, "prohibited_actions": [], **exp}
        expectations[cid] = exp
        status_counts[exp["label_status"]] += 1

    # Vendor-signal variants follow their base case (same incident).
    cases: list[dict[str, Any]] = [every[cid][1] for cid in sorted(batch_ids)]
    for cid, (_, c) in sorted(every.items()):
        base = c.get("variant_of")
        if base in expectations:
            cases.append(c)
            expectations[cid] = {
                **expectations[base],
                "case_id": cid,
                "note": VARIANT_NOTE.format(base=base),
            }
    review = {
        "batch_id": batch["batch_id"],
        "label_version": label_version,
        "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reviewers": handles,
        "adjudicators": sorted({str(d["adjudicator"]) for d in adjudications}),
        "label_status": dict(status_counts),
        "source_packs": batch.get("packs", {}),
        "per_case": dict(meta),
    }
    return cases, [expectations[c["case_id"]] for c in cases], review


def write_reviewed_pack(
    out: Path,
    cases: Sequence[Mapping[str, Any]],
    expectations: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
) -> Path:
    """The reviewed pack (private origins: owner-only, outside any checkout)."""
    private = any(c["origin"] in PRIVATE_ORIGINS for c in cases)
    out = _refuse_inside_checkout(out) if private else Path(out)
    if private:
        _private_dirs(out)
    else:
        out.mkdir(parents=True, exist_ok=True)

    def write(name: str, text: str) -> None:
        path = out / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)

    for name, rows in (("cases.jsonl", cases), ("expectations.jsonl", expectations)):
        write(name, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    write("review.json", json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write(
        "manifest.json",
        json.dumps(pack_manifest(out), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return out


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def check_review(
    expectations: Sequence[Mapping[str, Any]],
    answer_files: Sequence[Sequence[Mapping[str, Any]]] = (),
) -> dict[str, Any]:
    """Tier and unknown counts per task; kappa per task when two people answered.

    Vendor-signal variants carry a copy of their base case's labels; they are
    counted apart so a copy is never reported as another review.
    """
    prefix = VARIANT_NOTE.split("{base}")[0]
    copied = [e for e in expectations if str(e.get("note", "")).startswith(prefix)]
    expectations = [e for e in expectations if e not in copied]
    tiers = Counter(f"{e['label_origin']}/{e['label_status']}" for e in expectations)
    unknown: dict[str, int] = {}
    for task in TASK_OF_QUESTION.values():
        unknown[task] = sum(
            1 for e in expectations if e.get("expected") and e["expected"][task] == "unknown"
        )
    out: dict[str, Any] = {
        "evidence_class": (
            "TIER A CANDIDATE — CHECK ADJUDICATION"
            if any(e["label_origin"] == "double-human-reviewed" for e in expectations)
            else "TIER B — SINGLE REVIEWER"
            if any(e["label_origin"] == "single-human-reviewed" for e in expectations)
            else "NO HUMAN LABELS"
        ),
        "cases": len(expectations),
        "copied_to_vendor_variants": len(copied),
        "labels": dict(tiers),
        "unknown_per_task": unknown,
        "kappa_per_task": None,
        "minutes_spent": None,
    }
    if len(answer_files) == 2:
        a_rows = {r["case_id"]: r for r in answer_files[0]}
        b_rows = {r["case_id"]: r for r in answer_files[1]}
        common = sorted(set(a_rows) & set(b_rows))
        kappa: dict[str, float | None] = {}
        for qid in TASK_OF_QUESTION:
            xs = [json.dumps(_primary(a_rows[c]["answers"][qid])) for c in common]
            ys = [json.dumps(_primary(b_rows[c]["answers"][qid])) for c in common]
            k = cohen_kappa(xs, ys)
            kappa[TASK_OF_QUESTION[qid]] = None if k is None else round(k, 3)
        out["kappa_per_task"] = kappa
        out["kappa_note"] = "on the primary answer; owner and severity accept several"
    minutes: list[float] = [
        float(r["minutes_spent"])
        for rows in answer_files
        for r in rows
        if isinstance(r.get("minutes_spent"), (int, float))
    ]
    if minutes:
        out["minutes_spent"] = {
            "total": sum(minutes),
            "per_case_mean": round(sum(minutes) / len(minutes), 2),
        }
    return out
