"""Baselines for the five incident-triage tasks (plan 2026-10-20, PR-05).

A JEF number means something only next to transparent alternatives on the
same cases. Each baseline is an :class:`~jef_core.Engine` subclass, so the
harness in ``triage_eval`` asks it the scene's questions exactly as it asks
JEF, and scene mode runs it through the same gates:

B0 ``fail_closed``  every case to a person; measures workload, never wrong
B1 ``rules``        keyword rules per task, zh-TW and en, approval rule built in
B2 ``logreg``       TF-IDF character n-grams + logistic regression per task,
                    fitted on the lock's train split only
B3 ``jef``          the deployed engine (``--backbone/--head/--calibration``)

B1 was written by someone who had read the synthetic families, so its numbers
on them are optimistic; that is disclosed in every report, and it biases the
routing decision rule towards keeping a person in the loop, which is the safe
direction.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from jef_core import Engine
from jef_core.calibration import Calibrator
from jef_core.engine import SharedState
from jef_core.types import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    RawQuestion,
    ScoreAnswer,
    normalize,
)

from .triage import TASK_OF_QUESTION
from .triage_eval import EvalError, assert_no_family_overlap

__all__ = [
    "BASELINE_NOTES",
    "LogisticTriageEngine",
    "RuleTriageEngine",
    "answer_from",
    "fail_closed_summary",
    "routing_decision",
]

BASELINE_NOTES = {
    "fail_closed": "every case to a person: workload 100%, no automated outcome",
    "rules": (
        "keyword rules written after reading the synthetic families: optimistic on them, "
        "disclosed as such"
    ),
    "logreg": "TF-IDF char 2-4 grams + logistic regression per task, train split only",
    "jef": "the deployed engine as configured",
}


def answer_from(question: Question | RawQuestion, qid: str, dist: Mapping[str, float]) -> Answer:
    """An answer of the question's kind carrying ``dist`` (normalised)."""
    q = normalize(qid, question)
    keys = list(q.option_keys)
    if q.kind == "noul":
        p = float(dist.get("true", 0.5))
        return NoulAnswer(noul=p, confidence=abs(2 * p - 1))
    raw = np.array([max(0.0, float(dist.get(k, 0.0))) + 1e-6 for k in keys])
    raw /= raw.sum()
    probs = {k: round(float(v), 6) for k, v in zip(keys, raw, strict=True)}
    top = keys[int(raw.argmax())]
    if q.kind == "choice":
        return ChoiceAnswer(choice=top, probabilities=probs, confidence=float(raw.max()))
    return ScoreAnswer(
        score=float(sum(i * v for i, v in enumerate(raw))),
        legend=keys,
        probabilities=probs,
        confidence=float(raw.max()),
    )


# --------------------------------------------------------------------------- #
# B0
# --------------------------------------------------------------------------- #


def fail_closed_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Send everything to a person. The floor every other baseline is judged against."""
    return {
        "cases": len(cases),
        "review_required": len(cases),
        "workload_share": 1.0 if cases else None,
        "auto_allowed": 0,
        "dispatched": 0,
        "note": BASELINE_NOTES["fail_closed"],
    }


# --------------------------------------------------------------------------- #
# B1: rules
# --------------------------------------------------------------------------- #

_APPROVAL = re.compile(
    r"變更單|工單|CHG-\d|ticket|change \S+|核准|approved|允許清單|allowlist|已登記|registered|"
    r"排程|scheduled|服務台|service desk|派送單|request \S+",
    re.IGNORECASE,
)
_MISSING = re.compile(
    r"未說明|未包含|未附|查無|does not say|carries no|no parent|not in the asset inventory",
    re.IGNORECASE,
)
_MALICIOUS = re.compile(
    r"勒索|ransom|\.crypt|\.lockbit|decrypt|vssadmin delete shadows /all|mimikatz|lsass|qakbot|"
    r"cobalt|c2|beacon|DownloadString|authorized_keys|未經|outside the access|forbids|不得|"
    r"base64|DGA|新註冊|newly registered|less than a week|not approved|不在核准|"
    r"' OR 1=1|SLEEP\(|injection|注入|brute|失敗登入|failed .*login|push|推播|"
    r"轉出|move the whole balance|payee|structur|分批|略低於|just under",
    re.IGNORECASE,
)
_CONTAINED = re.compile(
    r"攔截|blocked|quarantin|隔離|還原|restored|after-action|事後|fail2ban|無任何成功|none successful",
    re.IGNORECASE,
)
_RECOVERED = re.compile(r"還原|restored|after-action|事後", re.IGNORECASE)
_CRITICAL = re.compile(
    r"勒索|ransom|\.crypt|mimikatz|lsass|qakbot|全部餘額|whole balance|authorized_keys",
    re.IGNORECASE,
)
_OWNER_WORDS: dict[str, re.Pattern[str]] = {
    "fraud": re.compile(
        r"客戶|customer|AML|風控|risk control|轉帳|轉出|withdraw|payee|存入|deposit|洗錢|約定帳戶",
        re.IGNORECASE,
    ),
    "appsec": re.compile(
        r"WAF|SQL|注入|injection|/api/|DAST|SCA|jackson|漏洞|deserializ|CVSS|staging",
        re.IGNORECASE,
    ),
    "identity": re.compile(
        r"帳號|account|MFA|sshd|登入|login|Administrators|特權|privileged|svc-|服務帳號|身分",
        re.IGNORECASE,
    ),
    "infra": re.compile(
        r"憑證|certificate|磁碟|disk|備份|backup|snapshot|快照|CDN|SCCM|msiexec|派送|修補|patch",
        re.IGNORECASE,
    ),
    "soc": re.compile(
        r"EDR|NDR|IDS|DNS|惡意|malicious|木馬|掃描|scan|beacon|勒索|ransom|proxy|郵件|mail|"
        r"powershell|LSASS",
        re.IGNORECASE,
    ),
}


class RuleTriageEngine(Engine):
    """B1. Keyword rules for the five tasks; the approval rule is built in."""

    def __init__(self, calibrator: Calibrator | None = None) -> None:
        super().__init__(calibrator=calibrator, model_name="baseline/rules")

    def answer(
        self, shared: SharedState, questions: Mapping[str, Question | RawQuestion]
    ) -> dict[str, Answer]:
        text = shared.text
        approval = bool(_APPROVAL.search(text))
        missing = bool(_MISSING.search(text))
        malicious = bool(_MALICIOUS.search(text))
        contained = bool(_CONTAINED.search(text))
        out: dict[str, Answer] = {}
        for qid, q in questions.items():
            task = TASK_OF_QUESTION.get(qid)
            if task == "false_positive":
                p = 0.9 if approval and not malicious and not missing else 0.1 if malicious else 0.3
                out[qid] = answer_from(q, qid, {"true": p})
            elif task == "evidence_sufficiency":
                out[qid] = answer_from(q, qid, {"true": 0.1 if missing else 0.85})
            elif task == "urgency":
                p = 0.85 if malicious and not contained else 0.1 if approval or contained else 0.4
                out[qid] = answer_from(q, qid, {"true": p})
            elif task == "owner":
                hits = {k: len(rx.findall(text)) for k, rx in _OWNER_WORDS.items()}
                # Fraud needs a customer-money word; host names alone never route there.
                out[qid] = answer_from(
                    q, qid, {k: v + (0.5 if k == "soc" else 0) for k, v in hits.items()}
                )
            elif task == "severity":
                keys = list(normalize(qid, q).option_keys)
                if approval and not malicious:
                    level = 0
                elif _CRITICAL.search(text) and not contained:
                    level = 4
                elif contained:
                    # Labelbook §5: each modifier takes one level off. Blocked
                    # before effect lands on 低; contained-and-restored on 中.
                    level = 2 if _RECOVERED.search(text) else 1
                elif malicious:
                    level = 3
                else:
                    level = 2
                out[qid] = answer_from(q, qid, {keys[level]: 1.0})
        return out


# --------------------------------------------------------------------------- #
# B2: TF-IDF + logistic regression per task
# --------------------------------------------------------------------------- #


def _train_target(task: str, gold: Any) -> str | None:
    """One training label per task; unknown gold is not a label."""
    if gold == "unknown":
        return None
    if task in ("owner", "severity"):
        return str(gold[0])
    return "true" if gold else "false"


class LogisticTriageEngine(Engine):
    """B2. One classifier per task, fitted on the train split's labels only."""

    def __init__(
        self,
        train_cases: Sequence[Mapping[str, Any]],
        expectations: Mapping[str, Mapping[str, Any]],
        *,
        calibrator: Calibrator | None = None,
    ) -> None:
        super().__init__(calibrator=calibrator, model_name="baseline/logreg")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.train_cases = list(train_cases)
        self.models: dict[str, tuple[Any, Any, list[str]]] = {}
        self.constant: dict[str, str] = {}
        for task in TASK_OF_QUESTION.values():
            xs, ys = [], []
            for c in self.train_cases:
                e = expectations[c["case_id"]]
                # Train only on labels the harness would score: pending and
                # disputed labels are excluded here as they are there.
                if e["label_status"] not in ("confirmed", "by_construction"):
                    continue
                y = _train_target(task, e["expected"][task])
                if y is not None:
                    xs.append(c["state"])
                    ys.append(y)
            if not ys:
                raise EvalError(f"no training labels for {task}")
            if len(set(ys)) == 1:
                self.constant[task] = ys[0]
                continue
            vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True
            )
            model = LogisticRegression(max_iter=2000, class_weight="balanced")
            model.fit(vec.fit_transform(xs), ys)
            self.models[task] = (vec, model, list(model.classes_))

    def check_no_leak(self, evaluated: Sequence[Mapping[str, Any]]) -> None:
        assert_no_family_overlap(self.train_cases, evaluated)

    def answer(
        self, shared: SharedState, questions: Mapping[str, Question | RawQuestion]
    ) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for qid, q in questions.items():
            task = TASK_OF_QUESTION.get(qid)
            if task is None:
                continue
            if task in self.constant:
                const = self.constant[task]
                if normalize(qid, q).kind == "noul":
                    # answer_from reads P(true); a constant "false" must be 0.
                    dist = {"true": 1.0 if const == "true" else 0.0}
                else:
                    dist = {const: 1.0}
                out[qid] = answer_from(q, qid, dist)
                continue
            vec, model, classes = self.models[task]
            probs = model.predict_proba(vec.transform([shared.text]))[0]
            out[qid] = answer_from(q, qid, dict(zip(classes, probs, strict=True)))
        return out


# --------------------------------------------------------------------------- #
# the pre-registered routing decision
# --------------------------------------------------------------------------- #


def routing_decision(by_baseline: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-registered: if B3 routing macro-F1 < B1's on the same split, routing stays
    human or rule-assisted, and no routing gate may be calibrated or enabled until a
    candidate beats B1 on independent human-labelled data."""
    rules = by_baseline.get("rules", {}).get("tasks", {}).get("owner", {}).get("macro_f1")
    jef = by_baseline.get("jef", {}).get("tasks", {}).get("owner", {}).get("macro_f1")
    if rules is None or jef is None:
        return {"decided": False, "why": "need both B1 (rules) and B3 (jef) routing macro-F1"}
    keep_human = jef < rules
    return {
        "decided": True,
        "b1_rules_macro_f1": rules,
        "b3_jef_macro_f1": jef,
        "routing": "human_or_rule_assisted" if keep_human else "may_proceed_to_human_labelled_test",
        "why": (
            "B3 below B1 on the same split: keep a person (or the rules) on routing"
            if keep_human
            else "B3 at or above B1 on synthetic labels: this permits a human-labelled test, "
            "nothing more"
        ),
    }


# --------------------------------------------------------------------------- #
# run every baseline on one locked split
# --------------------------------------------------------------------------- #


def _same(engine: Engine, _case: Mapping[str, Any]) -> Engine:
    return engine


def run_baselines(
    pack: Any,
    lock: Mapping[str, Any],
    scene: Any,
    *,
    split: str = "test",
    jef: Engine | None = None,
    include_pending: bool = False,
) -> dict[str, Any]:
    """B0-B3 on the same cases, questions mode, plus the routing decision."""
    from .triage_eval import run_questions, score_questions

    ids = set(lock["splits"].get(split, []))
    if not ids:
        raise EvalError(f"split {split!r} is empty or missing in the lock")
    evaluated = [c for c in pack.cases if c["case_id"] in ids]
    train = [c for c in pack.cases if c["case_id"] in set(lock["splits"].get("train", []))]

    engines: dict[str, Engine] = {"rules": RuleTriageEngine()}
    logreg = LogisticTriageEngine(train, pack.expectations)
    logreg.check_no_leak(evaluated)
    engines["logreg"] = logreg
    if jef is not None:
        engines["jef"] = jef

    out: dict[str, Any] = {
        "split": split,
        "cases": len(evaluated),
        "train_cases": len(train),
        "fail_closed": fail_closed_summary(evaluated),
        "baselines": {},
        "notes": dict(BASELINE_NOTES),
    }
    for name, engine in engines.items():
        rows = run_questions(functools.partial(_same, engine), evaluated, scene)
        out["baselines"][name] = score_questions(
            rows, pack.expectations, scene=scene, include_pending=include_pending
        )
    out["routing_decision"] = routing_decision(out["baselines"])
    return out


def render_baselines(report: Mapping[str, Any]) -> str:
    tasks = ["false_positive", "evidence_sufficiency", "owner", "severity", "urgency"]
    head = {
        "false_positive": ("attack→FP", lambda m: m.get("attack_called_false_positive")),
        "evidence_sufficiency": ("recall(false)", lambda m: m.get("false", {}).get("recall")),
        "owner": ("macro-F1", lambda m: m.get("macro_f1")),
        "severity": (
            "MAE / crit→low",
            lambda m: f"{m.get('mae')} / {m.get('critical_to_low_or_info')}",
        ),
        "urgency": ("missed urgent", lambda m: m.get("urgent_missed")),
    }
    lines = [
        f"# incident-triage baselines: split {report['split']} ({report['cases']} cases)",
        "",
        f"**Evidence class: {report['evidence_class']}** — {report['evidence_note']}",
        "",
        "| baseline | " + " | ".join(f"{t}: {head[t][0]}" for t in tasks) + " |",
        "|---|" + "---|" * len(tasks),
    ]
    for name, res in report["baselines"].items():
        cells = [
            str(head[t][1](res["tasks"][t])) + f" (n={res['tasks'][t]['scored']})" for t in tasks
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    fc = report["fail_closed"]
    lines += [
        "",
        f"B0 fail-closed: {fc['review_required']}/{fc['cases']} to a person, 0 automated.",
        "",
        f"**Routing decision (pre-registered):** {report['routing_decision'].get('routing', 'undecided')}"
        f" — {report['routing_decision']['why']}",
        "",
        *[f"- {k}: {v}" for k, v in report["notes"].items()],
    ]
    return "\n".join(lines) + "\n"
