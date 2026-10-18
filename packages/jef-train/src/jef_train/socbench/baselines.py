"""Baselines on the same frozen split (review 11.2).

Every scorer returns a probability distribution over the task's labels
(``annotate.TASK_LABELS``) for one case, so every metric is computed the same
way for each. The point of B0 and B1 is to make a JEF number mean
something: a result that does not beat transparent rules and a bag-of-words
classifier on the same cases is not a result.

B0 ``rules``     transparent per-task rules on the case's own fields
B1 ``logreg``    hashed bag-of-words logistic regression, fitted on the
                 lock's train split only
B2/B3 remote     named, pinned remote systems (see ``remote.py``): any
                 /v1/systemone service (Laya, official TypeSafe Jev, an LLM
                 wrapper, research/jev-local) or the AI Gateway
                 decision-model dialect; opt-in, the case leaves the machine
B4 ``jef``       JEF native: a choice question over the task labels, asked
                 of the case state through jef_core.Engine
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from .annotate import TASK_LABELS

__all__ = [
    "JefScorer",
    "LogisticScorer",
    "RuleScorer",
    "Scorer",
    "case_text",
    "label_text",
    "verdict_question",
]


class Scorer(Protocol):
    name: str

    def predict(self, case: Mapping[str, Any]) -> dict[str, float]: ...


def case_text(case: Mapping[str, Any]) -> str:
    """The case as one string: state then evidence, in order."""
    state = case["state"]
    lines = [f"{k}: {v}" for k, v in state.items()] if isinstance(state, dict) else [str(state)]
    lines += [f"{e['field']}: {e['value']}" for e in case.get("evidence", [])]
    return "\n".join(lines)


def _dist(labels: Sequence[str], weights: Mapping[str, float]) -> dict[str, float]:
    raw = np.array([max(0.0, weights.get(lab, 0.0)) + 1e-3 for lab in labels])
    raw = raw / raw.sum()
    return {lab: float(p) for lab, p in zip(labels, raw, strict=True)}


# --------------------------------------------------------------------------- #
# B0: rules
# --------------------------------------------------------------------------- #

_ATTACK_WORDS = re.compile(
    r"exploit|malware|trojan|backdoor|webshell|reverse shell|privilege|brute|cracking|"
    r"\bscan|nmap|sqlmap|wpscan|dirb|ransom|mimikatz|c2|beacon|exfil|tunnel",
    re.IGNORECASE,
)
_PHISH_WORDS = re.compile(
    r"verify|password|account|suspend|urgent|login|invoice|密碼|帳號|驗證|停用|緊急|點擊|登入",
    re.IGNORECASE,
)
_LEVEL = re.compile(r"\bL(\d{1,2})\b")


class RuleScorer:
    """Transparent per-task rules, the SIEM-style baseline (B0)."""

    name = "rules"

    def describe(self) -> dict[str, Any]:
        return {"type": "rules", "egress": "none"}

    def predict(self, case: Mapping[str, Any]) -> dict[str, float]:
        task = case["task_id"]
        labels = TASK_LABELS[task]
        text = case_text(case)
        state = case["state"] if isinstance(case["state"], dict) else {}
        if task == "soc.alert.verdict.v1":
            levels = [int(x) for x in _LEVEL.findall(text)]
            hits = len(_ATTACK_WORDS.findall(text))
            top = max(levels, default=0)
            if top >= 12 or hits >= 3:
                return _dist(
                    labels, {"confirmed_malicious": 0.5, "suspicious": 0.4, "unknown": 0.1}
                )
            if top >= 7 or hits >= 1 or "suricata" in text:
                return _dist(
                    labels, {"suspicious": 0.6, "unknown": 0.25, "confirmed_malicious": 0.15}
                )
            return _dist(labels, {"confirmed_benign": 0.45, "unknown": 0.35, "suspicious": 0.2})
        if task == "soc.ndr.technique.v1":
            flows = float(state.get("flows", 0) or 0)
            cv = float(state.get("interval_cv", 1) or 1)
            dport = str(state.get("dport", ""))
            if flows >= 5 and cv < 0.2 and dport in {"80", "443", "8080"}:
                return _dist(labels, {"c2": 0.6, "botnet": 0.2, "unknown": 0.2})
            if dport == "25" and flows >= 3:
                return _dist(labels, {"botnet": 0.6, "unknown": 0.3, "c2": 0.1})
            if dport == "53" and flows < 50:
                return _dist(labels, {"benign": 0.4, "unknown": 0.4, "dns_tunnel": 0.2})
            return _dist(labels, {"unknown": 0.6, "benign": 0.3, "botnet": 0.1})
        if task == "soc.phishing.verdict.v1":
            hits = len(_PHISH_WORDS.findall(text))
            link = "http" in text or "hxxp" in text
            if hits >= 3 and link:
                return _dist(labels, {"phishing": 0.6, "suspicious": 0.3, "legitimate": 0.1})
            if hits >= 1:
                return _dist(labels, {"suspicious": 0.5, "phishing": 0.2, "legitimate": 0.3})
            return _dist(labels, {"legitimate": 0.6, "spam_other": 0.2, "unknown": 0.2})
        return _dist(labels, {labels[-1]: 1.0})


# --------------------------------------------------------------------------- #
# B1: bag of words
# --------------------------------------------------------------------------- #


class LogisticScorer:
    """Hashed character n-grams + logistic regression, per task, train split only (B1)."""

    name = "logreg"

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self.trained_on: set[str] = set()

    def fit(
        self, cases: Sequence[Mapping[str, Any]], gold: Mapping[str, Mapping[str, Any]]
    ) -> LogisticScorer:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression

        by_task: dict[str, list[Mapping[str, Any]]] = {}
        for c in cases:
            by_task.setdefault(c["task_id"], []).append(c)
        for task, items in by_task.items():
            y = [gold[c["case_id"]]["gold"]["verdict"] for c in items]
            if len(set(y)) < 2:
                continue  # one class: nothing to learn; predict() falls back to the prior
            vec = HashingVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), n_features=2**16, alternate_sign=False
            )
            model = LogisticRegression(max_iter=2000, class_weight="balanced")
            model.fit(vec.transform([case_text(c) for c in items]), y)
            self._models[task] = (vec, model)
            self.trained_on |= {c["case_id"] for c in items}
        return self

    def describe(self) -> dict[str, Any]:
        return {"type": "logreg", "egress": "none", "trained_on": getattr(self, "n_train", None)}

    def predict(self, case: Mapping[str, Any]) -> dict[str, float]:
        labels = TASK_LABELS[case["task_id"]]
        fitted = self._models.get(case["task_id"])
        if fitted is None:
            return _dist(labels, {})
        vec, model = fitted
        probs = model.predict_proba(vec.transform([case_text(case)]))[0]
        return _dist(labels, dict(zip(model.classes_, probs, strict=True)))


# --------------------------------------------------------------------------- #
# B4: JEF native
# --------------------------------------------------------------------------- #

#: One question, asked identically of every scorer that takes a question:
#: the same instructions, the same options in TASK_LABELS order, the same
#: option text. A comparison where systems were asked different things is not
#: a comparison.
#: The question each task asks, as the scene would ask it. Two languages so a
#: zh-TW corpus is not a translated English corpus by accident.
TASK_QUESTIONS: Mapping[str, Mapping[str, str]] = {
    "soc.alert.verdict.v1": {
        "en": "Given this alert and its context, what is the verdict?",
        "zh-TW": "根據此告警與其脈絡，判定結果為何？",
    },
    "soc.ndr.technique.v1": {
        "en": "Which network behaviour does this evidence show?",
        "zh-TW": "此證據顯示何種網路行為？",
    },
    "soc.edr.execution.v1": {
        "en": "Is this process execution malicious?",
        "zh-TW": "此程序執行是否為惡意？",
    },
    "soc.identity.abuse.v1": {
        "en": "Is this identity activity compromised, suspicious or legitimate?",
        "zh-TW": "此身分活動屬於已遭入侵、可疑或合法？",
    },
    "soc.phishing.verdict.v1": {
        "en": "What is the verdict on this e-mail?",
        "zh-TW": "此郵件的判定結果為何？",
    },
    "soc.response.authority.v1": {
        "en": "May this response action proceed under policy?",
        "zh-TW": "此處置動作依政策是否可以執行？",
    },
}

#: Fallback for a task with no entry above.
VERDICT_INSTRUCTIONS = "Which verdict does this evidence support?"
HTTP_INSTRUCTIONS = VERDICT_INSTRUCTIONS  # kept for callers of the old name


def questions_digest() -> str:
    """Digest of every task's English instructions: what remote systems are asked."""
    return prompt_digest(
        json.dumps({t: task_instructions(t) for t in sorted(TASK_LABELS)}, sort_keys=True)
    )


def prompt_digest(instructions: str) -> str:
    """sha256 over the instructions and every label text: the prompt contract, pinned."""
    import hashlib

    raw = json.dumps({"instructions": instructions, "labels": dict(LABEL_TEXT)}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


LABEL_TEXT: Mapping[str, str] = {
    "confirmed_malicious": "confirmed attacker activity",
    "suspicious": "suspicious; needs investigation",
    "confirmed_benign": "benign: routine, approved or known-good activity",
    "insufficient_evidence": "not enough evidence to judge",
    "unknown": "cannot be determined from this evidence",
    "c2": "command-and-control channel",
    "botnet": "other botnet traffic (spam, scanning, click fraud)",
    "dns_tunnel": "DNS tunnelling or exfiltration",
    "normal_automation": "legitimate automated job",
    "benign": "legitimate traffic",
    "phishing": "phishing",
    "spam_other": "spam or unwanted but not phishing",
    "legitimate": "legitimate message",
    "malicious": "malicious execution",
    "compromised": "account is compromised",
}


def label_text(label: str) -> str:
    return LABEL_TEXT.get(label, label.replace("_", " "))


def task_instructions(task: str, lang: str = "en") -> str:
    """The instructions for ``task``: the exact text a JEF head is trained on."""
    asked = TASK_QUESTIONS.get(task, {})
    return asked.get(lang) or asked.get("en") or VERDICT_INSTRUCTIONS


def verdict_question(task: str) -> dict[str, Any]:
    """The choice question every scorer is asked for a case of ``task``.

    The same instructions ``export_corpus`` writes into training samples, so
    a head is scored on the question it learned, and every other system is
    asked that same question.
    """
    return {
        "type": "choice",
        "instructions": task_instructions(task),
        "criteria": {lab: label_text(lab) for lab in TASK_LABELS[task]},
    }


class JefScorer:
    """JEF native (B4): one choice question over the task labels, per case."""

    name = "jef"

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def describe(self) -> dict[str, Any]:
        eng = self.engine
        cal = getattr(eng, "calibrator", None)
        return {
            "type": "jef",
            "model_name": getattr(eng, "model_name", None),
            "backbone": getattr(getattr(eng, "backbone", None), "name", None),
            "head": getattr(getattr(eng, "head", None), "name", None),
            "calibration_schema": getattr(cal, "schema", None),
            "calibration_buckets": sorted(getattr(cal, "buckets", {}) or {}),
            "model_sha256": getattr(getattr(cal, "artifact", None), "model_sha256", None),
            "egress": "none",
        }

    def predict(self, case: Mapping[str, Any]) -> dict[str, float]:
        labels = TASK_LABELS[case["task_id"]]
        question = {**verdict_question(case["task_id"]), "task_id": case["task_id"]}
        # case_text, not the raw state: the rendering every scorer and every
        # exported training sample sees (state lines, then evidence lines).
        result = self.engine.evaluate(case_text(case), {"verdict": question})
        return {lab: float(result.answers["verdict"].probabilities[lab]) for lab in labels}


def __getattr__(name: str) -> Any:  # pragma: no cover - import shim
    # HttpScorer moved to remote.py with the other remote scorers; importing it
    # from here still works without a circular import at module load.
    if name == "HttpScorer":
        from .remote import HttpScorer

        return HttpScorer
    raise AttributeError(name)
