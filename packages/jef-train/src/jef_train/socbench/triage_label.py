"""An interactive terminal labeller for one reviewer's blind worksheet (zh-TW, with English terms).

``jef-bench triage review label --worksheet W`` walks a worksheet written by
``review export`` one case at a time: it shows the evidence, asks the two
truths and the five questions, warns on answers that contradict each other,
and saves after every case. Quit at any time; the next run resumes where it
stopped. The answer file it writes is what ``review import`` reads, and every
row is checked with the importer's own validation before it is saved.

Nothing here suggests an answer. The screen shows only what the worksheet
holds -- no model output, no detector output -- and ``unknown`` is always one
key away.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .triage_review import TRUTH_OF_ALERT, TRUTH_OF_EVENT, ReviewError, _check_answer

LABELBOOK = "benchmark/incident-triage/LABELBOOK.md"
BREAK_EVERY = 25

FIELD_NAMES = {
    "zone": "區域 Zone",
    "actor": "來源 Actor",
    "routes": "路徑 Routes",
    "window_utc": "時間窗 Window (UTC)",
    "requests": "請求數 Requests",
    "methods": "方法 Methods",
    "origin_status": "回應碼 Response status",
    "client_type": "用戶端 Client type",
    "query_parameters": "查詢參數名 Query keys",
    "security_events": "安全事件數 Security events",
    "approval": "核准/變更證據 Approval",
}

OPTION_NAMES: dict[Any, str] = {
    True: "是 yes",
    False: "否 no",
    "unknown": "不確定 unknown",
    "soc": "SOC 監控中心",
    "appsec": "應用安全 AppSec",
    "infra": "基礎架構 Infra",
    "identity": "身分與帳號 Identity",
    "fraud": "客戶詐欺 Fraud",
    "malicious": "惡意 malicious",
    "benign": "良性 benign",
    "true_positive": "告警正確 true positive",
    "false_positive": "誤報 false positive",
}

TRUTHS = {
    "truth_of_event": ("事件本身是攻擊嗎？ Was the activity an attack?", TRUTH_OF_EVENT),
    "truth_of_alert": ("這個偵測該觸發嗎？ Was the detection right to fire?", TRUTH_OF_ALERT),
}

CONTEXT_SUGGESTIONS = [
    ("change_ticket", "變更單 change ticket"),
    ("scanner_approval", "掃描器核准 scanner approval"),
    ("maintenance_window", "維護時段 maintenance window"),
    ("asset_owner", "資產負責人 asset owner"),
    ("origin_logs", "源站日誌 origin / app logs"),
    ("threat_intel", "威脅情資 threat intel on the source"),
]

# Short reminders from the labelbook, shown on "?". The labelbook is the rule.
HELP = {
    "truth_of_event": (
        "惡意 malicious：活動本身是攻擊或偵察（掃描 .env、路徑穿越、注入）。\n"
        "良性 benign：正常使用者、已核准的掃描或監控。\n"
        "不確定 unknown：畫面上的資訊無法判斷。unknown ≠ benign。"
    ),
    "truth_of_alert": (
        "告警正確 true positive：這個偵測該觸發（攻擊，或確實需要處理的狀況）。\n"
        "誤報 false positive：觸發在已知良性/已核准的活動上。\n"
        "is_false_positive 這一題跟著這個答案走。"
    ),
    "false_positive": (
        "是 yes：偵測打在良性或已核准活動上，**且核准證據在畫面中**（變更單、排程、掃描器核准）。\n"
        "否 no：是或可能是攻擊。\n"
        "不確定 unknown：看起來像已核准，但畫面沒有核准證據 → 選 unknown，並在 required_context 填缺什麼。\n"
        "「WAF 擋下了」不是惡意的證據，也不是良性的證據。"
    ),
    "evidence_sufficiency": (
        "是 yes：有具體細節可以判斷（路徑、時間、行為、來源；需要核准時有核准參照）。\n"
        "否 no：缺少關鍵脈絡（例如沒有核准參照、不知道資產重要性而嚴重度取決於此）。"
    ),
    "owner": (
        "可多選，第一個是主要負責 primary，其他是也可接受的團隊。\n"
        "Web/API 攻擊、注入、路徑穿越 → appsec（偵察掃描可加 soc）。\n"
        "可用性、憑證、設定 → infra。登入暴力破解、帳號 → identity（soc 可接受）。\n"
        "fraud 只用在**客戶**的帳戶與金流。"
    ),
    "severity": (
        "可能性 × 資產重要性；資產重要性不明時用「中或不明」那一欄：\n"
        "  確認有害且進行中 → 高；可能有害 → 中；只是可能 → 低；已核准/良性 → 資訊。\n"
        "已被阻擋、沒有效果 → 降一級。落在邊界時可選兩個等級（例如 2,3）。"
    ),
    "urgency": (
        "是 yes：損害正在發生或迫在眉睫（有效的入侵、外洩、帳號接管）。\n"
        "否 no：排程處理即可（未成功的嘗試、一般掃描、良性活動）。\n"
        "不確定 unknown：從畫面看不出是否正在發生。"
    ),
}

KEYS_LINE = "數字選擇 · u 不確定 · b 上一題 · ? 說明 · r 原始敘述 · s 略過此案 · q 儲存離開"


class Quit(Exception):
    """The reviewer asked to stop; everything confirmed so far is saved."""


class Skip(Exception):
    """The reviewer asked to skip this case for now."""


class Back(Exception):
    """Go back one question."""


Ask = Callable[[str], str]
Say = Callable[[str], None]


def default_answers_path(worksheet: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    first = rows[0]
    return worksheet.with_name(f"{first['batch_id']}-{first['reviewer']}.answers.jsonl")


def load_answers(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Answers saved by an earlier session, checked to belong to this worksheet."""
    if not path.exists():
        return {}
    ids = {r["case_id"] for r in rows}
    batch, reviewer = rows[0]["batch_id"], rows[0]["reviewer"]
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("batch_id") != batch or r.get("reviewer") != reviewer:
            raise ReviewError(f"{path} holds answers for another batch or reviewer")
        if r.get("case_id") not in ids:
            raise ReviewError(f"{path}: {r.get('case_id')} is not on this worksheet")
        out[r["case_id"]] = r
    return out


def save_answers(path: Path, rows: Sequence[Mapping[str, Any]], answers: Mapping[str, Any]) -> None:
    """Rewrite the answer file atomically, owner-only, in worksheet order."""
    text = "".join(
        json.dumps(answers[r["case_id"]], ensure_ascii=False) + "\n"
        for r in rows
        if r["case_id"] in answers
    )
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def consistency_warnings(row: Mapping[str, Any]) -> list[str]:
    """Answers that contradict each other or the labelbook's approval rule."""
    a = row["answers"]
    fp, alert, event = a.get("is_false_positive"), row["truth_of_alert"], row["truth_of_event"]
    out = []
    if fp is True and alert == "true_positive":
        out.append("誤報=是，但 truth_of_alert=告警正確（兩者應一致）")
    if fp is False and alert == "false_positive":
        out.append("誤報=否，但 truth_of_alert=誤報（兩者應一致）")
    if fp is True and event == "malicious":
        out.append("事件=惡意，卻標為誤報")
    if fp is True and a.get("has_evidence") is False:
        out.append("誤報=是 需要畫面中有核准證據，但證據充分=否（考慮改為 unknown）")
    if event == "benign" and alert == "true_positive":
        out.append("事件=良性 但告警正確：只有「良性但確實需要處理」才成立（例如憑證到期）")
    if a.get("urgent") is True and event == "benign":
        out.append("事件=良性 卻標為緊急")
    return out


class Labeller:
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        answers_path: Path,
        *,
        ask: Ask | None = None,
        say: Say | None = None,
        clock: Callable[[], float] = time.monotonic,
        color: bool | None = None,
        width: int | None = None,
    ) -> None:
        if not rows:
            raise ReviewError("the worksheet is empty")
        self.rows = list(rows)
        self.path = answers_path
        self.ask: Ask = ask or (lambda prompt: input(prompt))
        self.say: Say = say or print
        self.clock = clock
        self.color = (
            color if color is not None else sys.stdout.isatty() and "NO_COLOR" not in os.environ
        )
        self.width = width or min(shutil.get_terminal_size((100, 30)).columns, 110)
        self.answers = load_answers(answers_path, rows)
        self.drafts: dict[str, dict[str, Any]] = {}  # "edit again": prefill, never saved
        self.labelled_this_session = 0

    # ---- display --------------------------------------------------------- #

    def _c(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else text

    def _wrap(self, text: str, indent: str = "  ") -> str:
        return "\n".join(
            textwrap.fill(line, self.width, initial_indent=indent, subsequent_indent=indent + "  ")
            if line
            else ""
            for line in text.splitlines()
        )

    def show_case(self, row: Mapping[str, Any], n: int) -> None:
        done = len(self.answers)
        bar = "─" * min(self.width, 80)
        self.say("\n" + self._c(bar, "2"))
        self.say(
            self._c(f"案件 Case {n}/{len(self.rows)}", "1;36")
            + f"   {row['case_id']}   已完成 done {done}/{len(self.rows)}"
        )
        self.say(self._c(bar, "2"))
        if not row["evidence"]:
            self.say(self._c("  告警內容 Alert", "1"))
            self.say(self._wrap(str(row["state"]), "    "))
        for e in row["evidence"]:
            name = FIELD_NAMES.get(e["field"], e["field"])
            self.say(self._c(f"  {name}", "1"))
            self.say(self._wrap(str(e["value"]), "    "))
        self.say(self._c(f"\n  {KEYS_LINE}", "2"))

    def option_label(self, q: Mapping[str, Any], opt: Any) -> str:
        base = (
            OPTION_NAMES.get(opt, str(opt))
            if not isinstance(opt, int) or isinstance(opt, bool)
            else str(opt)
        )
        meaning = q.get("meanings", {}).get(opt) if isinstance(opt, str) else None
        return f"{base}（{meaning}）" if meaning and meaning not in (opt, base) else base

    # ---- one question ---------------------------------------------------- #

    def _read(self, prompt: str, row: Mapping[str, Any], help_key: str) -> str:
        while True:
            raw = self.ask(prompt).strip()
            low = raw.lower()
            if low == "q":
                raise Quit
            if low == "s":
                raise Skip
            if low == "b":
                raise Back
            if low == "?":
                self.say(self._c(self._wrap(HELP.get(help_key, "（沒有說明）")), "33"))
                self.say(self._c(f"  完整規則 full rules: {LABELBOOK}", "2"))
                continue
            if low == "r":
                self.say(self._c("  原始敘述 raw state:", "1"))
                self.say(self._wrap(str(row["state"]), "    "))
                continue
            return raw

    def ask_choice(
        self,
        title: str,
        options: Sequence[Any],
        q: Mapping[str, Any],
        row: Mapping[str, Any],
        help_key: str,
        *,
        multiple: bool,
        previous: Any = None,
    ) -> Any:
        self.say("\n" + self._c(title, "1;32"))
        choices = [o for o in options if o != "unknown"]
        for i, opt in enumerate(choices, 1):
            self.say(f"   {i}. {self.option_label(q, opt)}")
        self.say(f"   u. {OPTION_NAMES['unknown']}")
        if multiple:
            self.say(self._c("   可多選，用逗號分隔，第一個為主要 (e.g. 2,1)", "2"))
        hint = f" [Enter 保留 {self._fmt(previous)}]" if previous is not None else ""
        while True:
            raw = self._read(f"  > {hint} ", row, help_key)
            if not raw and previous is not None:
                return previous
            if raw.lower() == "u":
                return "unknown"
            parts = [p.strip() for p in raw.replace("，", ",").split(",") if p.strip()]
            try:
                picks = [choices[int(p) - 1] for p in parts if 1 <= int(p) <= len(choices)]
            except ValueError:
                picks = []
            if not picks or len(picks) != len(parts) or len(set(map(str, picks))) != len(picks):
                self.say(self._c("  請輸入列出的號碼 (enter a listed number), 或 u / ?", "31"))
                continue
            if not multiple and len(picks) > 1:
                self.say(self._c("  這題只能選一個 (one answer only)", "31"))
                continue
            return picks if multiple else picks[0]

    def _fmt(self, value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(OPTION_NAMES.get(v, str(v)) for v in value)
        return OPTION_NAMES.get(value, str(value))

    # ---- one case -------------------------------------------------------- #

    def label_case(self, row: Mapping[str, Any], n: int) -> dict[str, Any]:
        """Ask everything for one case; raises Quit or Skip."""
        prior = self.drafts.get(row["case_id"]) or self.answers.get(row["case_id"])
        questions = row["questions"]
        steps: list[tuple[str, Any]] = [("truth", k) for k in TRUTHS]
        steps += [("question", qid) for qid in questions]
        steps += [("context", None), ("notes", None)]
        got: dict[str, Any] = {}
        if prior:
            got.update({k: prior[k] for k in TRUTHS})
            got.update({f"q:{k}": v for k, v in prior["answers"].items()})
            got["required_context"] = prior.get("required_context", [])
            got["notes"] = prior.get("notes", "")
        start = self.clock()
        self.show_case(row, n)
        i = 0
        while i < len(steps):
            kind, key = steps[i]
            try:
                if kind == "truth":
                    title, opts = TRUTHS[key]
                    got[key] = self.ask_choice(
                        title, opts, {}, row, key, multiple=False, previous=got.get(key)
                    )
                elif kind == "question":
                    q = questions[key]
                    got[f"q:{key}"] = self.ask_choice(
                        f"{q['instructions']}  [{key}]",
                        q["options"],
                        q,
                        row,
                        q["task"],
                        multiple=q["multiple"],
                        previous=got.get(f"q:{key}"),
                    )
                elif kind == "context":
                    got["required_context"] = self.ask_context(row, got.get("required_context"))
                else:
                    prev = got.get("notes") or ""
                    hint = f" [Enter 保留「{prev}」]" if prev else ""
                    raw = self._read(f"\n備註 notes (可空白 optional){hint} > ", row, "notes")
                    got["notes"] = raw or prev
                i += 1
            except Back:
                i = max(0, i - 1)
        minutes = round(max(self.clock() - start, 6.0) / 60, 1)
        return {
            **row,
            "answers": {qid: got[f"q:{qid}"] for qid in questions},
            "truth_of_event": got["truth_of_event"],
            "truth_of_alert": got["truth_of_alert"],
            "required_context": got["required_context"],
            "notes": got["notes"],
            "minutes_spent": minutes,
        }

    def ask_context(self, row: Mapping[str, Any], previous: list[str] | None) -> list[str]:
        self.say("\n" + self._c("還缺什麼資訊才能確定？ required_context", "1;32"))
        for i, (_, name) in enumerate(CONTEXT_SUGGESTIONS, 1):
            self.say(f"   {i}. {name}")
        self.say(
            self._c("   號碼或自行輸入英文代號，逗號分隔；Enter = 不缺 (nothing missing)", "2")
        )
        hint = f" [Enter 保留 {', '.join(previous)}]" if previous else ""
        raw = self._read(f"  > {hint} ", row, "required_context")
        if not raw:
            return list(previous or [])
        out: list[str] = []
        for p in (x.strip() for x in raw.replace("，", ",").split(",")):
            if not p:
                continue
            if p.isdigit() and 1 <= int(p) <= len(CONTEXT_SUGGESTIONS):
                p = CONTEXT_SUGGESTIONS[int(p) - 1][0]
            if p not in out:
                out.append(p)
        return out

    def confirm(self, answer: Mapping[str, Any]) -> str:
        """``y`` save, ``e`` edit again, ``s`` skip, ``q`` quit without saving this case."""
        self.say("\n" + self._c("確認 Summary", "1;36"))
        for k, (title, _) in TRUTHS.items():
            self.say(f"  {title.split(' ')[0]:<16} {self._fmt(answer[k])}")
        for qid in answer["questions"]:
            self.say(f"  {qid:<18} {self._fmt(answer['answers'][qid])}")
        if answer["required_context"]:
            self.say(f"  {'required_context':<18} {', '.join(answer['required_context'])}")
        if answer["notes"]:
            self.say(f"  {'notes':<18} {answer['notes']}")
        self.say(f"  {'minutes':<18} {answer['minutes_spent']}")
        warnings = consistency_warnings(answer)
        for w in warnings:
            self.say(self._c(f"  ⚠ {w}", "33"))
        while True:
            raw = (
                self.ask("\n  Enter/y 儲存 save · e 修改 edit · s 略過 skip · q 離開 quit > ")
                .strip()
                .lower()
            )
            if raw in ("", "y"):
                if warnings:
                    sure = (
                        self.ask("  有矛盾提醒，仍要儲存？ save anyway? (y/N) > ").strip().lower()
                    )
                    if sure != "y":
                        return "e"
                return "y"
            if raw in ("e", "s", "q"):
                return raw

    # ---- the session ----------------------------------------------------- #

    def run(self, only: str | None = None) -> int:
        """Label until done or quit; returns how many cases are still open."""
        todo = [
            (n, r)
            for n, r in enumerate(self.rows, 1)
            if (r["case_id"] == only if only else r["case_id"] not in self.answers)
        ]
        if only and not todo:
            raise ReviewError(f"{only} is not on this worksheet")
        self.say(self._c("事件人工標註 Incident triage labelling", "1"))
        self.say(
            f"  批次 batch {self.rows[0]['batch_id']} · 審查者 reviewer {self.rows[0]['reviewer']}"
            f" · 已完成 {len(self.answers)}/{len(self.rows)} · 待標 {len(todo)}"
        )
        self.say(f"  答案存在 answers: {self.path}")
        self.say(self._c("  依畫面上的資訊作答；看不出來就選 u（不確定）。", "2"))
        skipped: list[str] = []
        for n, row in todo:
            try:
                while True:
                    answer = self.label_case(row, n)
                    _check_answer(answer, row["questions"])
                    choice = self.confirm(answer)
                    if choice == "e":
                        self.drafts[row["case_id"]] = answer
                        continue
                    if choice == "s":
                        raise Skip
                    if choice == "q":
                        raise Quit
                    break
                self.drafts.pop(row["case_id"], None)
                self.answers[row["case_id"]] = answer
                save_answers(self.path, self.rows, self.answers)
                self.labelled_this_session += 1
                self.say(self._c(f"  ✓ 已儲存 saved ({len(self.answers)}/{len(self.rows)})", "32"))
                if self.labelled_this_session % BREAK_EVERY == 0:
                    self.say(
                        self._c(
                            f"\n  已連續標註 {BREAK_EVERY} 案，建議休息一下 (take a break)。", "33"
                        )
                    )
                    if self.ask("  繼續？ continue? (Y/n) > ").strip().lower() == "n":
                        raise Quit
            except Skip:
                skipped.append(row["case_id"])
                self.say(self._c("  略過 skipped — 下次執行會再出現", "2"))
            except Quit:
                break
        remaining = len(self.rows) - len(self.answers)
        self.say("\n" + self._c("本次結束 Session ended", "1"))
        self.say(
            f"  本次標註 {self.labelled_this_session} 案 · 總計 {len(self.answers)}/{len(self.rows)}"
            f" · 尚餘 {remaining}"
        )
        if skipped:
            self.say(f"  略過 skipped: {', '.join(skipped)}")
        if remaining:
            self.say("  再執行同一個指令即可從中斷處繼續 (run the same command to resume).")
        else:
            self.say(self._c("  全部完成！下一步 next: jef-bench triage review import ...", "1;32"))
        return remaining
