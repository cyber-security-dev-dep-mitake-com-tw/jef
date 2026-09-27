"""MITRE ATT&CK -> typed decision samples.

The label is not an opinion: ATT&CK states which tactic (kill-chain phase) each
technique belongs to, so "which tactic does this behaviour serve?" has a ground
truth straight from the taxonomy. That is the whole reason this builder exists
instead of asking a language model to invent labels.

Techniques with multiple tactics are skipped rather than forced into one. A
question whose correct answer is genuinely ambiguous teaches the head to be
confident about a coin flip, which is the exact pathology the calibration work
is meant to remove.
"""

from __future__ import annotations

import logging
from typing import Any

from ..schema import Sample
from .sources import ATTACK_ENTERPRISE_URL, fetch_json

log = logging.getLogger("jef.train.corpus.attack")

__all__ = ["TACTICS_ZH", "build_attack_samples"]

#: The 14 Enterprise tactics with Traditional Chinese names. MITRE publishes no
#: official zh-TW translation, so these are curated here -- and they are part of
#: what makes zh-TW first-class rather than machine-translated at runtime.
TACTICS_ZH: dict[str, tuple[str, str]] = {
    "reconnaissance": ("偵察", "蒐集可用於後續攻擊的目標資訊"),
    "resource-development": ("資源開發", "建立或取得攻擊所需的基礎設施與工具"),
    "initial-access": ("初始存取", "取得目標網路的第一個立足點"),
    "execution": ("執行", "在目標系統上執行惡意程式碼"),
    "persistence": ("持久化", "在重開機或憑證變更後仍維持存取"),
    "privilege-escalation": ("提權", "取得更高層級的權限"),
    "defense-evasion": ("防禦規避", "躲避偵測或停用防護機制"),
    "credential-access": ("憑證存取", "竊取帳號名稱與密碼等憑證"),
    "discovery": ("環境探索", "了解所處環境與內部網路結構"),
    "lateral-movement": ("橫向移動", "在環境內移動到其他系統"),
    "collection": ("資料蒐集", "蒐集目標資料以供外傳"),
    "command-and-control": ("命令與控制", "與受害系統建立通訊通道"),
    "exfiltration": ("資料外傳", "將竊得的資料送出網路"),
    "impact": ("破壞影響", "操弄、中斷或破壞系統與資料"),
}


def _is_technique(obj: dict[str, Any]) -> bool:
    return (
        obj.get("type") == "attack-pattern"
        and not obj.get("revoked", False)
        and not obj.get("x_mitre_deprecated", False)
    )


def _tactics(obj: dict[str, Any]) -> list[str]:
    return [
        p["phase_name"]
        for p in obj.get("kill_chain_phases", [])
        if p.get("kill_chain_name") == "mitre-attack"
    ]


def _clean(text: str, limit: int = 1500) -> str:
    """Strip ATT&CK's citation markup, which is noise the model should not read."""
    import re

    text = re.sub(r"\(Citation:[^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # markdown links
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def build_attack_samples(*, refresh: bool = False, max_samples: int | None = None) -> list[Sample]:
    """Build tactic-classification and yes/no samples from Enterprise ATT&CK."""
    bundle = fetch_json(ATTACK_ENTERPRISE_URL, refresh=refresh)
    objects = bundle.get("objects", [])

    keys = list(TACTICS_ZH)
    labels = [f"{TACTICS_ZH[k][0]}：{TACTICS_ZH[k][1]}" for k in keys]

    samples: list[Sample] = []
    skipped_multi = 0
    #: Counts techniques, not samples. Deriving polarity from len(samples) is a
    #: trap: two samples are appended per technique, so the parity never flips
    #: and every yes/no question comes out with the same label.
    technique_index = 0

    for obj in objects:
        if not _is_technique(obj):
            continue
        tactics = _tactics(obj)
        if len(tactics) != 1:
            # Ambiguous by the taxonomy's own account -- see module docstring.
            skipped_multi += 1
            continue
        tactic = tactics[0]
        if tactic not in TACTICS_ZH:
            continue

        description = _clean(obj.get("description", ""))
        name = obj.get("name", "")
        if len(description) < 80:
            continue

        attack_id = next(
            (
                ref.get("external_id", "")
                for ref in obj.get("external_references", [])
                if ref.get("source_name") == "mitre-attack"
            ),
            "",
        )
        state = f"技術名稱：{name}\n\n技術描述：\n{description}"

        samples.append(
            Sample(
                state=state,
                kind="choice",
                instructions="這項攻擊技術主要服務於哪一個 ATT&CK 戰術階段？",
                option_keys=keys,
                option_labels=labels,
                label=keys.index(tactic),
                source="attack.tactic",
                lang="zh-TW",
                meta={"attack_id": attack_id, "state_lang": "en", "technique": name},
            )
        )

        # A paired yes/no over the same evidence. Polarity alternates per
        # technique to hold the true/false prior at 50%: a skewed prior lets a
        # head score well by ignoring the state entirely.
        positive = technique_index % 2 == 0
        technique_index += 1
        asked = tactic if positive else keys[(keys.index(tactic) + 7) % len(keys)]
        samples.append(
            Sample(
                state=state,
                kind="noul",
                instructions=f"這項技術是否屬於「{TACTICS_ZH[asked][0]}」戰術？",
                option_keys=["false", "true"],
                option_labels=[
                    f"不屬於{TACTICS_ZH[asked][0]}戰術",
                    f"屬於{TACTICS_ZH[asked][0]}戰術",
                ],
                label=1 if positive else 0,
                source="attack.tactic_noul",
                lang="zh-TW",
                meta={"attack_id": attack_id, "state_lang": "en", "asked_tactic": asked},
            )
        )

        if max_samples and len(samples) >= max_samples:
            break

    log.info(
        "attack: %d samples (%d techniques skipped for spanning multiple tactics)",
        len(samples),
        skipped_multi,
    )
    return samples
