"""Synthetic zh-TW SOAR alerts with labels that are true by construction.

ATT&CK and NVD supply ground truth but their prose is English. A model that only
ever reads English states cannot be claimed to treat zh-TW as first class, so
this generator produces the Chinese side: alert text assembled from a scenario
whose correct routing, severity and urgency are known because the scenario
*defines* them.

Two honesty constraints govern everything here:

* This data is for **training only**. `jef-bench-zh-tw` (P1.2) must be
  human-verified, and the headline numbers come from independent third-party
  benchmarks (P1.5), because a model evaluated on its own generator's templates
  measures the generator, not the model.
* Surface form is varied deliberately -- timestamps, hosts, accounts, vendors,
  phrasing -- so the head cannot shortcut to a template fingerprint. The
  ``noise`` scenarios exist for the same reason: without negatives that *look*
  urgent, "is this a false positive?" degenerates into keyword spotting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..schema import Sample

__all__ = ["SCENARIOS", "SEVERITY_LEVELS", "TEAMS", "Scenario", "build_soar_zh_samples"]

TEAMS: dict[str, str] = {
    "soc": "一般資安監控事件：告警分流、惡意連線、端點偵測",
    "appsec": "應用程式與程式碼相關：網頁漏洞、API 濫用、相依套件風險",
    "infra": "基礎設施層：主機、網路設備、憑證、備份與可用性",
    "identity": "身分與存取管理：帳號、權限、單一登入、特權帳號",
    "fraud": "金融詐欺與異常交易：盜用帳戶、異常下單、洗錢態樣",
}

SEVERITY_LEVELS: list[str] = ["資訊", "低", "中", "高", "危急"]

_HOSTS = [
    "twse-gw-01",
    "ap-trade-07",
    "db-core-02",
    "vpn-edge-03",
    "mail-relay-01",
    "k8s-node-11",
    "bastion-02",
    "web-front-05",
]
_USERS = ["chen.yw", "lin.ch", "huang.mt", "svc_batch", "admin_ops", "wu.jh", "tsai.pl"]
_IPS = ["203.0.113.47", "198.51.100.23", "192.0.2.180", "203.0.113.9", "198.51.100.77"]
_TIMES = ["02:14", "03:47", "09:05", "13:32", "17:58", "22:41", "23:19"]
_VENDORS = ["CrowdStrike", "Palo Alto", "Fortinet", "Microsoft Defender", "Trend Micro"]


@dataclass(frozen=True)
class Scenario:
    """One incident archetype and the answers it fixes by definition.

    Attributes:
        key: Identifier, also the provenance tag in sample metadata.
        templates: Alert bodies. ``{host}``/``{user}``/``{ip}``/``{time}``/
            ``{vendor}``/``{n}`` are filled per sample.
        team: The team that owns this archetype.
        severity: Index into :data:`SEVERITY_LEVELS`.
        urgent: Whether the archetype is genuinely time-critical.
        false_positive: Whether this is a known benign pattern.
    """

    key: str
    templates: list[str]
    team: str
    severity: int
    urgent: bool
    false_positive: bool = False


SCENARIOS: list[Scenario] = [
    Scenario(
        key="ransomware",
        templates=[
            "{vendor} 於 {time} 偵測到主機 {host} 出現大量檔案加密行為，副檔名遭統一改為 .lockbit，"
            "並在共用磁碟留下勒索訊息檔。該主機同時嘗試連線至 {ip}。",
            "端點防護回報 {host} 於 {time} 起在 {n} 分鐘內修改超過 12000 個檔案，"
            "並刪除磁碟區陰影複製（vssadmin delete shadows），研判為勒索軟體加密階段。",
        ],
        team="soc",
        severity=4,
        urgent=True,
    ),
    Scenario(
        key="sqli",
        templates=[
            "WAF 於 {time} 攔截針對 /api/v2/orders 的 SQL Injection 嘗試，"
            "來源 {ip} 在 {n} 分鐘內送出 340 次含 UNION SELECT 的請求，部分請求回應時間異常拉長。",
            "應用程式日誌顯示 {host} 上的查詢介面收到未經參數化的輸入，"
            "錯誤訊息將資料庫結構外洩至回應內容，來源 IP 為 {ip}。",
        ],
        team="appsec",
        severity=3,
        urgent=True,
    ),
    Scenario(
        key="dependency",
        templates=[
            "相依套件掃描發現交易前台使用的 log4j-core 版本為 2.14.1，"
            "存在遠端程式碼執行風險，影響 {n} 個服務，尚未有證據顯示遭到利用。",
            "SCA 工具回報 {host} 所部署的映像檔含有已知漏洞的 openssl 版本，"
            "CVSS 基礎分數 7.5，目前該服務僅對內部網段開放。",
        ],
        team="appsec",
        severity=2,
        urgent=False,
    ),
    Scenario(
        key="priv_esc",
        templates=[
            "帳號 {user} 於 {time} 被加入 Domain Admins 群組，"
            "該異動非由既有權限流程發起，操作來源為 {host}。",
            "特權帳號管理系統顯示 {user} 在未開立工單的情況下取出 {host} 的本機管理員密碼，"
            "並於 {n} 分鐘後登入該主機。",
        ],
        team="identity",
        severity=4,
        urgent=True,
    ),
    Scenario(
        key="impossible_travel",
        templates=[
            "身分平台偵測到 {user} 於 {time} 自台北登入後，"
            "{n} 分鐘內再由 {ip}（境外）成功登入，兩地距離無法於該時間內移動。",
            "單一登入紀錄顯示 {user} 同一時段存在兩個活躍工作階段，"
            "其中一個來源 {ip} 屬於已知的匿名代理網段。",
        ],
        team="identity",
        severity=3,
        urgent=True,
    ),
    Scenario(
        key="account_takeover",
        templates=[
            "風控系統於 {time} 攔截客戶帳戶異常行為：變更綁定手機後隨即申請提領，"
            "登入裝置指紋與過去 {n} 個月紀錄完全不符，來源 {ip}。",
            "同一 IP {ip} 於 {n} 分鐘內嘗試登入 87 個不同客戶帳號並成功 3 個，"
            "成功帳號隨即下單買進同一檔低流動性個股。",
        ],
        team="fraud",
        severity=4,
        urgent=True,
    ),
    Scenario(
        key="wash_trading",
        templates=[
            "交易監控標記帳戶群於 {time} 出現對敲態樣：{n} 組帳戶在同一檔標的上互為買賣對手，"
            "價格區間集中且未造成實質持股變動。",
            "AML 模組回報客戶於 {n} 日內以接近門檻金額分批匯入共 12 筆，"
            "資金隨即轉出至第三方帳戶，態樣符合結構化交易。",
        ],
        team="fraud",
        severity=3,
        urgent=False,
    ),
    Scenario(
        key="cert_expiry",
        templates=[
            "憑證監控顯示 {host} 使用的 TLS 憑證將於 {n} 天後到期，"
            "該主機承載對外行情服務，續期作業尚未排程。",
            "{host} 的中介憑證鏈不完整，部分舊版用戶端出現驗證失敗，"
            "自 {time} 起累積 {n} 筆連線錯誤。",
        ],
        team="infra",
        severity=2,
        urgent=False,
    ),
    Scenario(
        key="disk_pressure",
        templates=[
            "監控系統於 {time} 告警：{host} 根目錄使用率達 94%，"
            "日誌輪替未生效，預估 {n} 小時後將寫滿。",
            "Kubernetes 節點 {host} 進入 DiskPressure 狀態並開始驅逐 Pod，"
            "影響 {n} 個交易輔助服務的可用性。",
        ],
        team="infra",
        severity=3,
        urgent=True,
    ),
    Scenario(
        key="c2_beacon",
        templates=[
            "網路偵測於 {time} 發現 {host} 每 {n} 秒對 {ip} 發出固定長度的 HTTPS 請求，"
            "抖動極小，符合 C2 信標特徵，該網域於 {n} 天前始註冊。",
            "DNS 日誌顯示 {host} 持續查詢由演算法產生的網域名稱，"
            "{n} 分鐘內出現 240 筆 NXDOMAIN 回應，研判為 DGA 行為。",
        ],
        team="soc",
        severity=4,
        urgent=True,
    ),
    # ---- genuinely low severity. Without these the `score` task never observes
    # level 1, and an ordinal scale with a hole in the middle distorts the
    # expectation the engine computes over it.
    Scenario(
        key="stale_account",
        templates=[
            "帳號稽核發現 {user} 已連續 {n} 日未登入，"
            "但仍保有交易系統的唯讀權限，依政策應於離職或轉調後回收。",
            "{host} 上仍存在 {n} 個未停用的本機帳號，最後一次使用紀錄超過一年，無立即濫用跡象。",
        ],
        team="identity",
        severity=1,
        urgent=False,
    ),
    Scenario(
        key="weak_config",
        templates=[
            "組態基準掃描顯示 {host} 的 SSH 服務仍允許密碼登入，"
            "雖已限制來源網段，仍不符內部強化基準第 {n} 項。",
            "{host} 的稽核日誌保存期限設定為 {n} 天，低於法遵要求的 180 天，"
            "目前尚未發生需回溯調查的事件。",
        ],
        team="infra",
        severity=1,
        urgent=False,
    ),
    # ---- benign-but-alarming: these carry the "is this a false positive?" task
    Scenario(
        key="noise_scan",
        templates=[
            "IDS 於 {time} 回報來自 {ip} 的連接埠掃描，"
            "經比對 {ip} 屬於本公司委外之季度弱點掃描服務，掃描視窗已事先核准。",
            "{vendor} 告警指出 {host} 遭大量 TCP SYN 探測，"
            "來源為內部資安部門的 {n} 號掃描節點，屬例行合規作業。",
        ],
        team="soc",
        severity=0,
        urgent=False,
        false_positive=True,
    ),
    Scenario(
        key="noise_batch",
        templates=[
            "帳號 {user} 於 {time} 在 {n} 分鐘內存取 8400 筆客戶資料，"
            "該帳號為日結批次服務帳號，存取量與過去每月同期一致。",
            "{host} 於 {time} 對外傳輸 4.2GB 資料觸發 DLP 告警，"
            "經查為既有的異地備份排程，目的地為公司自有備援機房。",
        ],
        team="infra",
        severity=0,
        urgent=False,
        false_positive=True,
    ),
    Scenario(
        key="noise_patch",
        templates=[
            "端點於 {time} 出現大量檔案變更與服務重啟，"
            "經比對為當月例行修補程式部署，變更單編號 CHG-{n}。",
            "{user} 於 {time} 自 {ip} 登入 {host}，"
            "該來源為公司 VPN 出口位址，登入行為與其日常班表相符。",
        ],
        team="soc",
        severity=0,
        urgent=False,
        false_positive=True,
    ),
]


def _render(template: str, rng: random.Random) -> str:
    return template.format(
        host=rng.choice(_HOSTS),
        user=rng.choice(_USERS),
        ip=rng.choice(_IPS),
        time=rng.choice(_TIMES),
        vendor=rng.choice(_VENDORS),
        n=rng.randint(3, 96),
    )


def build_soar_zh_samples(*, per_scenario: int = 40, seed: int = 20260927) -> list[Sample]:
    """Generate zh-TW SOAR samples covering all three primitives."""
    rng = random.Random(seed)  # noqa: S311 -- reproducible synthetic data, not crypto

    team_keys = list(TEAMS)
    team_labels = list(TEAMS.values())
    samples: list[Sample] = []

    for scenario in SCENARIOS:
        for i in range(per_scenario):
            template_index = i % len(scenario.templates)
            state = _render(scenario.templates[template_index], rng)
            meta = {"scenario": scenario.key, "state_lang": "zh-TW", "synthetic": True}
            # Slot fills differ; the sentence does not. Two renders of the same
            # template are ~95% identical text, so the split unit is the
            # template, not the render. Without this, SOAR routing scored 1.000
            # by recalling the training set.
            group = f"soar:{scenario.key}:{template_index}"

            samples.append(
                Sample(
                    state=state,
                    kind="choice",
                    instructions="這則告警應由哪一個團隊處理？",
                    option_keys=team_keys,
                    option_labels=team_labels,
                    label=team_keys.index(scenario.team),
                    source="soar_zh.routing",
                    group=group,
                    meta=meta,
                )
            )
            samples.append(
                Sample(
                    state=state,
                    kind="score",
                    instructions="評估這則告警的嚴重程度。",
                    option_keys=SEVERITY_LEVELS,
                    option_labels=SEVERITY_LEVELS,
                    label=scenario.severity,
                    source="soar_zh.severity",
                    group=group,
                    meta=meta,
                )
            )
            # Only one yes/no per sample, alternating between the two questions:
            # asking both every time would make the two tasks perfectly
            # correlated in training and inflate apparent performance.
            if i % 2 == 0:
                samples.append(
                    Sample(
                        state=state,
                        kind="noul",
                        instructions="這則告警是否為已知的良性樣態（誤報）？",
                        option_keys=["false", "true"],
                        option_labels=["為真實事件，需進一步處理", "為已知良性樣態，可直接關閉"],
                        label=1 if scenario.false_positive else 0,
                        source="soar_zh.false_positive",
                        group=group,
                        meta=meta,
                    )
                )
            else:
                samples.append(
                    Sample(
                        state=state,
                        kind="noul",
                        instructions="這則告警是否需要立即處理？",
                        option_keys=["false", "true"],
                        option_labels=["可依排程處理", "需立即介入"],
                        label=1 if scenario.urgent else 0,
                        source="soar_zh.urgency",
                        group=group,
                        meta=meta,
                    )
                )

    rng.shuffle(samples)
    return samples
