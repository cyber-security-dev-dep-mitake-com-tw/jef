# zh-TW seed cases

Twenty-eight cases for the native Traditional Chinese (Taiwan) track
(review section 10.2), in twelve families:

| family | task | the decisive difference between the twins |
|---|---|---|
| bank-login | phishing | look-alike sender asking for password and OTP vs. an official notice that refuses to ask |
| einvoice-prize | phishing | prize claim that wants a card number vs. the platform's real claim flow |
| line-boss-giftcard | phishing | new LINE account posing as a manager asking for point cards vs. the official IT account's notice |
| parcel-fee | phishing | small "shipping fee" with a pay link vs. a pickup code with no link |
| traffic-fine | phishing | licence-suspension threat with a pay link vs. a renewal reminder with none |
| supplier-bank-change | phishing | same-day account change that forbids calling back vs. a stamped notice with a call-back step |
| mail-password-expiry | phishing | "keep your password" link from an outside domain vs. the documented in-house way |
| hr-bonus | phishing | outside page collecting company credentials vs. the internal HR portal |
| vpn-impossible-travel | alert | impossible travel, new device, MFA bypass, bulk download vs. approved travel and a registered laptop |
| backup-exfil | alert | off-schedule read to an outside cloud vs. the scheduled job to the internal NAS with a change ticket |
| office-powershell | alert | Word macro spawning encoded PowerShell to an outside IP vs. a signed inventory script from the management agent |
| firewall-rule-wipe | alert | a departed admin's account opening any-any outside a window vs. an approved, reviewed maintenance change |

Each family holds a positive and a **hard-negative twin** (`pair_kind:
hard-negative`) that differs only in the decisive detail. Four positives also
have an **English translation** (`pair_kind: translation`, same evidence) for
the parallel bilingual track. Twins and translations share a `family_id`, so
the lock keeps each family in one split.

**These are not gold.** They were written by Claude for this repository:
`telemetry_origin: synthetic`, `label_origin: llm-authored`, trust tier E,
`confidence_of_gold: low`. Every organisation, domain and person is fictional
(`.invalid`, `.example`), and every link is defanged. Run them through the
two-reviewer workflow (`jef-bench annotate`) before any number from them is
reported as anything but a smoke test, and report them apart from tiers A to
D.
