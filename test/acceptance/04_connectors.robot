*** Settings ***
Documentation    P4.3 acceptance: the SOAR connectors against a live server.
...
...              The pytest suite covers request shaping with the SDKs stubbed.
...              This covers the other half -- that the shapes those connectors
...              build are ones a real server accepts, and that what comes back
...              is what a playbook would actually branch on.
Library          ${CURDIR}/../resources/ConnectorLibrary.py
Library          Collections
Resource         ../resources/jef.resource
Force Tags       p4    connectors

*** Variables ***
${ALERT}     CrowdStrike 於 02:14 偵測到主機 db-core-02 出現大量檔案加密行為，副檔名遭統一改為 .lockbit。
${TEAMS}     soc=一般資安監控事件\nappsec=應用程式漏洞與程式碼相關\ninfra=基礎設施、網路、主機層
${LEVELS}    資訊\n低\n中\n高\n危急

*** Test Cases ***
Shuffle Health Gates The Workflow
    [Documentation]    A playbook should be able to refuse to automate on an
    ...                uncalibrated or test-backbone server, so both flags must
    ...                be reachable from the action's output.
    ${health}=    Shuffle Health    ${JEF_BASE_URL}
    Should Be Equal    ${health}[status]    ok
    Dictionary Should Contain Key    ${health}    calibrated
    Dictionary Should Contain Key    ${health}    test_backbone

Shuffle Run Scene Returns A Branchable Verdict
    # Tagged `scene` so runs against the Go server skip it: scenes live in
    # jef-scene, which has no Go port yet.
    [Tags]    scene
    ${trace}=    Shuffle Run Scene    ${JEF_BASE_URL}    incident-triage    ${ALERT}
    Should Not Be Equal    ${trace}[action]    ${None}
    Dictionary Should Contain Key    ${trace}    human_review
    # The whole playbook still costs one state read through the connector.
    Should Be Equal As Integers    ${trace}[state_encodes]    1

Shuffle Ask Choice Answers From The Supplied Options
    ${result}=    Shuffle Ask Choice    ${JEF_BASE_URL}    ${ALERT}    應由哪一個團隊處理？    ${TEAMS}
    ${answer}=    Set Variable    ${result}[answers][answer]
    Should Be Equal    ${answer}[type]    choice
    ${keys}=    Get Dictionary Keys    ${answer}[probabilities]    sort_keys=${False}
    List Should Contain Value    ${keys}    ${answer}[choice]
    # Option order must survive the app, the wire and the response. sort_keys
    # is off on purpose: the default sorts, which would make this assert
    # alphabetical order instead of the order that decides the option index.
    Should Be Equal    ${keys}    ${{ ["soc", "appsec", "infra"] }}

Shuffle Ask Score Returns A Continuous Score Within Its Legend
    ${result}=    Shuffle Ask Score    ${JEF_BASE_URL}    ${ALERT}    評估嚴重程度    ${LEVELS}
    ${answer}=    Set Variable    ${result}[answers][answer]
    Should Be Equal    ${answer}[type]    score
    Should Be True    0.0 <= ${answer}[score] <= 4.0
    Length Should Be    ${answer}[legend]    5

Shuffle Ask Yes No Returns A Probability
    ${result}=    Shuffle Ask Yes No    ${JEF_BASE_URL}    ${ALERT}    這則告警是否需要立即處理？
    ${answer}=    Set Variable    ${result}[answers][answer]
    Should Be Equal    ${answer}[type]    noul
    Should Be True    0.0 <= ${answer}[noul] <= 1.0

Shuffle Ask Many Costs One State Read
    [Documentation]    Four questions in one call, evaluated in parallel against
    ...                the same evidence. This is why a playbook should ask the
    ...                speculative questions too instead of round-tripping.
    ${questions}=    Catenate    SEPARATOR=
    ...    {"fp": {"type": "noul", "instructions": "是否為已知良性樣態？"},
    ...     "urgent": {"type": "noul", "instructions": "是否需要立即處理？"},
    ...     "team": {"type": "choice", "instructions": "誰處理？",
    ...              "criteria": {"soc": "監控", "infra": "網路", "appsec": "程式碼"}},
    ...     "sev": {"type": "score", "instructions": "嚴重度", "criteria": ["低","中","高"]}}
    ${before}=    Metric Value    jef_state_encodes_total
    ${result}=    Shuffle Ask Many    ${JEF_BASE_URL}    ${ALERT}    ${questions}
    Length Should Be    ${result}[answers]    4
    ${after}=    Metric Value    jef_state_encodes_total
    ${delta}=    Evaluate    ${after} - ${before}
    Should Be Equal As Numbers    ${delta}    1.0
    ...    msg=four questions through the connector cost ${delta} state encodes

Shuffle Surfaces A Rejection The Workflow Can Branch On
    ${result}=    Shuffle Ask Choice    ${JEF_BASE_URL}    ${ALERT}    誰處理？    only=one
    Should Be Equal    ${result}[success]    ${False}
    Should Contain    ${result}[error]    at least two

Shuffle Reports An Unreachable Server Rather Than Hanging
    ${result}=    Shuffle Health    http://127.0.0.1:59999
    Should Be Equal    ${result}[success]    ${False}
    Should Contain    ${result}[error]    unreachable

Cortex Taxonomies Describe A Real Trace
    [Documentation]    The verdict chip an analyst reads first comes from a real
    ...                scene run, not a hand-written fixture.
    [Tags]    scene
    ${trace}=    Shuffle Run Scene    ${JEF_BASE_URL}    incident-triage    ${ALERT}
    ${taxonomies}=    Cortex Scene Taxonomies    ${trace}
    Should Not Be Empty    ${taxonomies}
    ${predicates}=    Evaluate    [t["predicate"] for t in $taxonomies]
    List Should Contain Value    ${predicates}    action
    FOR    ${t}    IN    @{taxonomies}
        Should Be True    "${t}[level]" in ("safe", "info", "suspicious", "malicious")
    END

Cortex Never Renders Unknown Reliability As A Number
    [Documentation]    "Unknown" and "low" lead to different decisions, so an
    ...                uncalibrated server must not produce a plausible-looking
    ...                P(correct) chip.
    [Tags]    scene
    ${trace}=    Shuffle Run Scene    ${JEF_BASE_URL}    incident-triage    ${ALERT}
    ${taxonomies}=    Cortex Scene Taxonomies    ${trace}
    ${p}=    Evaluate    [t for t in $taxonomies if t["predicate"] == "p_correct"]
    IF    ${trace}[calibrated] == ${False} and ${p}
        Should Be Equal    ${p}[0][value]    unknown
    END
