*** Settings ***
Documentation    P2 acceptance: layered gates, decision traces, and the refusal
...              to automate without calibration.
...
...              This is the layer Jev leaves to the caller and every open
...              reimplementation therefore omits, so these cases are the ones
...              that describe what JEF is actually for.
Resource         ../resources/jef.resource
Suite Setup      Connect To JEF
Force Tags       p2    scene

*** Variables ***
${RANSOMWARE}     CrowdStrike 於 02:14 偵測到主機 db-core-02 出現大量檔案加密行為，副檔名遭統一改為 .lockbit，並刪除磁碟區陰影複製。
${BENIGN}         IDS 於 03:47 回報來自 198.51.100.23 的連接埠掃描，經比對該位址屬於本公司委外之季度弱點掃描服務，掃描視窗已事先核准。
${SCENE}          incident-triage

*** Keywords ***
Run Scene
    [Arguments]    ${scene}    ${state}
    ${body}=    Create Dictionary    state=${state}
    ${resp}=    POST On Session    ${SESSION}    /v1/scenes/${scene}:evaluate    json=${body}
    Should Be Equal As Integers    ${resp.status_code}    200
    RETURN    ${resp.json()}

*** Test Cases ***
Scenes Are Discoverable
    ${resp}=    GET On Session    ${SESSION}    /v1/scenes
    Should Be Equal As Integers    ${resp.status_code}    200
    ${ids}=    Evaluate    [e["id"] for e in $resp.json()["data"]]
    List Should Contain Value    ${ids}    ${SCENE}

A Whole Playbook Costs One State Read
    [Documentation]    Three layers and five questions, encoded once.
    ...                If this ever exceeds 1, a deep playbook costs one encode
    ...                per gate and scenes have no reason to live inside JEF.
    [Tags]    d3    performance
    ${trace}=    Run Scene    ${SCENE}    ${RANSOMWARE}
    Should Be Equal As Integers    ${trace}[state_encodes]    1

Trace Records The Evidence An Audit Needs
    ${trace}=    Run Scene    ${SCENE}    ${RANSOMWARE}
    ${layer}=    Set Variable    ${trace}[layers][0]
    ${question}=    Set Variable    ${layer}[questions][0]
    FOR    ${key}    IN    id    kind    instructions    allowed_answers    answer    confidence    p_correct
        Dictionary Should Contain Key    ${question}    ${key}
    END
    ${gate}=    Set Variable    ${layer}[gates][0]
    FOR    ${key}    IN    index    condition    fired    is_else
        Dictionary Should Contain Key    ${gate}    ${key}
    END

Answers Are Constrained To The Allowed Set
    ${trace}=    Run Scene    ${SCENE}    ${RANSOMWARE}
    FOR    ${layer}    IN    @{trace}[layers]
        FOR    ${q}    IN    @{layer}[questions]
            ${allowed}=    Set Variable    ${q}[allowed_answers]
            Should Not Be Empty    ${allowed}
            IF    "choice" == "${q}[kind]"
                List Should Contain Value    ${allowed}    ${q}[answer][choice]
            END
        END
    END

An Uncalibrated Deployment Cannot Automate
    [Documentation]    Without a fitted calibrator P(correct) is unavailable and
    ...                the conformal set excludes nothing, so no automating gate
    ...                can fire. This is the design working: you do not get
    ...                automation out of JEF without calibrating it.
    [Tags]    calibration    safety
    ${trace}=    Run Scene    ${SCENE}    ${RANSOMWARE}
    IF    ${trace}[calibrated] == ${False}
        Should Be True    ${trace}[human_review]
        ...    msg=an uncalibrated deployment routed a decision away from a human
        FOR    ${layer}    IN    @{trace}[layers]
            FOR    ${q}    IN    @{layer}[questions]
                Should Be Equal    ${q}[p_correct]    ${None}
                Should Not Be Empty    ${q}[prediction_set]
            END
        END
    END

Decided Layers Stop The Run
    [Documentation]    Layers after the decision must not be asked, and the
    ...                trace must distinguish "not asked" from "inconclusive".
    ${trace}=    Run Scene    ${SCENE}    ${BENIGN}
    ${asked}=    Evaluate    sum(len(l["questions"]) for l in $trace["layers"])
    Should Be Equal As Integers    ${trace}[questions_asked]    ${asked}
    ${skipped}=    Set Variable    ${trace}[layers_skipped]
    ${evaluated}=    Evaluate    [l["id"] for l in $trace["layers"]]
    FOR    ${id}    IN    @{skipped}
        List Should Not Contain Value    ${evaluated}    ${id}
    END

Every Run Reaches A Declared Action
    [Documentation]    A playbook that runs off the end has not made a decision.
    FOR    ${state}    IN    ${RANSOMWARE}    ${BENIGN}    系統異常。
        ${trace}=    Run Scene    ${SCENE}    ${state}
        Should Not Be Equal    ${trace}[action]    ${None}
        Should Be True    "${trace}[verdict]" in ("decided", "fallthrough")
    END

Unknown Scene Names The Ones It Knows
    ${body}=    Create Dictionary    state=x
    ${resp}=    POST On Session    ${SESSION}    /v1/scenes/nope:evaluate    json=${body}
    ...    expected_status=anything
    Should Be Equal As Integers    ${resp.status_code}    422
    Should Contain    ${resp.json()}[error][message]    ${SCENE}

Missing State Is Rejected
    ${body}=    Create Dictionary
    ${resp}=    POST On Session    ${SESSION}    /v1/scenes/${SCENE}:evaluate    json=${body}
    ...    expected_status=anything
    Should Be Equal As Integers    ${resp.status_code}    422

Object State Is Accepted
    ${state}=    Create Dictionary    alert=${RANSOMWARE}    host=db-core-02
    ${trace}=    Run Scene    ${SCENE}    ${state}
    Should Be Equal As Integers    ${trace}[state_encodes]    1
