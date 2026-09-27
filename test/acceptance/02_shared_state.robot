*** Settings ***
Documentation    D3 acceptance: the state is encoded once per request.
...
...              This is the project's central performance claim. If it ever
...              regresses into one encode per question, a twenty-gate SOAR
...              scene costs twenty state reads and JEF has no reason to exist.
...              The server exports the counters that make this observable in
...              production, so the same property is asserted here over HTTP.
Resource         ../resources/jef.resource
Suite Setup      Connect To JEF
Force Tags       p0    d3    performance

*** Test Cases ***
One Request With Many Questions Encodes The State Once
    ${before}=    Metric Value    jef_state_encodes_total
    ${questions}=    Create Dictionary
    FOR    ${i}    IN RANGE    25
        ${q}=    Noul Question    這是否緊急？
        Set To Dictionary    ${questions}    q${i}=${q}
    END
    ${body}=    Ask JEF    付款服務連續三天失敗    ${questions}
    Length Should Be    ${body}[answers]    25
    ${after}=    Metric Value    jef_state_encodes_total
    ${delta}=    Evaluate    ${after} - ${before}
    Should Be Equal As Numbers    ${delta}    1.0
    ...    msg=D3 violated: 25 questions triggered ${delta} state encodes

Adding Questions Does Not Multiply State Reads
    [Documentation]    One question and fifty questions must cost the same state reads.
    ${b1}=    Metric Value    jef_state_encodes_total
    ${one}=    Create Dictionary
    ${q}=    Noul Question    是否緊急？
    Set To Dictionary    ${one}    q0=${q}
    Ask JEF    ${{ "長文字狀態 " * 500 }}    ${one}
    ${a1}=    Metric Value    jef_state_encodes_total

    ${many}=    Create Dictionary
    FOR    ${i}    IN RANGE    50
        ${qq}=    Noul Question    是否緊急？
        Set To Dictionary    ${many}    q${i}=${qq}
    END
    Ask JEF    ${{ "長文字狀態 " * 500 }}    ${many}
    ${a2}=    Metric Value    jef_state_encodes_total

    ${cost_one}=     Evaluate    ${a1} - ${b1}
    ${cost_many}=    Evaluate    ${a2} - ${a1}
    Should Be Equal As Numbers    ${cost_one}     1.0
    Should Be Equal As Numbers    ${cost_many}    1.0

Questions Are Evaluated Independently
    [Documentation]    A question's answer must not change when unrelated siblings are added.
    ${team}=    Choice Question    誰處理？    billing=付款    infra=網路    appsec=漏洞
    ${alone}=    Create Dictionary    team=${team}
    ${solo}=    Ask JEF    付款服務失敗    ${alone}

    ${sev}=      Score Question    嚴重度    低    中    高
    ${urgent}=   Noul Question    是否緊急？
    ${crowded}=  Create Dictionary    team=${team}    sev=${sev}    urgent=${urgent}
    ${mixed}=    Ask JEF    付款服務失敗    ${crowded}

    Should Be Equal    ${solo}[answers][team][choice]    ${mixed}[answers][team][choice]
    Should Be Equal As Numbers
    ...    ${solo}[answers][team][confidence]    ${mixed}[answers][team][confidence]

Question Order Does Not Affect Answers
    ${team}=    Choice Question    誰處理？    billing=付款    infra=網路
    ${sev}=     Score Question    嚴重度    低    中    高
    ${forward}=    Create Dictionary    team=${team}    sev=${sev}
    ${reverse}=    Create Dictionary    sev=${sev}    team=${team}
    ${a}=    Ask JEF    付款失敗    ${forward}
    ${b}=    Ask JEF    付款失敗    ${reverse}
    Should Be Equal As Numbers    ${a}[answers][sev][score]    ${b}[answers][sev][score]

Batching Beats Asking One Question At A Time
    [Documentation]    The property shared-state encoding exists to deliver, and
    ...                the reason a playbook should ask its speculative questions
    ...                up front: eleven questions in one request must cost far
    ...                less than eleven separate requests.
    ...
    ...                Encode count alone does not prove this. A shared encoding
    ...                followed by expensive per-question work would still report
    ...                one encode while costing linearly.
    ...
    ...                The plan's other figure -- "ten extra questions add under
    ...                20% latency" -- is measured against a *real* backbone and
    ...                recorded in docs/RESULTS.md. It does not hold against the
    ...                hashing stub used here, where the state encode is nearly
    ...                free and so contributes almost none of the fixed cost.
    ...                Loosening that threshold until this suite passed would
    ...                have made the number meaningless rather than met.
    [Tags]    d3    performance
    ${state}=    Set Variable    ${{ "付款服務連續三天失敗，已影響營收。錯誤碼 502。" * 60 }}

    ${one}=    Create Dictionary
    ${q}=    Noul Question    這是否緊急？
    Set To Dictionary    ${one}    q0=${q}

    ${eleven}=    Create Dictionary
    FOR    ${i}    IN RANGE    11
        ${qq}=    Noul Question    這是否緊急？
        Set To Dictionary    ${eleven}    q${i}=${qq}
    END

    # Warm the server so first-call costs are not what gets measured.
    Ask JEF    ${state}    ${one}
    Ask JEF    ${state}    ${eleven}

    ${single}=      Median Latency    ${state}    ${one}
    ${batched}=     Median Latency    ${state}    ${eleven}
    ${separately}=  Evaluate    ${single} * 11
    ${speedup}=     Evaluate    ${separately} / ${batched} if ${batched} > 0 else 0
    ${growth}=      Evaluate    (${batched} - ${single}) / ${single} if ${single} > 0 else 0

    # One argument: Robot reads a second space-separated value as the log level.
    Log    ${{ f"one={$single:.4f}s batched11={$batched:.4f}s separate11={$separately:.4f}s speedup={$speedup:.2f}x growth={$growth:.3f}" }}
    Should Be True    ${speedup} > 3.0
    ...    msg=batching eleven questions was only ${speedup}x cheaper than asking separately; the state is not being shared
