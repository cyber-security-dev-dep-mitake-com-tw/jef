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
