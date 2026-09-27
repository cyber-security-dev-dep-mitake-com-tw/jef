*** Settings ***
Documentation    P0 acceptance: the /v1/systemone contract over real HTTP.
...
...              This is the executable form of the P0 exit criterion -- an
...              existing TypeSafe or Vercel AI SDK client must be able to point
...              at a JEF deployment without changing a line.
Resource         ../resources/jef.resource
Suite Setup      Connect To JEF
Force Tags       p0    contract

*** Variables ***
${ALERT}    客戶回報：連續三天付款失敗，已影響營收。錯誤碼 502。

*** Test Cases ***
Server Reports Healthy
    [Documentation]    A deployment must never hide that it is serving the test backbone.
    ${resp}=    GET On Session    ${SESSION}    /healthz
    Should Be Equal As Integers    ${resp.status_code}    200
    Should Be Equal    ${resp.json()}[status]    ok
    Dictionary Should Contain Key    ${resp.json()}    test_backbone

Model Discovery Advertises All Four Question Spellings
    ${resp}=    GET On Session    ${SESSION}    /v1/models
    ${types}=    Set Variable    ${resp.json()}[data][0][question_types]
    FOR    ${t}    IN    choice    score    noul    boolean
        List Should Contain Value    ${types}    ${t}
    END

Limits Are Discoverable
    ${resp}=    GET On Session    ${SESSION}    /v1/limits
    Should Be True    ${resp.json()}[max_questions_per_request] > 0

Choice Question Returns A Distribution Over The Supplied Options
    ${q}=    Choice Question    應由哪個團隊處理？
    ...    billing=付款、發票、退款    infra=基礎設施、網路、主機層    appsec=應用程式漏洞
    ${questions}=    Create Dictionary    team=${q}
    ${body}=    Ask JEF    ${ALERT}    ${questions}
    ${answer}=    Set Variable    ${body}[answers][team]
    Should Be Equal    ${answer}[type]    choice
    Probabilities Should Sum To One    ${answer}
    # The answer is constrained to the options the caller supplied -- never invented.
    List Should Contain Value    ${{ list($answer['probabilities']) }}    ${answer}[choice]
    Should Be True    0.0 <= ${answer}[confidence] <= 1.0

Score Question Returns A Continuous Score Within Its Legend
    ${q}=    Score Question    評估此事件的嚴重度    資訊    低    中    高    危急
    ${questions}=    Create Dictionary    sev=${q}
    ${body}=    Ask JEF    ${ALERT}    ${questions}
    ${answer}=    Set Variable    ${body}[answers][sev]
    Should Be Equal    ${answer}[type]    score
    Should Be True    0.0 <= ${answer}[score] <= 4.0
    Length Should Be    ${answer}[legend]    5
    Probabilities Should Sum To One    ${answer}

Noul Dialect Returns A Noul Field
    ${q}=    Noul Question    這則訊息是否表達時間緊迫？    noul
    ${questions}=    Create Dictionary    urgent=${q}
    ${body}=    Ask JEF    ${ALERT}    ${questions}
    ${answer}=    Set Variable    ${body}[answers][urgent]
    Should Be Equal    ${answer}[type]    noul
    Dictionary Should Contain Key        ${answer}    noul
    Dictionary Should Not Contain Key    ${answer}    probability

Boolean Dialect Returns A Probability Field
    [Documentation]    The AI SDK spells the same primitive `boolean`.
    ${q}=    Noul Question    這則訊息是否表達時間緊迫？    boolean
    ${questions}=    Create Dictionary    urgent=${q}
    ${body}=    Ask JEF    ${ALERT}    ${questions}
    ${answer}=    Set Variable    ${body}[answers][urgent]
    Should Be Equal    ${answer}[type]    boolean
    Dictionary Should Contain Key        ${answer}    probability
    Dictionary Should Not Contain Key    ${answer}    noul

Both Dialects Yield The Same Number
    ${n}=    Noul Question    這則訊息是否表達時間緊迫？    noul
    ${b}=    Noul Question    這則訊息是否表達時間緊迫？    boolean
    ${questions}=    Create Dictionary    a=${n}    b=${b}
    ${body}=    Ask JEF    ${ALERT}    ${questions}
    Should Be Equal As Numbers    ${body}[answers][a][noul]    ${body}[answers][b][probability]

Mixed Primitives Evaluate In A Single Request
    [Documentation]    All three types share one state, answered under caller-chosen ids.
    ${urgent}=    Noul Question    是否緊急？
    ${team}=      Choice Question    誰處理？    billing=付款    infra=網路
    ${sev}=       Score Question    嚴重度    低    中    高
    ${questions}=    Create Dictionary    urgent=${urgent}    team=${team}    sev=${sev}
    ${body}=    Ask JEF    ${ALERT}    ${questions}
    Length Should Be    ${body}[answers]    3
    Should Be Equal As Integers    ${body}[usage][outputTokens]    0

Object State Is Accepted
    ${state}=    Create Dictionary    alert=payout failed    count=${3}
    ${q}=    Noul Question    是否緊急？
    ${questions}=    Create Dictionary    urgent=${q}
    ${body}=    Ask JEF    ${state}    ${questions}
    Length Should Be    ${body}[answers]    1

Array State Is One Shared State Not A Batch
    [Documentation]    Three items in, still exactly one answer out.
    ${state}=    Create List    事件一    事件二    事件三
    ${q}=    Noul Question    是否緊急？
    ${questions}=    Create Dictionary    urgent=${q}
    ${body}=    Ask JEF    ${state}    ${questions}
    Length Should Be    ${body}[answers]    1

Unknown Question Type Is Rejected
    [Documentation]    Jev does not generate prose, and neither does JEF.
    ${q}=    Create Dictionary    type=freeform    instructions=寫一首詩
    ${questions}=    Create Dictionary    poem=${q}
    ${body}=    Ask JEF Expecting    ${ALERT}    ${questions}    422
    Dictionary Should Contain Key    ${body}[error]    code

Choice With A Single Option Is Rejected
    ${criteria}=    Create Dictionary    only=one
    ${q}=    Create Dictionary    type=choice    instructions=誰處理？    criteria=${criteria}
    ${questions}=    Create Dictionary    team=${q}
    Ask JEF Expecting    ${ALERT}    ${questions}    422

Numeric State Is Rejected
    ${q}=    Noul Question    是否緊急？
    ${questions}=    Create Dictionary    urgent=${q}
    Ask JEF Expecting    ${42}    ${questions}    422
