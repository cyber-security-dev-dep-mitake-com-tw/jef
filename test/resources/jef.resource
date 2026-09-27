*** Settings ***
Documentation    Shared keywords for JEF acceptance tests.
...              These drive the real HTTP surface, not the Python objects, so
...              they prove what an external SDK would actually experience.
Library          RequestsLibrary
Library          Collections
Library          OperatingSystem
Library          String

*** Variables ***
${JEF_BASE_URL}     http://127.0.0.1:8080
${SESSION}          jef

*** Keywords ***
Connect To JEF
    [Documentation]    Open a session against the server under test.
    Create Session    ${SESSION}    ${JEF_BASE_URL}    verify=${False}

Ask JEF
    [Documentation]    POST /v1/systemone and require success.
    [Arguments]    ${state}    ${questions}
    ${body}=    Create Dictionary    state=${state}    questions=${questions}
    ${resp}=    POST On Session    ${SESSION}    /v1/systemone    json=${body}
    Should Be Equal As Integers    ${resp.status_code}    200
    RETURN    ${resp.json()}

Ask JEF Expecting
    [Documentation]    POST /v1/systemone and require a specific failure status.
    [Arguments]    ${state}    ${questions}    ${status}
    ${body}=    Create Dictionary    state=${state}    questions=${questions}
    ${resp}=    POST On Session    ${SESSION}    /v1/systemone    json=${body}
    ...    expected_status=anything
    Should Be Equal As Integers    ${resp.status_code}    ${status}
    RETURN    ${resp.json()}

Noul Question
    [Arguments]    ${instructions}    ${type}=noul
    ${q}=    Create Dictionary    type=${type}    instructions=${instructions}
    RETURN    ${q}

Choice Question
    [Arguments]    ${instructions}    &{criteria}
    ${q}=    Create Dictionary    type=choice    instructions=${instructions}    criteria=${criteria}
    RETURN    ${q}

Score Question
    [Arguments]    ${instructions}    @{levels}
    ${q}=    Create Dictionary    type=score    instructions=${instructions}    criteria=${levels}
    RETURN    ${q}

Probabilities Should Sum To One
    [Arguments]    ${answer}
    ${values}=    Get Dictionary Values    ${answer}[probabilities]
    ${total}=    Evaluate    sum($values)
    Should Be True    abs(${total} - 1.0) < 1e-4
    ...    msg=probabilities must form a distribution, summed to ${total}

Median Latency
    [Documentation]    Median of five timed calls.
    ...                Median, not mean: one scheduling hiccup on a shared CI
    ...                runner would otherwise decide whether the build passes.
    [Arguments]    ${state}    ${questions}
    @{samples}=    Create List
    FOR    ${i}    IN RANGE    5
        ${start}=    Evaluate    time.perf_counter()    modules=time
        Ask JEF    ${state}    ${questions}
        ${end}=    Evaluate    time.perf_counter()    modules=time
        ${elapsed}=    Evaluate    ${end} - ${start}
        Append To List    ${samples}    ${elapsed}
    END
    ${median}=    Evaluate    sorted($samples)[len($samples) // 2]
    RETURN    ${median}

Metric Value
    [Documentation]    Read one Prometheus counter, 0.0 when not yet emitted.
    [Arguments]    ${name}
    ${resp}=    GET On Session    ${SESSION}    /metrics
    @{lines}=    Split To Lines    ${resp.text}
    FOR    ${line}    IN    @{lines}
        ${hit}=    Evaluate    $line.startswith("${name} ")
        IF    ${hit}
            ${value}=    Evaluate    float($line.rsplit(" ", 1)[1])
            RETURN    ${value}
        END
    END
    RETURN    ${0.0}
