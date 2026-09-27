"""MCP server: typed decisions as tools an agent can call.

The caller here is a System 2 agent, and the point is to let it hand off the
judgements it is bad at being cheap and consistent about. An LLM asked "route
this alert" will produce a confident sentence; JEF produces a distribution over
the options you named, with a number saying how often that answer is right.

Every tool description below says what `confidence` is and is not, because an
agent reading `confidence: 0.94` will otherwise treat it as a 94% chance of
being correct. It is not: it is how peaked the distribution was. `p_correct` is
the calibrated one, and it is `null` when the server has no data to support it
-- which an agent must be able to tell apart from a low value.

    pip install "jef[mcp]"
    jef-mcp                      # stdio, for Claude Desktop / Cursor
    jef-mcp --transport streamable-http --port 8765
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

log = logging.getLogger("jef.mcp")

__all__ = ["build_server", "main"]

_UNCALIBRATED_NOTE = (
    "This server has no calibration loaded, so p_correct is unavailable for "
    "every question and scene gates cannot automate. Treat the numbers as "
    "ordering, not as probabilities of being correct."
)

_TEST_BACKBONE_NOTE = (
    "This server runs the 'hashing' test backbone. Its answers are well-formed "
    "and carry no meaning whatsoever. Do not act on them."
)


def _client(args: argparse.Namespace) -> Any:
    from jef_sdk import Jef, JefClient

    if args.local:
        return Jef(
            args.backbone,
            head=args.head,
            calibration=args.calibration,
            scenes=args.scenes,
            threads=args.threads,
        )
    return JefClient(base_url=args.url, timeout=args.timeout)


def _health(client: Any) -> dict[str, Any]:
    if hasattr(client, "health"):
        return dict(client.health())
    return {
        "model": client.model,
        "calibrated": client.calibrated,
        "test_backbone": client.engine.backbone.name == "hashing",
    }


def _caveats(client: Any) -> list[str]:
    """Conditions under which the answer should not be acted on.

    Returned in the payload rather than logged, because an agent never reads
    the server's logs.
    """
    health = _health(client)
    notes = []
    if health.get("test_backbone"):
        notes.append(_TEST_BACKBONE_NOTE)
    if health.get("calibrated") is False:
        notes.append(_UNCALIBRATED_NOTE)
    return notes


def _answer_payload(client: Any, result: Any) -> dict[str, Any]:
    payload = result.model_dump(exclude_none=True) if hasattr(result, "model_dump") else result
    answer = payload["answers"]["answer"]
    caveats = _caveats(client)
    return {"answer": answer, "model": payload.get("model"), **({"caveats": caveats} if caveats else {})}


def build_server(args: argparse.Namespace) -> Any:
    from mcp.server.mcpserver import MCPServer

    from jef_sdk import boolean, choice, noul, score

    server = MCPServer(
        name="jef",
        title="JEF — typed decisions",
        instructions=(
            "JEF answers bounded questions about evidence you supply and returns "
            "a probability distribution over the options you named. It never "
            "generates prose, and it cannot answer a question whose options you "
            "have not listed.\n\n"
            "Use it instead of reasoning when the decision is a choice between "
            "known options, a rating against known levels, or a yes/no about the "
            "evidence in front of you. It is cheap enough to ask several "
            "questions at once.\n\n"
            "Do not use it for knowledge questions. It judges the text you give "
            "it; it does not know things that are not in that text.\n\n"
            "Reading the results: `confidence` is how peaked the distribution "
            "was, not the chance of being right. `p_correct` is the calibrated "
            "probability the answer is correct, and it is null when the server "
            "cannot support one -- null means unknown, not low."
        ),
    )

    client = _client(args)

    @server.tool(
        title="Choose one option",
        description=(
            "Pick exactly one of the options you supply, given some evidence. "
            "Returns the choice, the probability of every option, confidence "
            "(distribution sharpness) and p_correct (calibrated chance of being "
            "right, or null if unavailable). The answer is always one of your "
            "options -- nothing is invented."
        ),
    )
    def jef_choice(state: str, question: str, options: dict[str, str]) -> dict[str, Any]:
        """Args:
        state: The evidence to judge.
        question: What to ask of it, e.g. "Which team should handle this?".
        options: Option key to a description the model reads. At least two.
            Key order is the order the model sees them in.
        """
        if len(options) < 2:
            raise ValueError("a choice needs at least two options")
        return _answer_payload(client, client.evaluate(state, {"answer": choice(question, **options)}))

    @server.tool(
        title="Rate against levels",
        description=(
            "Rate evidence against ordered levels. Returns a continuous score "
            "that may land between levels, because it is the expectation over "
            "your ordering. Levels must be given lowest first -- reversing them "
            "inverts the scale silently."
        ),
    )
    def jef_score(state: str, question: str, levels: list[str]) -> dict[str, Any]:
        """Args:
        state: The evidence to judge.
        question: What to rate, e.g. "How severe is this?".
        levels: Ordered levels, LOWEST FIRST, e.g. ["low", "medium", "high"].
        """
        if len(levels) < 2:
            raise ValueError("a score needs at least two ordered levels")
        return _answer_payload(client, client.evaluate(state, {"answer": score(question, *levels)}))

    @server.tool(
        title="Yes or no",
        description=(
            "The probability that a statement holds against the evidence. Near "
            "0.5 means the evidence does not say, which is different from a no."
        ),
    )
    def jef_yes_no(
        state: str, question: str, if_true: str | None = None, if_false: str | None = None
    ) -> dict[str, Any]:
        """Args:
        state: The evidence to judge.
        question: The statement to test, e.g. "Is this urgent?".
        if_true: Optional description of what a yes means.
        if_false: Optional description of what a no means.
        """
        builder = boolean if args.dialect == "boolean" else noul
        question_spec = builder(question, true=if_true, false=if_false)
        return _answer_payload(client, client.evaluate(state, {"answer": question_spec}))

    @server.tool(
        title="Ask several questions at once",
        description=(
            "Evaluate many questions against the same evidence in one call. They "
            "run in parallel and the evidence is read once, so ten questions "
            "cost barely more than one -- ask the speculative ones too rather "
            "than calling back. Each question is a JEF question object."
        ),
    )
    def jef_ask_many(state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Args:
        state: The evidence to judge.
        questions: Question id to question object, e.g.
            {"urgent": {"type": "noul", "instructions": "Is this urgent?"},
             "team": {"type": "choice", "instructions": "Who owns it?",
                      "criteria": {"soc": "monitoring", "infra": "hosts"}}}
        """
        if not questions:
            raise ValueError("questions must not be empty")
        result = client.evaluate(state, questions)
        payload = result.model_dump(exclude_none=True) if hasattr(result, "model_dump") else result
        caveats = _caveats(client)
        return {
            "answers": payload["answers"],
            "model": payload.get("model"),
            **({"caveats": caveats} if caveats else {}),
        }

    @server.tool(
        title="Run a decision scene",
        description=(
            "Run a named playbook: layers of typed questions with calibrated "
            "gates between them. Returns the verdict and a full decision trace "
            "-- the evidence, every question, the permitted answers, where the "
            "probability mass fell, and which gate fired and why. Check "
            "`human_review`: when true the scene declined to decide and a person "
            "must look at it."
        ),
    )
    def jef_run_scene(scene: str, state: str) -> dict[str, Any]:
        """Args:
        scene: Scene id, from jef_list_scenes.
        state: The evidence to judge.
        """
        trace = client.run_scene(scene, state)
        payload = trace.to_dict() if hasattr(trace, "to_dict") else dict(trace)
        caveats = _caveats(client)
        return {**payload, **({"caveats": caveats} if caveats else {})}

    @server.tool(
        title="List scenes",
        description="The playbooks this server has loaded, with their shape.",
    )
    def jef_list_scenes() -> dict[str, Any]:
        if hasattr(client, "scenes") and not callable(client.scenes):
            registry = client.scenes
            return {
                "scenes": [
                    {"id": name, "layers": len(registry.get(name).layers),
                     "questions": registry.get(name).question_count,
                     "description": registry.get(name).description}
                    for name in registry.names()
                ]
            }
        return {"scenes": client.scenes()}

    @server.tool(
        title="Server status",
        description=(
            "Whether this server's answers can be trusted. `calibrated` false "
            "means p_correct is unavailable and nothing should be automated on "
            "these numbers; `test_backbone` true means the answers are "
            "meaningless."
        ),
    )
    def jef_health() -> dict[str, Any]:
        health = _health(client)
        caveats = _caveats(client)
        return {**health, **({"caveats": caveats} if caveats else {})}

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jef-mcp", description="Serve JEF's typed decisions over MCP"
    )
    parser.add_argument("--url", default=os.environ.get("JEF_URL", "http://localhost:8080"))
    parser.add_argument("--local", action="store_true", help="run the engine in-process")
    parser.add_argument("--backbone", default=os.environ.get("JEF_BACKBONE", "hashing"))
    parser.add_argument("--head", default=os.environ.get("JEF_HEAD_PATH"))
    parser.add_argument("--calibration", default=os.environ.get("JEF_CALIBRATION_PATH"))
    parser.add_argument("--scenes", default=os.environ.get("JEF_SCENES_DIR"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dialect", choices=["noul", "boolean"], default="noul")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    # stdio is the transport, so logs must not go to stdout.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        server = build_server(args)
    except ImportError:
        log.error("jef-mcp needs the mcp extra: pip install 'jef[mcp]'")
        return 1

    if args.transport == "stdio":
        server.run()
    else:
        server.run(transport=args.transport, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
