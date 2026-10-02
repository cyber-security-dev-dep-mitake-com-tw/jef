"""`jef` — the command line for JEF.

Defaults to talking to a server on localhost; `--local` runs the engine
in-process instead. Both paths go through the same `jef_sdk` objects, so the
answers do not depend on which you pick.

Human-readable by default, `--json` for scripts. Exit codes are meant to be
branched on: see `jef scene --fail-on-human-review` and `jef validate`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .render import Style, render_answer, render_trace, warn_if_untrustworthy

__all__ = ["main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
#: `jef scene --fail-on-human-review` uses this so a playbook can branch on it.
EXIT_HUMAN_REVIEW = 3
#: `jef validate` uses this when a scene is malformed.
EXIT_INVALID = 4


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #


def read_state(raw: str) -> Any:
    """Resolve a state argument.

    `-` reads stdin, which is how an alert usually arrives: piped from a SIEM
    query or a previous playbook step. `@path` reads a file. Text that parses as
    a JSON object or array is sent as structure, because a model reading
    `{"host": x, "severity": 4}` sees the fields a human sees.
    """
    if raw == "-":
        raw = sys.stdin.read()
    elif raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")

    text = raw.strip()
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Genuinely text that happens to start with a brace.
            return text
    return text


def parse_options(pairs: list[str]) -> dict[str, str | None]:
    """`key=description` arguments, order preserved.

    Order matters: it is the index each option occupies in the returned
    distribution.
    """
    criteria: dict[str, str | None] = {}
    for pair in pairs:
        key, sep, description = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"option {pair!r} has no key")
        criteria[key] = description.strip() if sep and description.strip() else None
    return criteria


def styler(args: argparse.Namespace) -> Style:
    """Resolve --color once, so every command honours it the same way.

    `auto` hands None to Style, which then decides from isatty and NO_COLOR.
    """
    choice = getattr(args, "color", "auto")
    return Style(enabled=None if choice == "auto" else choice == "always")


def build_client(args: argparse.Namespace) -> Any:
    from jef_sdk import Jef, JefClient

    from .defaults import resolve_model

    if args.local:
        backbone, head, calibration = resolve_model(args.backbone, args.head, args.calibration)
        return Jef(
            backbone,
            head=head,
            calibration=calibration,
            scenes=args.scenes,
            threads=args.threads,
        )
    return JefClient(base_url=args.url, timeout=args.timeout)


def server_info(client: Any) -> dict[str, Any]:
    """Health, from whichever transport is in use."""
    if hasattr(client, "health"):
        return dict(client.health())
    return {
        "model": client.model,
        "calibrated": client.calibrated,
        "test_backbone": client.engine.backbone.name == "hashing",
    }


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_ask(args: argparse.Namespace) -> int:
    from jef_sdk import boolean, choice, noul, score

    questions: dict[str, Any] = {}

    if args.questions_file:
        raw = Path(args.questions_file).read_text(encoding="utf-8")
        loaded = json.loads(raw)
        if not isinstance(loaded, dict) or not loaded:
            print("questions file must be a JSON object of id → question", file=sys.stderr)
            return EXIT_USAGE
        questions.update(loaded)

    if args.choice:
        instructions, *options = args.choice
        if len(options) < 2:
            print("--choice needs at least two 'key=description' options", file=sys.stderr)
            return EXIT_USAGE
        questions["answer"] = choice(instructions, **parse_options(options))

    if args.score:
        instructions, *levels = args.score
        if len(levels) < 2:
            print("--score needs at least two levels, lowest first", file=sys.stderr)
            return EXIT_USAGE
        questions["answer"] = score(instructions, *levels)

    if args.yes_no:
        questions["answer"] = (boolean if args.dialect == "boolean" else noul)(args.yes_no)

    if not questions:
        print(
            "nothing to ask: pass --choice, --score, --yes-no or --questions-file", file=sys.stderr
        )
        return EXIT_USAGE

    client = build_client(args)
    result = client.evaluate(read_state(args.state), questions)
    payload = result.model_dump(exclude_none=True) if hasattr(result, "model_dump") else result

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return EXIT_OK

    style = styler(args)
    warn_if_untrustworthy(server_info(client), style)
    for qid, answer in payload["answers"].items():
        print(render_answer(qid, answer, style))
        print()
    usage = payload.get("usage", {})
    print(
        style.dim(
            f"{usage.get('inputTokens', 0)} input tokens, "
            f"{usage.get('outputTokens', 0)} output tokens · {payload.get('model')}"
        )
    )
    return EXIT_OK


def cmd_scene(args: argparse.Namespace) -> int:
    client = build_client(args)
    trace = client.run_scene(args.scene, read_state(args.state))
    payload = trace.to_dict() if hasattr(trace, "to_dict") else dict(trace)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        style = styler(args)
        print(render_trace(payload, style))

    if args.fail_on_human_review and payload.get("human_review"):
        return EXIT_HUMAN_REVIEW
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    """Lint scenes without a model, so it can run in CI on every commit."""
    from jef_cli.validate import discover_scenes, validate_files

    files, problems = discover_scenes([Path(p) for p in args.paths])
    problems += validate_files(files)

    if args.json:
        print(json.dumps([p.as_dict() for p in problems], ensure_ascii=False, indent=2))
    else:
        style = styler(args)
        for problem in problems:
            colour = style.red if problem.severity == "error" else style.yellow
            location = f"{problem.path}" + (f":{problem.line}" if problem.line else "")
            print(f"{colour(problem.severity)} {location}: {problem.message}")
        errors = sum(1 for p in problems if p.severity == "error")
        warnings = len(problems) - errors
        if problems:
            print(f"\n{errors} error(s), {warnings} warning(s)")
        else:
            print(style.green(f"{len(files)} scene(s) checked, no problems"))

    return EXIT_INVALID if any(p.severity == "error" for p in problems) else EXIT_OK


def cmd_schema(args: argparse.Namespace) -> int:
    """Emit the scene JSON Schema, generated from the model that validates them.

    Hand-writing it would let the schema and the validator disagree, and the
    schema is what editors use to tell an author their scene is wrong before
    they ever run it.
    """
    from jef_scene import Scene

    schema = Scene.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = (
        "https://raw.githubusercontent.com/cyber-security-dev-dep-mitake-com-tw/"
        "jef/main/docs/scene.schema.json"
    )
    schema["title"] = "JEF scene"
    schema["description"] = (
        "Layers of typed questions with calibrated confidence gates. "
        "See https://github.com/cyber-security-dev-dep-mitake-com-tw/jef"
    )
    rendered = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"

    if args.out:
        target = Path(args.out)
        if args.check:
            current = target.read_text(encoding="utf-8") if target.is_file() else ""
            if current != rendered:
                print(
                    f"{target} is out of date; regenerate with: jef schema --out {target}",
                    file=sys.stderr,
                )
                return EXIT_INVALID
            print(f"{target} is current")
            return EXIT_OK
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        print(f"wrote {target}")
    else:
        print(rendered, end="")
    return EXIT_OK


def cmd_models(args: argparse.Namespace) -> int:
    client = build_client(args)
    info = server_info(client)

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return EXIT_OK

    style = styler(args)
    print(f"model       {style.bold(str(info.get('model')))}")
    print(f"calibrated  {info.get('calibrated')}")
    print(f"backbone    {'hashing (test stub)' if info.get('test_backbone') else 'real'}")
    if info.get("runtime"):
        print(f"runtime     {info['runtime']}")
    warn_if_untrustworthy(info, style)
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("jef serve needs the server extra: pip install 'jef[server]'", file=sys.stderr)
        return EXIT_ERROR

    import os

    for key, value in (
        ("JEF_BACKBONE", args.backbone),
        ("JEF_HEAD_PATH", args.head),
        ("JEF_CALIBRATION_PATH", args.calibration),
        ("JEF_SCENES_DIR", args.scenes),
        ("JEF_THREADS", str(args.threads) if args.threads else None),
    ):
        if value:
            os.environ[key] = str(value)

    uvicorn.run("jef_server.factory:app", host=args.host, port=args.port, workers=args.workers)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jef",
        description="Typed decisions with calibrated confidence. No text generation.",
        epilog="States: plain text, '-' for stdin, '@file' for a file, or JSON for structure.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--url", default="http://localhost:8080", help="JEF server (default: %(default)s)"
    )
    common.add_argument(
        "--local", action="store_true", help="run in-process instead of calling a server"
    )
    common.add_argument(
        "--backbone",
        default="auto",
        help="[--local] 'auto' (default) uses the real backbone with the "
        "published weights, or falls back to the 'hashing' test stub when the "
        "torch extra is missing. Pass a model id or 'hashing' to pin it.",
    )
    common.add_argument("--head", help="[--local] head.npz, or an hf:// path")
    common.add_argument("--calibration", help="[--local] calibration.json, or an hf:// path")
    common.add_argument("--scenes", help="[--local] scene directory")
    common.add_argument("--threads", type=int, help="[--local] intra-op threads")
    common.add_argument("--timeout", type=float, default=120.0, help="request timeout in seconds")
    common.add_argument("--json", action="store_true", help="machine-readable output")
    common.add_argument("--color", choices=["auto", "always", "never"], default="auto")

    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", parents=[common], help="ask one or more typed questions")
    ask.add_argument("state", help="the evidence to judge")
    ask.add_argument(
        "--choice",
        nargs="+",
        metavar=("INSTRUCTIONS", "KEY=DESC"),
        help="pick one option; order is the option index",
    )
    ask.add_argument(
        "--score",
        nargs="+",
        metavar=("INSTRUCTIONS", "LEVEL"),
        help="rate against ordered levels, LOWEST FIRST",
    )
    ask.add_argument("--yes-no", metavar="INSTRUCTIONS", help="probability a statement holds")
    ask.add_argument(
        "--dialect",
        choices=["noul", "boolean"],
        default="noul",
        help="yes/no spelling to answer in (default: %(default)s)",
    )
    ask.add_argument("-f", "--questions-file", help="JSON object of id → question")
    ask.set_defaults(func=cmd_ask)

    scene = sub.add_parser(
        "scene", parents=[common], help="run a scene and print its decision trace"
    )
    scene.add_argument("scene", help="scene id")
    scene.add_argument("state", help="the evidence to judge")
    scene.add_argument(
        "--fail-on-human-review",
        action="store_true",
        help=f"exit {EXIT_HUMAN_REVIEW} when the verdict needs a human",
    )
    scene.set_defaults(func=cmd_scene)

    validate = sub.add_parser("validate", help="lint scene files; needs no model")
    validate.add_argument("paths", nargs="+", help="scene files or directories")
    validate.add_argument("--json", action="store_true")
    validate.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    validate.set_defaults(func=cmd_validate)

    schema = sub.add_parser("schema", help="emit the scene JSON Schema")
    schema.add_argument("--out", help="write here instead of stdout")
    schema.add_argument("--check", action="store_true", help="verify --out is current")
    schema.set_defaults(func=cmd_schema)

    models = sub.add_parser(
        "models", parents=[common], help="show the model and whether to trust it"
    )
    models.set_defaults(func=cmd_models)

    serve = sub.add_parser("serve", help="run the HTTP server (needs jef[server])")
    serve.add_argument("--host", default="127.0.0.1", help="default: %(default)s")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--workers", type=int, default=1)
    serve.add_argument("--backbone", default="auto")
    serve.add_argument("--head")
    serve.add_argument("--calibration")
    serve.add_argument("--scenes")
    serve.add_argument("--threads", type=int)
    serve.set_defaults(func=cmd_serve)

    return parser


def _is_connection_failure(exc: BaseException) -> bool:
    """Whether an exception means "nothing is listening there".

    Matched on httpx's class hierarchy by name rather than by import: the CLI
    must keep working when the SDK's transport is swapped, and importing httpx
    here to compare types would pull it in on every error path.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names = {base.__name__ for base in type(current).__mro__}
        if {"ConnectError", "ConnectTimeout", "ConnectionRefusedError"} & names:
            return True
        current = current.__cause__ or current.__context__
    return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from jef_cli import __version__

        print(__version__)
        return EXIT_OK
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        style = styler(args)
        print(style.red(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        # The commands default to a server on localhost, so "connection
        # refused" is the single most likely first-run failure -- and
        # `ConnectError: [Errno 111] Connection refused` says nothing about
        # which address, or that running in-process is an option.
        if _is_connection_failure(exc):
            command = getattr(args, "command", "<command>")
            options = [
                ("jef serve", "start one (needs the server extra)"),
                (f"jef {command} --local", "run the engine in this process"),
                ("--url URL", "point at a server elsewhere"),
            ]
            width = max(len(option) for option, _ in options)
            print(f"\nNo JEF server at {getattr(args, 'url', '?')}. Either:", file=sys.stderr)
            for option, what in options:
                print(f"  {option:<{width}}  {what}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
