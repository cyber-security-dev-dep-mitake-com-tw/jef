# Agent skills

`jef/SKILL.md` teaches an agent *when* to reach for JEF and how to read what
comes back. It pairs with the MCP server, which gives the agent something to
call: without the skill the tools are four functions with no occasion for use,
and without the tools the skill is advice with nothing behind it.

## Claude Code

```bash
mkdir -p ~/.claude/skills
cp -r skills/jef ~/.claude/skills/
```

Or drop it in a project's `.claude/skills/` to scope it to one repository.

## Other agent environments

The file is plain markdown with YAML frontmatter, which is the common shape.
Codex and similar tools read the same structure; consult their docs for where
the directory lives.

## Why it exists separately from the tool descriptions

The MCP tool descriptions say what each tool does. The skill says which
questions are worth asking JEF at all, why `confidence` is not correctness, and
what to do when `p_correct` comes back null. That second set is judgement, and
it does not fit in a parameter description.
