# Harness notes

All skills are `SKILL.md` directories (YAML frontmatter `name` + `description`,
Markdown body, optional `scripts/` and `references/`). Harnesses that load that
format natively get the directories copied. Others get a generated rule or index
that tells the model when to read which `SKILL.md`.

Link rewriting: upstream links to `../../rules/<file>`. Installed copies put
rules in a sibling `_orp_rules/` directory and rewrite the links there. The
Antigravity target keeps upstream layout and does not rewrite.

## Verify before relying

I could not test every harness. Treat these as hypotheses and use `--dest`:

- **Codex CLI**: user skills in `~/.codex/skills`; project scope uses `./.agents/skills`.
- **Gemini CLI**: skills in `.gemini/skills`, context in `GEMINI.md`.
- **Cursor / Windsurf**: rules are short pointers; the model must choose to open
  the referenced SKILL.md, so results depend on rule-trigger behavior.

Verified here: the installer's own behavior (dry-run writes nothing, idempotent
marked blocks, exact uninstall, safety rule links resolve) via `tests/`.
Not verified here: any real robot, MuJoCo run, or live harness session.

## Adding a harness

Add a `t_<name>(plan, skills, scope, dest)` function to `tools/orp.py`, register
it in `TARGETS`, and extend `tests/test_orp.py` (the all-targets test picks it up).
