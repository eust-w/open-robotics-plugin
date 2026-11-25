#!/usr/bin/env python3
"""open-robotics-plugin installer: one skill source, many agent harnesses.

Usage:
  orp.py list
  orp.py install   --target TARGET [--scope user|project] [--dest DIR]
                   [--skills a,b,c] [--dry-run]
  orp.py uninstall --target TARGET [--scope user|project] [--dest DIR]

Stdlib only. Every install writes a manifest (.orp-manifest.json) so that
uninstall removes exactly what was installed, and edits to shared files
(CLAUDE.md, AGENTS.md, ...) are limited to a marked block.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS_SRC = REPO / "skills"
RULES_SRC = REPO / "rules"
SIDECARS_SRC = REPO / "sidecars"
MANIFEST = ".orp-manifest.json"
BEGIN = "<!-- BEGIN open-robotics-plugin -->"
END = "<!-- END open-robotics-plugin -->"
RULES_DIR_NAME = "_orp_rules"
SAFETY_RULE = "physical_robot_safety.md"

# Skills that move real hardware. Their safety rule must be reachable.
HARDWARE_SKILLS = {
    "so101-bringup", "i2rt-yam-bringup", "fr3-duo-control", "pick-and-place",
    "workspace-calibration", "visual-servoing", "iteration",
}


# --------------------------------------------------------------------------
# Skill discovery
# --------------------------------------------------------------------------
def parse_frontmatter(text: str) -> dict:
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    out = {"name": "", "description": ""}
    if not m:
        return out
    lines = m.group(1).split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        km = re.match(r"(name|description):\s*(.*)$", line)
        if km:
            key, val = km.group(1), km.group(2).strip()
            if val in (">-", ">", "|", "|-"):
                buf = []
                i += 1
                while i < len(lines) and (lines[i].startswith(" ") or not lines[i].strip()):
                    buf.append(lines[i].strip())
                    i += 1
                out[key] = " ".join(b for b in buf if b)
                continue
            out[key] = val.strip("'\"")
        i += 1
    return out


def discover_skills() -> dict[str, dict]:
    skills = {}
    for d in sorted(SKILLS_SRC.iterdir()):
        md = d / "SKILL.md"
        if md.is_file():
            fm = parse_frontmatter(md.read_text(encoding="utf-8"))
            skills[d.name] = {"dir": d, "name": fm["name"] or d.name,
                              "description": fm["description"]}
    return skills


# --------------------------------------------------------------------------
# File operations (all go through Plan so --dry-run is honest)
# --------------------------------------------------------------------------
class Plan:
    def __init__(self, dry: bool):
        self.dry = dry
        self.created: list[str] = []
        self.blocks: list[str] = []

    def write(self, path: Path, content: str | bytes):
        self.created.append(str(path))
        if self.dry:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_bytes(content)

    def block(self, path: Path, body: str):
        """Insert/replace the marked block in a shared file."""
        self.blocks.append(str(path))
        if self.dry:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        existing = strip_block(existing)
        sep = "\n\n" if existing.strip() else ""
        path.write_text(existing.rstrip() + sep + f"{BEGIN}\n{body.strip()}\n{END}\n",
                        encoding="utf-8")


def strip_block(text: str) -> str:
    return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", text, flags=re.S)


def rewrite_rule_links(text: str, depth: int) -> str:
    """Point `(../)+rules/` links at the sibling _orp_rules dir.

    depth = number of directories between the file and its skill dir.
    """
    prefix = "../" * (depth + 1) + RULES_DIR_NAME + "/"
    return re.sub(r"(?:\.\./)+rules/", prefix, text)


def copy_skill(plan: Plan, name: str, src: Path, dst: Path, rewrite: bool):
    for f in sorted(src.rglob("*")):
        if f.is_dir() or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(src)
        target = dst / rel
        if rewrite and f.suffix == ".md":
            text = f.read_text(encoding="utf-8")
            plan.write(target, rewrite_rule_links(text, len(rel.parts) - 1))
        else:
            plan.write(target, f.read_bytes())
        if f.suffix == ".py" or os.access(f, os.X_OK):
            if not plan.dry:
                target.chmod(f.stat().st_mode)


def copy_rules(plan: Plan, dst: Path):
    for f in sorted(RULES_SRC.glob("*.md")):
        plan.write(dst / f.name, f.read_text(encoding="utf-8"))


def save_manifest(plan: Plan, root: Path, target: str):
    if plan.dry:
        return
    data = {"target": target, "files": plan.created, "blocks": plan.blocks}
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
def index_block(skills: dict, skills_path: str, rules_path: str) -> str:
    lines = [
        "## Robotics skills (open-robotics-plugin)",
        "",
        "**Real-robot safety (mandatory):** before commanding ANY physical robot "
        f"motion, read `{rules_path}/{SAFETY_RULE}` and follow it. A moving robot "
        "can injure people; never run real-hardware motion without the user "
        "present and an emergency stop within reach. Simulation-only work does "
        "not need hardware confirmation.",
        "",
        f"Skills live in `{skills_path}/<name>/SKILL.md`. When a task matches a "
        "description below, read that SKILL.md fully before acting.",
        "",
    ]
    for name, s in skills.items():
        desc = s["description"]
        desc = desc if len(desc) <= 300 else desc[:297] + "..."
        hw = " **[real hardware]**" if name in HARDWARE_SKILLS else ""
        lines.append(f"- `{name}`{hw}: {desc}")
    return "\n".join(lines)


def native_skills(plan, skills, skills_dir: Path, memory_file: Path | None, rel_paths):
    """Harnesses that load SKILL.md directories natively."""
    for name, s in skills.items():
        copy_skill(plan, name, s["dir"], skills_dir / name, rewrite=True)
    copy_rules(plan, skills_dir / RULES_DIR_NAME)
    if memory_file is not None:
        plan.block(memory_file, safety_only_block(rel_paths))


def safety_only_block(rules_path: str) -> str:
    return (
        "## Robotics real-robot safety (open-robotics-plugin)\n\n"
        f"Before commanding ANY physical robot motion, read "
        f"`{rules_path}/{SAFETY_RULE}` and follow it. Never run real-hardware "
        "motion unless the user is present with an emergency stop in reach."
    )


def flat_install(plan, skills, root: Path, rel_root: str):
    """Copy skills+rules to <root>/orp/ for harnesses with no skill loader."""
    base = root / "orp"
    for name, s in skills.items():
        copy_skill(plan, name, s["dir"], base / "skills" / name, rewrite=False)
    copy_rules(plan, base / "rules")
    return f"{rel_root}/orp/skills", f"{rel_root}/orp/rules"


def t_claude(plan, skills, scope, dest):
    home = dest or (Path.home() / ".claude" if scope == "user" else Path.cwd() / ".claude")
    mem = home / "CLAUDE.md" if (scope == "user" or dest) else home.parent / "CLAUDE.md"
    rules_rel = str(home / "skills" / RULES_DIR_NAME)
    native_skills(plan, skills, home / "skills", mem, rules_rel)
    return home


def t_codex(plan, skills, scope, dest):
    # Best effort: Codex reads SKILL.md dirs plus AGENTS.md. Verify paths for
    # your Codex version; override with --dest.
    home = dest or (Path.home() / ".codex" if scope == "user" else Path.cwd() / ".agents")
    mem = home / "AGENTS.md" if (scope == "user" or dest) else home.parent / "AGENTS.md"
    rules_rel = str(home / "skills" / RULES_DIR_NAME)
    native_skills(plan, skills, home / "skills", mem, rules_rel)
    return home


def t_gemini(plan, skills, scope, dest):
    # Gemini CLI: GEMINI.md context + skills dir (best effort, see docs).
    home = dest or (Path.home() / ".gemini" if scope == "user" else Path.cwd() / ".gemini")
    mem = home / "GEMINI.md" if (scope == "user" or dest) else home.parent / "GEMINI.md"
    rules_rel = str(home / "skills" / RULES_DIR_NAME)
    native_skills(plan, skills, home / "skills", mem, rules_rel)
    return home


def t_antigravity(plan, skills, scope, dest):
    # Upstream layout, so ../../rules links and the sidecar work unmodified.
    home = dest or (Path.home() / ".gemini" / "config" / "plugins")
    root = home / "robotics"
    for name, s in skills.items():
        copy_skill(plan, name, s["dir"], root / "skills" / name, rewrite=False)
    copy_rules(plan, root / "rules")
    if SIDECARS_SRC.is_dir():
        for f in sorted(SIDECARS_SRC.rglob("*")):
            if f.is_file():
                plan.write(root / "sidecars" / f.relative_to(SIDECARS_SRC), f.read_bytes())
    manifest = REPO / "upstream-plugin.json"
    if manifest.exists():
        plan.write(root / "plugin.json", manifest.read_text(encoding="utf-8"))
    return root


def t_agents_md(plan, skills, scope, dest):
    """Generic: works for Codex, opencode, Zed, Aider-style, anything reading AGENTS.md."""
    root = dest or (Path.cwd() if scope == "project" else Path.home())
    skills_p, rules_p = flat_install(plan, skills, root, str(root))
    plan.block(root / "AGENTS.md", index_block(skills, skills_p, rules_p))
    return root


def per_skill_rules(plan, skills, root: Path, rule_dir: Path, ext: str, header_fn, rel_root):
    skills_p, rules_p = flat_install(plan, skills, root, rel_root)
    for name, s in skills.items():
        body = (
            f"Robotics skill `{name}`. When the task matches the description "
            f"above, read `{skills_p}/{name}/SKILL.md` fully before acting.\n"
        )
        if name in HARDWARE_SKILLS:
            body += (
                f"Real hardware: first read `{rules_p}/{SAFETY_RULE}`; never move a "
                "real robot without the user present and an e-stop in reach.\n"
            )
        plan.write(rule_dir / f"orp-{name}{ext}", header_fn(name, s["description"]) + body)


def t_cursor(plan, skills, scope, dest):
    root = dest or (Path.cwd() if scope == "project" else Path.home())
    def hdr(n, d):
        return f"---\ndescription: {json.dumps(d[:500])}\nalwaysApply: false\n---\n"
    per_skill_rules(plan, skills, root, root / ".cursor" / "rules", ".mdc", hdr, ".")
    plan.write(root / ".cursor" / "rules" / "orp-safety.mdc",
               "---\ndescription: Real-robot safety rules\nalwaysApply: true\n---\n"
               + safety_only_block("./orp/rules") + "\n")
    return root


def t_windsurf(plan, skills, scope, dest):
    root = dest or Path.cwd()
    def hdr(n, d):
        return f"---\ntrigger: model_decision\ndescription: {json.dumps(d[:500])}\n---\n"
    per_skill_rules(plan, skills, root, root / ".windsurf" / "rules", ".md", hdr, ".")
    plan.write(root / ".windsurf" / "rules" / "orp-safety.md",
               "---\ntrigger: always_on\n---\n" + safety_only_block("./orp/rules") + "\n")
    return root


def t_cline(plan, skills, scope, dest):
    root = dest or Path.cwd()
    skills_p, rules_p = flat_install(plan, skills, root, ".")
    plan.write(root / ".clinerules" / "orp-robotics.md",
               index_block(skills, skills_p, rules_p) + "\n")
    return root


def t_copilot(plan, skills, scope, dest):
    root = dest or Path.cwd()
    skills_p, rules_p = flat_install(plan, skills, root, ".")
    plan.block(root / ".github" / "copilot-instructions.md",
               index_block(skills, skills_p, rules_p))
    return root


def t_chatgpt(plan, skills, scope, dest):
    """Custom GPT / Project bundle. Upload by hand; ChatGPT cannot reach hardware."""
    root = (dest or Path.cwd() / "dist") / "chatgpt"
    rules_p = "knowledge/rules"
    instr = index_block(skills, "knowledge/skills", rules_p)
    instr += (
        "\n\nENVIRONMENT NOTE: ChatGPT cannot access USB/serial devices or the "
        "user's robot. Use these skills for MuJoCo simulation, planning, code "
        "generation, review and debugging (Code Interpreter). Give the user "
        "scripts to run locally for any real-hardware step, and require their "
        "confirmation per the safety rule."
    )
    if len(instr) > 7900:  # Custom GPT instructions limit is 8000 chars
        short = {n: {"description": s["description"][:120]} for n, s in skills.items()}
        instr = index_block(short, "knowledge/skills", rules_p)[:7500]
    plan.write(root / "instructions.md", instr + "\n")
    for name, s in skills.items():
        text = (s["dir"] / "SKILL.md").read_text(encoding="utf-8")
        plan.write(root / "knowledge" / "skills" / f"{name}.md",
                   rewrite_rule_links(text, 0).replace(f"../{RULES_DIR_NAME}/", "../rules/"))
    copy_rules(plan, root / "knowledge" / "rules")
    plan.write(root / "README.txt",
               "1) Create a Custom GPT (or Project). 2) Paste instructions.md into "
               "Instructions. 3) Upload everything under knowledge/ as Knowledge. "
               "4) Enable Code Interpreter. Scripts (.py) live in the repo; upload "
               "the ones you need or run them locally.\n")
    return root


TARGETS = {
    "claude": (t_claude, "Claude Code (~/.claude/skills or ./.claude/skills)"),
    "codex": (t_codex, "OpenAI Codex CLI (best effort paths, use --dest to override)"),
    "gemini": (t_gemini, "Gemini CLI (best effort paths, use --dest to override)"),
    "antigravity": (t_antigravity, "Google Antigravity (upstream plugin layout + sidecar)"),
    "agents-md": (t_agents_md, "Any AGENTS.md-aware agent (opencode, Zed, Codex, ...)"),
    "cursor": (t_cursor, "Cursor (.cursor/rules/*.mdc)"),
    "windsurf": (t_windsurf, "Windsurf (.windsurf/rules)"),
    "cline": (t_cline, "Cline / Roo (.clinerules)"),
    "copilot": (t_copilot, "GitHub Copilot (.github/copilot-instructions.md)"),
    "chatgpt": (t_chatgpt, "ChatGPT Custom GPT / Project bundle (manual upload)"),
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def find_manifest_root(target: str, scope: str, dest: Path | None) -> Path | None:
    plan = Plan(dry=True)
    skills = discover_skills()
    root = TARGETS[target][0](plan, skills, scope, dest)
    return root


def cmd_install(a) -> int:
    skills = discover_skills()
    if a.skills:
        want = [s.strip() for s in a.skills.split(",") if s.strip()]
        missing = [w for w in want if w not in skills]
        if missing:
            print(f"unknown skills: {', '.join(missing)}", file=sys.stderr)
            return 2
        skills = {k: v for k, v in skills.items() if k in want}
    plan = Plan(a.dry_run)
    dest = Path(a.dest).expanduser().resolve() if a.dest else None
    root = TARGETS[a.target][0](plan, skills, a.scope, dest)
    save_manifest(plan, root, a.target)
    verb = "Would write" if a.dry_run else "Wrote"
    print(f"{verb} {len(plan.created)} files, {len(plan.blocks)} marked blocks -> {root}")
    for b in sorted(set(plan.blocks)):
        print(f"  marked block in {b}")
    if a.target in ("claude", "codex", "gemini", "antigravity"):
        print("Start a NEW session so the agent reloads skills.")
    hw = [n for n in skills if n in HARDWARE_SKILLS]
    if hw:
        print("Hardware skills included. Read rules/physical_robot_safety.md before real-robot use.")
    return 0


def cmd_uninstall(a) -> int:
    dest = Path(a.dest).expanduser().resolve() if a.dest else None
    root = find_manifest_root(a.target, a.scope, dest)
    mf = root / MANIFEST
    if not mf.exists():
        print(f"no manifest at {mf}; nothing to uninstall", file=sys.stderr)
        return 1
    data = json.loads(mf.read_text(encoding="utf-8"))
    for f in data["files"]:
        p = Path(f)
        if p.exists():
            p.unlink()
    for b in data["blocks"]:
        p = Path(b)
        if p.exists():
            rest = strip_block(p.read_text(encoding="utf-8")).strip()
            if rest:
                p.write_text(rest + "\n", encoding="utf-8")
            else:
                p.unlink()  # we created it; nothing else in it
    mf.unlink()
    # prune empty dirs
    for f in sorted({Path(x).parent for x in data["files"]}, key=lambda p: -len(p.parts)):
        for d in [f, *f.parents]:
            try:
                d.rmdir()
            except OSError:
                break
    print(f"Removed {len(data['files'])} files and {len(data['blocks'])} marked blocks.")
    return 0


def cmd_list(_a) -> int:
    print("Targets:")
    for k, (_, d) in TARGETS.items():
        print(f"  {k:12} {d}")
    print("\nSkills:")
    for n, s in discover_skills().items():
        hw = "  [hardware]" if n in HARDWARE_SKILLS else ""
        print(f"  {n}{hw}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="orp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    for name, fn in (("install", cmd_install), ("uninstall", cmd_uninstall)):
        sp = sub.add_parser(name)
        sp.add_argument("--target", required=True, choices=sorted(TARGETS))
        sp.add_argument("--scope", choices=["user", "project"], default="project")
        sp.add_argument("--dest", help="override install root")
        if name == "install":
            sp.add_argument("--skills", help="comma-separated subset")
            sp.add_argument("--dry-run", action="store_true")
        sp.set_defaults(fn=fn)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
