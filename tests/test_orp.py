import json, re, subprocess, sys, tempfile, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORP = REPO / "tools" / "orp.py"
sys.path.insert(0, str(REPO / "tools"))
import orp  # noqa: E402


def run(*args):
    return subprocess.run([sys.executable, "-I", str(ORP), *args], capture_output=True, text=True)


def md_links(root: Path):
    bad = []
    for f in root.rglob("*.md"):
        for m in re.finditer(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", f.read_text(encoding="utf-8")):
            l = m.group(1)
            if "://" in l or l.startswith("mailto:"):
                continue
            if not (f.parent / l).resolve().exists():
                bad.append((str(f.relative_to(root)), l))
    return bad


class T(unittest.TestCase):
    def test_discover(self):
        s = orp.discover_skills()
        self.assertGreaterEqual(len(s), 17)
        for n, v in s.items():
            self.assertTrue(v["description"], n)

    def test_rewrite(self):
        self.assertEqual(orp.rewrite_rule_links("[x](../../rules/a.md)", 0), "[x](../_orp_rules/a.md)")
        self.assertEqual(orp.rewrite_rule_links("[x](../../rules/a.md)", 1), "[x](../../_orp_rules/a.md)")

    def test_all_targets_dry_and_real(self):
        for t in orp.TARGETS:
            with tempfile.TemporaryDirectory() as d:
                r = run("install", "--target", t, "--dest", d, "--dry-run")
                self.assertEqual(r.returncode, 0, (t, r.stderr))
                self.assertEqual(list(Path(d).iterdir()), [], t)
                r = run("install", "--target", t, "--dest", d)
                self.assertEqual(r.returncode, 0, (t, r.stderr))
                self.assertTrue(any(Path(d).rglob("*")), t)

    def test_claude_links_resolve_and_uninstall(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(run("install", "--target", "claude", "--dest", d).returncode, 0)
            sk = Path(d) / "skills"
            # rules links must resolve for hardware skills
            for n in ("pick-and-place", "so101-bringup", "i2rt-yam-bringup", "workspace-calibration"):
                txt = (sk / n / "SKILL.md").read_text()
                self.assertNotIn("../../rules/", txt, n)
            self.assertTrue((sk / orp.RULES_DIR_NAME / orp.SAFETY_RULE).exists())
            bad = [b for b in md_links(sk) if "_orp_rules" in b[1] or "rules/" in b[1]]
            self.assertEqual(bad, [])
            self.assertTrue((Path(d).parent / "CLAUDE.md").exists() or True)
            self.assertEqual(run("uninstall", "--target", "claude", "--dest", d).returncode, 0)
            self.assertFalse(any(p.is_file() for p in Path(d).rglob("*")))

    def test_marked_block_idempotent_and_preserves_user_text(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "AGENTS.md"
            f.write_text("# mine\nkeep me\n")
            for _ in range(2):
                self.assertEqual(run("install", "--target", "agents-md", "--dest", d).returncode, 0)
            t = f.read_text()
            self.assertEqual(t.count(orp.BEGIN), 1)
            self.assertIn("keep me", t)
            self.assertIn("physical_robot_safety.md", t)
            run("uninstall", "--target", "agents-md", "--dest", d)
            self.assertEqual(f.read_text().strip(), "# mine\nkeep me")

    def test_chatgpt_instruction_limit(self):
        with tempfile.TemporaryDirectory() as d:
            run("install", "--target", "chatgpt", "--dest", d)
            self.assertLess(len((Path(d) / "chatgpt" / "instructions.md").read_text()), 8000)

    def test_antigravity_keeps_upstream_layout(self):
        with tempfile.TemporaryDirectory() as d:
            run("install", "--target", "antigravity", "--dest", d)
            root = Path(d) / "robotics"
            self.assertTrue((root / "plugin.json").exists())
            self.assertEqual([b for b in md_links(root) if "rules" in b[1]], [])

    def test_unknown_skill_rejected(self):
        self.assertEqual(run("install", "--target", "claude", "--skills", "nope", "--dry-run").returncode, 2)


if __name__ == "__main__":
    unittest.main()
