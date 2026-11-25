# Run record: `mujoco-audit` installed via open-robotics-plugin into Claude Code

Date: 2026-11-25. Host: macOS (Darwin 24.6, arm64), system Python 3.9.6 (unused),
`uv 0.12.6`. Pure simulation: no robot, no API keys, no GPU.

## What was run

Skill `mujoco-audit` (copied by `tools/orp.py install --target claude --scope user`),
script `audit_scenarios.py`, which declares its own deps in a PEP 723 header, so
`uv run` provisioned MuJoCo and friends into an isolated cache. Nothing was
installed globally.

Two hand-written assets in `assets/`:

- `good_block`: 5 cm box, 50 g, collision geom in group 3, visual in group 2, root at origin.
- `flawed_block`: deliberately bad. Root body offset to z=0.3, 50 kg box, plus a
  0.2 m non-colliding sphere 0.4 m off-center that inflates the visual extent.

Commands (logs in `logs/`):

```bash
S=~/.claude/skills/mujoco-audit/scripts/audit_scenarios.py
uv run "$S" --xml_path=assets/good_block/model.xml --static_only
uv run "$S" --xml_path=assets/flawed_block/model.xml --static_only
uv run "$S" --xml_path=assets/<name>/model.xml --out_dir=audit_out/<name> --label=<name> --scenarios=all
```

## Results

| Check | good_block | flawed_block |
|---|---|---|
| Static audit | passed (1 warning: root body name != model name) | **failed**: root body has non-zero pos [0,0,0.3] |
| Dynamic scenario 1 (6-pose drop and hover) | PASS | **FAIL** (hover -180.91 mm, 647.59 N) |
| Scenario 8 (twin self-stack) | PASS | INFO |
| Scenario 10 (6x drop on thin slab) | PASS | PASS |
| Other scenarios | INFO | INFO |
| Reported mass | 50 g | 83,510 g (eff. density 668 g/cm3) |

Contact sheets: `good_scenarios.png`, `flawed_scenarios.png`.

## Timing

First run 2:41 wall clock, almost all `uv` downloading scipy, mujoco-core,
mujoco-studio (~40 MB). Afterwards a static check is ~2 s and the full 10-scenario
suite ~2-3 s per asset.

## Observations worth reusing

- The skill ran end to end from the installed copy with no path fixes needed.
- The suite caught the bad root offset in both the static pass and scenario 1.
- It did not flag the 50 kg mass as a failure; it only reported it. Mass sanity
  is left to the reader (the scorecard's `agent_guidance_disclaimer` says as much).
- Most scenarios report INFO, not PASS/FAIL, by design; they need engineering judgment.
- The static warning about root body vs model name is a naming convention, not a physics defect.

## Not covered

No real robot, no motion-planning or sim2real skill, no rendering via the
Studio sidecar, no run inside a live Claude session of an agent invoking the
skill by itself (this was driven manually from the shell). One asset pair is a
tiny sample, not a benchmark.
