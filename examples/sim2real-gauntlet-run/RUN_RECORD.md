# Run record: `sim2real-gauntlet` on a simulated I2RT YAM arm

Date: 2026-11-25. macOS arm64 (Darwin 24.6), Python 3.12.14 in an isolated uv
venv under /tmp. Pure simulation: no robot, no GPU, no API keys. Skill installed
into Claude Code with `tools/orp.py` (open-robotics-plugin). Driven from the
shell, not by an agent choosing the skill by itself.

## What was run

Task `cube_in_bowl` on the YAM arm (from the skill's `examples/playground_tasks.py`):
pick the red cube, put it in the blue bowl, leave the yellow bystander cube alone.
Controller: the packaged unprivileged RGB-D + IK controller
(`examples/color_blob_ik_controller.py`), two variants:

- `v0`: overhead perception only, coarse release.
- `v1`: wrist-camera refinement, one regrasp, camera-freeze and dead-gripper monitors.

Suite: tiers T0 (nominal), T1 (moderate perturbation), T2 (adversarial),
T3 (slip / push / knockout), T4 (self-model mismatch + injected camera freeze or
dead gripper). 4 episodes per tier, seed 0, so v0 and v1 see identical random
scenes (paired common random numbers). Logs: `logs/`. Reports: `yam_cube_in_bowl_{v0,v1,compare}.json`, `yam_preflight.json`.

## Results (20 paired episodes)

| Tier | v0 success | v1 success | v1 safe-pass |
|---|---|---|---|
| T0 | 0/4 | 4/4 | 4/4 |
| T1 | 0/4 | 4/4 | 4/4 |
| T2 | 0/4 | 4/4 | 3/4 (bystander moved once) |
| T3 | 0/4 | 4/4 | 4/4 |
| T4 | 0/4 | 2/4, 2 aborts | 4/4 |

- Preflight (scene invariants + goal controls): clean for all three YAM tasks.
- Paired comparison: **BETTER**, 18 episodes only v1 solved, 0 only v0 solved,
  p = 3.8e-6, 0 safety vetoes.
- v1 T4: camera_freeze and dead_gripper faults were aborted correctly; on two other
  T4 episodes it finished the task normally. Fault-abort precision 1.0 for both variants.
- v0 never succeeded, and on T4 it ignored a camera freeze (`fault_unhandled`).
- T0 single episode (`t0_cube_in_bowl_v1.json`): success, grasp force 18.7 N,
  impact force 18.7 N, no violations.

Wall time: about 20 s for the preflight or one episode, a few minutes per 20-episode suite.

## Getting it to run: the dependency path (the hard part)

The skill's own `pyproject.toml` could not be installed as written. What it took:

1. The skill depends on `mujoco-playground`, which is not on PyPI. The real
   package is named `playground` (github.com/google-deepmind/mujoco_playground).
2. `reaf` is on PyPI; `reaf-sim` is vendored in the playground repo
   (`third_party/reaf_sim`) and its PyPI version pins an older `warp-lang`.
3. playground's `[tool.uv.sources]` pull `gdm_robotics`, `mujoco_warp` (branch
   `deepmind`) and `i2rt` from git, and `mujoco` from a nightly index whose
   `mujoco-studio` pins cannot be resolved. GitHub git transport was also flaky here.
4. What worked: clone playground, gdm_robotics, mujoco_warp; then
   `uv pip install --no-sources <gdm_robotics> <mujoco_warp> <playground>/third_party/reaf_sim
   "reaf>=1.1.2" "torch>=2.7,<2.16" "mujoco[sysid]>=4.0.0" "warp-lang>=1.19.0,<1.19.1"
   absl-py dm-env numpy mcap pillow pytest`, then `uv pip install --no-deps -e <playground>`.
   This used stable PyPI mujoco 4.0.1 instead of the nightly. `i2rt` was not needed for sim.
5. The first run also sparse-cloned mujoco_menagerie (`i2rt_yam`, `aloha`) into
   the playground checkout's `.local/`.

## Caveats

- Success here means the task's goal predicate in simulation, not real-robot performance.
- `mujoco` stable 4.0.1 rather than the nightly the project targets; results may differ.
- 4 episodes per tier is small. Treat rates as a smoke test; the paired p-value is
  the more meaningful number and it only covers this one task and arm.
- One v1 T2 episode moved the bystander cube (safe-pass 3/4), so v1 is not safe everywhere.
- The skill's README refers to `tests/gauntlet_test.py`; no `tests/` directory
  exists in the upstream snapshot, so they were not run.
- Not tested: real mode, RL skills. SO-101 and the other two tasks are covered in the matrix below.
- `.mcap` logs for the first YAM `cube_in_bowl` suite only (41 files, 1.6 GB) are kept outside the repo in
  `~/workspace/orp-artifacts/sim2real-gauntlet-run/mcap` and ignored by git.

## Extended matrix: both arms, all three tasks (added later the same day)

Same suite for every row: tiers T0-T4, 4 episodes each (20 paired episodes), seed 0,
packaged controller v0 vs v1. No `.mcap` logs for these rows (disk was 99% full).
SO-101 preflight was clean for all three tasks (`so101_preflight.json`).
Success counts are per tier T0/T1/T2/T3/T4 out of 4.

| Arm | Task | v0 success | v1 success | v1 safe-pass | Paired verdict (p) |
|---|---|---|---|---|---|
| YAM | cube_in_bowl | 0/0/0/0/0 | 4/4/4/4/2 | 4/4/3/4/4 | BETTER (3.8e-6) |
| YAM | peg_in_plate | 0/0/0/0/0 | 4/4/4/2/3 | 4/4/4/2/4 | BETTER (3.8e-6) |
| YAM | two_block_stack | 0/0/0/0/0 | 4/4/4/3/1 | 4/4/4/3/4 | BETTER (3.8e-6) |
| SO-101 | cube_in_bowl | 4/3/2/4/2 | 4/4/4/4/2 | 4/4/3/4/4 | BETTER (2.4e-4) |
| SO-101 | peg_in_plate | 0/0/0/0/0 | 4/3/3/3/3 | 4/3/3/2/4 | BETTER (1.5e-5) |
| SO-101 | two_block_stack | 0/0/0/0/0 | 0/0/2/0/0 | 0/0/2/0/3 | INDISTINGUISHABLE (0.12) |

Safety vetoes: 0 in every comparison. Fault-abort precision was 1.0 for v1 on all six rows.

Observations:

- v1 beats v0 on 5 of 6 rows. SO-101 `two_block_stack` is the exception: v1 solved only
  2 of 20 episodes (both in T2), v0 none, and the difference is not significant.
  In that suite v1 never violated a safety limit on T0, T1 or T3; it simply did not
  complete the stack. T4 episodes there were safe because it aborted on injected
  faults or did nothing harmful.
- Hypothesis, not tested: the controller docstring says the wrist-camera refinement
  is tuned for the YAM, which could explain the weak SO-101 result. I did not
  investigate the cause.
- SO-101 `cube_in_bowl` is the one case where v0 works at all (T0 4/4), unlike on the YAM.
- v1 safety violations seen: bystander_moved (YAM cube_in_bowl T2), unsettled
  (YAM peg_in_plate T3), contact_force (SO-101 cube_in_bowl T2, peg_in_plate T3).
- Safe-pass rates for v1 across the six rows: 0.95, 0.90, 0.95, 0.95, 0.80, 0.25.
  The last is low mainly because failed stacks are not counted as safe passes.

Caveats for the matrix: 4 episodes per cell is small, so single cells are noisy;
the paired p-values are the sturdier signal. Same stable mujoco 4.0.1 caveat as above.
All episodes ran in simulation with the skill's example tasks and example controller.
