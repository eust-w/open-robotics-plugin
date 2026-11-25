---
name: sim2real-iteration
description: >-
  Guides disciplined MuJoCo robotics sim-to-real controller iteration using
  robosim_gauntlet verification, preflight scene/goal audits, deterministic
  episode forking, paired Common Random Numbers, and adversarial robotics critic
  reviews. Use only when iterating on MuJoCo robot manipulation controllers,
  triaging contact/safety failures, or comparing controller candidates. Don't
  use for RL training (use mujoco-playground-rl) or non-robotics tuning.
metadata:
  publisher: google
---

# Sim-to-Real Controller Iteration

Use alongside `sim2real-gauntlet` (`src/robosim_gauntlet` on `sys.path`) when tuning or debugging a manipulation controller for physical deployment. Never claim an improvement from single-seed runs, `T0`-only checks, or unpaired rollouts. Reference consumers: `../sim2real-gauntlet/examples/playground_tasks.py` and `../sim2real-gauntlet/examples/color_blob_ik_controller.py`.

## Loop

1. **Preflight gate**: Audit compiled `MjModel` physics and positive/negative goal controls (`safety.goal_setter` for articulated fixtures) before iterating:
   ```bash
   python -m robosim_gauntlet.cli --task-py=../sim2real-gauntlet/examples/playground_tasks.py --robot=yam --tasks=cube_in_bowl --preflight-only
   ```
   Fix every reported issue (manipuland welds, non-robot actuators, manipuland margin/gap, contact overrides, or vacuous/unsatisfiable `task.success`).
2. **Unprivileged controller contract**: Drive `proxy.RemoteDeviceLayer(os.environ["GAUNTLET_URL"])` out of process (or an in-process `fn(dl)` behind `wrappers.AllowlistWrapper`) using only `task.deployment_keys()` and `<cam>_timestamp`. On `T4` terminal faults (`camera_freeze` or `dead_gripper`), observe post-onset state and exit with `{"status": "abort", "fault": "<fault>"}`; aborting without an active fault fails with `spurious_abort`.
3. **Inner-loop diagnosis (`episode.fork`)**: Inspect `result.grade.violations`, `result.grade.metrics`, and `result.log` (`poses_at`, `FORCES`) to find the earliest causal failure tick, then replay deterministically on the exact seeded `T0`-`T4` plant to `tick` and hand off to a candidate command or callable:
   ```python
   forked = episode.fork(result, task, tick, candidate_cmd_or_fn)
   ```
   Confirm fixes on full-episode sweeps after fork debugging.
4. **Paired CRN hill-climbing**: Set `GAUNTLET_CRN_SECRET` (`>= 16` bytes) for HMAC-locked evaluation and compare baseline (`v0`) vs candidate (`v1`) on identical `--task-py`, `--robot`, `--tiers=T0,T1,T2,T3,T4`, `--episodes`, and `--seed`:
   ```bash
   GAUNTLET_CRN_SECRET="$KEY" python -m robosim_gauntlet.cli \
     --task-py=../sim2real-gauntlet/examples/playground_tasks.py --robot=yam \
     --compare-controller="$V0_CMD" --controller="$V1_CMD" \
     --tiers=T0,T1,T2,T3,T4 --episodes=4 --seed=0 --output-json=/tmp/crn.json
   ```
   Or compare saved runs via `--baseline-json=v0.json --candidate-json=v1.json`. Verdicts (`scorecard.compare`): `INVALID` (fix key/seed/`scene_sha256` mismatch), `WORSE` (revert; safety regressions veto task gains), `INDISTINGUISHABLE` (keep simpler controller or increase `--episodes` to reach >= 6 discordant pairs), `BETTER` (promote only after passing [critic_rubric.md](references/critic_rubric.md), then run `--mode=real --mcap-log-dir=...`).
