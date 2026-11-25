---
name: sim2real-gauntlet
description: >-
  Verifies a robot manipulation controller in MuJoCo simulation (robosim_gauntlet)
  across perturbation tiers T0-T4 with .mcap logging, physical safety grading,
  and paired Common Random Numbers before real-robot deployment. Use only when
  verifying, hill-climbing, or debugging a MuJoCo robot manipulation controller,
  auditing a task scene/goal predicate, or forking a failed episode. Don't use
  for RL training (use mujoco-playground-rl) or non-robotics tasks.
metadata:
  publisher: google
---

# Sim-to-Real Gauntlet (`robosim_gauntlet`)

Library and CLI in `src/robosim_gauntlet` (put `src/` on `sys.path`; `cli.ensure_gl_backend()` preserves explicit `MUJOCO_GL` or selects packaged `egl`/`osmesa`). Reference consumers: [playground_tasks.py](examples/playground_tasks.py) and [color_blob_ik_controller.py](examples/color_blob_ik_controller.py).

## Workflow

1. **Task** (`task.Task` protocol): `build_device_layer(mode="sim"|"real", extra_devices)` returns `(numpy DeviceLayer, reset, close)` (`Impl.CPP` in sim; real homes on `reset()` and parks on `close()`). Also `deployment_keys()`, `success(poses, poses0)`, and `safety` (`SafetyLimits`: 3D contact wrenches + indirect crush, 35 N grasp cap, bimanual `arm_prefixes` / `gripper_tokens` / `command_to_joint`, and `drop_z_tol_m=None` / `require_release=False` / `goal_setter` for insertions, hinged fixtures, or tool-in-hand tasks).
2. **Controller**: either an out-of-process command driving `proxy.RemoteDeviceLayer(os.environ["GAUNTLET_URL"])` (prompt in `GAUNTLET_PROMPT`, printing `{"status": "done"|"abort", "fault": ...}` last) or an in-process callable `fn(dl) -> dict` wrapped in `wrappers.AllowlistWrapper` (`include_latency=False` default). Both see only `deployment_keys()` + `<cam>_timestamp` (upstream timestamps forwarded/delayed; camera reads tracked via `last_cam_read_tick`).
3. **Preflight**: `preflight.run(task)` audits scene invariants + negative/positive goal controls (`safety.goal_setter` for articulated fixtures).
4. **Sweep / CLI**: `episode.run_suite([task], cmd_or_fn, episodes=..., mcap_dir=...)` or `python -m robosim_gauntlet.cli --task-py=path.py[:fn] --robot=yam --controller="..." --output-json=v1.json --baseline-json=v0.json --mode=sim|real --mcap-log-dir=...` (set `GAUNTLET_CRN_SECRET` for HMAC CRN).
5. **Debug**: `episode.fork(result, task, tick, cmd_or_fn)` replays logged commands to `tick` deterministically, then hands off; `McapWrapper` writes `.mcap` logs via `mujoco_playground.real.create_logger` identically in `sim` and `real`.

## Tiers

| Tier | Perturbation |
|---|---|
| T0 | nominal |
| T1 / T2 | moderate / adversarial pose, table, mass, friction, gains, noise, hue |
| T3 | slip, mid-approach push, or post-place knockout (`idx % 3`) |
| T4 | self-model mismatch + 50% terminal fault: `camera_freeze` (frozen or restamped) or `dead_gripper` |

T4 faults must end in `{"status": "abort", "fault": <fault>}`; aborting without
a fault is `spurious_abort`. Verdicts: BETTER / WORSE / INDISTINGUISHABLE
(one-sided exact McNemar, alpha 0.025, >= 6 discordant pairs) with safety veto.
