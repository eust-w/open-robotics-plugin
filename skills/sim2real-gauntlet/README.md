# sim2real-gauntlet (`robosim_gauntlet`)

Verify a manipulation controller in MuJoCo before running it on the real robot.

Everything sim-specific lives in the task's reaf-sim `DeviceLayer` and in
`perturb.py` (privileged monitor devices plus one static write through
`device_layer.sim` per episode). Everything above it (`McapWrapper`, proxy,
controller, grading of logged readings, scorecard, CLI) runs on the same
`DeviceLayer` seam in `sim` and `real`.

| Module | Role |
|---|---|
| `task.py` | `Task` protocol, `SafetyLimits` |
| `perturb.py` | Tiers, CRN streams, `apply_static`, `MonitorDevice`, `DisturbanceDevice` |
| `wrappers.py` | `McapWrapper`, `Recorder`, `CommandLimitWrapper`, `CameraTimestampWrapper`, sim-only `SensorWrapper` / `FaultWrapper` / `LatencyWrapper` |
| `proxy.py` | Loopback HTTP proxy with scrubbed env, deployment-key allowlist, lazy `/read`, `RemoteDeviceLayer` |
| `episode.py` | `run_episode`, `run_suite`, `fork`, `settle` |
| `grade.py` | Success, force, joint, bystander, and T4 fault-abort grading |
| `preflight.py` | Scene audits and goal controls |
| `scorecard.py` | Per-tier summary and paired CRN comparison |
| `cli.py` | Turnkey `--task-py <path.py[:callable]>` loader and `--mode=sim\|real` runner |

Examples: `examples/playground_tasks.py` (YAM / SO-101 `cube_in_bowl`,
`peg_in_plate`, `two_block_stack`) and `examples/color_blob_ik_controller.py`
(unprivileged RGB-D IK controller, `--variant=v0|v1`).

Tests (from the `skills/sim2real-gauntlet` package root):

```sh
pytest tests/gauntlet_test.py tests/solvability_test.py
```
