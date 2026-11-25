<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# Multi-Phase Task Execution (`Fr3Session`)

Use [`Fr3Session`](../scripts/fr3_session.py) to run multi-phase tasks inside
one persistent robot and planner session, avoiding per-stage `Fr3DuoRobot`
reconnects and MJCF rebuilds while recording unified phase history and optional
2x2 MP4s.

## 1. Persistent Session & Warm Planners

```python
from fr3_config import load_robot_config
from fr3_duo_robot import Fr3DuoRobot
from fr3_session import Fr3Session

cfg = load_robot_config("config/robot_config.json")
with Fr3Session(
    Fr3DuoRobot(cfg, mode="hardware", dry_run=False),
    record_dir="logs/run_01",
) as session:
  left_planner = session.planner("left", gripper_cmd=0.0)
  right_planner = session.planner("right", gripper_cmd=0.0)
  ...
```

- `session.planner(arm, gripper_cmd=...)` caches one `RobotArmPlanner` per arm
  and updates it via `set_gripper_opening(...)` when `gripper_cmd` changes.
- If `robot` is already connected when passed in, `Fr3Session` leaves its
  connection lifecycle to the caller; otherwise it connects on `__enter__` and
  closes on `__exit__`.
- `session.history` stores
  `PhaseResult(name, ok, attempts, elapsed_s, value, aborted, error)` per phase;
  `record_dir` tags MP4 segments via `recorder.set_phase(name)`.

## 2. Phase Gating & Retries (`run_phase`)

`session.run_phase(name, step, *, until=None, every_s=0.0, max_retries=0)` runs
`step(session)` until `until(val)` passes (polling `abort_watcher` before each
attempt):

- **Onboard-sensing gates:** Check `robot.read_state()` (joints, gripper
  ratio, external wrench) or geometry directly in `until(val)` when physical
  sensors suffice.
- **Visual decision gates:** Pair with
  [`scene_understanding.md`](scene_understanding.md) (`decide_scene` /
  `ask_scene_vqa`) when onboard sensing cannot confirm completion or safety.

```python
res = session.run_phase(
    "phase_1",
    step=lambda s: (
        s.robot.move_arm_waypoints("left", waypoints, abort_watcher=s.abort_watcher),
        s.robot.read_state(),  # or s.robot.observe_scene() when visual checks are needed
    )[-1],
    until=lambda state: ...,  # onboard state check or decide_scene(...) gate
    every_s=0.5,
    max_retries=2,
)
if not res.ok:
  raise RuntimeError(f"Phase {res.name!r} failed: {res}")
```
