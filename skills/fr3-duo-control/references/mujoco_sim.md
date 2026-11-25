# Franka Research 3 Duo Simulation & MuJoCo Studio Guide

Use this reference when running offline simulation trajectories, building
custom tabletop scenes, or visualizing the **FR3 Duo** live in **MuJoCo
Studio** before running physical hardware via `Fr3DuoRobot(cfg, mode="hardware")`.

---

## 1. Two Ways to Simulate FR3 Duo

| Mode | When to use | Entry point |
| :--- | :--- | :--- |
| **Control & Bring-up Dry-Run** | Offline collision/tabletop checks, `bringup.py --dry-run`, camera math, or `Fr3DuoRobot` tasks | `Fr3DuoRobot(cfg, mode="sim")` in [fr3_duo_robot.py](../scripts/fr3_duo_robot.py) |
| **Interactive 3D Studio / Scene Builder** | Custom objects/tasks built with `MjSpec`, live UI sliders, frame scrubbing (`mujoco-studio-sidecar` + `mujoco-spec-editing`) | `build_fr3_duo_spec()` / `build_fr3_duo_mjcf()` in [fr3_mjcf.py](../scripts/fr3_mjcf.py) |

Both use the same canonical `fr3_duo_with_spine` + dual Robotiq 2F-85 model,
16-DoF `state_16d` order `[left_q1..q7, left_grip, right_q1..q7, right_grip]`,
normalized `[0.0 = 85 mm open, 1.0 = 0 mm closed]` grippers,
`DEFAULT_REST_POSE_RAD` (MJCF keyframe `'home'`), and four camera names
(`head_camera_left`, `head_camera_right`, `wrist_camera_left`,
`wrist_camera_right`).

---

## 2. Offline Simulation with `Fr3DuoRobot`

Pass `mode="sim"` (using `Fr3DuoSimBackend`) to test planning, FK projections,
and camera rendering offline against the canonical fixed-environment scene
(`mj_forward` kinematics without dynamic `<freejoint/>` object physics):

```python
from fr3_config import load_robot_config
from fr3_duo_robot import Fr3DuoRobot

cfg = load_robot_config("config/robot_config.json")
with Fr3DuoRobot(cfg, mode="sim") as robot:
  obs = robot.observe_scene()  # returns SceneObservation (dict-compatible)
  state, frames = obs.state, obs.frames  # or read_state() + render_sim_cameras(width=1280, height=720)
  robot.activate_grippers()
  robot.move_arm_waypoints("left", [state.left.q_rad, q_target])  # returns MotionResult
  contact = robot.move_cartesian_until_contact("left", direction=(0.0, 0.0, -1.0), max_distance_m=0.03)  # returns ContactMoveResult
  robot.home("both")  # returns MotionResult; pass read_only=True in __init__ to block all actuation while capturing live HW state/cameras
```

1. **Compose scene automatically:** `Fr3DuoRobot(cfg, mode="sim")` calls
   `fr3_mjcf.build_fr3_duo_mjcf(cfg)` directly from `mujoco_menagerie`
   (`franka_fr3_duo/fr3_duo_with_spine.xml` + `robotiq_2f85/2f85.xml`).
   Standalone runs automatically sparse-clone and cache Menagerie into
   `$XDG_CACHE_HOME/mujoco_menagerie` (or `~/.cache/mujoco_menagerie`)—no manual
   clone or environment variable needed. The tabletop arena (`include_arena=True`)
   places an ALOHA `1.22 m x 0.74 m` (`table_width_m x table_depth_m`) surface
   `0.25 m` (`25 cm`, `table_front_offset_m`) ahead of the flat center bracket
   underside front lip (`x = 0.08437 m`), at `z_table_m = 0.163887 - (base_height_m - table_height_m)`
   (default `0.913887 m` base / `0.75 m` table -> `0.0 m`, matching
   `cfg.table_box_bounds()`), with pinch site offset `cfg.tcp_offset_z_m`
   (`0.145 m` stock 2F-85 vs `0.185 m` extended).
2. **Headless camera rendering:** Set `MUJOCO_GL=osmesa` (or `egl`) on headless
   Linux when calling `observe_scene(...)` or `render_sim_cameras(...)`.
3. **Dynamic object-manipulation sims (`<freejoint/>` props):**
   * Build your scene from `spec = build_fr3_duo_spec(cfg, include_arena=True,
     add_gravcomp=False)` (or `add_gravcomp=True` for Studio gravity-compensated
     playback), add props to `spec.worldbody`, and advance physics with
     `mujoco.mj_step`.
   * Clamp `RobotArmPlanner` paths with `clamp_q_to_padded_limits` (`0.100 rad`
     J4 margin) and validate trajectories with `generate_quintic_trajectory` and
     `Fr3DuoSafetyChecker`.
   * **Contact validation note:** `Fr3DuoSafetyChecker` forbids contacts between
     the arms and non-arm bodies. In dynamic object-manipulation sims,
     instantiate `Fr3DuoSafetyChecker` on the robot+tabletop
     `build_fr3_duo_spec(cfg)` model while stepping physics on the full scene
     `MjModel`.
4. **Keep sim-to-real parity:** Restrict perception and grasp checks strictly to
   the 4 RGB camera streams + Robotiq closure status (never read privileged
   object poses from `mj_data.xpos`).
5. **Episode recording & offline MuJoCo replay:** Wrap `sim` or `hardware`
   runs in `with robot.record_episode(out_dir, fps=10.0) as rec:` (or
   `start_recording()` / `stop_recording()`), switching phases with
   `rec.set_phase("grasp")`. Inspect logs with
   `load_recorded_episode(out_dir)` (`RecordedEpisode`) and step logged
   `state_16d` / `arms` poses back through MuJoCo with
   `replay_episode_in_sim(out_dir, output_mp4="replay.mp4")` (`ReplayResult`)
   from [fr3_recording.py](../scripts/fr3_recording.py) (re-exported by
   [fr3_duo_robot.py](../scripts/fr3_duo_robot.py)).

---

## 3. Canonical `MjSpec` Builder & Live MuJoCo Studio (`mujoco` Plugin)

When authoring interactive tabletop scenes or visual debugging workflows, load
**`mujoco-studio-sidecar`**, **`studio`**, and **`spec_editing`**
from the `robotics` plugin.

### Canonical `MjSpec` Contents
* `fr3_mjcf.build_fr3_duo_spec(cfg=None, *, menagerie_dir=None, include_arena=True, add_gravcomp=False)`
  returns a configurable `mujoco.MjSpec` with both `fr3v2` arms
  (`{left,right}/fr3_*`), tilted spine mount, dual Robotiq 2F-85 grippers
  (`pinch_site`), tabletop arena (`include_arena=True`), `'home'` keyframe
  (`DEFAULT_REST_POSE_RAD`), and calibrated head + wrist cameras.
* `fr3_mjcf.build_fr3_duo_mjcf(...)` compiles `(MjModel, mjcf_xml)` and strips
  unnamed `<default>` wrappers so `RobotArmPlanner` and `describe_robot` parse
  the XML cleanly.

### Minimal Live `SimPlugin` for MuJoCo Studio Web
Publish the `MjSpec` in `<workspace_dir>/sim_plugin.py` so Studio hot-reloads
the scene in the aux pane:

```python
from fr3_mjcf import build_fr3_duo_spec
from mujoco.studio import plugin as studio_plugin


class Fr3DuoSimPlugin(studio_plugin.SimPlugin):
  def on_load_model(self, event) -> bool:
    if event.path:
      return False
    spec = build_fr3_duo_spec(add_gravcomp=True)
    # Add tabletop props here with spec.worldbody.add_body(...)
    self.handle.set_model(spec)
    return True
```

Launch `<mujoco-studio-sidecar>/scripts/studiod.py --workspace_dir=<dir>` with
`skills/fr3-duo-control/scripts` on `PYTHONPATH`, then promote validated joint
trajectories to hardware via `Fr3DuoRobot(cfg, mode="hardware", dry_run=True)`
under `R1–R16` in [safety_rules.md](safety_rules.md).
