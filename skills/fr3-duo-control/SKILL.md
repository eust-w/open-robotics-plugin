---
name: fr3-duo-control
description: Franka Duo robot only (Franka Research 3 / FR3 Duo dual-arm cell with Robotiq 2F-85 grippers, ZED Mini and RealSense D405 cameras). Brings up, homes, and controls it via pylibfranka on hardware or in MuJoCo sim. Use only when the user names Franka Duo, FR3 Duo, or fr3_duo.
metadata:
  publisher: google
---

# Franka Research 3 Duo Control (`fr3_duo`)

Operate and program a dual-arm **FR3 Duo** (`left`/`right` 7-DOF FR3 arms, Robotiq 2F-85 grippers, overhead ZED Mini stereo camera, and wrist RealSense D405 RGB cameras) under `R1–R16` in [safety_rules.md](references/safety_rules.md) and [hardware_limits.md](references/hardware_limits.md). For offline `mode="sim"`, canonical `MjSpec` scenes, and live MuJoCo Studio workflows, see [mujoco_sim.md](references/mujoco_sim.md).

## 1. Setup Gate

1. **Offline sim only:** Run `Fr3DuoRobot(cfg, mode="sim")` directly (`default_robot_config()` works with no hardware or `config/robot_config.json`; set `MUJOCO_GL=egl` or `osmesa` on headless Linux for `render_sim_cameras`).
2. **Missing/incomplete `config/robot_config.json`, or bring-up/homing/diagnostics:** Follow [bringup_checklist.md](references/bringup_checklist.md) via [bringup.py](scripts/bringup.py) (`uv run scripts/bringup.py`, e.g. `--dry-run`, `--step geometry`, `--up-to all`, `--step home --yes`; orchestrates [discover_hardware.py](scripts/discover_hardware.py), [check_cameras.py](scripts/check_cameras.py), [check_robot.py](scripts/check_robot.py)). Split operator prompts cleanly across **two turns** during Stage 1 rather than dumping all checks at once:
   * **Turn 1 (connection prerequisite -> run `--step preflight --read-only`):** Confirm only left/right FR3 FCI IPs (stating robot vs. operator left/right) and that joints are unlocked, FCI is enabled, and physical E-stop is within reach. Let `--step preflight` auto-check tilted Desk gravity/payload, USB/RS-485 ports, cameras, and conflicting processes before asking the operator to intervene.
   * **Turn 2 (Stage 1b geometry -> run `--step geometry`):** Show/attach [cuff_touch_explainer.gif](assets/cuff_touch_explainer.gif) and [fr3_duo_dimensions_reference.png](assets/fr3_duo_dimensions_reference.png) inline **before** asking the operator to touch the closed fingertips to the table front edge (so they see the required pose and footprint values), and use **single-arm closed-gripper FR3 cuff touch** (`uv run scripts/bringup.py --step geometry` or `--touch-geometry`; `--touch-arm auto` samples `left` when enabled else `right`, with `--touch-arm both` opt-in). If a calibrated config already exists on disk, interactive `--step geometry` prompts `[1] Reuse existing config | [2] Recalibrate (start fresh)` (`--touch-geometry` / `--yes` skip the reuse prompt). Confirm closed `tcp_offset_z_m` (`0.145 m` stock 2F-85 pads vs `0.185 m` extended fingers) and tabletop footprint (`1.22 x 0.74 m` ALOHA preset or custom), have the operator rest closed tips on the table front edge with the cuff button and release to `Idle` for zero-motion FCI FK sampling—eliminating floor/pedestal tape measurements (explicit CLI measurement flags + `--confirm-geometry` remain supported for non-interactive overrides). Defer the optional `tmux` prompt (`R11`) until live motion stages.
   Before live hardware actuation or downstream skill composition, `config/robot_config.json` must satisfy `validate_robot_config_complete(cfg) == []` (`arms.{left,right}.{ip,gripper_port}`, all 3 `cameras.*` entries, and `geometry_confirmed=True`).
3. **Hardware already verified (`R2`, complete config):** Skip bring-up motion checks; run your task with `Fr3DuoRobot(cfg, mode="hardware", dry_run=True)` (or `Fr3DuoSafetyChecker`), then wait for explicit `"go"` (`R1`) before `dry_run=False`.

## 2. Canonical Contract, Coordinate Frames & Runtime Control API

* **Base frame (`base` / `franka_fr3_duo_world`):** `+X` forward across tabletop (`[0.15, 0.95] m`), `+Y` robot-left (`[-0.55, +0.55] m`; operator-right facing pedestal), `+Z` up (`z_table_m` typically `[-0.03, +0.05] m`). `left` and `right` are from the **robot's perspective (not the operator's when facing the pedestal)**—state both when prompting a human.
* **Joints, Grippers & Tool Center Point (TCP):** `state_16d` order is `[left_q1..q7, left_grip, right_q1..q7, right_grip]` (`{side}/fr3_joint1..7`). TCP is `{side}/robotiq_gripper_{side}/pinch_site` (`0.145 m` default `tcp_offset_z_m`; `Y_site` is the jaw-closing axis, `X_site` is across the pad width parallel to a straddled cylinder, and `Z_site` is the approach axis); gripper `0.0 = 85 mm open`, `1.0 = 0 mm closed` (empty pinch stalls at `0.82–0.91`).
* **4 Canonical RGB Streams (`1280x720` USB/UVC):** `head_camera_left`, `head_camera_right` (split from `2560x720` ZED Mini stereo), `wrist_camera_left`, `wrist_camera_right` (`{side}/wrist_camera/wrist_camera_{side}` D405).
* **Python / `uv` Environment (`pylibfranka`):** `pylibfranka` ships pre-built Linux x86_64 wheels for Python `3.9–3.12` (`requires-python = ">=3.9,<3.13"` in [bringup.py](scripts/bringup.py)). Run `uv run scripts/bringup.py ...` directly, or pass `uv run --python 3.12 <task_script>.py` / create `.venv` with `--python 3.12` for custom task scripts.
* **Architecture (`Fr3DuoRobot` -> `Fr3DuoRealBackend` / `Fr3DuoSimBackend` -> Hardware Controllers + Services):**
  1. **Robot & Backends:**
     * [fr3_duo_robot.py](scripts/fr3_duo_robot.py): `Fr3DuoRobot(cfg, mode="sim"|"hardware", dry_run=False, read_only=False)` context manager validating moves via `Fr3DuoSafetyChecker` and dispatching to `Fr3DuoRealBackend` or `Fr3DuoSimBackend` (`observe_scene`, `read_state`, `recover_errors`, `render_sim_cameras`, `activate_grippers`, `move_gripper`, `move_arm_waypoints`, `move_bimanual_waypoints`, `move_cartesian_until_contact` -> `ContactMoveResult`, `home`, `record_episode`, `start_recording`, `stop_recording`). Pass `read_only=True` with `mode="hardware", dry_run=False` to capture live FCI state and real USB frames via `observe_scene()` / `record_episode()` while structurally blocking all actuation with `RuntimeError`. **Always call `robot.observe_scene(*, output_dir=None, width=1280, height=720)`** whenever you need synchronized live joints, FK poses, all 4 camera RGB frames, OpenCV camera calibrations (`K`, `T_base_cam`), and table bounds—rather than writing custom multi-import `observe_scene.py` scripts combining `Fr3DuoCameraController`, `read_arm_joints_or_rest`, and `get_camera_calibration_opencv`. It works identically across `mode="sim"`, `mode="hardware", dry_run=True`, and `mode="hardware", dry_run=False` (lazily opening/reusing `Fr3DuoCameraController` on hardware and closing it cleanly in `robot.close()`), and returns `{"timestamp_s", "mode", "dry_run", "state", "state_16d", "frames", "camera_meta", "calibration", "z_table_m", "table_box_bounds", "saved_paths"}` (saving 4 stream PNGs + `contact_sheet_2x2.png` + `observation.json` when `output_dir` is given). `Fr3DuoRealBackend` reads `state.current_errors` and attempts `pylibfranka.Robot.automatic_error_recovery()` on recoverable errors/reflexes; extend `fr3_arm_controller.py` + `fr3_duo_robot.py` when adding `pylibfranka` or real-backend features.
     * [fr3_recording.py](scripts/fr3_recording.py): `EpisodeRecorder` (`with robot.record_episode(out_dir, fps=10.0) as rec:` + `rec.set_phase(...)` / `rec.log_state(...)`), `load_recorded_episode`, and `replay_episode_in_sim` (re-exported by `fr3_duo_robot.py`). Records thread-safe `2x2` `episode_mosaic.mp4`, `episode_frames.jsonl`, `episode_summary.json`, `first_frame.png`, and `last_frame.png` without colliding with `ArmWorkerProcess` or RS-485 gripper polling, and supports offline MuJoCo kinematic playback.
     * [fr3_session.py](scripts/fr3_session.py): `Fr3Session(Fr3DuoRobot(...), record_dir=...)` & `PhaseResult`. Keeps `Fr3DuoRobot`, warm `RobotArmPlanner` instances (`session.planner("left"|"right", gripper_cmd=0.0)`), `AbortWatcher`, and optional `EpisodeRecorder` open across multi-stage tasks; `session.run_phase(name, step, *, until=None, every_s=0.0, max_retries=0)` tags `recorder.set_phase(name)`, polls `abort_watcher`, and retries/measures completion via `until`.
  2. **Hardware Controllers:**
     * [fr3_arm_controller.py](scripts/fr3_arm_controller.py): Core `pylibfranka` wrapper (`ArmWorkerProcess` isolated 1 kHz FCI worker, `read_pylibfranka_state`, `recover_pylibfranka_errors`, `audit_arm_desk_config`, `validate_arm_readiness`, `probe_arm_readonly`, `read_arm_joints_or_rest`).
     * [robotiq_gripper_controller.py](scripts/robotiq_gripper_controller.py): `RobotiqGripper`, `RobotiqStatus`, `resolve_serial_ports`, `select_gripper_ports`.
     * [fr3_camera_controller.py](scripts/fr3_camera_controller.py): `Fr3DuoCameraController` (freshness `<= 0.30 s`, side-by-side ZED stereo split), `CameraCalibration`, `render_contact_sheet`, `discover_uvc_devices`, `check_linux_usb_permissions`.
  3. **Services / Shared Utilities:**
     * [fr3_config.py](scripts/fr3_config.py): `RobotConfig`, `ArmConfig`, `ArmsConfig`, `CamerasConfig`, `DEFAULT_CONFIG_PATH`, `default_robot_config`, `load_robot_config`, `save_robot_config`, `normalize_robot_config`, `table_box_bounds`, `validate_robot_config_complete`, `detect_preempt_rt`, `resolve_realtime_config`.
     * [fr3_mjcf.py](scripts/fr3_mjcf.py): `build_fr3_duo_spec(cfg)` & `build_fr3_duo_mjcf(cfg)` (single canonical Menagerie `MjSpec` builder + sparse GitHub cache), `DEFAULT_REST_POSE_RAD`.
     * [fr3_kinematics.py](scripts/fr3_kinematics.py): `compute_arm_fk`, `compute_arm_pinch_from_flange`, `estimate_external_wrench_base`, `get_camera_calibration_opencv`, `project_base_points_opencv`.
     * [fr3_safety.py](scripts/fr3_safety.py): `FR3_HW_MAX_VEL_RAD_S`, `FR3_HW_MAX_ACC_RAD_S2` (`7.5` J1–J2, `9.5` J3–J7), `FR3_HW_MAX_JERK_RAD_S3` (`3500` / `4500`), `MIN_INTER_ARM_CLEARANCE_M` (`0.030`), `generate_quintic_trajectory`, `validate_joint_waypoints`, `AbortWatcher`, `Fr3DuoSafetyChecker`.

## 3. Composition Map (`rgb-object-localization` -> `motion-planning` -> `fr3_duo_robot` -> `visual-servoing` / `iteration`)

* **Perception & planning (`rgb-object-localization`, `motion-planning`, `visual-servoing`):** Call `obs = robot.observe_scene()` and pass `obs["calibration"]` (`K`, `T_base_cam`) to `ray_plane_contour_pose`, `build_fr3_duo_mjcf(cfg)` to `mp.RobotArmPlanner(..., arm="left"|"right")` (`add_other_arm_to_scene`), and `wrist_camera_{side}` (`intrinsics_tuple`, `T_ee_cam`) to `ibvs.Ibvs`.
* **Iteration & Studio / Sim (`iteration`, `mujoco-studio-sidecar`, `sim2real-gauntlet`):** Stage scripts with 4-cam MP4 capture; build interactive Studio scenes via `fr3_mjcf.build_fr3_duo_spec` ([mujoco_sim.md](references/mujoco_sim.md)) or grade offline.
* **Multi-stage session (`Fr3Session`):** For multi-phase tasks, prefer `with Fr3Session(Fr3DuoRobot(cfg, mode=...), record_dir=...) as session:` over reconnecting `Fr3DuoRobot` or rebuilding `RobotArmPlanner` per stage ([multi_phase_task_execution.md](references/multi_phase_task_execution.md), [scene_understanding.md](references/scene_understanding.md)); use `session.planner("left"|"right")` and `session.run_phase(name, step, until=...)` so `history` and phase-tagged 2x2 MP4s stay unified.
* **Boundary `q_rad` sync (`R5`):** Always re-read `q_now = robot.read_state()[arm]["q_rad"]` across skill boundaries (`visual-servoing` / `ibvs` -> `motion-planning`) or between arm and gripper calls (`move_gripper()` -> `move_arm_waypoints` / `move_bimanual_waypoints` / `plan_cartesian_stroke`). Opening/closing the gripper or table contact can deflect the arm, and `Fr3DuoRobot` prepends live `q_start` when `waypoints[0]` differs by `> 1e-4 rad`—so reusing a cached pre-gripper `q_at_grasp` as `waypoints[0]` can command an unintended plunge into the table and trip `cartesian_reflex`.

Import sibling `robotics` scripts relative to `skills/fr3-duo-control/` (`../rgb-object-localization/scripts`, `../motion-planning/scripts`, `../visual-servoing/scripts`; include `pillow` and `scipy` via `--with pillow --with scipy` or `.venv` when composing `motion-planning`):

```python
from pathlib import Path
import sys
import tempfile

_SKILLS_DIR = Path(__file__).resolve().parents[2]
for rel in ("fr3-duo-control/scripts", "motion-planning/scripts", "rgb-object-localization/scripts", "visual-servoing/scripts"):
  sys.path.insert(0, str(_SKILLS_DIR / rel))

from fr3_config import load_robot_config
from fr3_mjcf import build_fr3_duo_mjcf
from fr3_duo_robot import Fr3DuoRobot
from fr3_safety import clamp_q_to_padded_limits
import motion_planner as mp
import rgb_localization
import ibvs

cfg = load_robot_config("config/robot_config.json")
_, mjcf_xml = build_fr3_duo_mjcf(cfg)
tmp_xml = Path(tempfile.mkdtemp(prefix="fr3_mp_")) / "fr3_duo.xml"
tmp_xml.write_text(mjcf_xml, encoding="utf-8")
left_planner = mp.RobotArmPlanner(mjcf_path=str(tmp_xml), arm="left", gripper_cmd=0.0)
scene = mp.Scene3D()
with Fr3DuoRobot(cfg, mode="sim") as robot:  # pass dry_run=True to validate real moves offline first
  obs = robot.observe_scene()  # pass output_dir=... to also write 4 PNGs, contact_sheet_2x2.png, observation.json
  state, frames, cal = obs["state"], obs["frames"], obs["calibration"]
  bounds = obs["table_box_bounds"]
  scene.add_aligned_box("table", center=bounds["center"], half_extents=bounds["half_extents"], group="table")
  left_planner.add_other_arm_to_scene(scene, other_arm_q=state["right"]["q_rad"], other_gripper_cmd=state["right"]["gripper_ratio"])

  # 1. Localize target (OpenCV T_base_cam on frames["head_camera_left"]) or triangulate (MuJoCo mj_pose for mp.*)
  pose_3d = rgb_localization.ray_plane_contour_pose(mask_u8, cal["head_camera_left"]["K"], cal["head_camera_left"]["T_base_cam"], z_plane_m=obs["z_table_m"])
  table_pt = mp.intersect_ray_plane(640.0, 360.0, cal["head_camera_left"]["mj_pose"], cal["head_camera_left"]["mp_intrinsics"], plane_z=obs["z_table_m"])

  # 2. Plan & execute sparse waypoints (clamping J4 safety margin to [−2.9421, −0.2518])
  plan = left_planner.plan_arm(scene, state["left"]["q_rad"], q_pregrasp)
  safe_wps = [clamp_q_to_padded_limits(q).tolist() for q in plan["waypoints"]]
  robot.move_arm_waypoints("left", safe_wps)

  # 3. Close-range eye-in-hand IBVS alignment before grasp
  ibvs_left = ibvs.Ibvs(*cal["wrist_camera_left"]["intrinsics_tuple"])
  step = ibvs_left.compute_step(u, v, z_depth, u_goal, v_goal, cal["wrist_camera_left"]["T_ee_cam"])

  # 4. Grasp and re-read live q_rad before planning lift/retract across skill/gripper boundaries
  robot.move_gripper("left", 1.0)
  state_now = robot.read_state()
  q_now = state_now["left"]["q_rad"]
  lift_plan = left_planner.plan_cartesian_stroke(q_now, lift_pos, scene=scene)
  if not lift_plan["solved"]:
    raise RuntimeError(f"Lift plan failed: {lift_plan}")
  robot.move_arm_waypoints("left", [clamp_q_to_padded_limits(q).tolist() for q in lift_plan["waypoints"]])
```

## 4. Scene VQA & Decisions API Gates ([fr3_vqa.py](scripts/fr3_vqa.py))

* **`ask_scene_vqa(obs, question, cameras=...)`:** Open-ended multimodal VQA via `google-genai` (`gemini-4-hydrogen` by default) over `observe_scene()` frames and robot state.
* **`decide_scene(obs, questions, cameras=..., resolution="medium")`:** Closed-set calibrated yes/no (`noul`), `choice`, and `score` gates via `client.decisions.create(model="gemini-decision-1", ...)` (zero output tokens) for grasp verification, arm routing, and stability/clearance checks.
* **Offline VLM fallback (`GEMINI_API_KEY` unset or `google-genai` missing):** Both functions write `head_camera_*.png`, `wrist_camera_*.png`, `contact_sheet_2x2.png`, `observation.json`, and `vqa_request.json` / `decision_request.json` into `output_dir` (defaulting to the stage VQA logs directory) and return `status="fallback_saved"` so the host agent can inspect the images directly via `read_file`.
