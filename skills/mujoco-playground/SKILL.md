---
name: mujoco-playground
description: >-
  Runs and zero-shot transfers robot controllers across MuJoCo Playground sim
  and physical SO-101 or I2RT YAM DeviceLayers (plus sim G1 and Trossen WXAI)
  via PlaygroundSim2Real and setup_scene(spec), composing Studio, perception,
  IK, motion-planning, and visual-servoing skills. Use by default for SO-101,
  YAM, G1, or WXAI sim or sim-to-real controllers. Don't use for custom raw
  drivers, GPU RL training (use mujoco-playground-rl), or non-robotics tasks.
metadata:
  publisher: google
---

# MuJoCo Playground Sim-to-Real Composition ([`playground_sim2real.py`](scripts/playground_sim2real.py))

Compose live MuJoCo Studio visualization, perception, IK, collision-aware planning, and visual servoing on `mujoco_playground`'s out-of-the-box `so101` and `yam` 50 Hz `DeviceLayer` stacks with identical code in `mode="sim"` and `mode="real"`:

1. **Workspace Setup**:
   - Clone & sync: `git clone https://github.com/google-deepmind/mujoco_playground.git && cd mujoco_playground && make sync-cpu` (or `make sync` for CUDA 13 / `make sync-cu12` for Linux CUDA 12; add `--extra so101` or `--extra yam` alongside the same `--group`, and pass `--no-sync` to `uv run`). For composed skills, run `uv pip install --python .venv/bin/python ruckig opencv-python-headless scipy mink daqp`. Set `MUJOCO_GL=osmesa` only on headless Linux (omit on macOS).
   - Hardware bringup & CLI (also runnable without cloning via `uvx --python 3.13 --from "playground[so101]>=1.1"`): identify/calibrate SO-101 with `uv run --no-sync python -m mujoco_playground.robots.so101.real.{get_ports --identify --roles=follower, get_cameras --identify, get_calibration --roles=follower}`, and run tasks via `uv run --no-sync playground <train|play|deploy|teleop|sysid|view_log>`.

2. **Unified `sim` / `real` `DeviceLayer` Execution ([`PlaygroundSim2Real`](scripts/playground_sim2real.py))**:
   - Wrap `SO101Follower` / `SO101FollowerDevice` (`so101_aloha_table`, arm in `rad`, gripper in `[0, 100]`) or `YamFollower` / `YamFollowerDevice` (`yam_aloha_table`, arm in `rad`, gripper in `[0, 1]`) via `with PlaygroundSim2Real(robot="yam", mode="sim", setup_scene=...) as env:`, passing `setup_scene(spec)` (`mujoco-spec-editing` / `mujoco-python`) to customize objects, containers, conveyors, or camera/base poses for any scene.
   - Never bypass the follower `DeviceLayer` or read privileged object `mj_data.xpos` inside controllers; `PlaygroundSim2Real` steps through `follower/goal_position/*`, reads `follower/joint_position/*` + camera streams, and cosine-interpolates safely to `FOLLOWER_HOME` on `close()`.

3. **Live MuJoCo Studio Visualization & Hot-Reload (`mujoco-studio-sidecar` + `mujoco-studio`)**:
   - Load **`mujoco-studio-sidecar`** and **`mujoco-studio`** whenever interactively building scenes, controllers, or Studio UI panels.
   - `mujoco_playground/.venv/bin/python` already ships `mujoco.studio`:
     publish `build_playground_spec(robot="yam", setup_scene=...)` from a
     `SimPlugin` in `<workspace_dir>/sim_plugin.py` (`on_load_model` calling
     `self.handle.set_model(spec)` and returning `True`, plus
     `if self in event.plugins: self.handle.reload()` on `HotReloadEvent`),
     place `viewer_plugin.py` / `messages.py` alongside it, launch `studiod.py`
     with `<playground_dir>/.venv/bin/python`
     (`<mujoco-studio-sidecar>/scripts/studiod.py --workspace_dir=<dir>`,
     `IsDaemon: true`), include
     `[MuJoCo Studio](sidecar://robotics/mujoco_studio/)`, and verify
     `/_studio/status` after edits (or stream `PlaygroundSim2Real` rollouts to
     a Studio `ViewerHandle` via `env.attach_studio(handle)`).

4. **Composing `robotics` Plugin Skills Zero-Shot (`sim` -> `real`)**:
   - **`rgb-object-localization`**: Call
     `cam = env.camera_opencv("wrist_camera")` (computes `K`, `T_world_cam`, and
     `T_ee_cam` from measured `q_arm` via `mj_kinematics` + `mj_camlight` with
     MuJoCo-to-OpenCV `R @ diag([1, -1, -1])` applied; `cam["T_ee_cam"]` is
     `None` for fixed `extra_cameras`) and pass
     `(mask, cam["K"], cam["T_world_cam"], z_plane_m)` (tabletop at
     `z_plane_m = env.table_height_m` in Playground world frame plus object
     height offset) to `ray_plane_contour_pose` or two views to
     `triangulate_matched_points`.
   - **`i2rt-yam-ik` & Arm IK**: On `yam`, map world targets into the arm base frame (computed dynamically from `env.mj_model` via `env.t_world_base`) with `p_b = env.world_to_base_point(p_w)` before `YamIk(xml_path=env.robot_xml_path).make_topdown_grasp_pose(*p_b)` + `ik.solve(...)` (or transform full poses with `env.world_to_base_pose`). On 5-DOF `so101` (or either arm), call `env.solve_arm_ik(pos_world, approach_dir_world=(0, 0, -1), jaw_yaw_rad=...)`.
   - **`motion-planning` (`motion_planner` `mp`)**: Transform world obstacles into the robot MJCF root frame via `env.world_to_base_point` / `env.world_to_base_pose` (`table` at `z = 0.0` in base frame), plan free-space detours and linear Cartesian strokes with `planner = mp.RobotArmPlanner(mjcf_path=env.robot_xml_path, joint_names=env.arm_joint_names, gripper_cmd=env.gripper_normalized)` (`planner.plan_arm` / `planner.plan_cartesian_stroke`, `control_dt_s=env.control_period_s`), and stream via `env.execute_trajectory(res["q_trajectory"])`.
   - **`visual-servoing` (`Ibvs`)**: Compute unoccluded hover height with `Ibvs.principal_ray_hover_distance(cam["T_ee_cam"], env.approach_axis_ee)`, refine wrist alignment at hover via `Ibvs.ray_plane_contour_pose(mask, cam["fx_px"], cam["fy_px"], cam["cx_px"], cam["cy_px"], cam["T_world_cam"], z_plane_m)` (or 6-DOF `Ibvs.compute_step`), then descend, close gripper (`env.set_gripper(...)`), and lift.
