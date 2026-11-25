# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified hardware/simulation backends and `Fr3DuoRobot` robot controller."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Any
import warnings
import weakref

from fr3_arm_controller import (
    ArmWorkerProcess,
    _recovery_result,
    _state_has_errors,
    audit_arm_desk_config,
    compute_ext_wrench_base_6d,
    compute_fk_pos_error_m,
    compute_link0_in_base_transform_for_model,
    disconnect_pylibfranka_robot,
    read_pylibfranka_state,
    recover_pylibfranka_errors,
)
import fr3_camera_controller
from fr3_camera_controller import (
    CANONICAL_CAMERAS,
    CameraFrame,
    build_sim_camera_frames,
    render_contact_sheet,
    scale_camera_calibration,
)
from fr3_config import (
    RobotConfig,
    normalize_robot_config,
    resolve_realtime_config,
    save_rgb_png,
    validate_robot_config_complete,
)
import fr3_kinematics
import fr3_mjcf
from fr3_recording import (
    EpisodeRecorder,
    EpisodeSummary,
    MosaicVideoSummary,
    MosaicVideoWriter,
    PhaseSlice,
    RecordedArmTelemetry,
    RecordedEpisode,
    RecordedFrame,
    ReplayResult,
    _DictCompatMixin,
    load_recorded_episode,
    record_mosaic_clip,
    replay_episode_in_sim,
)
from fr3_safety import (
    AbortWatcher,
    Fr3DuoSafetyChecker,
    clamp_q_to_padded_limits,
    generate_quintic_trajectory,
    simplify_joint_polyline,
    unpack_o_t_ee,
)
import numpy as np
import robotiq_gripper_controller

try:
  import mujoco  # pytype: disable=import-error
except ImportError:
  mujoco: Any = None

__all__ = [
    "ArmPoseState",
    "CameraMeta",
    "ContactMoveResult",
    "EpisodeRecorder",
    "EpisodeSummary",
    "Fr3DuoRealBackend",
    "Fr3DuoRobot",
    "Fr3DuoSimBackend",
    "MosaicVideoSummary",
    "MosaicVideoWriter",
    "MotionResult",
    "PhaseSlice",
    "RecordedArmTelemetry",
    "RecordedEpisode",
    "RecordedFrame",
    "ReplayResult",
    "RobotState",
    "SceneObservation",
    "TableBoxBounds",
    "load_recorded_episode",
    "record_mosaic_clip",
    "replay_episode_in_sim",
]


@dataclass
class ArmPoseState(_DictCompatMixin):
  """Full joint, TCP, wrench, gripper, and safety state for one FR3 arm."""

  q_rad: list[float]
  dq_rad: list[float]
  tau_j: list[float]
  tau_ext_base_6d: list[float]
  tcp_pos_base: list[float]
  tcp_rot_base: list[list[float]]
  o_t_ee_4x4: list[list[float]]
  fk_pos_error_m: float
  gripper_ratio: float
  gripper_status: dict[str, Any]
  current_errors: list[str]
  desk_audit: dict[str, Any]

  def to_dict(self) -> dict[str, Any]:
    """Converts the arm pose state to a plain dictionary."""
    return {
        "q_rad": [float(x) for x in self.q_rad],
        "dq_rad": [float(x) for x in self.dq_rad],
        "tau_j": [float(x) for x in self.tau_j],
        "tau_ext_base_6d": [float(x) for x in self.tau_ext_base_6d],
        "tcp_pos_base": [float(x) for x in self.tcp_pos_base],
        "tcp_rot_base": [list(r) for r in self.tcp_rot_base],
        "o_t_ee_4x4": [list(r) for r in self.o_t_ee_4x4],
        "fk_pos_error_m": float(self.fk_pos_error_m),
        "gripper_ratio": float(self.gripper_ratio),
        "gripper_status": dict(self.gripper_status),
        "current_errors": list(self.current_errors),
        "desk_audit": dict(self.desk_audit),
    }


@dataclass
class RobotState(_DictCompatMixin):
  """Synchronized bimanual state returned by `Fr3DuoRobot.read_state()`."""

  timestamp_s: float
  left: ArmPoseState
  right: ArmPoseState
  state_16d: list[float]

  def to_dict(self) -> dict[str, Any]:
    """Converts the bimanual robot state to a plain dictionary."""
    return {
        "timestamp_s": float(self.timestamp_s),
        "left": self.left.to_dict(),
        "right": self.right.to_dict(),
        "state_16d": [float(x) for x in self.state_16d],
    }


@dataclass
class CameraMeta(_DictCompatMixin):
  """Per-stream metadata attached to a `SceneObservation`."""

  backend: str
  width: int
  height: int
  timestamp_s: float | None = None
  age_s: float | None = None
  rectified: bool = False

  def to_dict(self) -> dict[str, Any]:
    """Converts camera metadata to a JSON-serializable dictionary."""
    out: dict[str, Any] = {
        "backend": str(self.backend),
        "width": int(self.width),
        "height": int(self.height),
    }
    if self.timestamp_s is not None:
      out["timestamp_s"] = float(self.timestamp_s)
    if self.age_s is not None:
      out["age_s"] = float(self.age_s)
    if self.rectified:
      out["rectified"] = True
    return out


@dataclass
class TableBoxBounds(_DictCompatMixin):
  """Axis-aligned table box center and half-extents in the base frame."""

  center: list[float]
  half_extents: list[float]

  def to_dict(self) -> dict[str, Any]:
    """Converts the table box bounds to a plain dictionary."""
    return {
        "center": [float(v) for v in self.center],
        "half_extents": [float(v) for v in self.half_extents],
    }


@dataclass
class SceneObservation(_DictCompatMixin):
  """Synchronized state, 4 RGB views, calibration, and table bounds."""

  timestamp_s: float
  mode: str
  dry_run: bool
  state: RobotState
  state_16d: list[float]
  frames: dict[str, np.ndarray]
  camera_meta: dict[str, CameraMeta]
  calibration: dict[str, Any]
  z_table_m: float
  table_box_bounds: TableBoxBounds
  saved_paths: dict[str, str] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Converts the scene observation to a plain dictionary."""
    return {
        "timestamp_s": float(self.timestamp_s),
        "mode": str(self.mode),
        "dry_run": bool(self.dry_run),
        "state": self.state.to_dict(),
        "state_16d": [float(x) for x in self.state_16d],
        "frames": dict(self.frames),
        "camera_meta": {
            k: v.to_dict() for k, v in self.camera_meta.items()
        },
        "calibration": {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in self.calibration.items()
        },
        "z_table_m": float(self.z_table_m),
        "table_box_bounds": self.table_box_bounds.to_dict(),
        "saved_paths": dict(self.saved_paths),
    }


@dataclass
class MotionResult(_DictCompatMixin):
  """Result returned by `move_arm_waypoints` / `move_bimanual_waypoints`."""

  ok: bool
  aborted: bool
  dry_run: bool
  reason: str
  unwound_to_start: bool
  steps_executed: int
  final_left_q_rad: list[float]
  final_right_q_rad: list[float]
  final_q_rad: list[float] | None = None

  def to_dict(self) -> dict[str, Any]:
    """Converts the motion result to a plain dictionary."""
    out: dict[str, Any] = {
        "ok": bool(self.ok),
        "aborted": bool(self.aborted),
        "dry_run": bool(self.dry_run),
        "reason": str(self.reason),
        "unwound_to_start": bool(self.unwound_to_start),
        "steps_executed": int(self.steps_executed),
        "final_left_q_rad": [float(x) for x in self.final_left_q_rad],
        "final_right_q_rad": [float(x) for x in self.final_right_q_rad],
    }
    if self.final_q_rad is not None:
      out["final_q_rad"] = [float(x) for x in self.final_q_rad]
    return out


@dataclass
class ContactMoveResult(_DictCompatMixin):
  """Result returned by `Fr3DuoRobot.move_cartesian_until_contact()`."""

  ok: bool
  contact_detected: bool
  aborted: bool
  dry_run: bool
  side: str
  reason: str
  steps_executed: int
  distance_moved_m: float
  final_force_n: float
  max_force_n: float
  final_tcp_pos_base: list[float]
  final_q_rad: list[float]

  def to_dict(self) -> dict[str, Any]:
    """Converts the contact move result to a plain dictionary."""
    return {
        "ok": bool(self.ok),
        "contact_detected": bool(self.contact_detected),
        "aborted": bool(self.aborted),
        "dry_run": bool(self.dry_run),
        "side": str(self.side),
        "reason": str(self.reason),
        "steps_executed": int(self.steps_executed),
        "distance_moved_m": float(self.distance_moved_m),
        "final_force_n": float(self.final_force_n),
        "max_force_n": float(self.max_force_n),
        "final_tcp_pos_base": [float(x) for x in self.final_tcp_pos_base],
        "final_q_rad": [float(x) for x in self.final_q_rad],
    }


def _build_bimanual_unwind_trajs(
    left_waypoints: Sequence[Sequence[float]],
    right_waypoints: Sequence[Sequence[float]],
    max_vel: float,
    max_acc: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Builds smooth quintic unwind trajectories from recorded path samples."""
  unwind_l = generate_quintic_trajectory(
      simplify_joint_polyline(left_waypoints),
      dt_s=0.001,
      max_vel_rad_s=max_vel,
      max_acc_rad_s2=max_acc,
  )
  unwind_r = generate_quintic_trajectory(
      simplify_joint_polyline(right_waypoints),
      dt_s=0.001,
      max_vel_rad_s=max_vel,
      max_acc_rad_s2=max_acc,
  )
  return unwind_l, unwind_r


def _resolve_sides(side: str | None) -> tuple[str, ...]:
  if side is None:
    return ("left", "right")
  if side not in ("left", "right"):
    raise ValueError(f"Invalid side={side!r}; expected 'left' or 'right'")
  return (side,)


def _to_json_safe(value: Any) -> Any:
  """Recursively converts dataclasses, numpy arrays/scalars, and Paths."""
  if hasattr(value, "to_dict") and callable(value.to_dict):
    return _to_json_safe(value.to_dict())
  if is_dataclass(value) and not isinstance(value, type):
    return _to_json_safe(asdict(value))
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, Mapping):
    return {str(k): _to_json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_to_json_safe(v) for v in value]
  return value


def _save_observation_outputs(
    output_dir: str | Path,
    observation: SceneObservation,
    contact_frames: Mapping[str, CameraFrame],
) -> dict[str, str]:
  """Writes stream PNGs, 2x2 contact sheet, and JSON summary to disk."""
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  saved_paths: dict[str, str] = {}
  for cam_name in CANONICAL_CAMERAS:
    cam_path = out_dir / f"{cam_name}.png"
    save_rgb_png(observation.frames[cam_name], cam_path)
    saved_paths[cam_name] = str(cam_path)

  sheet_path = out_dir / "contact_sheet_2x2.png"
  save_rgb_png(render_contact_sheet(contact_frames), sheet_path)
  saved_paths["contact_sheet_2x2"] = str(sheet_path)

  obs_json_path = out_dir / "observation.json"
  saved_paths["observation_json"] = str(obs_json_path)
  summary = {
      k: v
      for k, v in observation.to_dict().items()
      if k not in ("frames", "saved_paths")
  }
  summary["saved_paths"] = dict(saved_paths)
  obs_json_path.write_text(
      json.dumps(_to_json_safe(summary), indent=2) + "\n",
      encoding="utf-8",
  )
  return saved_paths


def _safe_close(obj: Any) -> None:
  close_fn = getattr(obj, "close", None)
  if callable(close_fn):
    try:
      close_fn()
    except Exception:  # pylint: disable=broad-exception-caught
      pass


class Fr3DuoRealBackend:
  """Manages 1 kHz `ArmWorkerProcess` workers and Robotiq RS-485 grippers."""

  def __init__(
      self,
      config: RobotConfig,
      *,
      realtime_cfg_name: str,
      read_only: bool = False,
      robot_factory: Any = None,
      gripper_factory: Any = None,
      camera_factory: Any = None,
  ) -> None:
    """Initializes hardware arm workers and optional gripper/camera setup."""
    self.config = config
    self.realtime_cfg_name = realtime_cfg_name
    self.read_only = bool(read_only)
    self._robot_factory = robot_factory
    self._gripper_factory = gripper_factory
    self._camera_factory = camera_factory
    self.workers: dict[str, ArmWorkerProcess] = {}
    self.injected_robots: dict[str, Any] = {}
    self.grippers: dict[str, Any] = {}
    self.camera_controller: Any = None

  def connect(self, realtime_cfg_name: str) -> None:
    """Connects enabled arm workers/factories and Robotiq serial grippers."""
    self.realtime_cfg_name = realtime_cfg_name
    try:
      for side, arm_cfg in self.config.arms.items():
        if not arm_cfg.enabled:
          continue
        if self._robot_factory is not None:
          self.injected_robots[side] = self._robot_factory(
              side, arm_cfg.ip, realtime_cfg_name
          )
        else:
          worker = ArmWorkerProcess(side, arm_cfg.ip, realtime_cfg_name)
          worker.start()
          self.workers[side] = worker

        g_port = arm_cfg.gripper_port.strip()
        if self._gripper_factory is not None:
          g_inst = self._gripper_factory(side, g_port)
          connect_fn = getattr(g_inst, "connect", None)
          if callable(connect_fn):
            connect_fn()
          self.grippers[side] = g_inst
        elif g_port:
          gripper = robotiq_gripper_controller.RobotiqGripper(
              port=g_port,
              baudrate=int(arm_cfg.gripper_baudrate),
              slave_id=int(arm_cfg.gripper_slave_id),
          )
          gripper.connect()
          self.grippers[side] = gripper

        if not self.read_only:
          rec = self.recover_errors(side).get(side)
          if rec and rec.get("had_errors") and not rec.get("recovered"):
            warnings.warn(
                f"FR3 {side} arm has unrecovered FCI errors on connect: "
                f"errors_after={rec.get('errors_after')}, "
                f"error={rec.get('error')}",
                RuntimeWarning,
                stacklevel=2,
            )
    except Exception:
      self.close()
      raise

  def close(self) -> None:
    """Stops child arm workers, disconnects robots, and closes grippers."""
    try:
      for worker in list(self.workers.values()):
        try:
          worker.stop()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      for robot in list(self.injected_robots.values()):
        disconnect_pylibfranka_robot(robot)
      for gripper in list(self.grippers.values()):
        _safe_close(gripper)
      if self.camera_controller is not None:
        _safe_close(self.camera_controller)
    finally:
      self.workers.clear()
      self.injected_robots.clear()
      self.grippers.clear()
      self.camera_controller = None

  def capture_cameras(self) -> dict[str, CameraFrame]:
    """Lazily opens camera controller on first call and captures all views."""
    if self.camera_controller is None:
      ctrl = (
          self._camera_factory(self.config)
          if self._camera_factory is not None
          else fr3_camera_controller.Fr3DuoCameraController(
              self.config.cameras
          )
      )
      open_fn = getattr(ctrl, "open", None)
      if callable(open_fn):
        try:
          open_fn()
        except Exception:
          _safe_close(ctrl)
          raise
      self.camera_controller = ctrl
    return self.camera_controller.capture_all()

  def get_rectified_camera_calibration(self) -> dict[str, Any]:
    """Returns post-rectification head-camera intrinsics if active."""
    if self.camera_controller is None:
      return {}
    getter = getattr(self.camera_controller, "get_rectified_calibration", None)
    if callable(getter):
      return getter()
    return {}

  def has_arm(self, side: str) -> bool:
    """Returns True if `side` has an active worker or injected robot."""
    return side in self.workers or side in self.injected_robots

  def read_arm_state(self, side: str) -> dict[str, Any]:
    """Reads the latest FCI state dictionary for arm `side`."""
    if side in self.workers:
      return self.workers[side].read_state()
    return read_pylibfranka_state(self.injected_robots[side])

  def recover_errors(
      self, side: str | None = None
  ) -> dict[str, dict[str, Any]]:
    """Checks for FCI errors/reflex and runs automatic recovery if needed."""
    out: dict[str, dict[str, Any]] = {}
    for arm_side in _resolve_sides(side):
      if arm_side in self.workers:
        worker = self.workers[arm_side]
        recover_fn = getattr(worker, "recover_errors", None)
        if callable(recover_fn):
          out[arm_side] = recover_fn()
        else:
          has_err, errs = _state_has_errors(worker.read_state())
          out[arm_side] = (
              _recovery_result(
                  had_errors=True,
                  errors_before=errs,
                  recovered=False,
                  errors_after=errs,
                  error="worker.recover_errors() is unavailable",
              )
              if has_err
              else _recovery_result()
          )
      elif arm_side in self.injected_robots:
        out[arm_side] = recover_pylibfranka_errors(
            self.injected_robots[arm_side]
        )
    return out

  def activate_grippers(
      self,
  ) -> dict[str, robotiq_gripper_controller.RobotiqStatus]:
    """Activates all connected Robotiq grippers."""
    return {side: g.activate() for side, g in sorted(self.grippers.items())}

  def move_gripper(
      self,
      arm: str,
      closure_0_to_1: float,
      *,
      speed: float = 0.5,
      force: float = 0.5,
      wait: bool = True,
  ) -> robotiq_gripper_controller.RobotiqStatus:
    """Commands normalized gripper closure `[0.0, 1.0]` on `arm`."""
    return self.grippers[arm].move_normalized(
        closure_0_to_1, speed=speed, force=force, wait=wait
    )

  def _run_worker_trajectories(
      self,
      trajs: Mapping[str, np.ndarray],
      *,
      poll_timeout_s: float,
      abort_watcher: AbortWatcher | None,
      error_prefix: str,
  ) -> tuple[dict[str, dict[str, Any]], bool]:
    for side, traj in trajs.items():
      if side in self.workers:
        self.workers[side].send_trajectory_async(traj)
    replies: dict[str, dict[str, Any]] = {}
    aborted = False
    while len(replies) < len(self.workers):
      if (
          not aborted
          and abort_watcher is not None
          and abort_watcher.poll()
      ):
        aborted = True
        for w in self.workers.values():
          w.signal_abort()
      for side, w in self.workers.items():
        if side not in replies:
          rep = w.poll_done(poll_timeout_s)
          if rep is not None:
            replies[side] = rep
            if not rep.get("ok") and not aborted:
              aborted = True
              for other_side, other_w in self.workers.items():
                if other_side != side and other_side not in replies:
                  other_w.signal_abort()
    for side, rep in replies.items():
      if not rep.get("ok"):
        raise RuntimeError(
            f"{error_prefix} on {side} arm: {rep.get('error')}"
        )
    return replies, aborted

  def execute_bimanual_trajectory(
      self,
      traj_left: np.ndarray,
      traj_right: np.ndarray,
      q_left_start: np.ndarray,
      q_right_start: np.ndarray,
      *,
      max_vel: float,
      max_acc: float,
      abort_watcher: AbortWatcher | None = None,
  ) -> bool:
    """Executes dual-arm 1 kHz splines and unwinds cleanly on abort."""
    for side in tuple(self.workers.keys()):
      rec = self.recover_errors(side).get(side)
      if rec and rec.get("had_errors") and not rec.get("recovered"):
        raise RuntimeError(
            f"FR3 {side} arm has unrecoverable FCI errors before trajectory "
            f"execution: errors_after={rec.get('errors_after')}, "
            f"error={rec.get('error')}"
        )

    replies, aborted = self._run_worker_trajectories(
        {"left": traj_left, "right": traj_right},
        poll_timeout_s=0.005,
        abort_watcher=abort_watcher,
        error_prefix="Hardware trajectory error",
    )

    if aborted:
      l_rep = replies.get("left", {})
      r_rep = replies.get("right", {})
      l_trav = list(l_rep.get("travelled", []))
      r_trav = list(r_rep.get("travelled", []))
      l_qd = (l_rep.get("state") or {}).get("q_d")
      r_qd = (r_rep.get("state") or {}).get("q_d")
      l_raw = ([l_qd] if l_qd else []) + l_trav[::-1] + [q_left_start.tolist()]
      r_raw = ([r_qd] if r_qd else []) + r_trav[::-1] + [q_right_start.tolist()]
      unwind_l, unwind_r = _build_bimanual_unwind_trajs(
          l_raw, r_raw, max_vel, max_acc
      )
      self._run_worker_trajectories(
          {"left": unwind_l, "right": unwind_r},
          poll_timeout_s=0.01,
          abort_watcher=None,
          error_prefix="Hardware unwind trajectory error",
      )

    return aborted


class Fr3DuoSimBackend:
  """Headless MuJoCo simulation and offscreen rendering backend."""

  def __init__(self, mj_model: Any, mj_data: Any) -> None:
    """Indexes MuJoCo arm joints, actuators, and initial gripper status."""
    if mujoco is None:
      raise RuntimeError(
          "`mujoco` is required for `Fr3DuoSimBackend`. "
          "Install it with `pip install mujoco`."
      )
    self.mj_model = mj_model
    self.mj_data = mj_data
    self._render_lock = threading.Lock()
    self._renderers: dict[int, tuple[Any, tuple[int, int], Any]] = {}
    self._arm_qpos_addrs: dict[str, list[int]] = {}
    self._arm_dof_addrs: dict[str, list[int]] = {}
    self._arm_act_ids: dict[str, list[int]] = {}
    self._gripper_act_id: dict[str, int] = {}
    self._gripper_status: dict[
        str, robotiq_gripper_controller.RobotiqStatus
    ] = {}

    for side in ("left", "right"):
      q_addrs, d_addrs = fr3_mjcf.arm_joint_qpos_and_dof_addrs(mj_model, side)
      a_ids = [
          int(
              mujoco.mj_name2id(
                  mj_model,
                  mujoco.mjtObj.mjOBJ_ACTUATOR,
                  f"{side}/fr3_joint{idx + 1}",
              )
          )
          for idx in range(7)
      ]
      self._arm_qpos_addrs[side] = q_addrs
      self._arm_dof_addrs[side] = d_addrs
      self._arm_act_ids[side] = a_ids

      self._gripper_act_id[side] = int(
          mujoco.mj_name2id(
              mj_model,
              mujoco.mjtObj.mjOBJ_ACTUATOR,
              fr3_mjcf.gripper_actuator_name(side),
          )
      )
      self._gripper_status[side] = robotiq_gripper_controller.RobotiqStatus(
          activated=True,
          goto_active=False,
          obj_status=3,
          fault=0,
          requested_pos_raw=0,
          measured_pos_raw=0,
          current_ma=0.0,
      )

    self.sync_ctrl_to_qpos()

  def sync_ctrl_to_qpos(self) -> None:
    """Copies current joint positions into position actuators and forwards."""
    for side in ("left", "right"):
      for qadr, aid in zip(
          self._arm_qpos_addrs[side], self._arm_act_ids[side]
      ):
        if aid >= 0:
          self.mj_data.ctrl[aid] = float(self.mj_data.qpos[qadr])
    mujoco.mj_forward(self.mj_model, self.mj_data)

  def get_arm_q(self, arm: str) -> np.ndarray:
    """Returns simulated 7-DOF joint positions (rad) for `arm`."""
    return np.array(
        [self.mj_data.qpos[i] for i in self._arm_qpos_addrs[arm]],
        dtype=np.float64,
    )

  def get_arm_dq(self, arm: str) -> np.ndarray:
    """Returns simulated 7-DOF joint velocities (rad/s) for `arm`."""
    return np.array(
        [self.mj_data.qvel[i] for i in self._arm_dof_addrs[arm]],
        dtype=np.float64,
    )

  def get_arm_tau(self, arm: str) -> np.ndarray:
    """Returns simulated bias torques (Nm) for `arm`."""
    return np.array(
        [self.mj_data.qfrc_bias[i] for i in self._arm_dof_addrs[arm]],
        dtype=np.float64,
    )

  def get_gripper_status(
      self, arm: str
  ) -> robotiq_gripper_controller.RobotiqStatus:
    """Returns simulated Robotiq status for `arm`."""
    return self._gripper_status[arm]

  def activate_grippers(
      self,
  ) -> dict[str, robotiq_gripper_controller.RobotiqStatus]:
    """Marks both simulated grippers as activated."""
    for side in ("left", "right"):
      prev = self._gripper_status[side]
      self._gripper_status[side] = robotiq_gripper_controller.RobotiqStatus(
          activated=True,
          goto_active=False,
          obj_status=3,
          fault=0,
          requested_pos_raw=prev.requested_pos_raw,
          measured_pos_raw=prev.measured_pos_raw,
          current_ma=10.0,
      )
    return dict(self._gripper_status)

  def move_gripper(
      self, arm: str, closure_0_to_1: float
  ) -> robotiq_gripper_controller.RobotiqStatus:
    """Steps simulated gripper actuator toward `closure_0_to_1` in `[0, 1]`."""
    raw_target = robotiq_gripper_controller.closure_to_raw(closure_0_to_1)
    aid = self._gripper_act_id[arm]
    if aid >= 0:
      self.mj_data.ctrl[aid] = float(raw_target)
      for _ in range(80):
        mujoco.mj_step(self.mj_model, self.mj_data)
    self._gripper_status[arm] = robotiq_gripper_controller.RobotiqStatus(
        activated=True,
        goto_active=False,
        obj_status=3,
        fault=0,
        requested_pos_raw=raw_target,
        measured_pos_raw=raw_target,
        current_ma=20.0 if raw_target > 0 else 0.0,
    )
    return self._gripper_status[arm]

  def step_bimanual_targets(
      self, q_left: np.ndarray, q_right: np.ndarray
  ) -> None:
    """Sets both arms' target qpos/ctrl and updates MuJoCo kinematics."""
    for side, q_vec in (("left", q_left), ("right", q_right)):
      for val, qadr, dadr, aid in zip(
          q_vec,
          self._arm_qpos_addrs[side],
          self._arm_dof_addrs[side],
          self._arm_act_ids[side],
      ):
        if aid >= 0:
          self.mj_data.ctrl[aid] = float(val)
        self.mj_data.qpos[qadr] = float(val)
        self.mj_data.qvel[dadr] = 0.0
    mujoco.mj_forward(self.mj_model, self.mj_data)

  def sync_qpos_no_actuators(self, side: str, q_rad: Sequence[float]) -> None:
    """Mirrors live hardware `q_rad` onto `mj_data.qpos` without stepping."""
    for val, qadr in zip(q_rad, self._arm_qpos_addrs[side]):
      self.mj_data.qpos[qadr] = float(val)
    mujoco.mj_forward(self.mj_model, self.mj_data)

  def render_cameras(
      self,
      width: int = 1280,
      height: int = 720,
      *,
      qpos_snapshot: np.ndarray | None = None,
  ) -> dict[str, np.ndarray]:
    """Renders the 4 canonical Duo cameras offscreen via `mujoco.Renderer`."""
    w, h = int(width), int(height)
    frames: dict[str, np.ndarray] = {}
    tid = threading.get_ident()
    with self._render_lock:
      entry = self._renderers.get(tid)
      if entry is not None and entry[1] != (h, w):
        _safe_close(self._renderers.pop(tid)[0])
        entry = None
      if entry is None:
        renderer = mujoco.Renderer(self.mj_model, height=h, width=w)
        scratch_data = mujoco.MjData(self.mj_model)
        self._renderers[tid] = (renderer, (h, w), scratch_data)
      else:
        renderer, _, scratch_data = entry

    render_data = self.mj_data
    if qpos_snapshot is not None:
      scratch_data.qpos[:] = qpos_snapshot
      render_data = scratch_data
    mujoco.mj_forward(self.mj_model, render_data)

    for cam_name in fr3_camera_controller.CANONICAL_CAMERAS:
      mjcf_cam = fr3_mjcf.resolve_mjcf_camera_name(self.mj_model, cam_name)
      renderer.update_scene(render_data, camera=mjcf_cam)
      rgb = np.ascontiguousarray(renderer.render(), dtype=np.uint8)
      frames[cam_name] = rgb
    return frames

  def close_renderer(self) -> None:
    """Closes all active offscreen MuJoCo GL renderers."""
    with self._render_lock:
      renderers = [entry[0] for entry in self._renderers.values()]
      self._renderers.clear()
    for renderer in renderers:
      _safe_close(renderer)


class Fr3DuoRobot:
  """Unified live hardware and MuJoCo simulation controller for FR3 Duo."""

  def __init__(
      self,
      config: RobotConfig | Mapping[str, Any] | None = None,
      *,
      mode: str = "hardware",
      dry_run: bool = False,
      read_only: bool = False,
      realtime_mode: str | None = None,
      menagerie_dir: str | Path | None = None,
      robot_factory: Any = None,
      gripper_factory: Any = None,
      camera_factory: Any = None,
  ) -> None:
    """Builds MJCF safety/sim models and selects real or simulated backend."""
    norm_mode = mode.strip().lower()
    if norm_mode not in ("sim", "hardware"):
      raise ValueError(
          f"Invalid robot mode={mode!r}; expected 'sim' or 'hardware'"
      )
    if mujoco is None:
      raise RuntimeError(
          "`mujoco` is required for `Fr3DuoRobot`. "
          "Install it with `pip install mujoco`."
      )
    self.mode = norm_mode
    self.dry_run = bool(dry_run)
    self.read_only = bool(read_only)
    self.config = normalize_robot_config(config)
    if self.mode == "hardware" and not self.dry_run:
      missing_cfg = validate_robot_config_complete(self.config)
      if missing_cfg:
        warnings.warn(
            "Incomplete/unconfirmed RobotConfig for live hardware mode "
            f"(missing: {', '.join(missing_cfg)}). Run `bringup.py --step "
            "geometry` and bind all hardware before live actuation.",
            UserWarning,
            stacklevel=2,
        )
    self.realtime_mode = str(realtime_mode or self.config.realtime_mode)
    self.realtime_info = resolve_realtime_config(
        self.realtime_mode
        if (self.mode == "hardware" and not self.dry_run)
        else (
            "ignore" if self.realtime_mode == "enforce" else self.realtime_mode
        )
    )
    self._sim_lock = threading.RLock()

    self._mj_model, self._mjcf_xml = fr3_mjcf.build_fr3_duo_mjcf(
        self.config, menagerie_dir=menagerie_dir
    )
    self._mj_data = mujoco.MjData(self._mj_model)
    self._sim_backend = Fr3DuoSimBackend(self._mj_model, self._mj_data)
    self._mj_data.qpos[:] = self._mj_model.qpos0
    for side in ("left", "right"):
      rest_q = self.config.arms[side].rest_pose_rad
      for qadr, val in zip(self._sim_backend._arm_qpos_addrs[side], rest_q):
        self._mj_data.qpos[qadr] = float(val)
    self._sim_backend.sync_ctrl_to_qpos()

    self._safety = Fr3DuoSafetyChecker(
        self.config, mj_model=self._mj_model, mjcf_xml=self._mjcf_xml
    )
    self._link0_in_base_cache: dict[str, np.ndarray] = {}
    self._link0_inv_cache: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
      t_bl = compute_link0_in_base_transform_for_model(
          self._mj_model, side, self._mj_data
      )
      self._link0_in_base_cache[side] = t_bl
      self._link0_inv_cache[side] = np.linalg.inv(t_bl)

    self._hw = (
        Fr3DuoRealBackend(
            self.config,
            realtime_cfg_name=str(self.realtime_info["realtime_config"]),
            read_only=self.read_only,
            robot_factory=robot_factory,
            gripper_factory=gripper_factory,
            camera_factory=camera_factory,
        )
        if (self.mode == "hardware" and not self.dry_run)
        else None
    )
    self._hw_finalizer: weakref.finalize | None = (
        weakref.finalize(self, self._hw.close)
        if self._hw is not None
        else None
    )
    self._connected = False
    self._state_snapshot_lock = threading.Lock()
    self._camera_capture_lock = threading.Lock()
    self._recorder_lock = threading.RLock()
    self._active_recorder: EpisodeRecorder | None = None
    self._latest_state_snapshot: RobotState = (
        self._build_initial_state_snapshot()
    )
    self._ik_scratch_data = mujoco.MjData(self._mj_model)
    self._ik_jacp = np.zeros((3, self._mj_model.nv), dtype=np.float64)
    self._ik_jacr = np.zeros((3, self._mj_model.nv), dtype=np.float64)
    self._ee_site_ids = {
        side: fr3_kinematics.resolve_ee_site_id(self._mj_model, side)
        for side in ("left", "right")
    }

  def _compute_tcp_and_otee(
      self, side: str, q_arr: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes `(tcp_pos_base, tcp_rot_base, o_t_ee_4x4)` from MJCF FK."""
    tcp_pos_base, tcp_rot_base = fr3_kinematics.compute_arm_fk(
        self._mj_model, side, q_arr
    )
    t_base_tcp = np.eye(4, dtype=np.float64)
    t_base_tcp[:3, :3] = tcp_rot_base
    t_base_tcp[:3, 3] = tcp_pos_base
    o_t_ee_4x4 = self._link0_inv_cache[side] @ t_base_tcp
    return tcp_pos_base, tcp_rot_base, o_t_ee_4x4

  def _build_arm_pose_entry(
      self,
      side: str,
      q_rad: np.ndarray,
      g_status: robotiq_gripper_controller.RobotiqStatus,
      *,
      dq_rad: Sequence[float] | None = None,
      tau_j: Sequence[float] | None = None,
      wrench_6d: Sequence[float] | None = None,
      fk_pos_error_m: float = 0.0,
      o_t_ee_4x4: np.ndarray | None = None,
      tcp_fk: tuple[np.ndarray, np.ndarray] | None = None,
      current_errors: Sequence[str] = (),
      desk_audit: Mapping[str, Any] | None = None,
  ) -> tuple[ArmPoseState, float]:
    if o_t_ee_4x4 is None:
      tcp_pos_base, tcp_rot_base, o_t_ee_4x4 = self._compute_tcp_and_otee(
          side, q_rad
      )
    elif tcp_fk is not None:
      tcp_pos_base, tcp_rot_base = tcp_fk
    else:
      tcp_pos_base, tcp_rot_base = fr3_kinematics.compute_arm_fk(
          self._mj_model, side, q_rad
      )
    g_ratio = float(g_status.position_ratio)
    entry = ArmPoseState(
        q_rad=[float(x) for x in q_rad],
        dq_rad=[
            float(x) for x in (dq_rad if dq_rad is not None else [0.0] * 7)
        ],
        tau_j=[
            float(x) for x in (tau_j if tau_j is not None else [0.0] * 7)
        ],
        tau_ext_base_6d=[
            float(x)
            for x in (wrench_6d if wrench_6d is not None else [0.0] * 6)
        ],
        tcp_pos_base=[float(x) for x in tcp_pos_base],
        tcp_rot_base=[
            [float(v) for v in row] for row in tcp_rot_base.tolist()
        ],
        o_t_ee_4x4=[
            [float(v) for v in row] for row in o_t_ee_4x4.tolist()
        ],
        fk_pos_error_m=float(fk_pos_error_m),
        gripper_ratio=g_ratio,
        gripper_status=g_status.to_dict(),
        current_errors=list(current_errors),
        desk_audit=(
            dict(desk_audit)
            if desk_audit is not None
            else audit_arm_desk_config(
                side, None, self._link0_in_base_cache[side][:3, :3]
            )
        ),
    )
    return entry, g_ratio

  def _build_initial_state_snapshot(self) -> RobotState:
    ts = time.monotonic()
    entries: dict[str, ArmPoseState] = {}
    state_16d: list[float] = []
    for side in ("left", "right"):
      with self._sim_lock:
        q_rad = self._sim_backend.get_arm_q(side)
        g_status = self._sim_backend.get_gripper_status(side)
      entry, g_ratio = self._build_arm_pose_entry(side, q_rad, g_status)
      entries[side] = entry
      state_16d.extend(entry.q_rad)
      state_16d.append(g_ratio)
    return RobotState(
        timestamp_s=ts,
        left=entries["left"],
        right=entries["right"],
        state_16d=state_16d,
    )

  def get_latest_state_snapshot(self) -> RobotState:
    """Returns a thread-safe copy of the most recently observed robot state."""
    with self._state_snapshot_lock:
      return copy.deepcopy(self._latest_state_snapshot)

  def _set_latest_state_snapshot(self, state: RobotState) -> None:
    snap = copy.deepcopy(state)
    with self._state_snapshot_lock:
      self._latest_state_snapshot = snap

  def _update_state_snapshot_gripper(
      self,
      arm: str,
      g_status: robotiq_gripper_controller.RobotiqStatus,
  ) -> None:
    g_ratio = float(g_status.position_ratio)
    g_dict = g_status.to_dict()
    now_s = time.monotonic()
    with self._state_snapshot_lock:
      snap = self._latest_state_snapshot
      snap.timestamp_s = now_s
      arm_state = snap.left if arm == "left" else snap.right
      arm_state.gripper_ratio = g_ratio
      arm_state.gripper_status = g_dict
      if len(snap.state_16d) == 16:
        snap.state_16d[7 if arm == "left" else 15] = g_ratio

  def _update_state_snapshot_kinematics(
      self, q_left: np.ndarray, q_right: np.ndarray
  ) -> None:
    now_s = time.monotonic()
    updates: dict[
        str,
        tuple[list[float], list[float], list[list[float]], list[list[float]]],
    ] = {}
    for side, q_vec in (("left", q_left), ("right", q_right)):
      q_arr = np.asarray(q_vec, dtype=np.float64)
      tcp_pos, tcp_rot, o_t_ee = self._compute_tcp_and_otee(side, q_arr)
      updates[side] = (
          [float(x) for x in q_arr],
          [float(x) for x in tcp_pos],
          [[float(v) for v in row] for row in tcp_rot.tolist()],
          [[float(v) for v in row] for row in o_t_ee.tolist()],
      )

    with self._state_snapshot_lock:
      snap = self._latest_state_snapshot
      snap.timestamp_s = now_s
      for side in ("left", "right"):
        q_list, tcp_list, rot_list, o_list = updates[side]
        arm_state = snap.left if side == "left" else snap.right
        arm_state.q_rad = q_list
        arm_state.tcp_pos_base = tcp_list
        arm_state.tcp_rot_base = rot_list
        arm_state.o_t_ee_4x4 = o_list
      snap.state_16d = (
          list(snap.left.q_rad)
          + [float(snap.left.gripper_ratio)]
          + list(snap.right.q_rad)
          + [float(snap.right.gripper_ratio)]
      )

  def _is_recorder_busy(self, recorder: EpisodeRecorder) -> bool:
    checker = getattr(recorder, "_is_active_or_starting", None)
    if callable(checker):
      return bool(checker())
    return bool(recorder.is_recording)

  def _register_active_recorder(self, recorder: EpisodeRecorder) -> None:
    with self._recorder_lock:
      if (
          self._active_recorder is not None
          and self._active_recorder is not recorder
          and self._is_recorder_busy(self._active_recorder)
      ):
        raise RuntimeError(
            "An episode recording is already active on this Fr3DuoRobot."
        )
      self._active_recorder = recorder

  def _unregister_active_recorder(self, recorder: EpisodeRecorder) -> None:
    with self._recorder_lock:
      if self._active_recorder is recorder:
        self._active_recorder = None

  def _notify_recorder_state(
      self, event: str, phase: str | None = None
  ) -> None:
    with self._recorder_lock:
      rec = self._active_recorder
    if rec is not None and rec.is_recording:
      rec.log_state(event=event, phase=phase)

  def record_episode(
      self,
      output_dir: str | Path,
      *,
      fps: float = 10.0,
      width: int = 1280,
      height: int = 720,
      phase: str = "episode",
      record_cameras: bool = True,
  ) -> EpisodeRecorder:
    """Starts and returns a context-managed `EpisodeRecorder` for this robot."""
    return self.start_recording(
        output_dir,
        fps=fps,
        width=width,
        height=height,
        phase=phase,
        record_cameras=record_cameras,
    )

  def start_recording(
      self,
      output_dir: str | Path,
      *,
      fps: float = 10.0,
      width: int = 1280,
      height: int = 720,
      phase: str = "episode",
      record_cameras: bool = True,
  ) -> EpisodeRecorder:
    """Starts background episode recording on this robot and returns it."""
    with self._recorder_lock:
      if (
          self._active_recorder is not None
          and self._is_recorder_busy(self._active_recorder)
      ):
        raise RuntimeError(
            "An episode recording is already active; call stop_recording() "
            "first."
        )
      rec = EpisodeRecorder(
          self,
          output_dir,
          fps=fps,
          width=width,
          height=height,
          phase=phase,
          record_cameras=record_cameras,
      )
      return rec.start()

  def stop_recording(self) -> EpisodeSummary | None:
    """Stops any active `EpisodeRecorder` and returns its `EpisodeSummary`."""
    with self._recorder_lock:
      rec = self._active_recorder
    if rec is None:
      return None
    return rec.stop()

  @property
  def mj_model(self) -> Any:
    """Returns the compiled MuJoCo `MjModel` for the Duo cell."""
    return self._mj_model

  @property
  def mj_data(self) -> Any:
    """Returns the active MuJoCo `MjData` state."""
    return self._mj_data

  @property
  def mjcf_xml(self) -> str:
    """Returns the resolved MJCF XML string for the Duo cell."""
    return self._mjcf_xml

  def connect(self) -> None:
    """Connects hardware workers/grippers when `mode == 'hardware'`."""
    if self._connected:
      return
    if self.dry_run or self._hw is None:
      self._connected = True
      return

    if self._hw_finalizer is None or not self._hw_finalizer.alive:
      self._hw_finalizer = weakref.finalize(self, self._hw.close)
    self.realtime_info = resolve_realtime_config(self.realtime_mode)
    self._hw.connect(str(self.realtime_info["realtime_config"]))
    self._connected = True

  def close(self) -> None:
    """Cleanly shuts down recordings, child processes, serial ports, and GL."""
    try:
      self.stop_recording()
    finally:
      if self._hw_finalizer is not None:
        self._hw_finalizer.detach()
        self._hw_finalizer = None
      try:
        if self._hw is not None:
          self._hw.close()
      finally:
        with self._sim_lock:
          self._sim_backend.close_renderer()
        self._connected = False

  def __enter__(self) -> Fr3DuoRobot:
    self.connect()
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    self.close()

  def _ensure_connected(self) -> None:
    if not self._connected:
      self.connect()

  def _require_writable(self) -> None:
    if self.read_only:
      raise RuntimeError(
          "Fr3DuoRobot is in read_only=True mode; actuation is disabled"
      )

  def recover_errors(
      self, side: str | None = None
  ) -> dict[str, dict[str, Any]]:
    """Checks for FCI errors/reflex and runs automatic recovery if needed."""
    self._require_writable()
    sides = _resolve_sides(side)
    if self.dry_run or self._hw is None:
      return {arm_side: _recovery_result() for arm_side in sides}
    self._ensure_connected()
    return self._hw.recover_errors(side=side)

  def read_state(self) -> RobotState:
    """Reads dual-arm joints, TCP, external wrench, gripper, and 16-D state."""
    if not self._connected and self.mode == "hardware" and not self.dry_run:
      self.connect()

    ts = time.monotonic()
    entries: dict[str, ArmPoseState] = {}
    state_16d: list[float] = []

    for side in ("left", "right"):
      t_base_link0 = self._link0_in_base_cache[side]
      r_base_link0 = t_base_link0[:3, :3]

      is_hw = self._hw is not None and self._hw.has_arm(side)
      if is_hw and self._hw is not None:
        hw = self._hw.read_arm_state(side)
        q_rad = np.asarray(hw["q"], dtype=np.float64)
        dq_rad = np.asarray(hw["dq"], dtype=np.float64)
        tau_j = np.asarray(hw["tau_j"], dtype=np.float64)
        o_t_ee_4x4 = unpack_o_t_ee(hw["o_t_ee"])

        with self._sim_lock:
          self._sim_backend.sync_qpos_no_actuators(side, q_rad)

        if side in self._hw.grippers:
          g_status = self._hw.grippers[side].read_status()
        else:
          g_status = self._sim_backend.get_gripper_status(side)
      else:
        hw = None
        with self._sim_lock:
          q_rad = self._sim_backend.get_arm_q(side)
          dq_rad = self._sim_backend.get_arm_dq(side)
          tau_j = self._sim_backend.get_arm_tau(side)
          g_status = self._sim_backend.get_gripper_status(side)
        o_t_ee_4x4 = None

      if o_t_ee_4x4 is None:
        tcp_pos_base, tcp_rot_base, o_t_ee_4x4 = self._compute_tcp_and_otee(
            side, q_rad
        )
        fk_pos_error_m = 0.0
      else:
        tcp_pos_base, tcp_rot_base = fr3_kinematics.compute_arm_fk(
            self._mj_model, side, q_rad
        )
        fk_pos_error_m = compute_fk_pos_error_m(
            tcp_pos_base,
            t_base_link0,
            o_t_ee_4x4,
            float(self.config.tcp_offset_z_m),
            hw.get("f_t_ee") if hw is not None else None,
            mj_model=self._mj_model,
            side=side,
            q_rad=q_rad,
        )

      wrench_6d = compute_ext_wrench_base_6d(
          hw if is_hw else None,
          self._mj_model,
          side,
          q_rad,
          tau_j,
          r_base_link0,
      )
      entry, g_ratio = self._build_arm_pose_entry(
          side,
          q_rad,
          g_status,
          dq_rad=dq_rad,
          tau_j=tau_j,
          wrench_6d=wrench_6d,
          fk_pos_error_m=fk_pos_error_m,
          o_t_ee_4x4=o_t_ee_4x4,
          tcp_fk=(tcp_pos_base, tcp_rot_base),
          current_errors=(
              list(hw.get("current_errors") or []) if hw is not None else ()
          ),
          desk_audit=audit_arm_desk_config(
              side, hw if is_hw else None, r_base_link0
          ),
      )
      entries[side] = entry
      state_16d.extend(entry.q_rad)
      state_16d.append(g_ratio)

    result = RobotState(
        timestamp_s=ts,
        left=entries["left"],
        right=entries["right"],
        state_16d=state_16d,
    )
    self._set_latest_state_snapshot(result)
    return result

  def render_sim_cameras(
      self,
      width: int = 1280,
      height: int = 720,
  ) -> dict[str, np.ndarray]:
    """Renders the 4 canonical Duo cameras offscreen via `Fr3DuoSimBackend`."""
    with self._camera_capture_lock:
      with self._sim_lock:
        qpos_snap = self._mj_data.qpos.copy()
      return self._sim_backend.render_cameras(
          width=width, height=height, qpos_snapshot=qpos_snap
      )

  def _build_camera_batch_outputs(
      self,
      raw_batch: Mapping[str, Any],
      *,
      is_live_hw: bool,
      timestamp_s: float,
  ) -> tuple[
      dict[str, np.ndarray],
      dict[str, CameraMeta],
      dict[str, CameraFrame],
  ]:
    frames: dict[str, np.ndarray] = {}
    camera_meta: dict[str, CameraMeta] = {}
    contact_frames: dict[str, CameraFrame] = {}
    for cam_name in CANONICAL_CAMERAS:
      if cam_name not in raw_batch:
        raise RuntimeError(
            f"Missing canonical camera frame {cam_name!r} from "
            "hardware capture_cameras()."
        )
      item = raw_batch[cam_name]
      if isinstance(item, CameraFrame):
        rgb = np.ascontiguousarray(item.rgb, dtype=np.uint8)
        cam_ts = float(item.timestamp_s)
        age_s = float(item.age_s(timestamp_s))
        backend_name = str(item.backend)
        frame_obj = item
      else:
        rgb = np.ascontiguousarray(item, dtype=np.uint8)
        cam_ts = float(timestamp_s)
        age_s = 0.0
        backend_name = "hardware" if is_live_hw else "mujoco_sim"
        frame_obj = CameraFrame(
            name=cam_name,
            rgb=rgb,
            timestamp_s=cam_ts,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            backend=backend_name,
        )
      frames[cam_name] = rgb
      camera_meta[cam_name] = CameraMeta(
          backend=backend_name,
          width=int(rgb.shape[1]),
          height=int(rgb.shape[0]),
          timestamp_s=cam_ts if is_live_hw else None,
          age_s=age_s if is_live_hw else None,
      )
      contact_frames[cam_name] = frame_obj
    return frames, camera_meta, contact_frames

  def _capture_canonical_batch(
      self,
      *,
      width: int = 1280,
      height: int = 720,
      timestamp_s: float | None = None,
  ) -> tuple[
      dict[str, np.ndarray],
      dict[str, CameraMeta],
      dict[str, CameraFrame],
      dict[str, Any],
      bool,
  ]:
    """Captures 4 canonical RGB frames, metadata, and rectified ZED cal."""
    if self.mode == "hardware" and not self.dry_run and self._hw is not None:
      with self._camera_capture_lock:
        captured = self._hw.capture_cameras()
        rect_cal = self._hw.get_rectified_camera_calibration()
      now_s = time.monotonic()
      frames, camera_meta, contact_frames = self._build_camera_batch_outputs(
          captured, is_live_hw=True, timestamp_s=now_s
      )
      return frames, camera_meta, contact_frames, rect_cal, True

    rendered = self.render_sim_cameras(width=width, height=height)
    sim_ts = time.monotonic() if timestamp_s is None else float(timestamp_s)
    sim_frames = build_sim_camera_frames(rendered, sim_ts)
    frames, camera_meta, contact_frames = self._build_camera_batch_outputs(
        sim_frames, is_live_hw=False, timestamp_s=sim_ts
    )
    return frames, camera_meta, contact_frames, {}, False

  def _capture_recording_frames(
      self,
      *,
      width: int = 1280,
      height: int = 720,
  ) -> tuple[dict[str, CameraFrame], dict[str, float]]:
    """Thread-safely captures 4 canonical `CameraFrame`s for recording."""
    _, _, contact_frames, _, _ = self._capture_canonical_batch(
        width=width, height=height, timestamp_s=None
    )
    return contact_frames, {
        cam_name: float(frm.timestamp_s)
        for cam_name, frm in contact_frames.items()
    }

  def observe_scene(
      self,
      *,
      output_dir: str | Path | None = None,
      width: int = 1280,
      height: int = 720,
  ) -> SceneObservation:
    """Returns synchronized state, 4 RGB views, calibration, and table."""
    state = self.read_state()
    ts = float(state.timestamp_s)
    frames, camera_meta, contact_frames, rect_cal, is_live_hw = (
        self._capture_canonical_batch(
            width=width, height=height, timestamp_s=ts
        )
    )
    res_wh = None if is_live_hw else (int(width), int(height))
    calibration = fr3_kinematics.get_camera_calibration_opencv(
        self._mj_model,
        state.left.q_rad,
        state.right.q_rad,
        resolution_wh=res_wh,
    )
    if is_live_hw:
      fr3_camera_controller.apply_rectified_camera_calibration(
          calibration, rect_cal
      )
      for cam_name, cam_cal in calibration.items():
        if bool(cam_cal.get("rectified")):
          if cam_name in camera_meta:
            camera_meta[cam_name].rectified = True
        elif cam_name in frames:
          rgb_arr = frames[cam_name]
          scale_camera_calibration(
              cam_cal, int(rgb_arr.shape[1]), int(rgb_arr.shape[0])
          )
    table_center, table_half_extents = self.config.table_box_bounds()
    observation = SceneObservation(
        timestamp_s=ts,
        mode=self.mode,
        dry_run=self.dry_run,
        state=state,
        state_16d=list(state.state_16d),
        frames=frames,
        camera_meta=camera_meta,
        calibration=calibration,
        z_table_m=float(self.config.z_table_m),
        table_box_bounds=TableBoxBounds(
            center=[float(v) for v in table_center],
            half_extents=[float(v) for v in table_half_extents],
        ),
        saved_paths={},
    )
    if output_dir is not None:
      observation.saved_paths = _save_observation_outputs(
          output_dir, observation, contact_frames
      )
    self._notify_recorder_state("observe")
    return observation

  def activate_grippers(
      self,
  ) -> dict[str, robotiq_gripper_controller.RobotiqStatus]:
    """Activates both Robotiq grippers (no-op on HW when `dry_run=True`)."""
    self._require_writable()
    self._ensure_connected()
    if self.dry_run:
      with self._sim_lock:
        return {
            side: self._sim_backend.get_gripper_status(side)
            for side in ("left", "right")
        }
    if self._hw is None:
      with self._sim_lock:
        return self._sim_backend.activate_grippers()
    if not self._hw.grippers:
      raise RuntimeError(
          "No hardware Robotiq grippers configured/connected in "
          "mode='hardware'."
      )
    return self._hw.activate_grippers()

  def move_gripper(
      self,
      arm: str,
      closure_0_to_1: float,
      *,
      speed: float = 0.5,
      force: float = 0.5,
      wait: bool = True,
  ) -> robotiq_gripper_controller.RobotiqStatus:
    """Moves `arm`'s Robotiq gripper to normalized closure `[0.0, 1.0]`."""
    self._require_writable()
    if arm not in ("left", "right"):
      raise ValueError(f"Invalid arm={arm!r}; expected 'left' or 'right'")
    self._ensure_connected()
    if not self.dry_run and self._hw is not None:
      if arm not in self._hw.grippers:
        raise RuntimeError(
            f"Hardware Robotiq gripper for '{arm}' is not configured/connected."
        )
      g_status = self._hw.move_gripper(
          arm, closure_0_to_1, speed=speed, force=force, wait=wait
      )
    else:
      with self._sim_lock:
        g_status = self._sim_backend.move_gripper(arm, closure_0_to_1)

    self._update_state_snapshot_gripper(arm, g_status)
    self._notify_recorder_state("gripper")
    return g_status

  def _resolve_speed_caps(
      self, speed_deg_s: float | None
  ) -> tuple[float, float]:
    if speed_deg_s is not None:
      default_deg_s = float(speed_deg_s)
    elif self.config.max_joint_speed_deg_s is not None:
      default_deg_s = float(self.config.max_joint_speed_deg_s)
    else:
      default_deg_s = float(
          self.realtime_info.get("max_joint_speed_deg_s", 15.0)
      )
    chosen_deg_s = min(max(float(default_deg_s), 1.0), 45.0)
    max_vel_rad_s = math.radians(chosen_deg_s)
    max_acc_rad_s2 = float(
        self.realtime_info.get("max_joint_acc_rad_s2", 2.0)
    )
    return max_vel_rad_s, max_acc_rad_s2

  def move_arm_waypoints(
      self,
      arm: str,
      waypoints_rad: Sequence[Sequence[float]],
      *,
      abort_watcher: AbortWatcher | None = None,
      speed_deg_s: float | None = None,
      allow_initial_table_lift: bool = False,
  ) -> MotionResult:
    """Executes a safety-checked single-arm trajectory with abort unwind."""
    self._require_writable()
    if arm not in ("left", "right"):
      raise ValueError(f"Invalid arm={arm!r}; expected 'left' or 'right'")
    self._ensure_connected()
    if self._hw is not None and arm not in self._hw.workers:
      raise RuntimeError(
          f"Hardware arm worker for '{arm}' is not configured/connected."
      )

    state_now = self.read_state()
    other_arm = state_now.right if arm == "left" else state_now.left
    hold_other = [clamp_q_to_padded_limits(other_arm.q_rad).tolist()]
    l_wps, r_wps = (
        (waypoints_rad, hold_other)
        if arm == "left"
        else (hold_other, waypoints_rad)
    )
    res = self._move_bimanual_waypoints_impl(
        l_wps,
        r_wps,
        abort_watcher=abort_watcher,
        speed_deg_s=speed_deg_s,
        required_arms=(arm,),
        allow_initial_table_lift=allow_initial_table_lift,
    )
    res.final_q_rad = list(
        res.final_left_q_rad if arm == "left" else res.final_right_q_rad
    )
    return res

  def _execute_sim_bimanual_trajectory(
      self,
      traj_left: np.ndarray,
      traj_right: np.ndarray,
      q_left_start: np.ndarray,
      q_right_start: np.ndarray,
      *,
      max_vel: float,
      max_acc: float,
      abort_watcher: AbortWatcher | None,
  ) -> tuple[bool, int]:
    n_max = int(traj_left.shape[0])
    num_chunks = min(40, n_max)
    chunk_indices = np.unique(
        np.linspace(0, n_max - 1, num_chunks, dtype=np.intp)
    )
    travelled_left: list[list[float]] = [q_left_start.tolist()]
    travelled_right: list[list[float]] = [q_right_start.tolist()]
    aborted = False
    steps_done = 0

    for step_idx, c_idx in enumerate(chunk_indices):
      if abort_watcher is not None and abort_watcher.poll():
        aborted = True
        break
      ql = traj_left[c_idx]
      qr = traj_right[c_idx]
      with self._sim_lock:
        self._sim_backend.step_bimanual_targets(ql, qr)
      self._update_state_snapshot_kinematics(ql, qr)
      if step_idx > 0 and step_idx % 4 == 0:
        self._notify_recorder_state("motion_step")
      travelled_left.append(ql.tolist())
      travelled_right.append(qr.tolist())
      steps_done = int(c_idx) + 1
      if abort_watcher is not None:
        time.sleep(0.004)

    if not aborted and abort_watcher is not None and abort_watcher.poll():
      aborted = True

    if aborted:
      unwind_l, unwind_r = _build_bimanual_unwind_trajs(
          travelled_left[::-1] + [q_left_start.tolist()],
          travelled_right[::-1] + [q_right_start.tolist()],
          max_vel,
          max_acc,
      )
      u_max = max(len(unwind_l), len(unwind_r))
      u_indices = np.unique(
          np.linspace(0, u_max - 1, min(16, u_max), dtype=np.intp)
      )
      for u_idx in u_indices:
        ql = unwind_l[min(u_idx, len(unwind_l) - 1)]
        qr = unwind_r[min(u_idx, len(unwind_r) - 1)]
        with self._sim_lock:
          self._sim_backend.step_bimanual_targets(ql, qr)
        self._update_state_snapshot_kinematics(ql, qr)
    return aborted, steps_done

  def move_bimanual_waypoints(
      self,
      left_waypoints_rad: Sequence[Sequence[float]],
      right_waypoints_rad: Sequence[Sequence[float]],
      *,
      abort_watcher: AbortWatcher | None = None,
      speed_deg_s: float | None = None,
      allow_initial_table_lift: bool = False,
  ) -> MotionResult:
    """Executes synchronized dual-arm splines with safety & abort unwind."""
    self._require_writable()
    return self._move_bimanual_waypoints_impl(
        left_waypoints_rad,
        right_waypoints_rad,
        abort_watcher=abort_watcher,
        speed_deg_s=speed_deg_s,
        required_arms=("left", "right"),
        allow_initial_table_lift=allow_initial_table_lift,
    )

  def _move_bimanual_waypoints_impl(
      self,
      left_waypoints_rad: Sequence[Sequence[float]],
      right_waypoints_rad: Sequence[Sequence[float]],
      *,
      abort_watcher: AbortWatcher | None = None,
      speed_deg_s: float | None = None,
      required_arms: tuple[str, ...] = ("left", "right"),
      allow_initial_table_lift: bool = False,
  ) -> MotionResult:
    self._require_writable()
    self._ensure_connected()

    state_before = self.read_state()
    q_left_start = clamp_q_to_padded_limits(state_before.left.q_rad)
    q_right_start = clamp_q_to_padded_limits(state_before.right.q_rad)

    l_wps = [list(map(float, wp)) for wp in left_waypoints_rad]
    r_wps = [list(map(float, wp)) for wp in right_waypoints_rad]
    if not l_wps or not r_wps:
      raise ValueError("Waypoints for both arms must be non-empty")

    if float(np.max(np.abs(np.asarray(l_wps[0]) - q_left_start))) > 1e-4:
      l_wps = [q_left_start.tolist()] + l_wps
    if float(np.max(np.abs(np.asarray(r_wps[0]) - q_right_start))) > 1e-4:
      r_wps = [q_right_start.tolist()] + r_wps

    max_vel, max_acc = self._resolve_speed_caps(speed_deg_s)
    traj_left = generate_quintic_trajectory(
        l_wps, dt_s=0.001, max_vel_rad_s=max_vel, max_acc_rad_s2=max_acc
    )
    traj_right = generate_quintic_trajectory(
        r_wps, dt_s=0.001, max_vel_rad_s=max_vel, max_acc_rad_s2=max_acc
    )

    n_max = max(int(traj_left.shape[0]), int(traj_right.shape[0]))
    if traj_left.shape[0] < n_max:
      pad = np.repeat(traj_left[-1:], n_max - traj_left.shape[0], axis=0)
      traj_left = np.vstack([traj_left, pad])
    if traj_right.shape[0] < n_max:
      pad = np.repeat(traj_right[-1:], n_max - traj_right.shape[0], axis=0)
      traj_right = np.vstack([traj_right, pad])

    with self._sim_lock:
      self._safety.verify_bimanual_trajectory(
          traj_left,
          traj_right,
          z_table_m=float(self.config.z_table_m),
          min_clearance_m=float(self.config.min_inter_arm_clearance_m),
          allow_initial_table_lift=bool(allow_initial_table_lift),
      )

    if self.dry_run:
      ql_end = traj_left[-1]
      qr_end = traj_right[-1]
      with self._sim_lock:
        self._sim_backend.step_bimanual_targets(ql_end, qr_end)
      self._update_state_snapshot_kinematics(ql_end, qr_end)
      self._notify_recorder_state("motion_start")
      self._notify_recorder_state("motion_end")
      return MotionResult(
          ok=True,
          aborted=False,
          dry_run=True,
          reason="",
          unwound_to_start=False,
          steps_executed=0,
          final_left_q_rad=ql_end.tolist(),
          final_right_q_rad=qr_end.tolist(),
      )

    self._notify_recorder_state("motion_start")
    if self._hw is not None:
      missing = [s for s in required_arms if s not in self._hw.workers]
      if missing:
        raise RuntimeError(
            f"Hardware arm worker(s) {missing} not configured/connected in "
            "mode='hardware'."
        )
      aborted = self._hw.execute_bimanual_trajectory(
          traj_left,
          traj_right,
          q_left_start,
          q_right_start,
          max_vel=max_vel,
          max_acc=max_acc,
          abort_watcher=abort_watcher,
      )
      steps_done = n_max
    else:
      aborted, steps_done = self._execute_sim_bimanual_trajectory(
          traj_left,
          traj_right,
          q_left_start,
          q_right_start,
          max_vel=max_vel,
          max_acc=max_acc,
          abort_watcher=abort_watcher,
      )

    after = self.read_state()
    self._notify_recorder_state("motion_end")
    return MotionResult(
        ok=not aborted,
        aborted=aborted,
        dry_run=False,
        reason=abort_watcher.reason if (aborted and abort_watcher) else "",
        unwound_to_start=aborted,
        steps_executed=steps_done,
        final_left_q_rad=list(after.left.q_rad),
        final_right_q_rad=list(after.right.q_rad),
    )

  def _lifted_waypoint(
      self, side: str, q_now: np.ndarray, lift_m: float
  ) -> list[float]:
    site_id = self._ee_site_ids[side]
    q_addrs = self._sim_backend._arm_qpos_addrs[side]
    with self._sim_lock:
      scratch = self._ik_scratch_data
      scratch.qpos[:] = self._mj_data.qpos
      for qadr, val in zip(q_addrs, q_now):
        scratch.qpos[qadr] = float(val)
      mujoco.mj_kinematics(self._mj_model, scratch)
      mujoco.mj_comPos(self._mj_model, scratch)
      target_pos = np.asarray(
          scratch.site_xpos[site_id], dtype=np.float64
      ).copy()
      target_pos[2] += float(lift_m)
      ee_rot = (
          np.asarray(scratch.site_xmat[site_id], dtype=np.float64)
          .reshape(3, 3)
          .copy()
      )
    return self._solve_cartesian_step_ik(
        side, q_now, target_pos, ee_rot
    ).tolist()

  def home(
      self,
      arm: str = "both",
      *,
      abort_watcher: AbortWatcher | None = None,
      speed_deg_s: float | None = None,
      allow_initial_table_lift: bool = False,
      lift_first_m: float = 0.0,
  ) -> MotionResult:
    """Moves `left`, `right`, or `both` arms smoothly to `rest_pose_rad`."""
    self._require_writable()
    if arm not in ("left", "right", "both"):
      raise ValueError(
          f"Invalid arm={arm!r}; expected 'left', 'right', or 'both'"
      )
    lift_m = float(lift_first_m)
    if not math.isfinite(lift_m) or lift_m < 0.0:
      raise ValueError(
          f"lift_first_m must be finite and >= 0, got {lift_first_m!r}"
      )
    self._ensure_connected()
    state_now = self.read_state()
    allow_lift = bool(allow_initial_table_lift or lift_m > 0.0)

    def _home_waypoints(side: str) -> list[list[float]]:
      q_now = clamp_q_to_padded_limits(state_now[side].q_rad)
      q_home = clamp_q_to_padded_limits(
          self.config.arms[side].rest_pose_rad
      ).tolist()
      wps = [q_now.tolist()]
      if lift_m > 0.0:
        wps.append(self._lifted_waypoint(side, q_now, lift_m))
      wps.append(q_home)
      return wps

    if arm == "both":
      return self.move_bimanual_waypoints(
          _home_waypoints("left"),
          _home_waypoints("right"),
          abort_watcher=abort_watcher,
          speed_deg_s=speed_deg_s,
          allow_initial_table_lift=allow_lift,
      )
    return self.move_arm_waypoints(
        arm,
        _home_waypoints(arm),
        abort_watcher=abort_watcher,
        speed_deg_s=speed_deg_s,
        allow_initial_table_lift=allow_lift,
    )

  def _solve_cartesian_step_ik(
      self,
      side: str,
      q_start: np.ndarray,
      target_pos_base: np.ndarray,
      target_rot_base: np.ndarray,
  ) -> np.ndarray:
    """Computes a damped-least-squares IK step from `q_start` to target."""
    site_id = self._ee_site_ids[side]
    q_addrs = self._sim_backend._arm_qpos_addrs[side]
    dof_addrs = self._sim_backend._arm_dof_addrs[side]
    q_work = clamp_q_to_padded_limits(q_start)
    damp_sq = 0.05**2
    with self._sim_lock:
      scratch = self._ik_scratch_data
      scratch.qpos[:] = self._mj_data.qpos
      for _ in range(10):
        for qadr, val in zip(q_addrs, q_work):
          scratch.qpos[qadr] = float(val)
        mujoco.mj_kinematics(self._mj_model, scratch)
        mujoco.mj_comPos(self._mj_model, scratch)
        pos_err = target_pos_base - np.asarray(
            scratch.site_xpos[site_id], dtype=np.float64
        )
        if float(np.linalg.norm(pos_err)) < 1e-4:
          break
        rot_curr = np.asarray(
            scratch.site_xmat[site_id], dtype=np.float64
        ).reshape(3, 3)
        rot_err = 0.5 * (
            np.cross(rot_curr[:, 0], target_rot_base[:, 0])
            + np.cross(rot_curr[:, 1], target_rot_base[:, 1])
            + np.cross(rot_curr[:, 2], target_rot_base[:, 2])
        )
        err_6d = np.concatenate([pos_err, 0.5 * rot_err])
        self._ik_jacp.fill(0.0)
        self._ik_jacr.fill(0.0)
        mujoco.mj_jacSite(
            self._mj_model,
            scratch,
            self._ik_jacp,
            self._ik_jacr,
            site_id,
        )
        jac = np.vstack(
            [self._ik_jacp[:, dof_addrs], self._ik_jacr[:, dof_addrs]]
        )
        jjt = jac @ jac.T + damp_sq * np.eye(6, dtype=np.float64)
        dq = jac.T @ np.linalg.solve(jjt, err_6d)
        q_work = clamp_q_to_padded_limits(q_work + dq)
    return q_work

  def move_cartesian_until_contact(
      self,
      side: str,
      direction: Sequence[float] = (0.0, 0.0, -1.0),
      max_distance_m: float = 0.05,
      step_m: float = 0.005,
      max_tau_ext_n: float = 9.0,
      *,
      speed_deg_s: float | None = None,
      abort_watcher: AbortWatcher | None = None,
  ) -> ContactMoveResult:
    """Steps `side` TCP along `direction` until `max_tau_ext_n` or limit."""
    self._require_writable()
    if side not in ("left", "right"):
      raise ValueError(f"Invalid side={side!r}; expected 'left' or 'right'")
    dir_arr = np.asarray(direction, dtype=np.float64).reshape(-1)
    if dir_arr.shape != (3,) or not np.all(np.isfinite(dir_arr)):
      raise ValueError(
          f"direction must be a finite 3-vector, got {direction!r}"
      )
    dir_norm = float(np.linalg.norm(dir_arr))
    if dir_norm <= 1e-9:
      raise ValueError("direction must be non-zero")
    unit_dir = dir_arr / dir_norm
    max_dist = float(max_distance_m)
    inc_step = float(step_m)
    force_limit = float(max_tau_ext_n)
    if not (math.isfinite(max_dist) and max_dist > 0.0):
      raise ValueError("max_distance_m must be finite and > 0")
    if not (math.isfinite(inc_step) and inc_step > 0.0):
      raise ValueError("step_m must be finite and > 0")
    if not (math.isfinite(force_limit) and force_limit > 0.0):
      raise ValueError("max_tau_ext_n must be finite and > 0")

    self._ensure_connected()
    if self._hw is not None and side not in self._hw.workers:
      raise RuntimeError(
          f"Hardware arm worker for '{side}' is not configured/connected."
      )

    state0 = self.read_state()
    arm0 = state0[side]
    f_ext = float(np.linalg.norm(arm0.tau_ext_base_6d[:3]))
    max_force_n = f_ext
    start_tcp = np.asarray(arm0.tcp_pos_base, dtype=np.float64)
    start_rot = np.asarray(arm0.tcp_rot_base, dtype=np.float64).reshape(3, 3)
    q_curr = clamp_q_to_padded_limits(arm0.q_rad)

    def _make_result(
        *,
        ok: bool,
        contact_detected: bool,
        aborted: bool,
        reason: str,
        steps: int,
        dist_m: float,
        arm_state: ArmPoseState,
    ) -> ContactMoveResult:
      return ContactMoveResult(
          ok=ok,
          contact_detected=contact_detected,
          aborted=aborted,
          dry_run=self.dry_run,
          side=side,
          reason=reason,
          steps_executed=steps,
          distance_moved_m=dist_m,
          final_force_n=f_ext,
          max_force_n=max_force_n,
          final_tcp_pos_base=[float(x) for x in arm_state.tcp_pos_base],
          final_q_rad=[float(x) for x in arm_state.q_rad],
      )

    if f_ext >= force_limit:
      return _make_result(
          ok=True,
          contact_detected=True,
          aborted=False,
          reason="contact_threshold_reached",
          steps=0,
          dist_m=0.0,
          arm_state=arm0,
      )

    if self.dry_run:
      q_end = self._solve_cartesian_step_ik(
          side, q_curr, start_tcp + unit_dir * max_dist, start_rot
      )
      self.move_arm_waypoints(
          side,
          [q_curr.tolist(), q_end.tolist()],
          abort_watcher=abort_watcher,
          speed_deg_s=speed_deg_s,
      )
      arm_after = self.read_state()[side]
      end_tcp = np.asarray(arm_after.tcp_pos_base, dtype=np.float64)
      return _make_result(
          ok=True,
          contact_detected=False,
          aborted=False,
          reason="dry_run",
          steps=0,
          dist_m=float(np.linalg.norm(end_tcp - start_tcp)),
          arm_state=arm_after,
      )

    traveled_m = 0.0
    steps_executed = 0
    curr_arm = arm0
    while traveled_m < max_dist - 1e-9:
      if abort_watcher is not None and abort_watcher.poll():
        return _make_result(
            ok=False,
            contact_detected=False,
            aborted=True,
            reason=abort_watcher.reason,
            steps=steps_executed,
            dist_m=traveled_m,
            arm_state=curr_arm,
        )
      delta_m = min(inc_step, max_dist - traveled_m)
      target_tcp = start_tcp + unit_dir * (traveled_m + delta_m)
      q_next = self._solve_cartesian_step_ik(
          side, q_curr, target_tcp, start_rot
      )
      step_res = self.move_arm_waypoints(
          side,
          [q_curr.tolist(), q_next.tolist()],
          abort_watcher=abort_watcher,
          speed_deg_s=speed_deg_s,
      )
      if step_res.aborted:
        curr_arm = self.read_state()[side]
        f_ext = float(np.linalg.norm(curr_arm.tau_ext_base_6d[:3]))
        max_force_n = max(max_force_n, f_ext)
        return _make_result(
            ok=False,
            contact_detected=False,
            aborted=True,
            reason=step_res.reason,
            steps=steps_executed,
            dist_m=traveled_m,
            arm_state=curr_arm,
        )
      steps_executed += 1
      traveled_m += delta_m
      curr_arm = self.read_state()[side]
      f_ext = float(np.linalg.norm(curr_arm.tau_ext_base_6d[:3]))
      max_force_n = max(max_force_n, f_ext)
      q_curr = clamp_q_to_padded_limits(curr_arm.q_rad)
      if f_ext >= force_limit:
        return _make_result(
            ok=True,
            contact_detected=True,
            aborted=False,
            reason="contact_threshold_reached",
            steps=steps_executed,
            dist_m=traveled_m,
            arm_state=curr_arm,
        )

    return _make_result(
        ok=True,
        contact_detected=False,
        aborted=False,
        reason="max_distance_reached",
        steps=steps_executed,
        dist_m=traveled_m,
        arm_state=curr_arm,
    )
