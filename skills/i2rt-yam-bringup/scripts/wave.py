# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bring-up 'cute wave' smoke test for I2RT YAM and YAM Pro arms."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import importlib
import sys
import time
from typing import Any

import numpy as np


def _optional_import(name: str, package: str | None = None) -> Any:
  try:
    return importlib.import_module(name, package=package)
  except ImportError:
    return None


macos_can: Any = (
    _optional_import(".macos_can", __package__) if __package__ else None
) or _optional_import("macos_can")
_get_robot_mod: Any = _optional_import("i2rt.robots.get_robot")
_robot_utils: Any = _optional_import("i2rt.robots.utils")
get_yam_robot: Any = getattr(_get_robot_mod, "get_yam_robot", None)
ArmType: Any = getattr(_robot_utils, "ArmType", None)
GripperType: Any = getattr(_robot_utils, "GripperType", None)

YAM_JOINT_MIN = np.array(
    [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944], dtype=np.float64
)
YAM_JOINT_MAX = np.array(
    [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944], dtype=np.float64
)
Q_HOME = np.array([-0.02, 0.01, 0.01, -0.07, -0.02, 0.00], dtype=np.float64)
Q_WAVE_READY = np.array(
    [0.00, 0.38, 0.62, 0.24, 0.00, 0.00], dtype=np.float64
)
Q_NOD_DOWN = np.array([0.00, 0.40, 0.65, 0.13, 0.00, 0.16], dtype=np.float64)
Q_NOD_UP = np.array([0.00, 0.37, 0.61, 0.28, 0.00, -0.14], dtype=np.float64)
Q_WINK = np.array([0.04, 0.37, 0.60, 0.29, 0.16, 0.28], dtype=np.float64)

GRIPPER_OPEN = 1.0
DT_S = 0.01
STOP_TOL_RAD = 0.08
MAX_TRACK_ERR_RAD = 0.45


def _resolve_enum(enum_cls: Any, value: str | Any) -> Any:
  if enum_cls is None or isinstance(value, enum_cls):
    return value
  key = str(value).strip().lower()
  for member in enum_cls:
    if key in (member.name.lower(), str(member.value).lower()):
      return member
  raise ValueError(f"Unsupported {enum_cls.__name__}: {value!r}")


def cosine_segment(
    q0: Sequence[float] | np.ndarray,
    q1: Sequence[float] | np.ndarray,
    min_duration_s: float,
    max_vel_rad_s: float = 1.10,
    dt_s: float = DT_S,
) -> np.ndarray:
  """Cosine-interpolated joint ramp with zero endpoint velocity."""
  if (
      not np.isfinite(min_duration_s)
      or min_duration_s <= 0.0
      or not np.isfinite(dt_s)
      or dt_s <= 0.0
  ):
    raise ValueError("min_duration_s and dt_s must be positive finite numbers")
  start = np.asarray(q0, dtype=np.float64)
  end = np.asarray(q1, dtype=np.float64)
  max_dq = float(np.max(np.abs(end[:6] - start[:6])))
  dur_s = max(
      min_duration_s, 0.5 * np.pi * max_dq / max(max_vel_rad_s, 0.1), 0.25
  )
  steps = max(2, int(np.ceil(dur_s / dt_s)))
  u = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
  s = (0.5 * (1.0 - np.cos(np.pi * u)))[:, None]
  return start[None, :] + s * (end - start)[None, :]


def build_cute_wave_stream(
    *, cycles: float = 3.5, dt_s: float = DT_S
) -> tuple[np.ndarray, np.ndarray]:
  """Builds the 100 Hz sin^2-enveloped overlapping-action wave stream."""
  if (
      not np.isfinite(cycles)
      or cycles <= 0.0
      or not np.isfinite(dt_s)
      or dt_s <= 0.0
  ):
    raise ValueError("cycles and dt_s must be positive finite numbers")
  duration_s = 4.4 * (cycles / 3.5)
  steps = max(2, int(round(duration_s / dt_s)))
  u = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
  env = np.sin(np.pi * u) ** 2
  phase = 2.0 * np.pi * cycles * u
  q6 = np.column_stack([
      0.10 * env * np.sin(phase),
      Q_WAVE_READY[1] + 0.03 * env * (1.0 - np.cos(2.0 * phase)),
      Q_WAVE_READY[2] + 0.04 * env * (1.0 - np.cos(2.0 * phase)),
      Q_WAVE_READY[3] + 0.05 * env * np.sin(2.0 * phase),
      0.44 * env * np.sin(phase),
      0.34 * env * np.sin(phase - 0.35),
  ])
  grip = 0.68 + 0.22 * env * np.sin(2.0 * phase)
  return q6, grip


def build_wave_trajectory(
    q_start: Sequence[float] | np.ndarray = Q_HOME,
    *,
    cycles: float = 3.5,
    animate_gripper: bool = True,
    dt_s: float = DT_S,
) -> np.ndarray:
  """Builds the full cute-wave trajectory (nod, wave, wink, park at Q_HOME)."""
  q0 = np.asarray(q_start, dtype=np.float64).copy()
  if q0.ndim != 1 or q0.size < 6 or not np.all(np.isfinite(q0)):
    raise ValueError(f"Expected finite >=6-DOF q_start, got {q0!r}")
  if np.any(q0[:6] < YAM_JOINT_MIN - STOP_TOL_RAD) or np.any(
      q0[:6] > YAM_JOINT_MAX + STOP_TOL_RAD
  ):
    raise ValueError(f"q_start outside YAM limits: {q0[:6].tolist()}")
  q0[:6] = np.clip(q0[:6], YAM_JOINT_MIN, YAM_JOINT_MAX)
  if q0.size > 6:
    q0[6] = float(np.clip(q0[6], 0.0, 1.0))

  def _pose(q_arm: np.ndarray, grip: float) -> np.ndarray:
    out = q0.copy()
    out[:6] = q_arm
    if animate_gripper and out.size > 6:
      out[6] = grip
    return out

  wave_q6, wave_grip = build_cute_wave_stream(cycles=cycles, dt_s=dt_s)
  wave_seg = np.tile(q0, (len(wave_q6), 1))
  wave_seg[:, :6] = wave_q6
  if animate_gripper and q0.size > 6:
    wave_seg[:, 6] = wave_grip

  p_ready_in = _pose(Q_WAVE_READY, 0.72)
  p_nod_dn = _pose(Q_NOD_DOWN, 0.28)
  p_nod_up = _pose(Q_NOD_UP, 0.82)
  p_wave_0 = _pose(Q_WAVE_READY, 0.68)
  p_wink = _pose(Q_WINK, 0.25)
  p_ready_out = _pose(Q_WAVE_READY, GRIPPER_OPEN)
  p_home = _pose(Q_HOME, GRIPPER_OPEN)

  segs = [
      cosine_segment(q0, p_ready_in, 1.55, 1.10, dt_s),
      cosine_segment(p_ready_in, p_nod_dn, 0.38, 1.35, dt_s)[1:],
      cosine_segment(p_nod_dn, p_nod_up, 0.38, 1.35, dt_s)[1:],
      cosine_segment(p_nod_up, p_wave_0, 0.36, 1.35, dt_s)[1:],
      wave_seg[1:],
      cosine_segment(p_wave_0, p_wink, 0.45, 1.25, dt_s)[1:],
      cosine_segment(p_wink, p_ready_out, 0.48, 1.20, dt_s)[1:],
      cosine_segment(p_ready_out, p_home, 2.10, 0.95, dt_s)[1:],
  ]
  return np.vstack(segs)


def _park_and_close(
    robot: Any,
    *,
    animate_gripper: bool,
    sim: bool,
    sleep_fn: Callable[[float], None],
    suppress_errors: bool = False,
) -> None:
  """Smoothly returns the arm to Q_HOME before calling robot.close()."""
  try:
    q_now = np.asarray(robot.get_joint_pos(), dtype=np.float64).copy()
    q_now[:6] = np.clip(q_now[:6], YAM_JOINT_MIN, YAM_JOINT_MAX)
    q_home = q_now.copy()
    q_home[:6] = Q_HOME
    if animate_gripper and q_home.size > 6:
      q_home[6] = GRIPPER_OPEN
    if float(np.max(np.abs(q_now[:6] - Q_HOME))) > 0.02:
      for q_cmd in cosine_segment(q_now, q_home, 1.5, 0.85, DT_S)[1:]:
        robot.command_joint_pos(q_cmd)
        if not sim:
          sleep_fn(DT_S)
    robot.command_joint_pos(q_home)
    if not sim:
      sleep_fn(0.25)
  except Exception:  # pylint: disable=broad-exception-caught
    if not suppress_errors:
      raise
  finally:
    robot.close()


def run_wave(
    *,
    channel: str = "can0",
    arm_type: str = "yam",
    gripper_type: str = "linear_4310",
    cycles: float = 3.5,
    sim: bool = False,
    interface: str | None = None,
    backend_channel: str | int | None = None,
    robot_factory: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
  """Executes the YAM cute-wave greeting in simulation or over CAN."""
  factory = robot_factory or get_yam_robot
  if factory is None:
    raise RuntimeError("i2rt is not installed; install i2rt before running.")
  if not sim and macos_can is not None:
    macos_can.install_mac_can(
        interface=interface, backend_channel=backend_channel
    )

  arm_arg = _resolve_enum(ArmType, arm_type)
  grip_arg = _resolve_enum(GripperType, gripper_type)
  grip_raw = grip_arg.value if hasattr(grip_arg, "value") else str(grip_arg)
  grip_name = str(grip_raw).strip().lower()
  animate_gripper = grip_name not in {
      "no_gripper",
      "none",
      "yam_teaching_handle",
      "yam_teaching_handle_v1",
      "yam_teaching_handle_v2",
  }

  if sim:
    robot = factory(arm_type=arm_arg, gripper_type=grip_arg, sim=True)
  else:
    robot = factory(
        channel=channel,
        arm_type=arm_arg,
        gripper_type=grip_arg,
        zero_gravity_mode=False,
    )

  failed = False
  try:
    q_init = np.asarray(robot.get_joint_pos(), dtype=np.float64)
    traj = build_wave_trajectory(
        q_init, cycles=cycles, animate_gripper=animate_gripper
    )
    if not sim and hasattr(robot, "move_joints"):
      robot.move_joints(traj[0], time_interval_s=0.4)
    t0 = time.monotonic()
    for step_idx, q_cmd in enumerate(traj):
      if (
          not sim
          and hasattr(robot, "motor_chain")
          and not robot.motor_chain.running
      ):
        raise RuntimeError("Motor chain stopped during wave.")
      robot.command_joint_pos(q_cmd)
      if not sim and step_idx % 5 == 0:
        q_live = np.asarray(robot.get_joint_pos(), dtype=np.float64)
        err = float(np.max(np.abs(q_live[:6] - q_cmd[:6])))
        if err > MAX_TRACK_ERR_RAD:
          hold = q_live.copy()
          if hold.size > 6 and q_cmd.size > 6:
            hold[6] = q_cmd[6]
          robot.command_joint_pos(hold)
          raise RuntimeError(
              f"Tracking error {err:.3f} > {MAX_TRACK_ERR_RAD} rad at step"
              f" {step_idx}"
          )
      if not sim:
        rem = (t0 + step_idx * DT_S) - time.monotonic()
        if rem > 0.0:
          sleep_fn(rem)
  except BaseException:
    failed = True
    raise
  finally:
    _park_and_close(
        robot,
        animate_gripper=animate_gripper,
        sim=sim,
        sleep_fn=sleep_fn,
        suppress_errors=failed,
    )


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Run a bring-up cute-wave smoke test on an I2RT YAM arm."
  )
  parser.add_argument("--channel", default="can0")
  parser.add_argument("--arm-type", default="yam")
  parser.add_argument("--gripper-type", default="linear_4310")
  parser.add_argument("--cycles", type=float, default=3.5)
  parser.add_argument("--sim", action="store_true")
  parser.add_argument("--interface", default=None)
  parser.add_argument("--backend-channel", default=None)
  args = parser.parse_args(argv)
  run_wave(
      channel=args.channel,
      arm_type=args.arm_type,
      gripper_type=args.gripper_type,
      cycles=args.cycles,
      sim=args.sim,
      interface=args.interface,
      backend_channel=args.backend_channel,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
