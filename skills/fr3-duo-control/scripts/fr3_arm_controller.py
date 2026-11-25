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

"""Franka Research 3 FCI controller, Desk audit, and 1 kHz worker process."""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import gc
import math
import multiprocessing as mp
import socket
from typing import Any

from fr3_config import (
    RobotConfig,
    normalize_robot_config,
    resolve_realtime_config,
)
import fr3_kinematics
import fr3_mjcf
from fr3_safety import (
    blend_start_pose_offset,
    clamp_q_to_padded_limits,
    compute_abort_deceleration_step,
    limit_rate_joint_positions,
    step_realtime_trajectory_index,
    unpack_o_t_ee,
)
import numpy as np

try:
  import mujoco  # pytype: disable=import-error
except ImportError:
  mujoco: Any = None

__all__ = [
    "ArmWorkerProcess",
    "audit_arm_desk_config",
    "compute_ext_wrench_base_6d",
    "compute_fk_pos_error_m",
    "compute_link0_in_base_transform_for_model",
    "disconnect_pylibfranka_robot",
    "probe_arm_readonly",
    "read_arm_joints_or_rest",
    "read_pylibfranka_state",
    "recover_pylibfranka_errors",
    "tcp_port_reachable",
    "validate_arm_readiness",
]


def _require_pylibfranka() -> Any:
  """Lazily imports Linux-only `pylibfranka` binding on demand."""
  import pylibfranka  # pytype: disable=import-error

  return pylibfranka


def _open_pylibfranka_robot(
    ip: str, realtime_cfg_name: str = "kIgnore"
) -> Any:
  """Opens a `pylibfranka.Robot` connection with the requested realtime mode."""
  pylibfranka = _require_pylibfranka()
  rt_enum = getattr(pylibfranka, "RealtimeConfig", None)
  if rt_enum is not None:
    rt_val = getattr(rt_enum, realtime_cfg_name, None)
    if rt_val is not None:
      return pylibfranka.Robot(ip, rt_val)
  return pylibfranka.Robot(ip)


def tcp_port_reachable(
    host: str, port: int = 1337, timeout_s: float = 0.25
) -> bool:
  """Checks whether TCP `host:port` accepts a connection within `timeout_s`."""
  try:
    with socket.create_connection((host, port), timeout=timeout_s):
      return True
  except OSError:
    return False


def disconnect_pylibfranka_robot(robot: Any) -> None:
  """Best-effort clean disconnect for a `pylibfranka.Robot` instance."""
  if robot is None:
    return
  for method_name in ("stop", "close", "disconnect"):
    fn = getattr(robot, method_name, None)
    if callable(fn):
      try:
        fn()
      except Exception:  # pylint: disable=broad-exception-caught
        pass


def _extract_pylibfranka_errors(raw_errors: Any) -> list[str]:
  """Extracts active errors from `pylibfranka.RobotState.current_errors`."""
  if raw_errors is None:
    return []
  if isinstance(raw_errors, bool):
    return ["unknown_error"] if raw_errors else []
  if isinstance(raw_errors, str):
    s = raw_errors.strip()
    return [s] if s else []
  if isinstance(raw_errors, Mapping):
    return [
        str(k).strip()
        for k, v in raw_errors.items()
        if bool(v) and str(k).strip()
    ]
  if isinstance(raw_errors, (Sequence, set)) and not isinstance(
      raw_errors, (str, bytes)
  ):
    return [str(x).strip() for x in raw_errors if str(x).strip()]

  found_flag = False
  active: list[str] = []
  for attr in dir(raw_errors):
    if attr.startswith("_"):
      continue
    try:
      val = getattr(raw_errors, attr)
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    if callable(val):
      continue
    if isinstance(val, (bool, int)):
      found_flag = True
      if val:
        active.append(attr)
  if active:
    return active
  if found_flag:
    return []
  try:
    has_truthy_bool = bool(raw_errors)
  except Exception:  # pylint: disable=broad-exception-caught
    has_truthy_bool = False
  if not has_truthy_bool:
    return []
  text = str(raw_errors).strip()
  if text and not text.startswith("<") and text != "namespace()":
    return [text]
  return ["unknown_error"]


def _state_has_errors(state: Mapping[str, Any]) -> tuple[bool, list[str]]:
  """Returns `(has_error_or_reflex, current_errors)` for a state dict."""
  errs = list(state.get("current_errors") or [])
  mode = str(state.get("robot_mode") or "").strip().lower()
  in_reflex = mode in (
      "kreflex",
      "reflex",
      "kautomaticerrorrecovery",
      "automaticerrorrecovery",
  )
  return bool(errs or in_reflex), errs


def _recovery_result(
    *,
    had_errors: bool = False,
    errors_before: Sequence[str] = (),
    recovered: bool = True,
    errors_after: Sequence[str] = (),
    error: str | None = None,
) -> dict[str, Any]:
  """Builds a uniform FCI error-recovery status payload."""
  return {
      "had_errors": had_errors,
      "errors_before": list(errors_before),
      "recovered": recovered,
      "errors_after": list(errors_after),
      "error": error,
  }


def read_pylibfranka_state(robot: Any) -> dict[str, Any]:
  """Reads one FCI state packet from a connected `pylibfranka.Robot`."""
  state = robot.read_once()
  q = [float(x) for x in getattr(state, "q")]
  q_d = [float(x) for x in getattr(state, "q_d", q)]
  dq = [float(x) for x in getattr(state, "dq", [0.0] * 7)]
  dq_d = [float(x) for x in getattr(state, "dq_d", dq)]
  ddq_d = [float(x) for x in getattr(state, "ddq_d", [0.0] * 7)]
  tau_j = [float(x) for x in getattr(state, "tau_J", [0.0] * 7)]
  o_t_ee = [
      float(x)
      for x in getattr(
          state,
          "O_T_EE",
          [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      )
  ]

  raw_mode = getattr(state, "robot_mode", None)
  if raw_mode is not None:
    robot_mode = (
        str(raw_mode.name)
        if hasattr(raw_mode, "name")
        else str(raw_mode).split(".")[-1]
    )
  else:
    robot_mode = ""

  current_errors = _extract_pylibfranka_errors(
      getattr(state, "current_errors", None)
  )

  m_ee_val = getattr(state, "m_ee", None)
  m_ee = float(m_ee_val) if m_ee_val is not None else None

  m_load_val = getattr(state, "m_load", None)
  m_load = float(m_load_val) if m_load_val is not None else None

  m_total_val = getattr(state, "m_total", None)
  if m_total_val is not None:
    m_total = float(m_total_val)
  elif m_ee is not None:
    m_total = float(m_ee + (m_load or 0.0))
  else:
    m_total = None

  f_x_cee_val = getattr(state, "F_x_Cee", None)
  f_x_cee = (
      [float(x) for x in f_x_cee_val] if f_x_cee_val is not None else None
  )

  i_ee_val = getattr(state, "I_ee", None)
  i_ee = [float(x) for x in i_ee_val] if i_ee_val is not None else None

  f_t_ee_val = getattr(state, "F_T_EE", None)
  f_t_ee = [float(x) for x in f_t_ee_val] if f_t_ee_val is not None else None

  o_ddp_o_val = getattr(state, "O_ddP_O", None)
  o_ddp_o = (
      [float(x) for x in list(o_ddp_o_val)[:3]]
      if o_ddp_o_val is not None
      else None
  )

  o_f_ext_val = getattr(state, "O_F_ext_hat_K", None)
  o_f_ext_hat_k = (
      [float(x) for x in o_f_ext_val] if o_f_ext_val is not None else None
  )

  tau_ext_val = getattr(state, "tau_ext_hat_filtered", None)
  tau_ext_hat_filtered = (
      [float(x) for x in tau_ext_val] if tau_ext_val is not None else None
  )

  return {
      "q": q,
      "q_d": q_d,
      "dq": dq,
      "dq_d": dq_d,
      "ddq_d": ddq_d,
      "tau_j": tau_j,
      "o_t_ee": o_t_ee,
      "robot_mode": robot_mode,
      "current_errors": current_errors,
      "m_ee": m_ee,
      "m_load": m_load,
      "m_total": m_total,
      "f_x_cee": f_x_cee,
      "i_ee": i_ee,
      "f_t_ee": f_t_ee,
      "o_ddp_o": o_ddp_o,
      "o_f_ext_hat_k": o_f_ext_hat_k,
      "tau_ext_hat_filtered": tau_ext_hat_filtered,
  }


def recover_pylibfranka_errors(robot: Any) -> dict[str, Any]:
  """Checks FCI `current_errors`/reflex and runs `automatic_error_recovery`."""
  has_err_before, before_errs = _state_has_errors(read_pylibfranka_state(robot))
  if not has_err_before:
    return _recovery_result()

  recovery_fn = getattr(robot, "automatic_error_recovery", None)
  if not callable(recovery_fn):
    return _recovery_result(
        had_errors=True,
        errors_before=before_errs,
        recovered=False,
        errors_after=before_errs,
        error="robot.automatic_error_recovery() is unavailable",
    )

  err_msg: str | None = None
  try:
    recovery_fn()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    err_msg = str(exc)

  try:
    has_err_after, after_errs = _state_has_errors(read_pylibfranka_state(robot))
  except Exception as exc:  # pylint: disable=broad-exception-caught
    combined = f"{err_msg}; {exc}" if err_msg else str(exc)
    return _recovery_result(
        had_errors=True,
        errors_before=before_errs,
        recovered=False,
        errors_after=before_errs,
        error=combined,
    )
  return _recovery_result(
      had_errors=True,
      errors_before=before_errs,
      recovered=(err_msg is None and not has_err_after),
      errors_after=after_errs,
      error=err_msg,
  )


def _audit_desk_gravity(
    arm: str,
    o_ddp_o: Sequence[float] | None,
    exp_g_arr: np.ndarray,
    expected_mount_tilt_deg: float,
    gravity_tol_deg: float,
    fci_ext_force_norm_n: float | None,
    fci_tau_ext_norm_nm: float | None,
    force_tol_n: float,
    tau_tol_nm: float,
    warnings: list[str],
    advisories: list[str],
) -> float | None:
  if o_ddp_o is None or len(o_ddp_o) < 3:
    return None
  v_meas = np.asarray(o_ddp_o[:3], dtype=np.float64)
  norm_meas = float(np.linalg.norm(v_meas))
  norm_exp = float(np.linalg.norm(exp_g_arr))
  if norm_meas <= 1e-3 or norm_exp <= 1e-3:
    return None
  cos_ang = np.clip(
      float(np.dot(v_meas, exp_g_arr) / (norm_meas * norm_exp)), -1.0, 1.0
  )
  gravity_angle_err_deg = float(math.degrees(math.acos(cos_ang)))
  if gravity_angle_err_deg <= gravity_tol_deg:
    return gravity_angle_err_deg

  z_down_angle_deg = float(
      math.degrees(math.acos(np.clip(float(-v_meas[2] / norm_meas), -1.0, 1.0)))
  )
  wrench_healthy = (
      fci_ext_force_norm_n is not None
      and fci_ext_force_norm_n <= force_tol_n
      and (fci_tau_ext_norm_nm is None or fci_tau_ext_norm_nm <= tau_tol_nm)
  )
  meas_rounded = [round(float(x), 4) for x in v_meas]
  exp_rounded = [round(float(x), 3) for x in exp_g_arr]
  if z_down_angle_deg <= 2.0 and wrench_healthy:
    advisories.append(
        f"{arm}: RobotState.O_ddP_O={meas_rounded} differs by "
        f"{gravity_angle_err_deg:.1f} deg from expected tilted mount gravity "
        f"{exp_rounded} (tilt={expected_mount_tilt_deg:.1f} deg), but "
        f"||O_F_ext_hat_K[:3]|| = {fci_ext_force_norm_n:.2f} N <= "
        f"{force_tol_n:.2f} N (firmware reports nominal [0, 0, -9.81] while "
        "Desk gravity compensation appears active)."
    )
  else:
    warnings.append(
        f"{arm}: RobotState.O_ddP_O={meas_rounded} differs by "
        f"{gravity_angle_err_deg:.1f} deg from expected tilted mount gravity "
        f"{exp_rounded} (tilt={expected_mount_tilt_deg:.1f} deg). Verify "
        "tilted shoulder mounting angle in Franka Desk (note: older libfranka "
        "builds may report static [0, 0, -9.81]; check fci_ext_force_norm_n)."
    )
  return gravity_angle_err_deg


def audit_arm_desk_config(
    arm: str,
    hw_state: dict[str, Any] | None,
    r_base_link0: np.ndarray,
    *,
    expected_mass_range_kg: tuple[float, float] = (0.5, 3.0),
    gravity_tol_deg: float = 8.0,
    force_tol_n: float = 6.5,
    tau_tol_nm: float = 4.5,
) -> dict[str, Any]:
  """Audits Franka Desk configuration (mode, payload, tilted mount, wrench)."""
  r_mat = np.asarray(r_base_link0, dtype=np.float64)
  exp_g_arr = r_mat.T @ np.array([0.0, 0.0, -9.81], dtype=np.float64)
  expected_mount_tilt_deg = math.degrees(
      math.acos(np.clip(float(r_mat[2, 2]), -1.0, 1.0))
  )

  if hw_state is None:
    return {
        "checked": False,
        "fci_connected": False,
        "robot_mode": "",
        "m_ee_kg": None,
        "m_total_kg": None,
        "f_x_cee_m": None,
        "i_ee_trace_kgm2": None,
        "o_ddp_o": None,
        "expected_o_ddp_o": [round(float(x), 4) for x in exp_g_arr],
        "expected_mount_tilt_deg": round(float(expected_mount_tilt_deg), 2),
        "gravity_angle_err_deg": None,
        "fci_ext_force_norm_n": None,
        "fci_tau_ext_norm_nm": None,
        "ok": True,
        "warnings": [],
        "advisories": [],
    }

  warnings: list[str] = []
  advisories: list[str] = []

  robot_mode = str(hw_state.get("robot_mode") or "").strip()
  if robot_mode and robot_mode not in ("kIdle", "Idle", "IDLE"):
    warnings.append(
        f"{arm}: FCI RobotState.robot_mode is {robot_mode!r} (expected 'kIdle'"
        " with joints unlocked and E-stop released)."
    )

  m_ee = hw_state.get("m_ee")
  m_total = hw_state.get("m_total")
  if m_total is not None:
    lo, hi = expected_mass_range_kg
    if not (lo <= float(m_total) <= hi):
      warnings.append(
          f"{arm}: Franka Desk m_total={float(m_total):.3f} kg outside expected"
          f" [{lo:.2f}, {hi:.2f}] kg for Robotiq 2F-85 + coupling + wrist"
          " camera (is Desk EE payload still 0 kg?)."
      )

  f_x_cee = hw_state.get("f_x_cee")
  if f_x_cee is not None:
    if float(np.linalg.norm(np.asarray(f_x_cee, dtype=np.float64))) < 0.005:
      warnings.append(
          f"{arm}: Franka Desk F_x_Cee={list(f_x_cee)} m is near zero"
          " (configure Robotiq + wrist camera CoM in Desk)."
      )

  i_ee = hw_state.get("i_ee")
  i_ee_trace: float | None = None
  if i_ee is not None and len(i_ee) == 9:
    i_ee_trace = float(abs(i_ee[0]) + abs(i_ee[4]) + abs(i_ee[8]))
    if i_ee_trace <= 1e-5:
      warnings.append(
          f"{arm}: Franka Desk rotational inertia I_ee trace is ~0 kg*m^2"
          " (configure EE inertia in Desk)."
      )

  o_f_ext_hat_k = hw_state.get("o_f_ext_hat_k")
  fci_ext_force_norm_n: float | None = None
  if o_f_ext_hat_k is not None and len(o_f_ext_hat_k) >= 3:
    fci_ext_force_norm_n = float(
        np.linalg.norm(np.asarray(o_f_ext_hat_k[:3], dtype=np.float64))
    )
    if fci_ext_force_norm_n > force_tol_n:
      warnings.append(
          f"{arm}: Franka controller external force residual"
          f" ||O_F_ext_hat_K[:3]|| = {fci_ext_force_norm_n:.2f} N exceeds"
          f" tolerance {force_tol_n:.2f} N (indicates unmodeled payload mass"
          " or tilted mount mismatch)."
      )

  tau_ext_hat_filtered = hw_state.get("tau_ext_hat_filtered")
  fci_tau_ext_norm_nm: float | None = None
  if tau_ext_hat_filtered is not None and len(tau_ext_hat_filtered) >= 7:
    fci_tau_ext_norm_nm = float(
        np.linalg.norm(np.asarray(tau_ext_hat_filtered, dtype=np.float64))
    )
    if fci_tau_ext_norm_nm > tau_tol_nm:
      warnings.append(
          f"{arm}: Franka controller external torque residual"
          f" ||tau_ext_hat_filtered|| = {fci_tau_ext_norm_nm:.2f} Nm exceeds"
          f" tolerance {tau_tol_nm:.2f} Nm."
      )

  o_ddp_o = hw_state.get("o_ddp_o")
  gravity_angle_err_deg = _audit_desk_gravity(
      arm,
      o_ddp_o,
      exp_g_arr,
      expected_mount_tilt_deg,
      gravity_tol_deg,
      fci_ext_force_norm_n,
      fci_tau_ext_norm_nm,
      force_tol_n,
      tau_tol_nm,
      warnings,
      advisories,
  )

  return {
      "checked": True,
      "fci_connected": True,
      "robot_mode": robot_mode,
      "m_ee_kg": float(m_ee) if m_ee is not None else None,
      "m_total_kg": float(m_total) if m_total is not None else None,
      "f_x_cee_m": [float(x) for x in f_x_cee] if f_x_cee is not None else None,
      "i_ee_trace_kgm2": (
          round(float(i_ee_trace), 6) if i_ee_trace is not None else None
      ),
      "o_ddp_o": (
          [round(float(x), 4) for x in o_ddp_o[:3]]
          if o_ddp_o is not None
          else None
      ),
      "expected_o_ddp_o": [round(float(x), 4) for x in exp_g_arr],
      "expected_mount_tilt_deg": round(float(expected_mount_tilt_deg), 2),
      "gravity_angle_err_deg": (
          round(float(gravity_angle_err_deg), 2)
          if gravity_angle_err_deg is not None
          else None
      ),
      "fci_ext_force_norm_n": (
          round(float(fci_ext_force_norm_n), 3)
          if fci_ext_force_norm_n is not None
          else None
      ),
      "fci_tau_ext_norm_nm": (
          round(float(fci_tau_ext_norm_nm), 3)
          if fci_tau_ext_norm_nm is not None
          else None
      ),
      "ok": len(warnings) == 0,
      "warnings": warnings,
      "advisories": advisories,
  }


def compute_link0_in_base_transform_for_model(
    mj_model: Any, side: str, mj_data: Any = None
) -> np.ndarray:
  """Computes `T_base_link0` 4x4 transform for `side` from an MjModel."""
  if mujoco is None:
    raise RuntimeError(
        "`mujoco` is required to compute T_base_link0 from an MjModel."
    )
  data = mj_data if mj_data is not None else mujoco.MjData(mj_model)
  if mj_data is None:
    data.qpos[:] = mj_model.qpos0
    mujoco.mj_forward(mj_model, data)

  link0_id = int(
      mujoco.mj_name2id(
          mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}/fr3_link0"
      )
  )
  t_base_link0 = np.eye(4, dtype=np.float64)
  if link0_id >= 0:
    t_base_link0[:3, :3] = np.asarray(
        data.xmat[link0_id], dtype=np.float64
    ).reshape(3, 3)
    t_base_link0[:3, 3] = np.asarray(data.xpos[link0_id], dtype=np.float64)
  return t_base_link0


def compute_fk_pos_error_m(
    tcp_pos_base: np.ndarray,
    t_base_link0: np.ndarray,
    o_t_ee_4x4: np.ndarray,
    tcp_offset_z_m: float,
    raw_f_t_ee: Sequence[float] | None,
    *,
    mj_model: Any = None,
    side: str = "",
    q_rad: Sequence[float] | None = None,
) -> float:
  """Returns min Euclidean discrepancy (m) between MuJoCo FK and FCI O_T_EE."""
  rq_pinch_from_flange = None
  if mj_model is not None and side and q_rad is not None:
    try:
      rq_pinch_from_flange = fr3_kinematics.compute_arm_pinch_from_flange(
          mj_model, side, q_rad
      )
    except (ValueError, KeyError, RuntimeError):
      rq_pinch_from_flange = None
  if rq_pinch_from_flange is None:
    rq_pinch_from_flange = np.array(
        [0.0, 0.0, 0.0108 + float(tcp_offset_z_m), 1.0], dtype=np.float64
    )
  candidate_errs = [
      float(np.linalg.norm(tcp_pos_base - (t_base_link0 @ o_t_ee_4x4)[:3, 3])),
      float(
          np.linalg.norm(
              tcp_pos_base
              - (t_base_link0 @ o_t_ee_4x4 @ rq_pinch_from_flange)[:3]
          )
      ),
  ]
  if raw_f_t_ee is not None and len(raw_f_t_ee) == 16:
    try:
      f_t_ee_4x4 = unpack_o_t_ee(raw_f_t_ee)
      o_t_f = o_t_ee_4x4 @ np.linalg.inv(f_t_ee_4x4)
      fci_from_flange = (t_base_link0 @ o_t_f @ rq_pinch_from_flange)[:3]
      candidate_errs.append(
          float(np.linalg.norm(tcp_pos_base - fci_from_flange))
      )
    except np.linalg.LinAlgError:
      pass
  return min(candidate_errs)


def compute_ext_wrench_base_6d(
    hw: Mapping[str, Any] | None,
    mj_model: Any,
    side: str,
    q_rad: np.ndarray,
    tau_j: np.ndarray,
    r_base_link0: np.ndarray,
) -> np.ndarray:
  """Resolves 6-D external wrench in Duo `base` frame from FCI or MJCF SVD."""
  raw_f_ext = hw.get("o_f_ext_hat_k") if hw is not None else None
  if raw_f_ext is not None and len(raw_f_ext) >= 6:
    w_o = -np.asarray(raw_f_ext[:6], dtype=np.float64)
    return np.concatenate(
        [r_base_link0 @ w_o[:3], r_base_link0 @ w_o[3:6]]
    ).astype(np.float64)
  return fr3_kinematics.estimate_external_wrench_base(
      mj_model, side, q_rad, tau_j
  )


def validate_arm_readiness(
    arm: str, arm_state: Mapping[str, Any]
) -> list[str]:
  """Returns FK discrepancy, free-air force, and Desk config warnings."""
  warnings: list[str] = []
  desk_audit = arm_state.get("desk_audit")
  if isinstance(desk_audit, Mapping) and desk_audit.get("warnings"):
    for w in desk_audit["warnings"]:
      if w not in warnings:
        warnings.append(str(w))

  fk_err_m = float(arm_state.get("fk_pos_error_m", 0.0))
  if fk_err_m > 0.005:
    warnings.append(
        f"{arm}: O_T_EE vs MJCF FK discrepancy {fk_err_m * 1000.0:.2f} mm > 5.0"
        " mm."
    )

  if "tau_ext_base_6d" in arm_state:
    force_norm_n = float(
        np.linalg.norm(
            np.asarray(arm_state["tau_ext_base_6d"], dtype=np.float64)[:3]
        )
    )
  else:
    force_norm_n = float(arm_state.get("free_air_force_norm_n", 0.0))
  if force_norm_n > 6.5:
    warnings.append(
        f"{arm}: free-air tau_ext force norm {force_norm_n:.2f} N > 6.5 N "
        "(verify tilted Duo mounting gravity and EE payload in Franka Desk)."
    )
  cur_errs = arm_state.get("current_errors") or []
  if cur_errs:
    warnings.append(
        f"{arm}: active FCI current_errors: "
        f"{', '.join(str(e) for e in cur_errs)}."
    )
  return warnings


def probe_arm_readonly(
    side: str,
    ip: str,
    mj_model: Any,
    *,
    rest_pose_rad: Sequence[float] | None = None,
    tcp_offset_z_m: float = 0.145,
    realtime_cfg_name: str = "kIgnore",
    timeout_s: float = 0.25,
    tcp_reachable: bool | None = None,
) -> dict[str, Any]:
  """Probes one FR3 arm read-only via pylibfranka, falling back to rest FK."""
  reachable = (
      tcp_port_reachable(ip, timeout_s=timeout_s)
      if tcp_reachable is None
      else bool(tcp_reachable)
  )
  q_fallback = np.asarray(
      rest_pose_rad
      if rest_pose_rad is not None
      else fr3_mjcf.DEFAULT_REST_POSE_RAD[side],
      dtype=np.float64,
  )

  if not reachable:
    tcp_pos, _ = fr3_kinematics.compute_arm_fk(mj_model, side, q_fallback)
    err_msg = f"FR3 {side} arm ({ip}): TCP port 1337 unreachable."
    return {
        "connected": False,
        "connection_error": err_msg,
        "q_rad": [round(float(x), 6) for x in q_fallback],
        "tcp_pos_base_m": [round(float(x), 5) for x in tcp_pos],
        "fk_pos_error_m": 0.0,
        "free_air_force_norm_n": 0.0,
        "current_errors": [],
        "desk_audit": None,
        "warnings": [err_msg],
        "advisories": [],
    }

  robot: Any = None
  try:
    robot = _open_pylibfranka_robot(ip, realtime_cfg_name)
    hw = read_pylibfranka_state(robot)
    q_rad = np.asarray(hw["q"], dtype=np.float64)
    tau_j = np.asarray(hw["tau_j"], dtype=np.float64)
    o_t_ee_4x4 = unpack_o_t_ee(hw["o_t_ee"])

    t_base_link0 = compute_link0_in_base_transform_for_model(mj_model, side)
    r_base_link0 = t_base_link0[:3, :3]
    tcp_pos_base, _ = fr3_kinematics.compute_arm_fk(mj_model, side, q_rad)
    fk_err_m = compute_fk_pos_error_m(
        tcp_pos_base,
        t_base_link0,
        o_t_ee_4x4,
        tcp_offset_z_m,
        hw.get("f_t_ee"),
        mj_model=mj_model,
        side=side,
        q_rad=q_rad,
    )
    wrench_6d = compute_ext_wrench_base_6d(
        hw, mj_model, side, q_rad, tau_j, r_base_link0
    )
    force_norm_n = float(np.linalg.norm(wrench_6d[:3]))
    desk_audit = audit_arm_desk_config(side, hw, r_base_link0)
    readout = {
        "connected": True,
        "connection_error": None,
        "q_rad": [round(float(x), 6) for x in q_rad],
        "tcp_pos_base_m": [round(float(x), 5) for x in tcp_pos_base],
        "fk_pos_error_m": round(fk_err_m, 6),
        "free_air_force_norm_n": round(force_norm_n, 3),
        "current_errors": list(hw.get("current_errors") or []),
        "desk_audit": desk_audit,
        "advisories": list((desk_audit or {}).get("advisories") or []),
    }
    readout["warnings"] = validate_arm_readiness(side, readout)
    return readout
  except Exception as exc:  # pylint: disable=broad-exception-caught
    tcp_pos, _ = fr3_kinematics.compute_arm_fk(mj_model, side, q_fallback)
    err_msg = str(exc)
    return {
        "connected": False,
        "connection_error": err_msg,
        "q_rad": [round(float(x), 6) for x in q_fallback],
        "tcp_pos_base_m": [round(float(x), 5) for x in tcp_pos],
        "fk_pos_error_m": 0.0,
        "free_air_force_norm_n": 0.0,
        "current_errors": [],
        "desk_audit": None,
        "warnings": [err_msg],
        "advisories": [],
    }
  finally:
    disconnect_pylibfranka_robot(robot)


def read_arm_joints_or_rest(
    config: RobotConfig | Mapping[str, Any],
    *,
    realtime_mode: str = "auto",
    timeout_s: float = 0.25,
) -> tuple[list[float], list[float], list[str]]:
  """Reads live joint angles per reachable arm, else `rest_pose_rad`."""
  cfg = normalize_robot_config(config)
  rt_info = resolve_realtime_config(realtime_mode)
  rt_name = str(rt_info["realtime_config"])
  out_q: dict[str, list[float]] = {}
  warnings: list[str] = []

  for side in ("left", "right"):
    arm_cfg = cfg.arms[side]
    fallback = list(arm_cfg.rest_pose_rad)
    if not arm_cfg.enabled:
      out_q[side] = fallback
      continue
    if not tcp_port_reachable(arm_cfg.ip, timeout_s=timeout_s):
      warnings.append(
          f"{side} ({arm_cfg.ip}): TCP port 1337 unreachable; using rest pose"
      )
      out_q[side] = fallback
      continue
    robot: Any = None
    try:
      robot = _open_pylibfranka_robot(arm_cfg.ip, rt_name)
      hw = read_pylibfranka_state(robot)
      out_q[side] = [float(x) for x in hw["q"]]
    except Exception as exc:  # pylint: disable=broad-exception-caught
      warnings.append(f"{side} ({arm_cfg.ip}): {exc}; using rest pose")
      out_q[side] = fallback
    finally:
      disconnect_pylibfranka_robot(robot)

  return out_q["left"], out_q["right"], warnings


def _read_ctrl_state(
    st: Any,
    fallback_q: np.ndarray,
    fallback_dq: np.ndarray,
    fallback_ddq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  return (
      np.asarray(getattr(st, "q_d", fallback_q), dtype=np.float64),
      np.asarray(getattr(st, "dq_d", fallback_dq), dtype=np.float64),
      np.asarray(getattr(st, "ddq_d", fallback_ddq), dtype=np.float64),
  )


def _handle_worker_abort(
    ctrl: Any,
    pylibfranka: Any,
    ref_q: np.ndarray,
    ref_dq: np.ndarray,
    ref_ddq: np.ndarray,
) -> np.ndarray:
  """Executes 250 ms Hermite deceleration and settling to zero velocity."""
  stop_steps = 250
  hermite_q = ref_q.copy()
  dq_at_abort = ref_dq.copy()
  cur_q, cur_dq, cur_ddq = ref_q.copy(), ref_dq.copy(), ref_ddq.copy()
  for s_idx in range(1, stop_steps + 1):
    if s_idx > 1:
      abort_state, _ = ctrl.readOnce()
      cur_q, cur_dq, cur_ddq = _read_ctrl_state(
          abort_state, cur_q, cur_dq, cur_ddq
      )
    hermite_q, hermite_dq = compute_abort_deceleration_step(
        hermite_q,
        dq_at_abort,
        s_idx,
        stop_steps=stop_steps,
        dt_s=0.001,
    )
    cmd_q, cur_dq, cur_ddq = limit_rate_joint_positions(
        hermite_q,
        cur_q,
        cur_dq,
        cur_ddq,
        dq_ff=hermite_dq,
        dt_s=0.001,
    )
    cur_q = cmd_q
    ctrl.writeOnce(pylibfranka.JointPositions(cmd_q.tolist()))

  q_hold = cur_q.copy()
  for settle_idx in range(250):
    settle_state, _ = ctrl.readOnce()
    cur_q, cur_dq, cur_ddq = _read_ctrl_state(
        settle_state, cur_q, cur_dq, cur_ddq
    )
    if (
        float(np.max(np.abs(cur_dq))) <= 1e-5
        and float(np.max(np.abs(cur_ddq))) <= 1e-2
    ):
      jp_done = pylibfranka.JointPositions(cur_q.tolist())
      jp_done.motion_finished = True
      ctrl.writeOnce(jp_done)
      break
    cmd_q, cur_dq, cur_ddq = limit_rate_joint_positions(
        q_hold,
        cur_q,
        cur_dq,
        cur_ddq,
        dq_ff=np.zeros(7, dtype=np.float64),
        dt_s=0.001,
    )
    cur_q = cmd_q
    jp = pylibfranka.JointPositions(cmd_q.tolist())
    if settle_idx == 249:
      jp.motion_finished = True
    ctrl.writeOnce(jp)
  return cur_q


def _run_worker_trajectory(
    robot: Any,
    pylibfranka: Any,
    traj_raw: np.ndarray,
    abort_event: Any,
) -> dict[str, Any]:
  """Streams one `(N, 7)` spline over 1 kHz FCI with abort decel & trace."""
  travelled: list[list[float]] = []
  aborted = False
  tick_trace: collections.deque[str] = collections.deque(maxlen=8)
  gc.disable()
  try:
    coll_fn = getattr(robot, "set_collision_behavior", None)
    if callable(coll_fn):
      coll_fn([20.0] * 7, [20.0] * 7, [20.0] * 6, [20.0] * 6)

    ctrl = robot.start_joint_position_control(
        pylibfranka.ControllerMode.JointImpedance
    )
    n_total = int(traj_raw.shape[0])
    traj_exec = traj_raw
    time_elapsed_s = 0.0
    is_first_tick = True
    prev_q = traj_raw[0].copy()
    prev_dq = np.zeros(7, dtype=np.float64)
    prev_ddq = np.zeros(7, dtype=np.float64)
    tick_num = 0

    while True:
      state, duration = ctrl.readOnce()
      tick_num += 1
      dur_ms = (
          float(duration.to_sec()) * 1000.0
          if hasattr(duration, "to_sec")
          else 1.0
      )

      if is_first_tick:
        q_anchor = getattr(state, "q_d", getattr(state, "q", traj_raw[0]))
        traj_exec = blend_start_pose_offset(traj_raw, q_anchor)
        prev_q = np.asarray(q_anchor, dtype=np.float64).copy()
        prev_dq = np.asarray(
            getattr(state, "dq_d", np.zeros(7)), dtype=np.float64
        ).copy()
        prev_ddq = np.asarray(
            getattr(state, "ddq_d", np.zeros(7)), dtype=np.float64
        ).copy()

      time_elapsed_s, idx, _ = step_realtime_trajectory_index(
          time_elapsed_s,
          duration,
          is_first_tick=is_first_tick,
          n_total=n_total,
          dt_s=0.001,
      )
      is_first_tick = False

      if idx % 25 == 0 or idx == n_total - 1:
        travelled.append(clamp_q_to_padded_limits(prev_q).tolist())

      ref_q, ref_dq, ref_ddq = _read_ctrl_state(
          state, prev_q, prev_dq, prev_ddq
      )

      if abort_event.is_set():
        aborted = True
        prev_q = _handle_worker_abort(ctrl, pylibfranka, ref_q, ref_dq, ref_ddq)
        travelled.append(clamp_q_to_padded_limits(prev_q).tolist())
        break

      target_q = traj_exec[idx]
      target_dq_ff = (
          (traj_exec[idx] - traj_exec[idx - 1]) / 0.001
          if 0 < idx < n_total - 1
          else np.zeros(7, dtype=np.float64)
      )
      cmd_q, prev_dq, prev_ddq = limit_rate_joint_positions(
          target_q, ref_q, ref_dq, ref_ddq, dq_ff=target_dq_ff, dt_s=0.001
      )
      prev_q = cmd_q
      at_goal = (
          idx >= n_total - 1
          and float(np.max(np.abs(ref_q - traj_exec[-1]))) <= 1e-4
          and float(np.max(np.abs(ref_dq))) <= 1e-5
          and float(np.max(np.abs(prev_dq))) <= 1e-5
          and float(np.max(np.abs(ref_ddq))) <= 1e-2
      )
      if at_goal or time_elapsed_s >= (n_total * 0.001 + 5.0):
        jp = pylibfranka.JointPositions(ref_q.tolist())
        jp.motion_finished = True
        tick_trace.append(
            f"t={tick_num} idx={idx}/{n_total} dur={dur_ms:.2f}ms "
            f"FIN d_q={float(np.max(np.abs(cmd_q - ref_q))):.2e} "
            f"ref_dq={float(np.max(np.abs(ref_dq))):.2e} "
            f"ref_ddq={float(np.max(np.abs(ref_ddq))):.2e}"
        )
        ctrl.writeOnce(jp)
        break
      tick_trace.append(
          f"t={tick_num} idx={idx}/{n_total} dur={dur_ms:.2f}ms "
          f"d_q={float(np.max(np.abs(cmd_q - ref_q))):.2e} "
          f"ref_dq={float(np.max(np.abs(ref_dq))):.2e} "
          f"cmd_dq={float(np.max(np.abs(prev_dq))):.2e} "
          f"ref_ddq={float(np.max(np.abs(ref_ddq))):.2e} "
          f"cmd_ddq={float(np.max(np.abs(prev_ddq))):.2e}"
      )
      ctrl.writeOnce(pylibfranka.JointPositions(cmd_q.tolist()))

    final_state = read_pylibfranka_state(robot)
    return {
        "ok": True,
        "aborted": aborted,
        "travelled": travelled,
        "state": final_state,
    }
  except Exception as exc:  # pylint: disable=broad-exception-caught
    trace_tail = " | ".join(tick_trace)
    err_str = f"{exc} [trace: {trace_tail}]" if trace_tail else str(exc)
    return {"ok": False, "error": err_str, "travelled": travelled}
  finally:
    gc.enable()
    gc.collect()


def _arm_worker_loop(
    ip: str,
    realtime_cfg_name: str,
    cmd_conn: Any,
    abort_event: Any,
) -> None:
  """Dedicated 1 kHz `pylibfranka` child process avoiding GIL contention."""
  robot: Any = None
  try:
    pylibfranka = _require_pylibfranka()
    robot = _open_pylibfranka_robot(ip, realtime_cfg_name)

    # Initial worker spawn only opens pylibfranka; callers send op="recover"
    # explicitly when clearing FCI errors before actuation.
    cmd_conn.send({"ok": True})
  except Exception as exc:  # pylint: disable=broad-exception-caught
    disconnect_pylibfranka_robot(robot)
    cmd_conn.send({"ok": False, "error": str(exc)})
    return

  try:
    while True:
      msg = cmd_conn.recv()
      op = msg.get("op")
      if op == "close":
        disconnect_pylibfranka_robot(robot)
        robot = None
        cmd_conn.send({"ok": True})
        break
      if op in ("read", "recover"):
        try:
          if op == "read":
            cmd_conn.send({"ok": True, "state": read_pylibfranka_state(robot)})
          else:
            cmd_conn.send(
                {"ok": True, "result": recover_pylibfranka_errors(robot)}
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
          cmd_conn.send({"ok": False, "error": str(exc)})
        continue
      if op == "trajectory":
        traj_raw = np.asarray(msg["traj"], dtype=np.float64)
        cmd_conn.send(
            _run_worker_trajectory(robot, pylibfranka, traj_raw, abort_event)
        )
  finally:
    disconnect_pylibfranka_robot(robot)
    robot = None


class ArmWorkerProcess:
  """Parent-side handle to a dedicated 1 kHz per-arm child process."""

  def __init__(self, arm: str, ip: str, realtime_cfg_name: str) -> None:
    """Configures IPC pipe and abort event for one arm worker process."""
    self.arm = arm
    self.ip = ip
    self.realtime_cfg_name = realtime_cfg_name
    self._ctx = mp.get_context("spawn")
    self._parent_conn: Any = None
    self._proc: Any = None
    self._abort_event = self._ctx.Event()

  def start(self, timeout_s: float = 10.0) -> None:
    """Spawns the 1 kHz child process and waits for FCI connection."""
    parent_conn, child_conn = self._ctx.Pipe()
    self._parent_conn = parent_conn
    self._proc = self._ctx.Process(
        target=_arm_worker_loop,
        args=(self.ip, self.realtime_cfg_name, child_conn, self._abort_event),
        daemon=True,
    )
    self._proc.start()
    if not parent_conn.poll(timeout_s):
      self.stop()
      raise TimeoutError(
          f"Timed out connecting to FR3 {self.arm} arm at {self.ip}"
      )
    reply = parent_conn.recv()
    if not reply.get("ok"):
      self.stop()
      raise RuntimeError(
          f"Failed to initialize FR3 {self.arm} arm ({self.ip}): "
          f"{reply.get('error')}"
      )

  def _send_rpc(self, op: str, action: str) -> dict[str, Any]:
    self._parent_conn.send({"op": op})
    reply = self._parent_conn.recv()
    if not reply.get("ok"):
      raise RuntimeError(
          f"FR3 {self.arm} {action} failed: {reply.get('error')}"
      )
    return reply

  def read_state(self) -> dict[str, Any]:
    """Requests one read-only FCI state snapshot from the worker."""
    return self._send_rpc("read", "state read")["state"]

  def recover_errors(self) -> dict[str, Any]:
    """Checks FCI errors/reflex and runs automatic recovery if needed."""
    return self._send_rpc("recover", "error recovery")["result"]

  def send_trajectory_async(self, traj: np.ndarray) -> None:
    """Clears the abort flag and sends a 1 kHz joint spline to execute."""
    self._abort_event.clear()
    self._parent_conn.send({"op": "trajectory", "traj": traj})

  def signal_abort(self) -> None:
    """Signals the active worker loop to decelerate smoothly and stop."""
    self._abort_event.set()

  def poll_done(self, timeout_s: float = 0.005) -> dict[str, Any] | None:
    """Polls for trajectory completion and returns the worker reply if ready."""
    if self._parent_conn.poll(timeout_s):
      return self._parent_conn.recv()
    return None

  def stop(self) -> None:
    """Aborts any motion, closes the worker process, and cleans up IPC."""
    try:
      if self._proc is not None and self._proc.is_alive():
        try:
          self._abort_event.set()
          if self._parent_conn is not None:
            self._parent_conn.send({"op": "close"})
            if self._parent_conn.poll(1.0):
              self._parent_conn.recv()
        except (OSError, EOFError):
          # Ignore broken/closed pipe errors when worker has already exited.
          pass
        self._proc.join(timeout=1.5)
        if self._proc.is_alive():
          self._proc.terminate()
          self._proc.join(timeout=0.5)
    finally:
      if self._parent_conn is not None:
        try:
          self._parent_conn.close()
        except OSError:
          pass
        self._parent_conn = None
      if self._proc is not None:
        try:
          self._proc.close()
        except (OSError, ValueError):
          pass
        self._proc = None
