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

"""Trajectory planning, rate limiting, abort watching, and collision checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import select
import signal
import sys
import threading
from typing import Any

from fr3_config import (
    MIN_INTER_ARM_CLEARANCE_M,
    RobotConfig,
    normalize_robot_config,
)
import fr3_mjcf
import numpy as np

try:
  import mujoco  # pytype: disable=import-error
except ImportError:
  mujoco: Any = None

FR3_HW_MAX_VEL_RAD_S = np.array(
    [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], dtype=np.float64
)
# J1-J2 capped conservatively at 7.5 rad/s^2 to avoid
# communication_constraints_violation under 1 kHz position streaming.
FR3_HW_MAX_ACC_RAD_S2 = np.array(
    [7.5, 7.5, 9.5, 9.5, 9.5, 9.5, 9.5], dtype=np.float64
)
FR3_HW_MAX_JERK_RAD_S3 = np.array(
    [3500.0, 3500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0],
    dtype=np.float64,
)
INITIAL_TABLE_LIFT_TOLERANCE_M = 0.015

__all__ = [
    "AbortWatcher",
    "Fr3DuoSafetyChecker",
    "FR3_HW_MAX_ACC_RAD_S2",
    "FR3_HW_MAX_JERK_RAD_S3",
    "FR3_HW_MAX_VEL_RAD_S",
    "MIN_INTER_ARM_CLEARANCE_M",
    "blend_start_pose_offset",
    "clamp_q_to_padded_limits",
    "compute_abort_deceleration_step",
    "generate_quintic_trajectory",
    "limit_rate_joint_positions",
    "simplify_joint_polyline",
    "step_realtime_trajectory_index",
    "unpack_o_t_ee",
    "validate_joint_waypoints",
]


def unpack_o_t_ee(o_t_ee_col_major_16: Sequence[float]) -> np.ndarray:
  """Unpacks Franka's 16-element column-major `O_T_EE` to `(4, 4)`."""
  arr = np.asarray(o_t_ee_col_major_16, dtype=np.float64).reshape(-1)
  if arr.shape != (16,):
    raise ValueError(
        f"Expected 16 elements for column-major O_T_EE, got {arr.shape}"
    )
  return np.ascontiguousarray(arr.reshape((4, 4), order="F"))


def _broadcast_7d_positive(
    val: float | Sequence[float], name: str
) -> np.ndarray:
  arr = np.asarray(val, dtype=np.float64)
  if arr.ndim == 0:
    out = np.full(7, float(arr), dtype=np.float64)
  else:
    out = arr.reshape(-1)
    if out.shape != (7,):
      raise ValueError(f"{name} must be a scalar or 7-element sequence")
  if np.any(out <= 0.0) or not np.all(np.isfinite(out)):
    raise ValueError(f"{name} values must be finite and strictly positive")
  return out


def _padded_joint_bounds(
    *, apply_margins: bool = True
) -> tuple[np.ndarray, np.ndarray]:
  margins = (
      np.asarray(fr3_mjcf.FR3_IK_MARGINS_RAD, dtype=np.float64)
      if apply_margins
      else np.zeros(7, dtype=np.float64)
  )
  return (
      fr3_mjcf.FR3_JOINT_LIMITS_RAD[:, 0] + margins,
      fr3_mjcf.FR3_JOINT_LIMITS_RAD[:, 1] - margins,
  )


def validate_joint_waypoints(
    waypoints_rad: np.ndarray, *, apply_margins: bool = True
) -> None:
  """Verifies all `(M, 7)` waypoints lie inside padded FR3 joint limits."""
  q_min, q_max = _padded_joint_bounds(apply_margins=apply_margins)

  arr = np.asarray(waypoints_rad, dtype=np.float64)
  if arr.size == 0:
    return
  non_finite = np.argwhere(~np.all(np.isfinite(arr), axis=1))
  if non_finite.size > 0:
    wp_idx = int(non_finite[0, 0])
    raise ValueError(
        f"Non-finite joint angle at waypoint {wp_idx}: {arr[wp_idx]}"
    )
  bad = np.argwhere((arr < q_min - 1e-9) | (arr > q_max + 1e-9))
  if bad.size > 0:
    wp_idx, j_idx = int(bad[0, 0]), int(bad[0, 1])
    raise ValueError(
        f"Waypoint {wp_idx} joint J{j_idx + 1}={arr[wp_idx, j_idx]:.4f} rad "
        f"violates safe FR3 margin bounds [{q_min[j_idx]:.4f}, "
        f"{q_max[j_idx]:.4f}] rad"
    )


def generate_quintic_trajectory(
    waypoints_rad: Sequence[Sequence[float]],
    *,
    dt_s: float = 0.001,
    max_vel_rad_s: float | Sequence[float] = 0.26,
    max_acc_rad_s2: float | Sequence[float] = 2.0,
    check_limits: bool = True,
) -> np.ndarray:
  """Generates a `C^2`-continuous `(N, 7)` rest-to-rest quintic trajectory.

  Guarantees:
  - Exact zero velocity and acceleration at start, transitions, and end.
  - Peak velocity `<= min(max_vel_rad_s, FR3_HW_MAX_VEL_RAD_S)`.
  - Peak acceleration `<= min(max_acc_rad_s2, FR3_HW_MAX_ACC_RAD_S2)`.
  - Padded FR3 joint-limit enforcement (`0.100 rad` margin on J4, `0.025 rad`
    on J1-J3/J5-J7) when `check_limits=True`.
  """
  dt = float(dt_s)
  if dt <= 0.0 or not math.isfinite(dt):
    raise ValueError(f"dt_s must be positive and finite, got {dt_s}")

  wps = np.asarray(waypoints_rad, dtype=np.float64)
  if wps.ndim != 2 or wps.shape[0] < 1 or wps.shape[1] != 7:
    raise ValueError(
        f"Expected (M, 7) waypoints array with M >= 1, got shape {wps.shape}"
    )

  if check_limits:
    validate_joint_waypoints(wps, apply_margins=True)

  v_cap = np.minimum(
      _broadcast_7d_positive(max_vel_rad_s, "max_vel_rad_s"),
      FR3_HW_MAX_VEL_RAD_S,
  )
  a_cap = np.minimum(
      _broadcast_7d_positive(max_acc_rad_s2, "max_acc_rad_s2"),
      FR3_HW_MAX_ACC_RAD_S2,
  )

  if wps.shape[0] == 1:
    return np.repeat(wps[:1], 5, axis=0)

  # Quintic s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5:
  # max |s'(tau)| = 1.875, max |s''(tau)| = 10 / sqrt(3) ~= 5.77350269.
  vel_factor = 1.875 * 1.02
  acc_factor = (10.0 / math.sqrt(3.0)) * 1.02

  segments: list[np.ndarray] = [wps[0:1], wps[0:1]]
  for seg_idx in range(wps.shape[0] - 1):
    q0 = wps[seg_idx]
    q1 = wps[seg_idx + 1]
    dq = q1 - q0
    abs_dq = np.abs(dq)
    if float(np.max(abs_dq)) < 1e-9:
      continue

    t_vel = float(np.max(vel_factor * abs_dq / v_cap))
    t_acc = float(np.max(np.sqrt(acc_factor * abs_dq / a_cap)))
    duration_s = max(t_vel, t_acc, 4.0 * dt)
    num_steps = max(4, int(math.ceil(duration_s / dt)))

    tau = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float64)[1:]
    tau3 = tau * tau * tau
    s = tau3 * (10.0 + tau * (-15.0 + 6.0 * tau))
    seg_points = q0[None, :] + s[:, None] * dq[None, :]
    segments.append(seg_points)

  segments.append(wps[-1:])
  segments.append(wps[-1:])
  traj = np.vstack(segments)
  if check_limits:
    validate_joint_waypoints(traj, apply_margins=True)
  return traj


class AbortWatcher:
  """Monitors `SIGINT` and non-blocking `stdin` (`q`+Enter) for user abort."""

  def __init__(self, *, watch_stdin: bool = True) -> None:
    """Initializes the abort watcher state and optional stdin polling."""
    self._watch_stdin = bool(watch_stdin)
    self._aborted = False
    self._reason = ""
    self._lock = threading.Lock()
    self._prev_sigint: Any = None
    self._installed_sigint = False

  @property
  def aborted(self) -> bool:
    """Returns True once an abort signal or `q` input has been triggered."""
    with self._lock:
      return self._aborted

  @property
  def reason(self) -> str:
    """Returns the human-readable reason string for the triggered abort."""
    with self._lock:
      return self._reason

  def trigger(self, reason: str = "manual") -> None:
    """Marks the watcher aborted so active motion loops unwind cleanly."""
    with self._lock:
      if not self._aborted:
        self._aborted = True
        self._reason = str(reason or "manual")

  def poll(self) -> bool:
    """Checks abort flag and non-blocking stdin for `q`+`Enter`."""
    if self.aborted:
      return True
    if not self._watch_stdin:
      return False
    stdin = sys.stdin
    if stdin is None:
      return False
    try:
      fd = stdin.fileno()
      ready, _, _ = select.select([fd], [], [], 0.0)
      if ready:
        line = stdin.readline()
        if line and line.strip().lower().startswith("q"):
          self.trigger("stdin_q")
          return True
    except (OSError, ValueError, AttributeError):
      # Non-file stdin (e.g. redirected StringIO in tests) has no fileno.
      pass
    return self.aborted

  def __enter__(self) -> AbortWatcher:
    if threading.current_thread() is threading.main_thread():
      try:
        self._prev_sigint = signal.getsignal(signal.SIGINT)

        def _on_sigint(_signum: int, _frame: Any) -> None:
          self.trigger("SIGINT")

        signal.signal(signal.SIGINT, _on_sigint)
        self._installed_sigint = True
      except (OSError, ValueError):
        self._installed_sigint = False
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    if self._installed_sigint and self._prev_sigint is not None:
      try:
        signal.signal(signal.SIGINT, self._prev_sigint)
      except (OSError, ValueError):
        # Ignore signal restoration errors on teardown outside main loop.
        pass
      self._installed_sigint = False


def blend_start_pose_offset(
    traj: np.ndarray,
    q_d_initial: Sequence[float],
    *,
    max_offset_rad: float = 0.10,
) -> np.ndarray:
  """Blends q_d initial offset to zero over the first 250 ms of trajectory."""
  traj_arr = np.asarray(traj, dtype=np.float64)
  if traj_arr.ndim != 2 or traj_arr.shape[1] != 7:
    raise ValueError(
        f"Expected trajectory of shape (N, 7), got {traj_arr.shape}"
    )
  if len(traj_arr) == 0:
    return traj_arr
  if len(traj_arr) == 1:
    return np.asarray(q_d_initial, dtype=np.float64).reshape(1, 7)

  delta0 = np.asarray(q_d_initial, dtype=np.float64).reshape(7) - traj_arr[0]
  max_err = float(np.max(np.abs(delta0)))
  if max_err <= 1e-6:
    return traj_arr
  if max_err > max_offset_rad:
    raise RuntimeError(
        f"Initial robot q_d differs from trajectory start by {max_err:.4f} rad"
        f" > {max_offset_rad:.4f} rad"
    )

  blend_steps = min(max(1, len(traj_arr) - 1), 250)
  s = np.clip(
      np.arange(len(traj_arr), dtype=np.float64) / float(blend_steps), 0.0, 1.0
  )
  w = 1.0 - 10.0 * s**3 + 15.0 * s**4 - 6.0 * s**5
  blended = traj_arr.copy()
  blended += w[:, None] * delta0[None, :]
  return blended


def clamp_q_to_padded_limits(
    q_vec: Sequence[float] | np.ndarray,
) -> np.ndarray:
  """Clamps a 7-DOF joint vector into FR3's padded IK margin bounds."""
  q_min, q_max = _padded_joint_bounds(apply_margins=True)
  return np.clip(np.asarray(q_vec, dtype=np.float64).reshape(7), q_min, q_max)


def _rdp(seq: list[np.ndarray], tol_rad: float) -> list[np.ndarray]:
  if len(seq) <= 2:
    return seq
  a = seq[0]
  b = seq[-1]
  ab = b - a
  ab_norm_sq = float(np.dot(ab, ab))
  max_dist = -1.0
  split_idx = 0
  for idx in range(1, len(seq) - 1):
    ap = seq[idx] - a
    if ab_norm_sq <= 1e-12:
      dist = float(np.linalg.norm(ap))
    else:
      t = min(1.0, max(0.0, float(np.dot(ap, ab)) / ab_norm_sq))
      dist = float(np.linalg.norm(ap - t * ab))
    if dist > max_dist:
      max_dist = dist
      split_idx = idx
  if max_dist > tol_rad:
    left_part = _rdp(seq[: split_idx + 1], tol_rad)
    right_part = _rdp(seq[split_idx:], tol_rad)
    return left_part[:-1] + right_part
  return [a, b]


def simplify_joint_polyline(
    waypoints_rad: Sequence[Sequence[float]], *, tol_rad: float = 2e-3
) -> list[list[float]]:
  """Simplifies dense joint samples via Ramer-Douglas-Peucker and clamps.

  Collapses collinear `25 ms` trajectory samples back to `[q_stop, q_start]` so
  `generate_quintic_trajectory` executes one smooth return stroke instead of
  hundreds of start-stop stutters, while preserving true dogleg corners.
  """
  pts = [clamp_q_to_padded_limits(wp) for wp in waypoints_rad]
  if not pts:
    return []
  deduped: list[np.ndarray] = [pts[0]]
  for p in pts[1:]:
    if float(np.linalg.norm(p - deduped[-1])) > 1e-6:
      deduped.append(p)
  if len(deduped) <= 2:
    return [p.tolist() for p in deduped]
  return [p.tolist() for p in _rdp(deduped, tol_rad)]


def limit_rate_joint_positions(
    q_target: np.ndarray,
    q_last: np.ndarray,
    dq_last: np.ndarray,
    ddq_last: np.ndarray,
    *,
    dq_ff: np.ndarray | None = None,
    dt_s: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Critically-damped per-tick joint position rate limiter."""
  err = q_target - q_last
  a_brake = 0.5 * FR3_HW_MAX_ACC_RAD_S2
  tau_settle = 0.025
  v_brake = np.sqrt(2.0 * a_brake * np.abs(err))
  v_fb = np.sign(err) * np.minimum(np.abs(err) / tau_settle, v_brake)
  dq_cmd = v_fb if dq_ff is None else (dq_ff + v_fb)

  jerk_raw = ((dq_cmd - dq_last) / dt_s - ddq_last) / dt_s
  jerk_lim = np.clip(
      jerk_raw, -FR3_HW_MAX_JERK_RAD_S3, FR3_HW_MAX_JERK_RAD_S3
  )
  acc_cmd = ddq_last + jerk_lim * dt_s

  ratio = FR3_HW_MAX_JERK_RAD_S3 / FR3_HW_MAX_ACC_RAD_S2
  safe_max_acc = np.minimum(
      ratio * (FR3_HW_MAX_VEL_RAD_S - dq_last), FR3_HW_MAX_ACC_RAD_S2
  )
  safe_min_acc = np.maximum(
      ratio * (-FR3_HW_MAX_VEL_RAD_S - dq_last), -FR3_HW_MAX_ACC_RAD_S2
  )
  acc_lim = np.clip(acc_cmd, safe_min_acc, safe_max_acc)
  dq_lim = dq_last + acc_lim * dt_s
  q_lim = q_last + dq_lim * dt_s
  return q_lim, dq_lim, acc_lim


def step_realtime_trajectory_index(
    time_elapsed_s: float,
    duration_obj: Any,
    *,
    is_first_tick: bool,
    n_total: int,
    dt_s: float = 0.001,
) -> tuple[float, int, float]:
  """Advances monotonically by 1 sample per closed-loop FCI packet.

  Synchronous `start_joint_position_control` (`writeOnce`) does not run
  `ControlLoop`'s internal rate limiter; advancing one 1 ms spline sample per
  received FCI packet prevents multi-step position jumps after OS scheduler
  jitter on non-`PREEMPT_RT` kernels.
  """
  del duration_obj  # Handled via monotonic 1-sample progression + rate limiter.
  if is_first_tick:
    return 0.0, 0, dt_s

  new_elapsed = time_elapsed_s + dt_s
  idx = min(max(0, n_total - 1), int(round(new_elapsed / dt_s)))
  return new_elapsed, idx, dt_s


def compute_abort_deceleration_step(
    q_prev: np.ndarray,
    dq_at_abort: np.ndarray,
    step_idx: int,
    stop_steps: int = 250,
    dt_s: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes smooth C1/C2 Hermite velocity decay from `dq_at_abort`."""
  tau = min(1.0, max(0.0, float(step_idx) / float(max(1, stop_steps))))
  decay = 1.0 - 3.0 * tau * tau + 2.0 * tau * tau * tau
  cmd_dq = dq_at_abort * decay
  q_next = clamp_q_to_padded_limits(q_prev + cmd_dq * dt_s)
  return q_next, cmd_dq


class Fr3DuoSafetyChecker:
  """Offline and pre-execution MuJoCo tabletop/inter-arm collision checker."""

  def __init__(
      self,
      config: RobotConfig | Mapping[str, Any] | None = None,
      *,
      mj_model: Any | None = None,
      mjcf_xml: str = "",
      menagerie_dir: str | Path | None = None,
  ) -> None:
    """Initializes the MuJoCo collision checker and caches geom/body maps."""
    if mujoco is None:
      raise RuntimeError(
          "`mujoco` is required for `Fr3DuoSafetyChecker`. "
          "Install it with `pip install mujoco`."
      )
    self.config = normalize_robot_config(config)
    if mj_model is None:
      self.mj_model, self.mjcf_xml = fr3_mjcf.build_fr3_duo_mjcf(
          self.config, menagerie_dir=menagerie_dir
      )
    else:
      self.mj_model = mj_model
      self.mjcf_xml = mjcf_xml
    self._lock = threading.RLock()
    self._safety_scratch_data = mujoco.MjData(self.mj_model)

    self._table_gid = int(
        mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    )
    self._left_site_id = fr3_mjcf.resolve_ee_site_id(self.mj_model, "left")
    self._right_site_id = fr3_mjcf.resolve_ee_site_id(self.mj_model, "right")
    self._arm_qpos_addrs = {
        side: fr3_mjcf.arm_joint_qpos_and_dof_addrs(self.mj_model, side)[0]
        for side in ("left", "right")
    }

    nbody = int(self.mj_model.nbody)
    body_arm_side: list[str] = [""] * nbody
    is_moving_arm_body: list[bool] = [False] * nbody
    arm_gids: list[int] = []
    arm_body_geoms: list[tuple[int, np.ndarray]] = []
    fixed_roots = ("left/fr3_link0", "right/fr3_link0")
    for bid in range(nbody):
      bname = self._body_name(bid, default="")
      if bname.startswith("left/"):
        side_tag = "left"
      elif bname.startswith("right/"):
        side_tag = "right"
      else:
        continue
      body_arm_side[bid] = side_tag
      is_moving_arm_body[bid] = bname not in fixed_roots
      g_start = int(self.mj_model.body_geomadr[bid])
      g_num = int(self.mj_model.body_geomnum[bid])
      coll_gids = [
          gid
          for gid in range(g_start, g_start + g_num)
          if int(self.mj_model.geom_contype[gid]) != 0
          or int(self.mj_model.geom_conaffinity[gid]) != 0
      ]
      if coll_gids:
        arm_gids.extend(coll_gids)
        arm_body_geoms.append((bid, np.asarray(coll_gids, dtype=np.int32)))

    self._body_arm_side: tuple[str, ...] = tuple(body_arm_side)
    self._is_moving_arm_body: tuple[bool, ...] = tuple(is_moving_arm_body)
    self._arm_collision_gids = np.asarray(arm_gids, dtype=np.int32)
    self._arm_body_geoms = tuple(arm_body_geoms)
    self._additive_margin_gap = fr3_mjcf.uses_additive_margin_gap()
    self._clearance_scale = (
        0.5 if fr3_mjcf.uses_pairwise_additive_margin() else 1.0
    )
    resolved_clearance = max(0.0, float(self.config.min_inter_arm_clearance_m))
    self._configured_half_clearance: float | None = (
        None
        if mj_model is not None
        else self._clearance_scale * resolved_clearance
    )

  def collect_arm_collision_geom_ids(self, side: str) -> list[int]:
    """Collects moving collision geom IDs (`contype > 0`, no `fr3_link0`)."""
    gids: list[int] = []
    for bid, coll_gids in self._arm_body_geoms:
      if self._body_arm_side[bid] == side and self._is_moving_arm_body[bid]:
        gids.extend(int(g) for g in coll_gids)
    return gids

  def _sync_clearance_margins(self, half_clearance: float) -> None:
    if self._arm_collision_gids.size == 0 or (
        self._configured_half_clearance is not None
        and abs(half_clearance - self._configured_half_clearance) <= 1e-12
    ):
      return
    self.mj_model.geom_gap[self._arm_collision_gids] = half_clearance
    if not self._additive_margin_gap:
      self.mj_model.geom_margin[self._arm_collision_gids] = half_clearance
    for bid, b_gids in self._arm_body_geoms:
      if self._additive_margin_gap:
        self.mj_model.body_margin[bid] = float(
            np.max(
                self.mj_model.geom_margin[b_gids]
                + self.mj_model.geom_gap[b_gids]
            )
        )
      else:
        self.mj_model.body_margin[bid] = float(
            np.max(self.mj_model.geom_margin[b_gids])
        )
    self._configured_half_clearance = half_clearance

  def _body_name(self, bid: int, default: str | None = None) -> str:
    name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return name or (f"body_{bid}" if default is None else default)

  def _geom_name(self, gid: int) -> str:
    return (
        mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        or f"geom_{gid}"
    )

  def _check_sample_contacts(
      self,
      scratch: Any,
      idx: int,
      eff_clearance: float,
      *,
      allow_table_tolerance_m: float = 0.0,
  ) -> set[str]:
    mujoco.mj_collision(self.mj_model, scratch)
    table_gid = self._table_gid
    body_arm_side = self._body_arm_side
    is_moving_arm_body = self._is_moving_arm_body
    geom_bodyid = self.mj_model.geom_bodyid

    table_contact_sides: set[str] = set()
    best_dist = float("inf")
    best_pair: tuple[int, int, int, int] | None = None
    for con_i in range(int(scratch.ncon)):
      con = scratch.contact[con_i]
      dist = float(con.dist)
      if dist >= eff_clearance and dist >= 0.0:
        continue
      g1, g2 = int(con.geom1), int(con.geom2)
      b1, b2 = int(geom_bodyid[g1]), int(geom_bodyid[g2])
      s1, s2 = body_arm_side[b1], body_arm_side[b2]
      if s1 and s2 and s1 != s2:
        if dist < best_dist:
          best_dist = dist
          best_pair = (g1, g2, b1, b2)
        continue

      m1, m2 = is_moving_arm_body[b1], is_moving_arm_body[b2]
      if dist < 0.0 and (m1 or m2):
        moving_bid, other_bid = (b1, b2) if (m1 and not m2) else (b2, b1)
        is_table_hit = table_gid >= 0 and (g1 == table_gid or g2 == table_gid)
        moving_side = body_arm_side[moving_bid]
        if (
            is_table_hit
            and allow_table_tolerance_m > 0.0
            and dist >= -allow_table_tolerance_m - 1e-6
            and moving_side in ("left", "right")
        ):
          table_contact_sides.add(moving_side)
          continue
        other_name = "tabletop" if is_table_hit else self._body_name(other_bid)
        raise ValueError(
            f"Arm link {self._body_name(moving_bid)!r} collides with "
            f"{other_name!r} (dist={dist:.4f} m) at sample {idx}"
        )

    if best_pair is not None and best_dist < eff_clearance:
      g1, g2, b1, b2 = best_pair
      if best_dist < 0.0:
        raise ValueError(
            f"Inter-arm collision between {self._body_name(b1)!r} and "
            f"{self._body_name(b2)!r} "
            f"(dist={best_dist:.4f} m < {eff_clearance:.4f} m) "
            f"at sample {idx}"
        )
      raise ValueError(
          f"Inter-arm clearance violation: {best_dist:.4f} m < "
          f"{eff_clearance:.4f} m at sample {idx} "
          f"({self._geom_name(g1)} <-> {self._geom_name(g2)})"
      )
    return table_contact_sides

  def verify_bimanual_trajectory(
      self,
      traj_left: np.ndarray,
      traj_right: np.ndarray,
      *,
      z_table_m: float | None = None,
      min_clearance_m: float | None = None,
      allow_initial_table_lift: bool = False,
  ) -> None:
    """Verifies `z >= z_table_m`, table collision, and arm clearance."""
    eff_z_table = (
        float(self.config.z_table_m) if z_table_m is None else float(z_table_m)
    )
    eff_clearance = max(
        0.0,
        float(self.config.min_inter_arm_clearance_m)
        if min_clearance_m is None
        else float(min_clearance_m),
    )

    n_total = max(int(traj_left.shape[0]), int(traj_right.shape[0]))
    sample_count = min(n_total, max(25, n_total // 25))
    indices = np.unique(
        np.linspace(0, n_total - 1, sample_count, dtype=np.intp)
    )

    left_qaddrs = self._arm_qpos_addrs["left"]
    right_qaddrs = self._arm_qpos_addrs["right"]
    left_site_id = self._left_site_id
    right_site_id = self._right_site_id
    table_tol = (
        INITIAL_TABLE_LIFT_TOLERANCE_M if allow_initial_table_lift else 0.0
    )

    with self._lock:
      self._sync_clearance_margins(self._clearance_scale * eff_clearance)
      scratch = self._safety_scratch_data
      scratch.qpos[:] = self.mj_model.qpos0

      lift_active = {"left": False, "right": False}
      needs_clear = {"left": False, "right": False}
      z_start: dict[str, float] = {}
      z_prev: dict[str, float] = {}
      q_prev: dict[str, np.ndarray] = {}

      for sample_pos, raw_idx in enumerate(indices):
        idx = int(raw_idx)
        ql = traj_left[min(idx, len(traj_left) - 1)]
        qr = traj_right[min(idx, len(traj_right) - 1)]
        for qadr, val in zip(left_qaddrs, ql):
          scratch.qpos[qadr] = float(val)
        for qadr, val in zip(right_qaddrs, qr):
          scratch.qpos[qadr] = float(val)
        mujoco.mj_kinematics(self.mj_model, scratch)

        z_by_side: dict[str, float] = {}
        if left_site_id >= 0:
          z_by_side["left"] = float(scratch.site_xpos[left_site_id, 2])
        if right_site_id >= 0:
          z_by_side["right"] = float(scratch.site_xpos[right_site_id, 2])

        for side, p_z in z_by_side.items():
          if p_z < eff_z_table - table_tol - 1e-4:
            raise ValueError(
                f"{side.capitalize()} TCP violates tabletop floor "
                f"z={p_z:.4f} m < z_table_m={eff_z_table:.4f} m at sample {idx}"
            )

        table_contact_sides = self._check_sample_contacts(
            scratch,
            idx,
            eff_clearance,
            allow_table_tolerance_m=table_tol,
        )
        if not allow_initial_table_lift:
          continue

        for side in ("left", "right"):
          p_z = z_by_side.get(side)
          q_cur = ql if side == "left" else qr
          floor_viol = p_z is not None and p_z < eff_z_table - 1e-4
          has_viol = floor_viol or (side in table_contact_sides)
          if sample_pos == 0:
            if has_viol:
              if p_z is None:
                raise ValueError(
                    f"{side.capitalize()} tabletop violation without TCP site "
                    f"at sample {idx}"
                )
              lift_active[side] = True
              needs_clear[side] = True
              z_start[side] = p_z
              z_prev[side] = p_z
              q_prev[side] = np.asarray(q_cur, dtype=np.float64)
          elif has_viol:
            if not lift_active[side]:
              if floor_viol:
                raise ValueError(
                    f"{side.capitalize()} TCP violates tabletop floor "
                    f"z={p_z:.4f} m < z_table_m={eff_z_table:.4f} m "
                    f"at sample {idx}"
                )
              raise ValueError(
                  f"{side.capitalize()} arm collides with 'tabletop' "
                  f"at sample {idx}"
              )
            assert p_z is not None
            is_dup_start = idx == 1 and np.allclose(
                q_cur, q_prev[side], atol=1e-9
            )
            if p_z < z_prev[side] - 1e-9 or (
                p_z <= z_prev[side] + 1e-9 and not is_dup_start
            ):
              raise ValueError(
                  f"{side.capitalize()} TCP failed to lift strictly away from "
                  f"tabletop (z={p_z:.4f} m <= prev={z_prev[side]:.4f} m) "
                  f"at sample {idx}"
              )
            z_prev[side] = p_z
            q_prev[side] = np.asarray(q_cur, dtype=np.float64)
          elif lift_active[side]:
            if p_z is not None and p_z <= z_start[side] + 1e-9:
              raise ValueError(
                  f"{side.capitalize()} TCP failed to lift strictly away from "
                  f"tabletop at sample {idx}"
              )
            lift_active[side] = False
            needs_clear[side] = False

      if allow_initial_table_lift:
        for side in ("left", "right"):
          if needs_clear[side]:
            raise ValueError(
                f"{side.capitalize()} TCP remained in tabletop violation "
                "without clearing by end of trajectory"
            )
