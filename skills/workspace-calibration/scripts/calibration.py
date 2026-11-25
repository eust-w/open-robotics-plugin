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
"""Embodiment-agnostic workspace calibration and contact-probing primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
import pathlib
from typing import Any

import cv2
import numpy as np


def load_calibration(path: str | pathlib.Path) -> dict[str, Any]:
  """Loads `.robot_calibration.json` if present, else returns an empty dict."""
  p = pathlib.Path(path)
  if not p.is_file():
    return {}
  return json.loads(p.read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  if isinstance(obj, np.generic):
    return obj.item()
  raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_calibration(
    path: str | pathlib.Path, data: Mapping[str, Any]
) -> pathlib.Path:
  """Writes calibration dict as formatted JSON, converting NumPy arrays."""
  p = pathlib.Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(
      json.dumps(dict(data), indent=2, sort_keys=True, default=_json_default),
      encoding="utf-8",
  )
  return p


def scale_intrinsics(
    k_mat: Sequence[Sequence[float]] | np.ndarray,
    from_wh: tuple[int, int],
    to_wh: tuple[int, int],
) -> np.ndarray:
  """Rescales 3x3 pinhole intrinsics `K` proportionally for a new resolution."""
  k = np.asarray(k_mat, dtype=np.float64).copy()
  sx = float(to_wh[0]) / float(from_wh[0])
  sy = float(to_wh[1]) / float(from_wh[1])
  k[0, 0] *= sx
  k[0, 2] *= sx
  k[1, 1] *= sy
  k[1, 2] *= sy
  return k


def border_fingerprint(
    img: np.ndarray, border_frac: float = 0.08
) -> list[float]:
  """Computes a lightweight outer-border intensity heuristic for camera shifts.

  Note: mean-intensity bins can be sensitive to large lighting changes and may
  miss very small shifts; treat `is_border_stale` as a fast heuristic check.
  """
  gray = (
      cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
  ).astype(np.float64)
  h, w = gray.shape[:2]
  bh = max(1, int(round(h * border_frac)))
  bw = max(1, int(round(w * border_frac)))
  strips = (
      gray[:bh, :],
      gray[h - bh :, :],
      gray[:, :bw],
      gray[:, w - bw :],
  )
  bins: list[float] = []
  for strip in strips:
    parts = np.array_split(strip.ravel(), 4)
    bins.extend(float(np.mean(part)) for part in parts)
  return bins


def is_border_stale(
    fp_a: Sequence[float], fp_b: Sequence[float], threshold: float = 18.0
) -> bool:
  """Returns True when outer-border fingerprints diverge (camera shifted)."""
  a = np.asarray(fp_a, dtype=np.float64)
  b = np.asarray(fp_b, dtype=np.float64)
  if a.shape != b.shape or a.size == 0:
    return True
  return float(np.mean(np.abs(a - b))) > float(threshold)


def probe_surface_z(
    move_z_fn: Callable[[float], None],
    read_z_fk_fn: Callable[[], float],
    z_start: float,
    z_min: float,
    *,
    mode: str = "compliant",
    read_force_fn: Callable[[], float] | None = None,
    step_m: float | None = None,
    min_upward_err_m: float | None = None,
    force_threshold_n: float = 3.0,
    stall_steps: int = 2,
    retreat_m: float = 0.06,
) -> dict[str, Any]:
  """Probes a support surface height via guarded vertical descent.

  Contact triggers either when `read_force_fn()` reaches `force_threshold_n`, or
  when for `stall_steps` consecutive steps all three physics conditions hold:
    1. Strictly positive upward reaction: `z_fk - z_cmd >= min_upward_err_m`
       (never triggers while `z_cmd` is merely catching up to negative sag).
    2. Stalled downward motion: `prev_z_fk - z_fk <= 0.25 * step_m`.
    3. Growing reaction: `(err_z - prev_err_z) >= 0.60 * step_m`.

  Args:
    move_z_fn: Callback smoothly moving the TCP to `(x, y, z_cmd)` and settling.
    read_z_fk_fn: Callback returning measured FK TCP height `z_fk` in meters.
    z_start: Starting clear height above the surface (m).
    z_min: Lowest allowed commanded height (m) before aborting.
    mode: `'compliant'` (default `step_m=0.004`, `min_upward_err_m=0.006`;
      presses ~8-12 mm past contact to clear gravity sag reliably—pass
      `read_force_fn` or `'stiff'` / smaller `min_upward_err_m` for delicate
      surfaces) or `'stiff'` (default `step_m=0.002`, `min_upward_err_m=0.002`).
    read_force_fn: Optional callback returning upward contact force (N).
    step_m: Optional vertical step override (m, positive).
    min_upward_err_m: Optional minimum positive `z_fk - z_cmd` override (m).
    force_threshold_n: Upward force threshold (N) when `read_force_fn` is set.
    stall_steps: Consecutive stalled steps required to confirm contact.
    retreat_m: Vertical retreat height above latched `surface_z` (m).

  Returns:
    Dict with `surface_z`, `surface_z_cmd`, `free_air_droop_z`, and `steps`.
  """
  if mode not in ("compliant", "stiff"):
    raise ValueError(f"Unsupported mode {mode!r}; use 'compliant' or 'stiff'.")
  step = float(
      step_m if step_m is not None else (0.004 if mode == "compliant" else 0.002)
  )
  if step <= 0.0:
    raise ValueError(f"step_m must be positive, got {step}")
  upward_tol = float(
      min_upward_err_m
      if min_upward_err_m is not None
      else (0.006 if mode == "compliant" else 0.002)
  )
  if z_start <= z_min:
    raise ValueError(f"z_start ({z_start}) must be > z_min ({z_min}).")

  n_steps = int(math.floor((z_start - z_min) / step)) + 1
  z_cmds = [float(z_start - i * step) for i in range(n_steps)]

  prev_z_fk: float | None = None
  prev_err_z: float | None = None
  stall_count = 0
  free_air_errs: list[float] = []
  last_z_fk = float(z_start)
  result: dict[str, Any] | None = None

  try:
    for idx, z_cmd in enumerate(z_cmds):
      move_z_fn(z_cmd)
      z_fk = float(read_z_fk_fn())
      last_z_fk = z_fk
      err_z = z_fk - z_cmd

      if idx > 0 and read_force_fn is not None:
        if float(read_force_fn()) >= float(force_threshold_n):
          droop = float(np.median(free_air_errs)) if free_air_errs else 0.0
          result = {
              "surface_z": z_fk,
              "surface_z_cmd": z_cmd,
              "free_air_droop_z": droop,
              "steps": idx,
          }
          break

      if idx > 0 and prev_z_fk is not None and prev_err_z is not None:
        dz_fk = prev_z_fk - z_fk
        d_err = err_z - prev_err_z
        is_stall = (
            err_z >= upward_tol
            and dz_fk <= 0.25 * step
            and d_err >= 0.60 * step
        )
        if is_stall:
          stall_count += 1
        else:
          stall_count = 0
          if dz_fk >= 0.50 * step:
            free_air_errs.append(err_z)

        if stall_count >= max(1, int(stall_steps)):
          droop = float(np.median(free_air_errs)) if free_air_errs else 0.0
          onset_cmd = min(float(z_start), z_cmd + float(stall_count) * step)
          result = {
              "surface_z": z_fk,
              "surface_z_cmd": onset_cmd,
              "free_air_droop_z": droop,
              "steps": idx,
          }
          break

      prev_z_fk = z_fk
      prev_err_z = err_z
  finally:
    if retreat_m > 0.0:
      move_z_fn(min(float(z_start), last_z_fk + float(retreat_m)))

  if result is None:
    raise RuntimeError(
        f"Surface probe reached z_min={z_min:.4f} m without detecting contact."
    )
  return result


def _unproject_pixel_to_plane(
    u_px: float,
    v_px: float,
    k_mat: np.ndarray,
    t_world_cam: np.ndarray,
    z_plane_m: float,
) -> np.ndarray | None:
  """Intersects an OpenCV optical ray through `(u_px, v_px)` with `z=z_plane`."""
  fx, fy = float(k_mat[0, 0]), float(k_mat[1, 1])
  cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])
  ray_cam = np.array([(u_px - cx) / fx, (v_px - cy) / fy, 1.0], dtype=np.float64)
  with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    ray_w = t_world_cam[:3, :3] @ ray_cam
  if abs(float(ray_w[2])) < 1e-8:
    return None
  origin = t_world_cam[:3, 3]
  t_param = (float(z_plane_m) - float(origin[2])) / float(ray_w[2])
  if t_param <= 0.0:
    return None
  return origin + t_param * ray_w


def verify_wrist_extrinsic_sign(
    t_ee_cam: Sequence[Sequence[float]] | np.ndarray,
    k_mat: Sequence[Sequence[float]] | np.ndarray,
    t_world_ee_1: Sequence[Sequence[float]] | np.ndarray,
    uv_1: tuple[float, float],
    t_world_ee_2: Sequence[Sequence[float]] | np.ndarray,
    uv_2: tuple[float, float],
    z_plane_m: float,
    *,
    max_err_m: float = 0.015,
) -> dict[str, Any]:
  """Selects the optical-Z roll (`0/90/180/270 deg`) making a landmark static.

  Given two wrist-camera observations `(t_world_ee_1, uv_1)` and
  `(t_world_ee_2, uv_2)` of the same stationary landmark on `z_plane_m` across a
  small horizontal translation, tests in-plane optical roll offsets and returns
  the consistent `T_ee_cam`. Sets `valid=False` when `consistency_err_m >
  max_err_m` (e.g., wrong mount tilt, translation, or mirrored axis—fall back
  to `fit_wrist_jacobian`).
  """
  t_base = np.asarray(t_ee_cam, dtype=np.float64).reshape(4, 4)
  k = np.asarray(k_mat, dtype=np.float64).reshape(3, 3)
  t_ee1 = np.asarray(t_world_ee_1, dtype=np.float64).reshape(4, 4)
  t_ee2 = np.asarray(t_world_ee_2, dtype=np.float64).reshape(4, 4)

  best_err = float("inf")
  best_deg = 0.0
  best_t = t_base.copy()

  for deg in (0.0, 90.0, 180.0, 270.0):
    th = math.radians(deg)
    r_z = np.array(
        [
            [math.cos(th), -math.sin(th), 0.0, 0.0],
            [math.sin(th), math.cos(th), 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cand = t_base @ r_z
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
      t_wc1 = t_ee1 @ cand
      t_wc2 = t_ee2 @ cand
    p1 = _unproject_pixel_to_plane(uv_1[0], uv_1[1], k, t_wc1, z_plane_m)
    p2 = _unproject_pixel_to_plane(uv_2[0], uv_2[1], k, t_wc2, z_plane_m)
    if p1 is None or p2 is None:
      continue
    err = float(np.linalg.norm(p1[:2] - p2[:2]))
    if err < best_err:
      best_err = err
      best_deg = deg
      best_t = cand

  if not math.isfinite(best_err):
    raise RuntimeError("Could not unproject landmark onto z_plane_m.")
  det_ok = abs(float(np.linalg.det(t_base[:3, :3])) - 1.0) < 1e-2
  return {
      "T_ee_cam": best_t,
      "roll_offset_deg": best_deg,
      "consistency_err_m": best_err,
      "valid": bool(det_ok and best_err <= float(max_err_m)),
  }


def fit_wrist_jacobian(
    duv_dx: tuple[float, float],
    duv_dy: tuple[float, float],
    step_x_m: float,
    step_y_m: float,
) -> np.ndarray:
  """Fits `2x2` `J_xy_uv = d(x, y) / d(u, v)` from two orthogonal hover nudges.

  Args:
    duv_dx: Observed landmark pixel shift `(du, dv)` after moving `+step_x_m`.
    duv_dy: Observed landmark pixel shift `(du, dv)` after moving `+step_y_m`.
    step_x_m: Horizontal X nudge in meters (non-zero).
    step_y_m: Horizontal Y nudge in meters (non-zero).

  Returns:
    2x2 matrix mapping pixel error `(u_target - u_ref, v_target - v_ref)` to
    `(dx_m, dy_m)` robot motion (note: moving the robot by `+dxy` shifts a
    stationary landmark by `duv`, so `dxy_to_center = -J_xy_uv @ (uv_t - uv_r)`
    or `J_xy_uv @ (uv_r - uv_t)`).
  """
  if abs(step_x_m) < 1e-9 or abs(step_y_m) < 1e-9:
    raise ValueError("step_x_m and step_y_m must be non-zero.")
  j_uv_xy = np.array(
      [
          [float(duv_dx[0]) / float(step_x_m), float(duv_dy[0]) / float(step_y_m)],
          [float(duv_dx[1]) / float(step_x_m), float(duv_dy[1]) / float(step_y_m)],
      ],
      dtype=np.float64,
  )
  if abs(float(np.linalg.det(j_uv_xy))) < 1e-9:
    raise ValueError("Degenerate pixel shifts; cannot invert J_uv_xy.")
  return np.linalg.inv(j_uv_xy)


def jacobian_step_xy(
    j_xy_uv: Sequence[Sequence[float]] | np.ndarray,
    uv_target: tuple[float, float],
    uv_ref: tuple[float, float],
    *,
    gain: float = 0.8,
    max_step_m: float = 0.04,
) -> np.ndarray:
  """Computes a clamped `(dx_m, dy_m)` robot step moving `uv_target` -> `uv_ref`.

  Uses `dxy = gain * J_xy_uv @ (uv_ref - uv_target)` (with `J_xy_uv` from
  `fit_wrist_jacobian`) so the servoing sign is never flipped.
  """
  j_mat = np.asarray(j_xy_uv, dtype=np.float64).reshape(2, 2)
  duv = np.array(
      [float(uv_ref[0]) - float(uv_target[0]), float(uv_ref[1]) - float(uv_target[1])],
      dtype=np.float64,
  )
  with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    dxy = float(gain) * (j_mat @ duv)
  norm = float(np.linalg.norm(dxy))
  if norm > float(max_step_m) > 0.0:
    dxy = dxy * (float(max_step_m) / norm)
  return dxy
