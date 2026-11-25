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

"""Forward kinematics, wrench estimation, and OpenCV camera calibration."""

from __future__ import annotations

import collections
from collections.abc import Sequence
import math
import threading
from typing import Any
import weakref

import numpy as np

from fr3_camera_controller import CameraCalibration
from fr3_mjcf import (
    DEFAULT_REST_POSE_RAD,
    _require_mujoco,
    arm_joint_qpos_and_dof_addrs,
    canonical_camera_name,
    resolve_ee_site_id,
)


class _ModelKinematicsContext:
  """Reusable per-MjModel workspace and pre-indexed kinematics handles."""

  def __init__(self, mj_model: Any) -> None:
    mj = _require_mujoco()
    self.model_ref = weakref.ref(mj_model)
    self.lock = threading.RLock()
    self.data = mj.MjData(mj_model)
    self.arm_addrs: dict[str, tuple[list[int], list[int]]] = {}
    self.ee_site_ids: dict[str, int] = {}
    for arm in ("left", "right"):
      try:
        self.arm_addrs[arm] = arm_joint_qpos_and_dof_addrs(mj_model, arm)
      except KeyError:
        # Partial or custom test models may omit one arm's joints.
        pass
      try:
        self.ee_site_ids[arm] = resolve_ee_site_id(mj_model, arm)
      except KeyError:
        self.ee_site_ids[arm] = -1
    nv = int(mj_model.nv)
    self.rne_tau = np.zeros(nv, dtype=np.float64)
    self.jacp = np.zeros((3, nv), dtype=np.float64)
    self.jacr = np.zeros((3, nv), dtype=np.float64)


_KIN_CTX_LOCK = threading.Lock()
_KIN_CTX_CACHE: collections.OrderedDict[int, _ModelKinematicsContext] = (
    collections.OrderedDict()
)
_KIN_CTX_CACHE_MAX = 16


def _get_model_kinematics_ctx(mj_model: Any) -> _ModelKinematicsContext:
  key = id(mj_model)
  with _KIN_CTX_LOCK:
    entry = _KIN_CTX_CACHE.get(key)
    if entry is not None and entry.model_ref() is mj_model:
      _KIN_CTX_CACHE.move_to_end(key)
      return entry
    entry = _ModelKinematicsContext(mj_model)
    _KIN_CTX_CACHE[key] = entry
    _KIN_CTX_CACHE.move_to_end(key)
    while len(_KIN_CTX_CACHE) > _KIN_CTX_CACHE_MAX:
      _KIN_CTX_CACHE.popitem(last=False)
    return entry


def _set_ctx_arm_q(
    ctx: _ModelKinematicsContext,
    mj_model: Any,
    arm: str,
    q_arr: np.ndarray,
) -> list[int]:
  other_arm = "right" if arm == "left" else "left"
  if arm not in ctx.arm_addrs:
    ctx.arm_addrs[arm] = arm_joint_qpos_and_dof_addrs(mj_model, arm)
  if other_arm not in ctx.arm_addrs:
    try:
      ctx.arm_addrs[other_arm] = arm_joint_qpos_and_dof_addrs(
          mj_model, other_arm
      )
    except KeyError:
      # Partial or single-arm models may omit the opposite arm's joints.
      pass
  qpos_addrs, dof_addrs = ctx.arm_addrs[arm]

  ctx.data.qpos[:] = mj_model.qpos0
  if other_arm in ctx.arm_addrs:
    other_qpos_addrs, _ = ctx.arm_addrs[other_arm]
    for qadr, val in zip(other_qpos_addrs, DEFAULT_REST_POSE_RAD[other_arm]):
      ctx.data.qpos[qadr] = float(val)
  for qadr, val in zip(qpos_addrs, q_arr):
    ctx.data.qpos[qadr] = float(val)
  return dof_addrs


def _validate_arm_and_q(arm: str, q_rad: Sequence[float]) -> np.ndarray:
  if arm not in ("left", "right"):
    raise ValueError(f"Invalid arm={arm!r}; expected 'left' or 'right'")
  q_arr = np.asarray(q_rad, dtype=np.float64).reshape(-1)
  if q_arr.shape != (7,):
    raise ValueError(f"Expected 7 joint angles for {arm!r}, got {q_arr.shape}")
  return q_arr


def _prepare_arm_kinematics_ctx(
    ctx: _ModelKinematicsContext,
    mj_model: Any,
    arm: str,
    q_arr: np.ndarray,
) -> tuple[int, list[int]]:
  """Sets `ctx.data.qpos` for `arm` and runs `mj_kinematics` (lock held)."""
  mj = _require_mujoco()
  site_id = ctx.ee_site_ids.get(arm, -1)
  if site_id < 0:
    site_id = resolve_ee_site_id(mj_model, arm)
    ctx.ee_site_ids[arm] = site_id
  dof_addrs = _set_ctx_arm_q(ctx, mj_model, arm, q_arr)
  mj.mj_kinematics(mj_model, ctx.data)
  return site_id, dof_addrs


def compute_arm_fk(
    mj_model: Any, arm: str, q_rad: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
  """Computes `(pos_3, rot_3x3)` of `{arm}/pinch_site` in `base` frame."""
  _require_mujoco()
  q_arr = _validate_arm_and_q(arm, q_rad)
  ctx = _get_model_kinematics_ctx(mj_model)
  with ctx.lock:
    site_id, _ = _prepare_arm_kinematics_ctx(ctx, mj_model, arm, q_arr)
    pos_world = np.asarray(
        ctx.data.site_xpos[site_id], dtype=np.float64
    ).copy()
    rot_world = (
        np.asarray(ctx.data.site_xmat[site_id], dtype=np.float64)
        .reshape(3, 3)
        .copy()
    )
    return pos_world, rot_world


def compute_arm_pinch_from_flange(
    mj_model: Any, arm: str, q_rad: Sequence[float]
) -> np.ndarray | None:
  """Returns homogeneous `[x, y, z, 1]` of `pinch_site` in flange frame."""
  mj = _require_mujoco()
  q_arr = _validate_arm_and_q(arm, q_rad)
  ctx = _get_model_kinematics_ctx(mj_model)
  with ctx.lock:
    pinch_sid, _ = _prepare_arm_kinematics_ctx(ctx, mj_model, arm, q_arr)
    pinch_pos = np.asarray(ctx.data.site_xpos[pinch_sid], dtype=np.float64)

    flange_sid = -1
    for candidate_site in (f"{arm}/attachment_site", f"{arm}_attachment_site"):
      sid = int(mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SITE, candidate_site))
      if sid >= 0:
        flange_sid = sid
        break

    if flange_sid >= 0:
      flange_pos = np.asarray(ctx.data.site_xpos[flange_sid], dtype=np.float64)
      flange_rot = np.asarray(
          ctx.data.site_xmat[flange_sid], dtype=np.float64
      ).reshape(3, 3)
    else:
      link7_id = int(
          mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, f"{arm}/fr3_link7")
      )
      if link7_id < 0:
        return None
      flange_rot = np.asarray(
          ctx.data.xmat[link7_id], dtype=np.float64
      ).reshape(3, 3)
      pos7 = np.asarray(ctx.data.xpos[link7_id], dtype=np.float64)
      flange_pos = pos7 + flange_rot @ np.array(
          [0.0, 0.0, 0.107], dtype=np.float64
      )

    local_pos = flange_rot.T @ (pinch_pos - flange_pos)
    return np.array(
        [local_pos[0], local_pos[1], local_pos[2], 1.0], dtype=np.float64
    )


def estimate_external_wrench_base(
    mj_model: Any,
    arm: str,
    q_rad: Sequence[float],
    tau_j_measured: Sequence[float],
    *,
    s_floor: float = 0.05,
) -> np.ndarray:
  """Estimates 6-D `+w_env_on_robot` wrench `[fx, fy, fz, tx, ty, tz]`."""
  mj = _require_mujoco()
  q_arr = _validate_arm_and_q(arm, q_rad)
  tau_meas = np.asarray(tau_j_measured, dtype=np.float64).reshape(-1)
  if tau_meas.shape != (7,):
    raise ValueError(
        f"Expected 7-D tau_j_measured, got {tau_meas.shape}"
    )

  ctx = _get_model_kinematics_ctx(mj_model)
  with ctx.lock:
    site_id, dof_addrs = _prepare_arm_kinematics_ctx(ctx, mj_model, arm, q_arr)
    ctx.data.qvel[:] = 0.0
    ctx.data.qacc[:] = 0.0
    ctx.data.cvel[:] = 0.0
    ctx.data.cdof_dot[:] = 0.0

    mj.mj_comPos(mj_model, ctx.data)
    ctx.rne_tau.fill(0.0)
    mj.mj_rne(mj_model, ctx.data, 0, ctx.rne_tau)

    tau_ext = ctx.rne_tau[dof_addrs] - tau_meas
    ctx.jacp.fill(0.0)
    ctx.jacr.fill(0.0)
    mj.mj_jacSite(mj_model, ctx.data, ctx.jacp, ctx.jacr, site_id)
    j_arm = np.vstack([ctx.jacp[:, dof_addrs], ctx.jacr[:, dof_addrs]])

  u_mat, s_vals, vt_mat = np.linalg.svd(j_arm.T, full_matrices=False)
  floor_sq = max(float(s_floor), 1e-6) ** 2
  damped_inv_s = s_vals / np.maximum(s_vals * s_vals, floor_sq)
  return ((vt_mat.T * damped_inv_s) @ (u_mat.T @ tau_ext)).astype(np.float64)


def get_camera_calibration_opencv(
    mj_model: Any,
    left_q_rad: Sequence[float],
    right_q_rad: Sequence[float],
    *,
    resolution_wh: tuple[int, int] | None = None,
) -> dict[str, CameraCalibration]:
  """Computes OpenCV camera intrinsics and extrinsics in `base` frame."""
  mj = _require_mujoco()
  q_l = np.asarray(left_q_rad, dtype=np.float64).reshape(-1)
  q_r = np.asarray(right_q_rad, dtype=np.float64).reshape(-1)
  if q_l.shape != (7,) or q_r.shape != (7,):
    raise ValueError(
        "Expected 7-D joint angles for left and right arms, got "
        f"{q_l.shape} and {q_r.shape}"
    )

  ctx = _get_model_kinematics_ctx(mj_model)
  cv_diag = np.diag([1.0, -1.0, -1.0])
  cal: dict[str, CameraCalibration] = {}
  num_cams = int(mj_model.ncam)

  with ctx.lock:
    for side in ("left", "right"):
      if side not in ctx.arm_addrs:
        ctx.arm_addrs[side] = arm_joint_qpos_and_dof_addrs(mj_model, side)
    left_qpos_addrs, _ = ctx.arm_addrs["left"]
    right_qpos_addrs, _ = ctx.arm_addrs["right"]
    ctx.data.qpos[:] = mj_model.qpos0
    ctx.data.qvel[:] = 0.0
    ctx.data.qacc[:] = 0.0
    for qadr, val in zip(left_qpos_addrs, q_l):
      ctx.data.qpos[qadr] = float(val)
    for qadr, val in zip(right_qpos_addrs, q_r):
      ctx.data.qpos[qadr] = float(val)
    mj.mj_kinematics(mj_model, ctx.data)
    mj.mj_camlight(mj_model, ctx.data)

    t_base_pinch: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
      site_id = ctx.ee_site_ids.get(side, -1)
      if site_id >= 0:
        t_mat = np.eye(4, dtype=np.float64)
        t_mat[:3, :3] = np.asarray(
            ctx.data.site_xmat[site_id], dtype=np.float64
        ).reshape(3, 3)
        t_mat[:3, 3] = np.asarray(
            ctx.data.site_xpos[site_id], dtype=np.float64
        )
        t_base_pinch[side] = t_mat

    for cid in range(num_cams):
      raw_cam_name = (
          mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_CAMERA, cid)
          or f"camera_{cid}"
      )
      cam_name = canonical_camera_name(raw_cam_name)
      if resolution_wh is not None:
        w, h = int(resolution_wh[0]), int(resolution_wh[1])
      elif (
          hasattr(mj_model, "cam_resolution")
          and mj_model.cam_resolution is not None
          and mj_model.cam_resolution[cid][0] > 0
          and mj_model.cam_resolution[cid][1] > 0
      ):
        w = int(mj_model.cam_resolution[cid][0])
        h = int(mj_model.cam_resolution[cid][1])
      else:
        w, h = 1280, 720

      has_sensor = False
      if hasattr(mj_model, "cam_sensorsize") and hasattr(
          mj_model, "cam_intrinsic"
      ):
        ss = mj_model.cam_sensorsize[cid]
        intr = mj_model.cam_intrinsic[cid]
        if ss[0] > 0 and ss[1] > 0 and intr[0] > 0:
          fx = float(intr[0]) / float(ss[0]) * w
          fy = float(intr[1]) / float(ss[1]) * h
          cx = w * (0.5 - float(intr[2]) / float(ss[0]))
          cy = h * (0.5 - float(intr[3]) / float(ss[1]))
          has_sensor = True

      if not has_sensor:
        fovy_rad = math.radians(float(mj_model.cam_fovy[cid]))
        fy = 0.5 * h / max(math.tan(0.5 * fovy_rad), 1e-3)
        fx = fy
        cx = 0.5 * w
        cy = 0.5 * h

      k_mat = np.array(
          [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
      )
      p_cam_base = np.asarray(ctx.data.cam_xpos[cid], dtype=np.float64).copy()
      r_cam_mj_base = (
          np.asarray(ctx.data.cam_xmat[cid], dtype=np.float64)
          .reshape(3, 3)
          .copy()
      )
      t_base_cam = np.eye(4, dtype=np.float64)
      t_base_cam[:3, :3] = r_cam_mj_base @ cv_diag
      t_base_cam[:3, 3] = p_cam_base

      if cam_name == "wrist_camera_left" and "left" in t_base_pinch:
        t_ee_cam = np.linalg.inv(t_base_pinch["left"]) @ t_base_cam
      elif cam_name == "wrist_camera_right" and "right" in t_base_pinch:
        t_ee_cam = np.linalg.inv(t_base_pinch["right"]) @ t_base_cam
      else:
        t_ee_cam = None

      cal[cam_name] = CameraCalibration(
          width=w,
          height=h,
          K=k_mat,
          intrinsics_tuple=(fx, fy, cx, cy),
          mp_intrinsics={
              "focal_px": [fx, fy],
              "principal_point_px": [cx, cy],
          },
          T_base_cam=t_base_cam,
          T_ee_cam=t_ee_cam,
          mj_pose={
              "pos": p_cam_base,
              "rot": r_cam_mj_base,
          },
      )
  return cal


def project_base_points_opencv(
    points_base: np.ndarray,
    k_mat: np.ndarray,
    t_base_cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | float]:
  """Projects 3D points in `base` frame to OpenCV `(u, v)` and depth."""
  pts = np.asarray(points_base, dtype=np.float64)
  is_1d = pts.ndim == 1
  pts_2d = pts.reshape(1, 3) if is_1d else pts
  n = pts_2d.shape[0]

  t_cam_base = np.linalg.inv(t_base_cam)
  pts_h = np.hstack([pts_2d, np.ones((n, 1), dtype=np.float64)])
  p_cam = (t_cam_base @ pts_h.T).T[:, :3]

  z_depth = p_cam[:, 2].copy()
  denom = np.where(np.abs(z_depth) > 1e-9, z_depth, 1e-9)

  fx = float(k_mat[0, 0])
  fy = float(k_mat[1, 1])
  cx = float(k_mat[0, 2])
  cy = float(k_mat[1, 2])

  u = fx * (p_cam[:, 0] / denom) + cx
  v = fy * (p_cam[:, 1] / denom) + cy
  uv = np.column_stack([u, v])

  if is_1d:
    return uv[0], float(z_depth[0])
  return uv, z_depth
