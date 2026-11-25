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
"""6-DOF Mink + DAQP inverse kinematics and grasp-pose builder for I2RT YAM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

try:
  import mink
except ImportError:
  mink = None

try:
  from i2rt.robots.utils import ArmType
  from i2rt.robots.utils import GripperType
  from i2rt.robots.utils import combine_arm_and_gripper_xml
except ImportError:
  ArmType = None
  GripperType = None
  combine_arm_and_gripper_xml = None

Q_DEFAULT_SEED = np.array([0.0, 1.2, 1.2, -1.2, 0.0, 0.0], dtype=np.float64)

# Rigid flange-to-D405 optical-frame CAD mounts from i2rt station URDF/MJCFs
# (`{side}_gripper` joint6 output frame -> `{side}_camera` OpenCV optical frame,
# invariant to `<site name="grasp_site">` quaternion conventions).
_FLANGE_D405_MOUNTS: dict[str, tuple[float, tuple[float, float, float]]] = {
    "linear_4310": (25.0, (-7.6568e-08, -0.07043498, 0.07700593)),
    "linear_3507": (25.0, (-7.6568e-08, -0.07043498, 0.07700593)),
    "crank_4310": (50.0, (0.00169990, -0.07972864, 0.06602077)),
}


def resolve_i2rt_enum(enum_cls: Any, value: str | Any) -> Any:
  """Resolves an i2rt ArmType or GripperType from string name/value or enum."""
  if enum_cls is None or isinstance(value, enum_cls):
    return value
  key = str(value).strip().lower()
  for member in enum_cls:
    if key in (member.name.lower(), str(member.value).lower()):
      return member
  valid = [m.value for m in enum_cls]
  raise ValueError(
      f"Unsupported {enum_cls.__name__} {value!r}; expected one of {valid}."
  )


def _rx_extrinsic(
    cant_deg: float, xyz: tuple[float, float, float]
) -> np.ndarray:
  """Builds 4x4 T_flange_cam for a D405 canted by Rx(-cant_deg) toward +Z."""
  th = float(np.deg2rad(cant_deg))
  return np.array(
      [
          [1.0, 0.0, 0.0, xyz[0]],
          [0.0, np.cos(th), np.sin(th), xyz[1]],
          [0.0, -np.sin(th), np.cos(th), xyz[2]],
          [0.0, 0.0, 0.0, 1.0],
      ],
      dtype=np.float64,
  )


def _wrist_cam_extrinsic(
    pitch_deg: float,
    xyz: tuple[float, float, float],
    jaw_along_x: bool = False,
) -> np.ndarray:
  """Builds T_grasp_site_cam for +/-X_site (v1.4.7+) or +/-Y_site jaws."""
  if jaw_along_x:
    return _rx_extrinsic(pitch_deg, xyz)
  th = float(np.deg2rad(pitch_deg))
  return np.array(
      [
          [0.0, np.cos(th), np.sin(th), xyz[0]],
          [-1.0, 0.0, 0.0, xyz[1]],
          [0.0, -np.sin(th), np.cos(th), xyz[2]],
          [0.0, 0.0, 0.0, 1.0],
      ],
      dtype=np.float64,
  )


def _detect_installed_i2rt_flange_tcp() -> bool:
  """Returns whether installed i2rt uses v1.4.7+ flange-aligned grasp_site.

  Defaults to True (current v1.4.7+ convention) when `i2rt` is not installed;
  callers loading custom or legacy XML paths directly should call
  `ik.wrist_cam_extrinsic()` instead of reading `STATION_D405_EXTRINSICS`.
  """
  if GripperType is None:
    return True
  try:
    lin_xml = GripperType.LINEAR_4310.get_xml_path()
    root = ET.parse(lin_xml).getroot()
    for site in root.iter("site"):
      if site.get("name") == "grasp_site":
        quat_vals = [float(v) for v in site.get("quat", "1 0 0 0").split()]
        # MuJoCo quat=(w, x, y, z): pre-v1.4.7 quat="0 1 0 0" has q_y=0
        # (index 2); v1.4.7+ quat="0 0.70711 0.70711 0" has |q_y|≈0.707.
        return len(quat_vals) == 4 and abs(quat_vals[2]) > 0.3
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return True


def _default_station_extrinsics() -> dict[str, np.ndarray]:
  if _detect_installed_i2rt_flange_tcp():
    return {
        "linear_4310": _wrist_cam_extrinsic(
            25.0, (-3.89e-05, -0.070532, -0.069767), jaw_along_x=True
        ),
        "crank_4310": _wrist_cam_extrinsic(
            50.0, (0.001788, -0.124319, -0.080744), jaw_along_x=True
        ),
    }
  return {
      "linear_4310": _wrist_cam_extrinsic(
          25.0, (-0.070535, 0.0, -0.067642), jaw_along_x=False
      ),
  }


# Prefer `ik.wrist_cam_extrinsic()` so T_grasp_site_cam is always derived from
# the exact loaded MJCF regardless of i2rt or mujoco_menagerie version.
STATION_D405_EXTRINSICS: dict[str, np.ndarray] = _default_station_extrinsics()


class YamIk:
  """Mink + DAQP 6-DOF FK/IK solver for YAM and YAM Pro arms and grippers."""

  def __init__(
      self,
      xml_path: str | None = None,
      arm_type: str | Any = "yam",
      gripper_type: str | Any = "linear_4310",
      site_name: str | None = "grasp_site",
      open_finger_pos: float | None = None,
  ) -> None:
    self.arm_type = arm_type
    self.gripper_type = gripper_type
    if xml_path is None:
      if combine_arm_and_gripper_xml is None:
        raise RuntimeError(
            "Pass xml_path or install i2rt for combine_arm_and_gripper_xml."
        )
      arm_enum = resolve_i2rt_enum(ArmType, arm_type)
      grip_enum = resolve_i2rt_enum(GripperType, gripper_type)
      xml_path = combine_arm_and_gripper_xml(arm_enum, grip_enum)
    self.xml_path = xml_path
    self.model = mujoco.MjModel.from_xml_path(xml_path)
    self.data = mujoco.MjData(self.model)
    self.arm_dof = min(6, int(self.model.nv))

    if site_name is None and self.model.nsite > 0:
      site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, 0)
    self.site_name = site_name or "grasp_site"
    self.site_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_SITE, self.site_name
    )
    if self.site_id < 0:
      raise ValueError(
          f"Site {self.site_name!r} not found in {xml_path}. Use"
          " combine_arm_and_gripper_xml(arm_type, gripper_type) or pass a"
          " valid site_name."
      )

    self._default_qpos = self.model.qpos0.copy()
    for j_id in range(self.model.njnt):
      q_adr = int(self.model.jnt_qposadr[j_id])
      if q_adr >= self.arm_dof and bool(self.model.jnt_limited[j_id]):
        lo, hi = float(self.model.jnt_range[j_id, 0]), float(
            self.model.jnt_range[j_id, 1]
        )
        target_val = hi if open_finger_pos is None else float(open_finger_pos)
        self._default_qpos[q_adr] = float(np.clip(target_val, lo, hi))

    self.closing_axis_site, self.jaw_perp_axis_site = (
        self._introspect_gripper_axes_in_site()
    )

    non_adj_pairs = self._build_non_adjacent_geom_pairs()
    self._non_adj_pairs = non_adj_pairs
    if mink is not None:
      self.cfg = mink.Configuration(self.model)
      freeze_dofs = list(range(self.arm_dof, self.model.nv))
      self._freeze_task = (
          mink.DofFreezingTask(self.model, dof_indices=freeze_dofs)
          if (freeze_dofs and hasattr(mink, "DofFreezingTask"))
          else None
      )
      self._limits: list[Any] = [mink.ConfigurationLimit(self.model)]
      if non_adj_pairs:
        self._limits.append(
            mink.CollisionAvoidanceLimit(
                model=self.model,
                geom_pairs=non_adj_pairs,
                minimum_distance_from_collisions=0.003,
                collision_detection_distance=0.03,
            )
        )
    else:
      self.cfg = None
      self._freeze_task = None
      self._limits = []

  def _is_descendant_body(self, body_id: int, ancestor_id: int) -> bool:
    """Returns True if `body_id` is `ancestor_id` or in its body subtree."""
    cur = int(body_id)
    while cur > 0:
      if cur == int(ancestor_id):
        return True
      cur = int(self.model.body_parentid[cur])
    return False

  def _is_rigid_descendant_body(self, body_id: int, ancestor_id: int) -> bool:
    """Returns True if `body_id` attaches to `ancestor_id` with no joints."""
    cur = int(body_id)
    while cur > 0:
      if cur == int(ancestor_id):
        return True
      if int(self.model.body_jntnum[cur]) > 0:
        return False
      cur = int(self.model.body_parentid[cur])
    return False

  def _introspect_gripper_axes_in_site(self) -> tuple[np.ndarray, np.ndarray]:
    """Introspects jaw-closing and jaw-perpendicular unit axes in site frame."""
    data_lo = mujoco.MjData(self.model)
    data_hi = mujoco.MjData(self.model)
    data_lo.qpos[:] = self.model.qpos0.copy()
    data_hi.qpos[:] = self.model.qpos0.copy()

    finger_roots: list[int] = []
    for j_id in range(self.model.njnt):
      q_adr = int(self.model.jnt_qposadr[j_id])
      if q_adr >= self.arm_dof and bool(self.model.jnt_limited[j_id]):
        data_lo.qpos[q_adr] = float(self.model.jnt_range[j_id, 0])
        data_hi.qpos[q_adr] = float(self.model.jnt_range[j_id, 1])
        finger_roots.append(int(self.model.jnt_bodyid[j_id]))

    mujoco.mj_kinematics(self.model, data_lo)
    mujoco.mj_kinematics(self.model, data_hi)
    r_site = data_hi.site_xmat[self.site_id].reshape(3, 3)

    best_xy = np.zeros(3, dtype=np.float64)
    best_norm = 0.0
    for g_id in range(self.model.ngeom):
      b_id = int(self.model.geom_bodyid[g_id])
      if not any(self._is_descendant_body(b_id, root) for root in finger_roots):
        continue
      d_site = r_site.T @ (data_hi.geom_xpos[g_id] - data_lo.geom_xpos[g_id])
      d_xy = np.array([d_site[0], d_site[1], 0.0], dtype=np.float64)
      n_xy = float(np.linalg.norm(d_xy))
      if n_xy > best_norm:
        best_norm = n_xy
        best_xy = d_xy

    if best_norm <= 1e-6:
      for b_id in finger_roots:
        d_site = r_site.T @ (data_hi.xpos[b_id] - data_lo.xpos[b_id])
        d_xy = np.array([d_site[0], d_site[1], 0.0], dtype=np.float64)
        n_xy = float(np.linalg.norm(d_xy))
        if n_xy > best_norm:
          best_norm = n_xy
          best_xy = d_xy

    if best_norm > 1e-6 and abs(float(best_xy[0])) > abs(float(best_xy[1])):
      # i2rt >= v1.4.7 flange-aligned TCP: +-X_site is jaw stroke, +Y_site perp.
      return (
          np.array([1.0, 0.0, 0.0], dtype=np.float64),
          np.array([0.0, 1.0, 0.0], dtype=np.float64),
      )
    # Legacy i2rt / mujoco_menagerie TCP: +-Y_site is jaw stroke, +X_site perp.
    return (
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )

  def wrist_cam_extrinsic(
      self, gripper_type: str | Any | None = None
  ) -> np.ndarray:
    """Returns 4x4 T_site_cam (OpenCV optical axes) matched to `self.model`.

    Uses an explicit wrist `<camera>` rigidly mounted to `site_body` in
    `self.model` when present; otherwise maps the rigid flange-to-D405 station
    CAD mount into `self.site_name`'s exact position and axis orientation.

    Args:
      gripper_type: Optional gripper key override (e.g. `'linear_4310'` or
        `'crank_4310'`). Defaults to `self.gripper_type`.
    """
    self.data.qpos[:] = self._default_qpos
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_camlight(self.model, self.data)
    r_site = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
    p_site = self.data.site_xpos[self.site_id].copy()
    site_body = int(self.model.site_bodyid[self.site_id])

    for cam_id in range(self.model.ncam):
      cam_body = int(self.model.cam_bodyid[cam_id])
      if self._is_rigid_descendant_body(cam_body, site_body):
        r_mj = self.data.cam_xmat[cam_id].reshape(3, 3)
        r_cv = r_mj @ np.diag([1.0, -1.0, -1.0])
        p_cam = self.data.cam_xpos[cam_id]
        t_out = np.eye(4, dtype=np.float64)
        t_out[:3, :3] = r_site.T @ r_cv
        t_out[:3, 3] = r_site.T @ (p_cam - p_site)
        return t_out

    raw_key = gripper_type if gripper_type is not None else self.gripper_type
    key = (
        str(raw_key.value).lower()
        if hasattr(raw_key, "value")
        else str(raw_key).strip().lower()
    )
    cant_deg, flange_cam_xyz = _FLANGE_D405_MOUNTS.get(
        key, _FLANGE_D405_MOUNTS["linear_4310"]
    )
    t_flange_cam = _rx_extrinsic(cant_deg, flange_cam_xyz)

    tcp_sid = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site"
    )
    if tcp_sid >= 0:
      p_site_flange = r_site.T @ (self.data.site_xpos[tcp_sid] - p_site)
    else:
      p_site_flange = np.array([0.0, 0.0, -0.146773], dtype=np.float64)

    jaw_along_x = bool(
        abs(float(self.closing_axis_site[0]))
        >= abs(float(self.closing_axis_site[1]))
    )
    t_site_flange = np.eye(4, dtype=np.float64)
    if not jaw_along_x:
      t_site_flange[:3, :3] = np.array(
          [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
          dtype=np.float64,
      )
    t_site_flange[:3, 3] = p_site_flange
    return t_site_flange @ t_flange_cam

  def _build_non_adjacent_geom_pairs(self) -> list[tuple[list[int], list[int]]]:
    """Builds self-collision geom pairs excluding same/parent-child bodies."""
    pairs: list[tuple[list[int], list[int]]] = []
    for g1 in range(self.model.ngeom):
      if (
          self.model.geom_contype[g1] == 0
          and self.model.geom_conaffinity[g1] == 0
      ):
        continue
      b1 = int(self.model.geom_bodyid[g1])
      for g2 in range(g1 + 1, self.model.ngeom):
        if (
            self.model.geom_contype[g2] == 0
            and self.model.geom_conaffinity[g2] == 0
        ):
          continue
        b2 = int(self.model.geom_bodyid[g2])
        if b1 == b2:
          continue
        if (
            b1 > 0
            and b2 > 0
            and (
                int(self.model.body_parentid[b2]) == b1
                or int(self.model.body_parentid[b1]) == b2
            )
        ):
          continue
        pairs.append(([g1], [g2]))
    return pairs

  def _full_q(self, q_arm: Sequence[float] | np.ndarray) -> np.ndarray:
    q = self._default_qpos.copy()
    q[: self.arm_dof] = np.asarray(q_arm, dtype=np.float64)[: self.arm_dof]
    return q

  def fk(
      self, q_arm: Sequence[float] | np.ndarray, site_name: str | None = None
  ) -> np.ndarray:
    """Returns 4x4 T_world_site for q_arm."""
    if site_name is None:
      sid = self.site_id
    else:
      sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
      if sid < 0:
        raise ValueError(
            f"Unknown MuJoCo site {site_name!r} in {self.xml_path}."
        )
    self.data.qpos[:] = self._full_q(q_arm)
    mujoco.mj_kinematics(self.model, self.data)
    t_mat = np.eye(4, dtype=np.float64)
    t_mat[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
    t_mat[:3, 3] = self.data.site_xpos[sid]
    return t_mat

  def _neutral_site_axis(self, x: float, y: float, col: int = 0) -> np.ndarray:
    """Computes neutral site column direction at J6=0 from loaded MJCF."""
    q_ref = Q_DEFAULT_SEED.copy()
    q_ref[0] = float(np.arctan2(y, x))
    q_ref[5] = 0.0
    q_ref = np.clip(
        q_ref[: self.arm_dof],
        self.model.jnt_range[: self.arm_dof, 0],
        self.model.jnt_range[: self.arm_dof, 1],
    )
    return self.fk(q_ref)[:3, col].copy()

  def _neutral_x_axis(self, x: float, y: float) -> np.ndarray:
    """Computes neutral +X_site direction at J6=0 directly from loaded MJCF."""
    return self._neutral_site_axis(x, y, col=0)

  @staticmethod
  def orthonormal_grasp_frame(
      pos: Sequence[float] | np.ndarray,
      approach_dir: Sequence[float] | np.ndarray = (0.0, 0.0, -1.0),
      x_align_dir: Sequence[float] | np.ndarray | None = None,
      neutral_x_dir: Sequence[float] | np.ndarray | None = None,
      *,
      align_y_instead_of_x: bool = False,
  ) -> np.ndarray:
    """Builds a guaranteed right-handed SO(3) grasp_site pose.

    Supports both legacy YAM grippers (`+-Y_site` jaw stroke, `align_y=False`)
    and `i2rt >= v1.4.7` flange-aligned grippers (`+-X_site` jaw stroke,
    `align_y_instead_of_x=True`).
    """
    p_vec = np.asarray(pos, dtype=np.float64)[:3]
    z_raw = np.asarray(approach_dir, dtype=np.float64)[:3]
    z_norm = float(np.linalg.norm(z_raw))
    if z_norm < 1e-9:
      raise ValueError("approach_dir must be non-zero.")
    z_site = z_raw / z_norm

    if neutral_x_dir is None:
      radial = float(np.arctan2(p_vec[1], p_vec[0])) + np.pi
      ref_vec = np.array(
          [np.cos(radial), np.sin(radial), 0.0], dtype=np.float64
      )
    else:
      ref_vec = np.asarray(neutral_x_dir, dtype=np.float64)[:3]
    ref_proj = ref_vec - float(np.dot(ref_vec, z_site)) * z_site

    if x_align_dir is None:
      perp_proj = ref_proj
    else:
      cand = np.asarray(x_align_dir, dtype=np.float64)[:3]
      perp_proj = cand - float(np.dot(cand, z_site)) * z_site
      if (
          float(np.linalg.norm(perp_proj)) > 1e-6
          and float(np.linalg.norm(ref_proj)) > 1e-6
          and float(np.dot(perp_proj, ref_proj)) < 0.0
      ):
        perp_proj = -perp_proj

    perp_norm = float(np.linalg.norm(perp_proj))
    if perp_norm <= 1e-6:
      fallback = (
          np.array([0.0, 0.0, 1.0], dtype=np.float64)
          if abs(float(z_site[2])) < 0.9
          else np.array([1.0, 0.0, 0.0], dtype=np.float64)
      )
      y_site = np.cross(z_site, fallback)
      y_site /= float(np.linalg.norm(y_site))
      x_site = np.cross(y_site, z_site)
      x_site /= float(np.linalg.norm(x_site))
    elif align_y_instead_of_x:
      y_site = perp_proj / perp_norm
      x_site = np.cross(y_site, z_site)
      x_site /= float(np.linalg.norm(x_site))
    else:
      x_site = perp_proj / perp_norm
      y_site = np.cross(z_site, x_site)
      y_site /= float(np.linalg.norm(y_site))

    t_target = np.eye(4, dtype=np.float64)
    t_target[:3, 0] = x_site
    t_target[:3, 1] = y_site
    t_target[:3, 2] = z_site
    t_target[:3, 3] = p_vec
    return t_target

  def make_grasp_pose(
      self,
      pos: Sequence[float] | np.ndarray,
      approach_dir: Sequence[float] | np.ndarray = (0.0, 0.0, -1.0),
      jaw_perp_axis: Sequence[float] | np.ndarray | None = None,
  ) -> np.ndarray:
    """Builds a 4x4 grasp_site target for arbitrary approach & jaw orientation.

    Args:
      pos: Target [x, y, z] position in robot base frame (m).
      approach_dir: Approach direction (+Z_site) in base frame.
      jaw_perp_axis: Optional 3D axis perpendicular to jaw closing (e.g., an
        elongated object's major axis). Automatically aligns whichever site axis
        (`+Y_site` on i2rt >= v1.4.7, `+X_site` on legacy XMLs) is orthogonal
        to the introspected finger stroke while keeping J6 near neutral 0.
    """
    p_vec = np.asarray(pos, dtype=np.float64)[:3]
    jaw_along_x = bool(
        abs(float(self.closing_axis_site[0]))
        >= abs(float(self.closing_axis_site[1]))
    )
    perp_col = 1 if jaw_along_x else 0
    neutral_perp = self._neutral_site_axis(
        float(p_vec[0]), float(p_vec[1]), col=perp_col
    )
    return self.orthonormal_grasp_frame(
        pos=p_vec,
        approach_dir=approach_dir,
        x_align_dir=jaw_perp_axis,
        neutral_x_dir=neutral_perp,
        align_y_instead_of_x=jaw_along_x,
    )

  def make_topdown_grasp_pose(
      self,
      x: float,
      y: float,
      z: float,
      obj_yaw_rad: float | None = None,
      tilt_deg: float = 0.0,
  ) -> np.ndarray:
    """Convenience builder for vertical or radially-tilted top-down grasps."""
    q1_radial = float(np.arctan2(y, x))
    tilt_rad = float(np.deg2rad(tilt_deg))
    approach = np.array(
        [
            np.sin(tilt_rad) * np.cos(q1_radial),
            np.sin(tilt_rad) * np.sin(q1_radial),
            -np.cos(tilt_rad),
        ],
        dtype=np.float64,
    )
    jaw_perp = (
        None
        if obj_yaw_rad is None
        else np.array(
            [np.cos(float(obj_yaw_rad)), np.sin(float(obj_yaw_rad)), 0.0],
            dtype=np.float64,
        )
    )
    return self.make_grasp_pose(
        pos=(x, y, z), approach_dir=approach, jaw_perp_axis=jaw_perp
    )

  def _solve_jacobian_fallback(
      self,
      t_goal: np.ndarray,
      q_start: np.ndarray,
      q_posture: np.ndarray,
      pos_cost: float,
      ori_cost: float,
      max_iters: int,
  ) -> np.ndarray:
    """Damped least-squares MuJoCo Jacobian solver used when mink is absent."""
    dof = self.arm_dof
    lo = self.model.jnt_range[:dof, 0].copy()
    hi = self.model.jnt_range[:dof, 1].copy()
    for j in range(dof):
      if not bool(self.model.jnt_limited[j]):
        lo[j], hi[j] = -np.pi, np.pi
    q_cur = np.clip(q_start[:dof], lo, hi)
    jacp = np.zeros((3, self.model.nv), dtype=np.float64)
    jacr = np.zeros((3, self.model.nv), dtype=np.float64)
    q_err = np.zeros(4, dtype=np.float64)
    vel_body = np.zeros(3, dtype=np.float64)
    eye_dof = np.eye(dof, dtype=np.float64)
    w_p = float(max(pos_cost, 0.0))
    w_r = float(max(ori_cost, 0.0))

    for _ in range(max_iters):
      self.data.qpos[:] = self._full_q(q_cur)
      mujoco.mj_kinematics(self.model, self.data)
      mujoco.mj_comPos(self.model, self.data)
      p_cur = self.data.site_xpos[self.site_id].copy()
      r_cur = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
      e_pos = t_goal[:3, 3] - p_cur
      r_err = r_cur.T @ t_goal[:3, :3]
      mujoco.mju_mat2Quat(q_err, r_err.ravel())
      mujoco.mju_quat2Vel(vel_body, q_err, 1.0)
      e_rot = r_cur @ vel_body
      pos_done = float(np.linalg.norm(e_pos)) <= 5e-4
      ori_done = w_r <= 0.0 or float(np.linalg.norm(e_rot)) <= 2e-3
      if pos_done and ori_done:
        break
      mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
      j_arm = np.vstack([w_p * jacp[:, :dof], w_r * jacr[:, :dof]])
      err6 = np.concatenate([w_p * e_pos, w_r * e_rot])
      lhs = j_arm.T @ j_arm + 1e-3 * eye_dof
      rhs = j_arm.T @ err6 + 2e-4 * (q_posture - q_cur)
      dq = np.linalg.solve(lhs, rhs)
      step_norm = float(np.linalg.norm(dq))
      if step_norm > 0.25:
        dq *= 0.25 / step_norm
      q_cur = np.clip(q_cur + dq, lo, hi)
    return q_cur

  def solve(
      self,
      target_pose: np.ndarray,
      q_seed: Sequence[float] | np.ndarray | None = None,
      pos_cost: float = 1.0,
      ori_cost: float = 1.0,
      pos_tol_m: float = 2e-3,
      ori_tol_rad: float = 1e-2,
      max_iters: int = 300,
      dt_s: float = 0.05,
  ) -> np.ndarray:
    """Solves 6-DOF IK with DAQP, re-seeding every call and verifying FK."""
    seed_arm = (
        Q_DEFAULT_SEED[: self.arm_dof]
        if q_seed is None
        else np.asarray(q_seed, dtype=np.float64)[: self.arm_dof]
    )
    q_init = self._full_q(seed_arm)
    t_goal = np.asarray(target_pose, dtype=np.float64)
    posture_ref = (
        self._full_q(Q_DEFAULT_SEED[: self.arm_dof])
        if q_seed is None
        else q_init
    )

    if mink is not None and self.cfg is not None:
      self.cfg.update(q_init)
      ee_task = mink.FrameTask(
          frame_name=self.site_name,
          frame_type="site",
          position_cost=pos_cost,
          orientation_cost=ori_cost,
          lm_damping=1e-3,
      )
      ee_task.set_target(mink.SE3.from_matrix(t_goal))
      posture_task = mink.PostureTask(self.model, cost=1e-3)
      posture_task.set_target(posture_ref)

      tasks: list[Any] = [ee_task, posture_task]
      if self._freeze_task is not None:
        tasks.append(self._freeze_task)

      for _ in range(max_iters):
        vel = mink.solve_ik(
            self.cfg,
            tasks,
            dt=dt_s,
            solver="daqp",
            damping=1e-3,
            limits=self._limits,
        )
        self.cfg.integrate_inplace(vel, dt_s)
        err = ee_task.compute_error(self.cfg)
        pos_done = float(np.linalg.norm(err[:3])) <= 5e-4
        ori_done = ori_cost <= 0.0 or float(np.linalg.norm(err[3:])) <= 2e-3
        if pos_done and ori_done:
          break
      q_sol = self.cfg.q[: self.arm_dof].copy()
    else:
      q_sol = self._solve_jacobian_fallback(
          t_goal=t_goal,
          q_start=seed_arm.copy(),
          q_posture=posture_ref[: self.arm_dof].copy(),
          pos_cost=pos_cost,
          ori_cost=ori_cost,
          max_iters=max_iters,
      )
    t_achieved = self.fk(q_sol)
    pos_err = float(np.linalg.norm(t_achieved[:3, 3] - t_goal[:3, 3]))
    rot_err_mat = t_achieved[:3, :3].T @ t_goal[:3, :3]
    cos_ang = np.clip(0.5 * (np.trace(rot_err_mat) - 1.0), -1.0, 1.0)
    ori_err = float(np.arccos(cos_ang))
    if pos_err > pos_tol_m or (ori_cost > 0.0 and ori_err > ori_tol_rad):
      raise ValueError(
          f"IK residual too large for {t_goal[:3, 3].tolist()}: "
          f"pos_err={pos_err*1000:.2f} mm (tol {pos_tol_m*1000:.1f} mm), "
          f"ori_err={np.rad2deg(ori_err):.2f} deg."
      )
    return q_sol
