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
"""Unified MuJoCo Playground sim/real DeviceLayer bridge for skill composition."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
import mujoco
import numpy as np

ArrayLike = Sequence[float] | np.ndarray
_CV_FROM_MJ_CAM = np.diag([1.0, -1.0, -1.0])
_SO101_URI = (
    "github:google-deepmind/mujoco_menagerie/robotstudio_so101/so101.xml"
)


def camera_intrinsics_from_model(
    model: mujoco.MjModel, cam_id: int, width: int, height: int
) -> tuple[np.ndarray, float, float, float, float]:
  """Computes exact MuJoCo pinhole intrinsics K matching mjr_render."""
  ss, intr = model.cam_sensorsize[cam_id], model.cam_intrinsic[cam_id]
  if (
      float(ss[0]) > 0.0
      and float(ss[1]) > 0.0
      and float(intr[0]) > 0.0
      and float(intr[1]) > 0.0
  ):
    fx, fy = float(intr[0] / ss[0] * width), float(intr[1] / ss[1] * height)
    cx = float((0.5 - intr[2] / ss[0]) * width)
    cy = float((0.5 - intr[3] / ss[1]) * height)
  else:
    fy = float(0.5 * height / np.tan(0.5 * np.deg2rad(model.cam_fovy[cam_id])))
    fx, cx, cy = fy, 0.5 * float(width), 0.5 * float(height)
  k = np.array(
      [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
  )
  return k, fx, fy, cx, cy


def _mat4(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
  t = np.eye(4, dtype=np.float64)
  t[:3, :3], t[:3, 3] = rot.reshape(3, 3), pos
  return t


def camera_world_pose_opencv(data: mujoco.MjData, cam_id: int) -> np.ndarray:
  """Returns 4x4 T_world_cam in OpenCV (+X right, +Y down, +Z forward) frame."""
  r_cv = data.cam_xmat[cam_id].reshape(3, 3) @ _CV_FROM_MJ_CAM
  return _mat4(r_cv, data.cam_xpos[cam_id])


def _compute_rigid_wrist_bodies(
    model: mujoco.MjModel, last_arm_body: int
) -> set[int]:
  """Returns bodies rigidly attached to last_arm_body without moving joints."""
  rigid_wrist = {int(last_arm_body)}
  for b in range(1, int(model.nbody)):
    cur, rigid = b, True
    while cur > 0 and cur != last_arm_body:
      if int(model.body_jntnum[cur]) > 0:
        rigid = False
        break
      cur = int(model.body_parentid[cur])
    if cur == last_arm_body and rigid:
      rigid_wrist.add(b)
  return rigid_wrist


def solve_site_ik(
    model: mujoco.MjModel,
    site_id: int,
    qpos_indices: Sequence[int],
    dof_indices: Sequence[int],
    q_min: np.ndarray,
    q_max: np.ndarray,
    pos_world: ArrayLike,
    *,
    q_seed: ArrayLike,
    q_posture: ArrayLike | None = None,
    context_qpos: np.ndarray | None = None,
    approach_axis_ee: ArrayLike = (0.0, 0.0, 1.0),
    approach_dir_world: ArrayLike | None = (0.0, 0.0, -1.0),
    jaw_yaw_rad: float | None = None,
    target_rot_world: np.ndarray | None = None,
    pos_tol_m: float = 1.5e-3,
    ori_tol_rad: float = 2e-2,
    max_iters: int = 200,
    damping: float = 2e-2,
) -> np.ndarray:
  """Damped least-squares site IK for 5-DOF and 6-DOF arms with limit clamping."""
  data = mujoco.MjData(model)
  pos_des = np.asarray(pos_world, dtype=np.float64)[:3]
  ax_ee = np.asarray(approach_axis_ee, dtype=np.float64)[:3]
  ax_ee /= float(np.linalg.norm(ax_ee))
  seed0 = np.asarray(q_seed, dtype=np.float64).copy()
  q_nom = seed0 if q_posture is None else np.asarray(q_posture, dtype=float)
  a_des = None
  if approach_dir_world is not None:
    a_des = np.asarray(approach_dir_world, dtype=np.float64)[:3]
    a_des /= float(np.linalg.norm(a_des))

  n_dof = len(dof_indices)
  ba, bb = np.zeros(n_dof), np.zeros(n_dof)
  if n_dof >= 3:
    ba[1], ba[2], bb[1], bb[2] = 1.0, -1.6, -1.0, 1.6
  seeds = [seed0, q_nom, 0.5 * (q_min + q_max), ba, bb]
  seeds += [np.array([p, *q_nom[1:]]) for p in (-0.6, 0.6)]

  jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
  q_idx, eye_d = list(qpos_indices), np.eye(n_dof)
  best_q, best_err = seed0, float("inf")

  for s in seeds:
    q = np.clip(s.copy(), q_min, q_max)
    for _ in range(max_iters):
      data.qpos[:] = model.qpos0 if context_qpos is None else context_qpos
      data.qpos[q_idx] = q
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)
      r_cur = data.site_xmat[site_id].reshape(3, 3)
      e_pos = pos_des - data.site_xpos[site_id]

      if target_rot_world is not None:
        r_des = np.asarray(target_rot_world, dtype=np.float64).reshape(3, 3)
        e_rot = 0.5 * sum(np.cross(r_cur[:, i], r_des[:, i]) for i in range(3))
      elif a_des is not None:
        e_rot = np.cross(r_cur @ ax_ee, a_des)
        if jaw_yaw_rad is not None:
          perp = np.array(
              [0.0, 1.0, 0.0] if abs(ax_ee[0]) > 0.9 else [1.0, 0.0, 0.0]
          )
          yd = np.array([np.cos(jaw_yaw_rad), np.sin(jaw_yaw_rad), 0.0])
          yd -= float(np.dot(yd, a_des)) * a_des
          if float(np.linalg.norm(yd)) > 1e-6:
            yd, pa = yd / float(np.linalg.norm(yd)), r_cur @ perp
            yd = -yd if float(np.dot(pa, yd)) < 0.0 else yd
            e_rot = e_rot + 0.5 * float(np.dot(np.cross(pa, yd), a_des)) * a_des
      else:
        e_rot = np.zeros(3)

      p_err = float(np.linalg.norm(e_pos))
      if p_err < best_err:
        best_err, best_q = p_err, q.copy()
      if p_err <= pos_tol_m and float(np.linalg.norm(e_rot)) <= ori_tol_rad:
        return q

      mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
      jp, jr = jacp[:, dof_indices], jacr[:, dof_indices]
      pos_only = target_rot_world is None and a_des is None
      j_mat = jp if pos_only else np.vstack([jp, 0.35 * jr])
      err = e_pos if pos_only else np.concatenate([e_pos, 0.35 * e_rot])
      jjt = j_mat @ j_mat.T + (damping**2) * np.eye(j_mat.shape[0])
      j_pinv = j_mat.T @ np.linalg.solve(jjt, np.eye(j_mat.shape[0]))
      dq = j_pinv @ err + 0.05 * ((eye_d - j_pinv @ j_mat) @ (q_nom - q))
      step = float(np.max(np.abs(dq)))
      q = np.clip(q + (dq * (0.15 / step) if step > 0.15 else dq), q_min, q_max)

  if best_err > 1.5e-2:
    raise RuntimeError(f"solve_site_ik failed: pos_err={best_err:.4f} m")
  return best_q


def build_playground_spec(
    robot: Literal["yam", "so101"] = "yam",
    setup_scene: Callable[[mujoco.MjSpec], None] | None = None,
) -> mujoco.MjSpec:
  """Builds the SO-101 or YAM ALOHA-table MjSpec for Studio or DeviceLayer."""
  if robot == "so101":
    from mujoco_playground.robots.so101.sim import scene as s_sc

    spec = s_sc.so101_aloha_table()
  elif robot == "yam":
    from mujoco_playground.robots.yam.sim import scene as s_sc

    spec = s_sc.yam_aloha_table()
  else:
    raise ValueError(f"Unsupported robot {robot!r}; use 'yam' or 'so101'.")
  if setup_scene is not None:
    setup_scene(spec)
  return spec


class PlaygroundSim2Real:
  """Out-of-the-box sim/real SO-101 and YAM DeviceLayer bridge."""

  def __init__(
      self,
      robot: Literal["yam", "so101"] = "yam",
      mode: Literal["sim", "real"] = "sim",
      *,
      control_period_s: float = 0.02,
      setup_scene: Callable[[mujoco.MjSpec], None] | None = None,
      extra_cameras: Mapping[str, tuple[str, int, int]] | None = None,
      extra_real_devices: Sequence[Any] = (),
      device_layer: Any | None = None,
      studio_handle: Any | None = None,
      can_channel: str = "can0",
      reset_to_hover: bool = True,
  ) -> None:
    from mujoco_playground import asset, real, wrapper
    from reaf.simulation import torch as reaf_sim
    from reaf.simulation.torch import flat_scene

    if mode not in ("sim", "real"):
      raise ValueError(f"Unsupported mode {mode!r}; expected 'sim' or 'real'.")
    self.robot, self.mode = robot, mode
    self.control_period_s = float(control_period_s)
    self._cosine_interpolate = real.cosine_interpolate
    self._owned_real_devices: list[Any] = []
    self._closed, self._sim, self._studio_handle = False, None, studio_handle
    spec = build_playground_spec(robot, setup_scene)

    def add_cam(
        cfg: Any, key: str, mj_cam: str, w: int, h: int, fps: int = 30
    ) -> None:
      cfg.device_layer.devices[key] = flat_scene.Component(
          reaf_sim.MujocoCameraDevice,
          camera_name=mj_cam,
          width=int(w),
          height=int(h),
          render_rate_hz=fps,
          image_output_name_prefix=key,
          rgb_output_name_suffix="",
      )

    if robot == "so101":
      from mujoco_playground.robots.so101.real import constant as r_c
      from mujoco_playground.robots.so101.sim import constant as s_c, device as s_dev, scene as s_sc

      self.robot_xml_path = str(asset.get(_SO101_URI).path)
      self.hover_pose = np.array([*r_c.HOVER_POSE[:5], 80.0], dtype=np.float64)
      self.gripper_open, self.gripper_closed = 85.0, 12.0
      self.approach_axis_ee = np.array([1.0, 0.0, 0.0])
      pfx, base_body = s_c.ATTACHMENT_PREFIX, f"{s_c.ATTACHMENT_PREFIX}base"
      fingers = (f"{pfx}{r_c.GRIPPER_JOINT}",)
      build_cfg = lambda sp: s_dev.follower_device_config(
          sp, self.control_period_s
      )
    else:
      from mujoco_playground.robots.yam.real import constant as r_c
      from mujoco_playground.robots.yam.sim import constant as s_c, device as s_dev, scene as s_sc

      self.robot_xml_path = str(asset.get(s_sc.YAM_XML_URI).path)
      self.hover_pose = np.array(r_c.HOVER_POSE, dtype=np.float64)
      self.gripper_open, self.gripper_closed = 1.0, 0.0
      self.approach_axis_ee = np.array([0.0, 0.0, 1.0])
      pfx = s_c.ATTACHMENT_PREFIX
      base_body = f"{pfx}{s_c.BASE_BODY}"
      fingers = tuple(f"{pfx}{f}" for f in s_c.FINGER_JOINTS)

      def build_cfg(sp: mujoco.MjSpec) -> Any:
        cfg = flat_scene.Environment(base_scene=sp)
        cfg.sim.nworld, cfg.sim.xpu_device, cfg.sim.impl = 1, "cpu", "cpp"
        cfg.sim.control_timestep = self.control_period_s
        yam_dev = flat_scene.Component(s_dev.YamFollower)
        cfg.device_layer.devices["follower"] = yam_dev
        w_sz = (r_c.CAMERA_WIDTH, r_c.CAMERA_HEIGHT, r_c.CAMERA_FPS)
        add_cam(cfg, r_c.WRIST_CAMERA, s_c.WRIST_CAM_FULLNAME, *w_sz)
        return cfg

    self.table_height_m = float(s_sc.TABLE_HEIGHT)
    self.joint_names = tuple(r_c.JOINTS)
    self.arm_joint_names = tuple(r_c.JOINTS[:-1])
    self._goal_keys = dict(r_c.GOAL_POSITION_KEYS)
    self._meas_keys = dict(r_c.JOINT_POSITION_KEYS)
    self.home_pose = np.array(r_c.FOLLOWER_HOME, dtype=np.float64)
    self._camera_mj_names = {r_c.WRIST_CAMERA: s_c.WRIST_CAM_FULLNAME}
    for k, (mj_cam, _, _) in (extra_cameras or {}).items():
      self._camera_mj_names[k] = mj_cam

    self.mj_model = spec.compile()
    m = self.mj_model
    self._fk_data = mujoco.MjData(m)
    self._base_body_id = int(m.body(base_body).id)
    self._ee_site_id = int(m.site(s_c.EE_SITE).id)
    rb = {self._base_body_id}
    for bid in range(1, m.nbody):
      if int(m.body_parentid[bid]) in rb:
        rb.add(bid)
    gb, bp = m.geom_bodyid, m.body_parentid
    self._ignored_pairs = {
        (g1, g2)
        for g1 in range(m.ngeom)
        for g2 in range(g1 + 1, m.ngeom)
        if (int(gb[g1]) not in rb and int(gb[g2]) not in rb)
        or (
            int(gb[g1]) in rb and int(gb[g2]) in rb and bp[gb[g1]] == bp[gb[g2]]
        )
    }
    arm_j = [m.joint(f"{pfx}{j}") for j in self.arm_joint_names]
    self.arm_qpos_indices = [int(j.qposadr[0]) for j in arm_j]
    self.arm_dof_indices = [int(j.dofadr[0]) for j in arm_j]
    self.q_min = np.array([float(j.range[0]) for j in arm_j])
    self.q_max = np.array([float(j.range[1]) for j in arm_j])
    f_j = [m.joint(fn) for fn in fingers]
    self._finger_qpos_indices = [int(j.qposadr[0]) for j in f_j]
    self._finger_ranges = [(float(j.range[0]), float(j.range[1])) for j in f_j]

    last_arm_body = int(m.jnt_bodyid[int(arm_j[-1].id)])
    self._rigid_wrist_bodies = _compute_rigid_wrist_bodies(m, last_arm_body)

    if device_layer is not None:
      self.device_layer = device_layer
    elif mode == "sim":
      cfg = build_cfg(spec)
      for k, (mj_cam, w, h) in (extra_cameras or {}).items():
        add_cam(cfg, k, mj_cam, w, h)
      sim_dl = flat_scene.build_device_layer(cfg)
      self._sim = sim_dl.sim
      self.device_layer = wrapper.NumpyDeviceLayerWrapper(
          wrapper.SingleDeviceLayerWrapper(sim_dl)
      )
    else:
      if robot == "so101":
        from mujoco_playground.robots.so101.real import device as r_dev

        follower = r_dev.create_follower_device()
      else:
        from mujoco_playground.robots.yam.real import device as r_dev

        follower = r_dev.create_follower_device(channel=can_channel)
      wrist = r_dev.create_wrist_camera_device()
      self._owned_real_devices = [follower, wrist]
      dl_kw = {"control_period_seconds": self.control_period_s}
      self.device_layer = r_dev.create_device_layer(
          follower, wrist, *extra_real_devices, **dl_kw
      )

    self._measurements = dict(self.device_layer.begin_stepping())
    self._sync_fk()
    if reset_to_hover:
      self.move_joint(
          self.hover_pose[:-1], float(self.hover_pose[-1]), duration_s=1.2
      )

  def __enter__(self) -> "PlaygroundSim2Real":
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    del exc_type, exc, tb
    self.close(park=True)

  @property
  def mj_data(self) -> mujoco.MjData | None:
    return None if self._sim is None else self._sim.mj_data

  def attach_studio(self, handle: Any | None) -> None:
    """Attaches a MuJoCo Studio ViewerHandle for live 50 Hz rollout sync."""
    self._studio_handle = handle
    if handle is not None and self.mj_data is not None:
      handle.sync(self.mj_model, self.mj_data)

  @property
  def measurements(self) -> Mapping[str, Any]:
    return self._measurements

  @property
  def q_full(self) -> np.ndarray:
    m = self._measurements
    return np.array([float(m[self._meas_keys[j]]) for j in self.joint_names])

  @property
  def q_arm(self) -> np.ndarray:
    return self.q_full[:-1]

  @property
  def gripper(self) -> float:
    return float(self.q_full[-1])

  @property
  def gripper_normalized(self) -> float:
    s = 100.0 if self.robot == "so101" else 1.0
    return float(np.clip(self.gripper / s, 0.0, 1.0))

  def _finger_qpos(self, g_val: float) -> list[float]:
    f = np.clip(g_val / (100.0 if self.robot == "so101" else 1.0), 0.0, 1.0)
    return [lo + float(f) * (hi - lo) for lo, hi in self._finger_ranges]

  def _sync_fk(
      self, q_arm: ArrayLike | None = None, gripper: float | None = None
  ) -> None:
    d = self._fk_data
    d.qpos[:] = self.mj_model.qpos0
    d.qpos[self.arm_qpos_indices] = self.q_arm if q_arm is None else q_arm
    g = self.gripper if gripper is None else float(gripper)
    d.qpos[self._finger_qpos_indices] = self._finger_qpos(g)
    mujoco.mj_kinematics(self.mj_model, d)
    mujoco.mj_camlight(self.mj_model, d)

  @property
  def t_world_base(self) -> np.ndarray:
    b = self._base_body_id
    return _mat4(self._fk_data.xmat[b], self._fk_data.xpos[b])

  def world_to_base_pose(self, t_world_target: np.ndarray) -> np.ndarray:
    return np.linalg.inv(self.t_world_base) @ np.asarray(
        t_world_target, dtype=np.float64
    )

  def world_to_base_point(self, pos_world: ArrayLike) -> np.ndarray:
    t = np.linalg.inv(self.t_world_base)
    return t[:3, :3] @ np.asarray(pos_world, dtype=np.float64)[:3] + t[:3, 3]

  def fk_world_ee(self, q_arm: ArrayLike | None = None) -> np.ndarray:
    self._sync_fk(q_arm=q_arm)
    s = self._ee_site_id
    return _mat4(self._fk_data.site_xmat[s], self._fk_data.site_xpos[s])

  def camera_opencv(
      self,
      camera_key: str = "wrist_camera",
      mj_camera_name: str | None = None,
      q_arm: ArrayLike | None = None,
      gripper: float | None = None,
  ) -> dict[str, Any]:
    """Returns calibrated RGB frame, K, T_world_cam, and wrist T_ee_cam.

    Args:
      camera_key: Key in measurements dictionary (e.g. 'wrist_camera').
      mj_camera_name: Optional explicit MuJoCo camera name.
      q_arm: Optional arm joint positions for kinematics evaluation.
      gripper: Optional gripper scalar position.

    Returns:
      Dict with 'rgb', 'K', 'fx_px', 'fy_px', 'cx_px', 'cy_px', 'T_world_cam',
      and 'T_ee_cam' (None for fixed scene cameras not on the wrist).
    """
    rgb = np.asarray(self._measurements[camera_key], dtype=np.uint8)
    cname = mj_camera_name or self._camera_mj_names[camera_key]
    cid = int(self.mj_model.camera(cname).id)
    self._sync_fk(q_arm=q_arm, gripper=gripper)
    k, fx, fy, cx, cy = camera_intrinsics_from_model(
        self.mj_model, cid, rgb.shape[1], rgb.shape[0]
    )
    t_wc = camera_world_pose_opencv(self._fk_data, cid)
    cam_body = int(self.mj_model.cam_bodyid[cid])
    ee_site_body = int(self.mj_model.site_bodyid[self._ee_site_id])
    if (
        cam_body in self._rigid_wrist_bodies
        and ee_site_body in self._rigid_wrist_bodies
    ):
      s = self._ee_site_id
      t_we = _mat4(self._fk_data.site_xmat[s], self._fk_data.site_xpos[s])
      t_ee_cam = np.linalg.inv(t_we) @ t_wc
    else:
      t_ee_cam = None
    return dict(
        rgb=rgb,
        K=k,
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        T_world_cam=t_wc,
        T_ee_cam=t_ee_cam,
    )

  def _context_qpos(self, gripper: float | None = None) -> np.ndarray:
    ctx = self.mj_model.qpos0.copy()
    ctx[self.arm_qpos_indices] = self.q_arm
    g = self.gripper if gripper is None else float(gripper)
    ctx[self._finger_qpos_indices] = self._finger_qpos(g)
    return ctx

  def planner_inputs(self, gripper: float | None = None) -> dict[str, Any]:
    """Returns whole-scene model, arm indices/limits, and non-robot ignores."""
    return dict(
        model=self.mj_model, qpos_indices=list(self.arm_qpos_indices),
        q_min=self.q_min.copy(), q_max=self.q_max.copy(),
        context_qpos=self._context_qpos(gripper),
        ignored_geom_pairs=set(self._ignored_pairs),
    )

  def solve_arm_ik(
      self,
      pos_world: ArrayLike,
      *,
      q_seed: ArrayLike | None = None,
      approach_dir_world: ArrayLike | None = (0.0, 0.0, -1.0),
      jaw_yaw_rad: float | None = None,
      target_rot_world: np.ndarray | None = None,
  ) -> np.ndarray:
    """Solves arm IK in world coordinates for either SO-101 or YAM."""
    seed = self.q_arm if q_seed is None else q_seed
    return solve_site_ik(
        self.mj_model, self._ee_site_id, self.arm_qpos_indices,
        self.arm_dof_indices, self.q_min, self.q_max, pos_world,
        q_seed=seed, q_posture=self.hover_pose[:-1],
        context_qpos=self._context_qpos(),
        approach_axis_ee=self.approach_axis_ee,
        approach_dir_world=approach_dir_world,
        jaw_yaw_rad=jaw_yaw_rad, target_rot_world=target_rot_world,
    )

  def step_joint(
      self, q_arm: ArrayLike, gripper: float | None = None
  ) -> Mapping[str, Any]:
    g = self.gripper if gripper is None else float(gripper)
    vals = [*np.asarray(q_arm, dtype=np.float32).tolist(), g]
    cmd = {
        self._goal_keys[j]: np.float32(v)
        for j, v in zip(self.joint_names, vals, strict=True)
    }
    self._measurements = dict(self.device_layer.step(cmd))
    self._sync_fk()
    if self._studio_handle is not None and self.mj_data is not None:
      self._studio_handle.sync(self.mj_model, self.mj_data)
    return self._measurements

  def _interp_to(self, target_full: np.ndarray, duration_s: float) -> None:
    self._cosine_interpolate(
        lambda p: self.step_joint(p[:-1], float(p[-1])),
        start_pose=self.q_full.astype(np.float32),
        target_pose=target_full.astype(np.float32),
        duration_secs=duration_s,
        control_timestep=self.control_period_s,
        real_time=(self.mode == "real"),
    )

  def move_joint(
      self,
      q_arm_goal: ArrayLike,
      gripper: float | None = None,
      *,
      duration_s: float = 1.0,
      settle_steps: int = 8,
  ) -> None:
    g = self.gripper if gripper is None else float(gripper)
    self._interp_to(np.array([*np.asarray(q_arm_goal).tolist(), g]), duration_s)
    for _ in range(settle_steps):
      self.step_joint(q_arm_goal, g)

  def execute_trajectory(
      self, q_trajectory: np.ndarray, gripper: float | None = None
  ) -> None:
    g = self.gripper if gripper is None else float(gripper)
    traj = np.asarray(q_trajectory, dtype=np.float64)
    for wp in [*traj, *([traj[-1]] * 6 if len(traj) > 0 else [])]:
      self.step_joint(wp, g)

  def set_gripper(self, value: float, *, duration_s: float = 0.45) -> None:
    self.move_joint(self.q_arm, float(value), duration_s=duration_s)

  def close(self, *, park: bool = True, duration_s: float = 1.5) -> None:
    if self._closed:
      return
    self._closed = True
    if park:
      self._interp_to(self.home_pose, duration_s)
    self.device_layer.end_stepping()
    for dev in self._owned_real_devices:
      dev.close()
