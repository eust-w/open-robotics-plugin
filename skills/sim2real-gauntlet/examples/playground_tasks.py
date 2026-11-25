# Copyright 2026 DeepMind Technologies Limited
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
"""Example gauntlet tasks on the playground YAM and SO-101 ALOHA-table stacks.

`cube_in_bowl`, `peg_in_plate`, and `two_block_stack` implement the
`robosim_gauntlet.task.Task` contract. Sim builds a reaf-sim `flat_scene`
`DeviceLayer` (`Impl.CPP`); real builds the playground hardware stack. Only
`robosim_gauntlet.task.SafetyLimits` is imported from the gauntlet.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
import pathlib
import sys
from typing import Any

import mujoco
from mujoco_playground import real as mp_real
from mujoco_playground import wrapper
from mujoco_playground.robots.so101.real import constant as so101_real
from mujoco_playground.robots.so101.real import device as so101_real_device
from mujoco_playground.robots.so101.sim import constant as so101_sim
from mujoco_playground.robots.so101.sim import device as so101_sim_device
from mujoco_playground.robots.yam.real import constant as yam_real
from mujoco_playground.robots.yam.real import device as yam_real_device
from mujoco_playground.robots.yam.sim import constant as yam_sim
from mujoco_playground.robots.yam.sim import device as yam_sim_device
import numpy as np
from reaf.simulation import torch as reaf_sim
from robosim_gauntlet import task as task_lib

_HERE = pathlib.Path(__file__).absolute().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

# pylint: disable=g-import-not-at-top
import rgbd_ik_perception as perc

# pylint: enable=g-import-not-at-top

TASK_IDS = ("cube_in_bowl", "peg_in_plate", "two_block_stack")
OVERHEAD_CAM = perc.OVERHEAD_CAM
WRIST_CAM = perc.WRIST_CAM
CAM_W, CAM_H = 160, 120
CONTROL_DT_S = perc.CONTROL_DT_S
nominal_robot_spec = perc.nominal_robot_spec
_HOME = {
    "yam": dict(
        zip(
            (*yam_real.ARM_JOINTS, "left_finger", "right_finger"),
            (0.0, 1.6, 1.23, -1.19, 0.0, 0.0, 0.0475, 0.0475),
        )
    ),
    "so101": dict(
        zip(so101_real.JOINTS, (-0.192, -0.262, 0.253, 1.13, 0.0793, 1.7453292))
    ),
}


def _box(
    spec: mujoco.MjSpec,
    name: str,
    pos: Sequence[float],
    size: Sequence[float],
    rgba: Sequence[float],
    mass: float,
    friction: Sequence[float] = (1.4, 0.03, 0.005),
    geom_type: mujoco.mjtGeom = mujoco.mjtGeom.mjGEOM_BOX,
) -> None:
  body = spec.worldbody.add_body(name=name, pos=list(pos))
  body.add_freejoint(name=f"{name}_free")
  body.add_geom(
      name=f"{name}_geom",
      type=geom_type,
      size=list(size),
      mass=mass,
      friction=list(friction),
      solref=[0.012, 1.0],
      rgba=list(rgba),
  )


def _bowl(spec: mujoco.MjSpec, pos: Sequence[float], inner_r: float) -> None:
  """A round bowl: a floor disc and 16 rim boxes."""
  rgba, thick, half_h, n = [0.16, 0.42, 0.88, 1.0], 0.008, 0.022, 16
  bowl = spec.worldbody.add_body(name="bowl", pos=list(pos))
  bowl.add_geom(
      name="bowl_floor",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=[inner_r + 2 * thick, 0.004, 0],
      pos=[0, 0, 0.004],
      rgba=rgba,
  )
  ring = inner_r + thick
  half_len = ring * math.tan(math.pi / n) + 0.001
  for i in range(n):
    th = 2 * math.pi * i / n
    bowl.add_geom(
        name=f"bowl_rim_{i:02d}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[thick, half_len, half_h],
        pos=[ring * math.cos(th), ring * math.sin(th), half_h],
        quat=[math.cos(th / 2), 0, 0, math.sin(th / 2)],
        rgba=rgba,
    )


def _socket(spec: mujoco.MjSpec, pos: Sequence[float], inner_r: float) -> None:
  """An octagonal socket plate."""
  rgba = [0.32, 0.36, 0.42, 1.0]
  sock = spec.worldbody.add_body(name="plate_socket", pos=list(pos))
  sock.add_geom(
      name="socket_base",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=[max(0.052, inner_r + 0.028), 0.004, 0],
      pos=[0, 0, 0.004],
      friction=[0.7, 0.01, 0.002],
      rgba=rgba,
  )
  r = inner_r + 0.010
  for k in range(8):
    a = k * math.pi / 4
    sock.add_geom(
        name=f"socket_wall_{k}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.010, r * math.tan(math.pi / 8) + 0.0015, 0.012],
        pos=[r * math.cos(a), r * math.sin(a), 0.020],
        quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)],
        friction=[0.6, 0.01, 0.002],
        rgba=rgba,
    )


def _tilt_deg(pose: np.ndarray) -> float:
  """Angle between the body z axis and world z, from `[pos, wxyz]`."""
  qx, qy = pose[4], pose[5]
  return math.degrees(math.acos(float(np.clip(1 - 2 * (qx**2 + qy**2), -1, 1))))


@dataclasses.dataclass(frozen=True)
class PlaygroundTask:
  """A `robosim_gauntlet.task.Task` on the YAM or SO-101 ALOHA table."""

  task_id: str
  robot: str
  prompt: str
  target_body: str
  support_body: str
  scene_fn: Callable[[mujoco.MjSpec, float], None]
  success_fn: Callable[[Mapping[str, np.ndarray]], bool]
  bystander_bodies: tuple[str, ...] = ("bystander_cube",)
  control_dt_s: float = CONTROL_DT_S
  episode_s: float = 45.0
  # Rigid blocks: opt up from the fail-closed 35 N fragile-object default.
  safety: task_lib.SafetyLimits = task_lib.SafetyLimits(max_grasp_force_n=150.0)
  robot_pattern: str = r"^robot/"
  include_overhead: bool = True
  include_depth: bool = True
  real_devices: tuple[Any, ...] = ()
  real_time_homing: bool = True
  homing_s: float = 1.5

  @property
  def pad_pattern(self) -> str:
    if self.robot == "yam":
      return r"^robot/(left|right)_(finger|pad)$"
    return r"^robot/(gripper|moving_jaw_so101_v1)$"

  @property
  def table_z_m(self) -> float:
    return perc.nominal_table_z(self.robot)

  def deployment_keys(self) -> tuple[frozenset[str], frozenset[str]]:
    real = yam_real if self.robot == "yam" else so101_real
    cams = {WRIST_CAM}
    if self.include_overhead:
      cams.add(OVERHEAD_CAM)
      if self.include_depth:
        cams.add(f"{OVERHEAD_CAM}_depth")
    if self.robot == "yam" and self.include_depth:
      cams.add(f"{WRIST_CAM}_depth")  # The YAM wrist camera is a D405.
    return (
        frozenset(real.JOINT_POSITION_KEYS.values()) | cams,
        frozenset(real.GOAL_POSITION_KEYS.values()),
    )

  def success(self, poses: Mapping[str, np.ndarray], poses0: Any) -> bool:
    del poses0
    return bool(self.success_fn(poses))

  def build_device_layer(
      self,
      *,
      mode: str,
      extra_devices: Mapping[str, Any] = {},  # pylint: disable=dangerous-default-value
      real_devices: Sequence[Any] = (),
  ) -> tuple[Any, Callable[[], None], Callable[[], None]]:
    """Returns `(numpy DeviceLayer, reset, close)` for `mode="sim"|"real"`.

    Args:
      mode: `"sim"` or `"real"`.
      extra_devices: Sim-only `flat_scene` components merged into the scene.
      real_devices: Real devices; defaults to the follower and wrist camera. The
        rig's overhead RGB-D camera must be passed here in real mode.
    """
    if mode == "sim":
      return self._build_sim(extra_devices)
    real = yam_real if self.robot == "yam" else so101_real
    dev = yam_real_device if self.robot == "yam" else so101_real_device
    devices = list(real_devices or self.real_devices) or [
        dev.create_follower_device(),
        dev.create_wrist_camera_device(),
    ]
    dl = dev.create_device_layer(
        *devices, control_period_seconds=self.control_dt_s
    )
    keys = [real.GOAL_POSITION_KEYS[j] for j in real.JOINTS]
    jkeys = [real.JOINT_POSITION_KEYS[j] for j in real.JOINTS]
    open_grip = 1.0 if self.robot == "yam" else 100.0
    hover = np.array([
        open_grip if j == real.GRIPPER_JOINT else _HOME[self.robot].get(j, 0.0)
        for j in real.JOINTS
    ])
    rest = np.asarray(real.FOLLOWER_HOME, dtype=float)

    def _interp(meas: Mapping[str, Any], target: np.ndarray) -> None:
      mp_real.cosine_interpolate(
          lambda p: dl.step({k: np.float32(v) for k, v in zip(keys, p)}),
          start_pose=np.array([float(meas[k]) for k in jkeys]),
          target_pose=target,
          duration_secs=self.homing_s,
          control_timestep=self.control_dt_s,
          real_time=self.real_time_homing,
      )

    def reset() -> None:
      _interp(dl.begin_stepping(), hover)
      dl.end_stepping()

    def close() -> None:
      _interp(dl.begin_stepping(), rest)
      dl.end_stepping()
      for d in devices:
        d.close()

    return dl, reset, close

  def _build_sim(self, extra: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    spec = nominal_robot_spec(self.robot)
    self.scene_fn(spec, self.table_z_m)
    if self.robot == "yam":
      cfg = yam_sim_device.follower_device_config(spec, self.control_dt_s)
      wrist = yam_sim.WRIST_CAM_FULLNAME
      prefix = yam_sim.ATTACHMENT_PREFIX
    else:
      cfg = so101_sim_device.follower_device_config(spec, self.control_dt_s)
      wrist = so101_sim.WRIST_CAM_FULLNAME
      prefix = so101_sim.ATTACHMENT_PREFIX
    cfg.sim.impl = reaf_sim.Impl.CPP.value
    cfg.sim.simulation_timestep = spec.option.timestep
    rgbd = reaf_sim.RenderMode.RGB | reaf_sim.RenderMode.DEPTH
    rgb = reaf_sim.RenderMode.RGB
    wrist_mode = rgbd if (self.robot == "yam" and self.include_depth) else rgb
    cam_specs = [(WRIST_CAM, wrist, wrist_mode)]
    if self.include_overhead:
      cam_specs.append(
          (OVERHEAD_CAM, OVERHEAD_CAM, rgbd if self.include_depth else rgb)
      )
    for name, cam, mode in cam_specs:
      cfg.device_layer.devices[name] = reaf_sim.flat_scene.Component(
          reaf_sim.MujocoCameraDevice,
          camera_name=cam,
          width=CAM_W,
          height=CAM_H,
          render_rate_hz=10.0,
          render_mode=mode,
          image_output_name_prefix=name,
          rgb_output_name_suffix="",
      )
    cfg.device_layer.devices.update(extra)
    raw = reaf_sim.flat_scene.build_device_layer(cfg)
    dl = wrapper.NumpyDeviceLayerWrapper(wrapper.SingleDeviceLayerWrapper(raw))
    home = _HOME[self.robot]
    grip = 0.0475 if self.robot == "yam" else 1.7453292

    def reset() -> None:
      raw.reset()
      mjm, d = raw.sim.mj_model, raw.sim.mj_data
      for name, q in home.items():
        d.qpos[mjm.jnt_qposadr[mjm.joint(prefix + name).id]] = q
      for a in range(mjm.nu):
        joint = mjm.joint(mjm.actuator_trnid[a, 0]).name.removeprefix(prefix)
        d.ctrl[a] = (
            grip
            if mjm.actuator(a).name == prefix + "gripper"
            else (home.get(joint, d.ctrl[a]))
        )
      raw.forward()

    def close() -> None:
      for d in raw.devices:
        if isinstance(d, reaf_sim.MujocoCameraDevice):
          d.close()

    return dl, reset, close


def _inside_bowl(inner_r: float) -> Callable[..., bool]:
  def ok(p: Mapping[str, np.ndarray]) -> bool:
    d = p["target_cube"][:3] - p["bowl"][:3]
    return bool(np.hypot(d[0], d[1]) < inner_r and 0.0 < d[2] < 0.064)

  return ok


def _in_socket(tol: float) -> Callable[..., bool]:
  def ok(p: Mapping[str, np.ndarray]) -> bool:
    d = p["peg"][:3] - p["plate_socket"][:3]
    return bool(
        np.hypot(d[0], d[1]) < tol and d[2] < 0.057 and _tilt_deg(p["peg"]) < 8
    )

  return ok


def _stacked(p: Mapping[str, np.ndarray]) -> bool:
  d = p["block_top"][:3] - p["block_base"][:3]
  return bool(
      np.hypot(d[0], d[1]) < 0.022
      and 0.03 < d[2] < 0.05
      and _tilt_deg(p["block_top"]) < 10
      and _tilt_deg(p["block_base"]) < 10
  )


def get_task(task_id: str, robot: str = "yam") -> PlaygroundTask:
  """Returns `task_id` on `robot` (`"yam"` or `"so101"`)."""
  yam = robot == "yam"
  tx, ty = (-0.09, 0.02) if yam else (-0.07, -0.05)
  dx, dy = (0.10, 0.03) if yam else (0.07, -0.04)
  half = 0.018 if yam else 0.015
  (bx, by), bh = ((0.18, 0.14), 0.018) if yam else ((0.0, 0.08), 0.016)

  def bystander(spec: mujoco.MjSpec, tz: float) -> None:
    _box(
        spec,
        "bystander_cube",
        (bx, by, tz + bh),
        (bh, bh, bh),
        (0.74, 0.68, 0.22, 1),
        0.06,
        friction=(0.9, 0.01, 0.002),
    )

  if task_id == "cube_in_bowl":
    inner_r = 0.068 if yam else 0.058

    def cube_scene(spec: mujoco.MjSpec, tz: float) -> None:
      _box(
          spec,
          "target_cube",
          (tx, ty, tz + 0.02),
          (half, half, 0.02),
          (0.9, 0.16, 0.16, 1),
          0.06 if yam else 0.045,
      )
      _bowl(
          spec,
          (dx + 0.01 if yam else dx, dy + 0.01 if yam else dy, tz),
          inner_r,
      )
      bystander(spec, tz)

    return PlaygroundTask(
        task_id,
        robot,
        "Place the red cube inside the blue bowl; leave the yellow bystander"
        " untouched.",
        "target_cube",
        "bowl",
        cube_scene,
        _inside_bowl(inner_r),
    )
  if task_id == "peg_in_plate":
    r = 0.013 if yam else 0.015

    def peg_scene(spec: mujoco.MjSpec, tz: float) -> None:
      _box(
          spec,
          "peg",
          (tx - 0.01 if yam else tx, ty - 0.01 if yam else ty, tz + 0.04),
          [r, 0.04, 0] if yam else [r, r, 0.04],
          (0.2, 0.82, 0.3, 1),
          0.075 if yam else 0.055,
          friction=(1.5, 0.04, 0.006),
          geom_type=(
              mujoco.mjtGeom.mjGEOM_CYLINDER
              if yam
              else mujoco.mjtGeom.mjGEOM_BOX
          ),
      )
      _socket(spec, (dx, dy, tz), 0.022 if yam else 0.026)
      bystander(spec, tz)

    return PlaygroundTask(
        task_id,
        robot,
        "Insert the green cylindrical peg upright into the square socket"
        " plate.",
        "peg",
        "plate_socket",
        peg_scene,
        _in_socket(0.012 if yam else 0.016),
    )
  if task_id == "two_block_stack":

    def stack_scene(spec: mujoco.MjSpec, tz: float) -> None:
      _box(
          spec,
          "block_top",
          (tx, ty, tz + 0.02),
          (half, half, 0.02),
          (0.92, 0.36, 0.16, 1),
          0.06 if yam else 0.045,
      )
      _box(
          spec,
          "block_base",
          (dx, dy, tz + 0.02),
          (0.03, 0.03, 0.02),
          (0.22, 0.5, 0.86, 1),
          0.14,
          friction=(1.2, 0.03, 0.004),
      )
      bystander(spec, tz)

    return PlaygroundTask(
        task_id,
        robot,
        "Stack the orange top block cleanly onto the blue base block without"
        " tipping it.",
        "block_top",
        "block_base",
        stack_scene,
        _stacked,
    )
  raise ValueError(f"Unknown task_id {task_id!r}; expected one of {TASK_IDS}.")
