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
"""Unprivileged RGB-D / RGB + damped-least-squares IK pick-and-place controller.

Runs on any numpy `DeviceLayer` with real-robot keys: the sim layer, the real
YAM / SO-101 layer, or the gauntlet's `RemoteDeviceLayer`. It knows only the
calibrated robot rig (`rgbd_ik_perception.nominal_robot_spec`) and the prompt in
`GAUNTLET_PROMPT` / `--prompt`, never task scene modules or object coordinates.

v0: overhead perception only, coarse release. v1: YAM wrist-camera refinement,
wrist-camera overview fallback when `overhead_cam` is absent, one regrasp, and
fault monitors that abort on a frozen camera or a dead gripper.

CLI: reads `GAUNTLET_URL` and prints `{"status": "done"|"abort", "fault": ...}`
as its last stdout line.
"""

from collections.abc import Iterator, Mapping, Sequence
import json
import os
import pathlib
import sys
from typing import Any

from absl import app
from absl import flags
import mujoco
from mujoco_playground.robots.so101.real import constant as so101_real
from mujoco_playground.robots.yam.real import constant as yam_real
import numpy as np

_HERE = pathlib.Path(__file__).absolute().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "src")]

# pylint: disable=g-import-not-at-top,g-bad-import-order
import rgbd_ik_perception as perc
from robosim_gauntlet import proxy

# pylint: enable=g-import-not-at-top,g-bad-import-order

_FALLBACK_PROMPTS = {
    "cube_in_bowl": (
        "Place the red cube inside the blue bowl; leave the yellow bystander"
        " untouched."
    ),
    "peg_in_plate": (
        "Insert the green cylindrical peg upright into the square socket plate."
    ),
    "two_block_stack": (
        "Stack the orange top block cleanly onto the blue base block without"
        " tipping it."
    ),
}

_GRIP_PROBE_INTERVAL_S = 0.30
_FREEZE_POLL_S = 0.2
_FREEZE_MIN_MOTION_RAD = 0.03
_NO_OPEN_PULSE_RULES = frozenset({
    "pre_grasp", "grasp", "close_jaws", "gentle_release", "retract",
    "reopen_after_miss",
})


class AbortEpisodeError(RuntimeError):
  """A terminal sensor or actuator fault was detected."""

  def __init__(self, fault: str, *, already_stopped: bool = False):
    super().__init__(fault)
    self.fault = fault
    self.already_stopped = already_stopped


class _Snap(Mapping[str, np.ndarray]):
  """Eagerly copies scalar/1-D readings while keeping 2-D/3-D cameras lazy."""

  def __init__(self, meas: Mapping[str, Any], specs_map: Mapping[str, Any]):
    self._raw = meas
    self._keys = tuple(specs_map)
    self._eager = {
        k: np.array(meas[k])
        for k, s in specs_map.items()
        if len(s.shape) < 2 and k in meas
    }

  def __getitem__(self, key: str) -> np.ndarray:
    if key in self._eager:
      return self._eager[key]
    return np.asarray(self._raw[key])

  def __contains__(self, key: object) -> bool:
    return key in self._eager or key in self._raw

  def __iter__(self) -> Iterator[str]:
    return iter(self._keys)

  def __len__(self) -> int:
    return len(self._keys)


class _Arm:
  """Stateful controller over a numpy `DeviceLayer` with real-robot keys."""

  def __init__(self, dl: Any, robot: str, variant: str):
    self.dl, self.robot, self.variant = dl, robot, variant
    self._mspec = dl.measurements_spec()
    real = yam_real if robot == "yam" else so101_real
    self.arm_joints = [j for j in real.JOINTS if j != real.GRIPPER_JOINT]
    self.joint_keys, self.goal_keys = (
        real.JOINT_POSITION_KEYS,
        real.GOAL_POSITION_KEYS,
    )
    self.grip_joint = real.GRIPPER_JOINT
    self.grip_max = 1.0 if robot == "yam" else 100.0
    spec = perc.nominal_robot_spec(robot)
    self.table_z = perc.nominal_table_z(robot)
    self.model = spec.compile()
    self.data = mujoco.MjData(self.model)
    self.dt = perc.CONTROL_DT_S
    self.ticks = 0
    self.meas = self._copy(dl.begin_stepping())
    kin = perc.parse_spec_kinematics(self.model, self.data, self.q_robot())
    self.site_id, self.wrist_xy_off = kin.site_id, kin.wrist_xy_off
    self.arm_jnt_ids, self.arm_qpos_ids = kin.arm_jnt_ids, kin.arm_qpos_ids
    self.arm_dof_ids, self.all_qpos_adrs = kin.arm_dof_ids, kin.all_qpos_adrs
    g0 = kin.grip_act_ids[0]
    self.n_arm = len(kin.arm_act_ids)
    self.jaw_open = float(self.model.actuator_ctrlrange[g0, 1])
    self.jaw_closed = float(self.model.actuator_ctrlrange[g0, 0])
    self.last_jaw = self.jaw_open
    self.q_home = self.q_robot()[: self.n_arm].copy()
    self.q = self.q_home.copy()
    self.q_scan = perc.compute_scan_pose(
        self.q_home, self.model.jnt_range[self.arm_jnt_ids[0]]
    )
    self.guard_zones: list[tuple[float, float, float]] = []
    self.lock_yaw = False
    self._q_hist: list[np.ndarray] = [self.q_home.copy()]
    self._frames: list[tuple[int, float | None, np.ndarray]] = []
    self._next_probe_s = 0.28
    self._open_ticks = 0
    self.last_probe_end_s = -1.0

  def _copy(self, meas: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    return _Snap(meas, self._mspec)

  @property
  def t(self) -> float:
    return self.ticks * self.dt

  def q_robot(self) -> np.ndarray:
    """Arm joints, then the nominal model's gripper joint coordinates."""
    arm = [float(self.meas[self.joint_keys[j]]) for j in self.arm_joints]
    g = float(
        np.clip(self.meas[self.joint_keys[self.grip_joint]], 0, self.grip_max)
    )
    g /= self.grip_max
    if self.robot == "yam":
      return np.array(arm + [0.0475 * g] * 2)
    lo, hi = self.model.jnt_range[self.model.joint("robot/gripper").id]
    return np.array(arm + [lo + g * (hi - lo)])

  def jaw_mm(self, meas: Mapping[str, Any] | None = None) -> float:
    g = float((meas or self.meas)[self.joint_keys[self.grip_joint]])
    return (95.0 if self.robot == "yam" else 80.0) * g / self.grip_max

  def act(self, q_arm: np.ndarray, jaw: float) -> Mapping[str, np.ndarray]:
    span = max(1e-9, self.jaw_open - self.jaw_closed)
    g = float(np.clip((jaw - self.jaw_closed) / span, 0.0, 1.0)) * self.grip_max
    cmd = {
        self.goal_keys[j]: np.float32(q_arm[i])
        for i, j in enumerate(self.arm_joints)
    }
    cmd[self.goal_keys[self.grip_joint]] = np.float32(g)
    self.last_jaw = jaw
    self._open_ticks = (
        self._open_ticks + 1 if abs(jaw - self.jaw_open) < 1e-6 else 0
    )
    self.meas = self._copy(self.dl.step(cmd))
    self.ticks += 1
    self._q_hist = [*self._q_hist[-29:], self.q_robot()[: self.n_arm]]
    return self.meas

  def check_camera_freeze(self) -> None:
    """Aborts if the camera timestamp stalls or frames repeat during motion."""
    cam = (
        perc.OVERHEAD_CAM if perc.OVERHEAD_CAM in self.meas else perc.WRIST_CAM
    )
    if cam not in self.meas:
      return
    if self._frames and self.t - self._frames[-1][0] * self.dt < _FREEZE_POLL_S:
      return
    ts_key = f"{cam}_timestamp"
    ts = float(self.meas[ts_key]) if ts_key in self.meas else None
    frame = self.meas[cam]
    if self._frames:
      tick0, ts0, frame0 = self._frames[-1]
      if ts is not None and ts0 is not None and ts <= ts0:
        raise AbortEpisodeError("camera_freeze")
      lag = self.ticks - tick0
      if lag <= len(self._q_hist) - 7:
        moved = np.max(np.abs(self._q_hist[-7] - self._q_hist[-1 - lag]))
        if moved > _FREEZE_MIN_MOTION_RAD and np.array_equal(frame, frame0):
          raise AbortEpisodeError("camera_freeze")
    self._frames = [(self.ticks, ts, frame)]

  def probe_open_jaw(self) -> None:
    """Pulses the open jaw partly closed in free space; aborts if it is dead."""
    w0, w_mid = self.jaw_mm(), self.jaw_mm()
    probe = self.jaw_open + (0.28 if self.n_arm == 5 else 0.18) * (
        self.jaw_closed - self.jaw_open
    )
    n_ticks = 15 if self.n_arm == 5 else 9
    for k in range(n_ticks):
      self.act(self.q, probe)
      if k == n_ticks - 2:
        w_mid = self.jaw_mm()
    if abs(0.5 * (w_mid + self.jaw_mm()) - w0) < (
        1.4 if self.n_arm == 5 else 2.8
    ):
      self.act(self.q, self.jaw_open)
      raise AbortEpisodeError("dead_gripper", already_stopped=True)
    for _ in range(4):
      self.act(self.q, self.jaw_open)
    self.last_probe_end_s = self.t
    self._next_probe_s = self.t + _GRIP_PROBE_INTERVAL_S

  def close_on_object(self) -> float:
    """Closes until the jaw stalls on the object; returns the hold command.

    The SO-101 jaw is a torque-limited position servo: commanding it fully
    closed saturates its stall torque (~200 N on a block), so it creeps from
    just above a 36 mm object, stops when blocked, and holds 1.2% past contact.
    """
    span = self.jaw_open - self.jaw_closed
    if self.n_arm != 5:
      self.move_q(self.q, self.jaw_closed, hold=18, rule="close_jaws")
      return self.jaw_closed
    jaw = self.jaw_closed + (36.0 / 80.0) * span
    self.move_q(self.q, jaw, hold=6, rule="close_jaws")
    prev, blocked = self.jaw_mm(), 0
    for _ in range(150):
      jaw = max(self.jaw_closed, jaw - 0.002 * span)
      self.act(self.q, jaw)
      w = self.jaw_mm()
      lag = w - 80.0 * (jaw - self.jaw_closed) / span
      # Blocked by the object: stalled while lagging the command by 2 mm.
      blocked = blocked + 1 if lag > 2.0 and prev - w < 0.05 else 0
      prev = w
      if blocked >= 2:
        break
    jaw = self.jaw_closed + (self.jaw_mm() / 80.0 - 0.012) * span
    for _ in range(8):
      self.act(self.q, jaw)
    return jaw

  def _monitors(self, jaw: float, rule: str) -> None:
    if self.variant == "v0":
      return
    self.check_camera_freeze()
    near = perc.tcp_near_zones(
        self.model,
        self.data,
        self.site_id,
        self.arm_qpos_ids,
        self.q,
        self.guard_zones,
    )
    if (
        rule not in _NO_OPEN_PULSE_RULES
        and self._open_ticks >= (26 if self.n_arm == 5 else 14)
        and abs(jaw - self.jaw_open) < 1e-6
        and self.t >= self._next_probe_s
        and not near
    ):
      self.probe_open_jaw()

  def move_q(
      self,
      q_goal: np.ndarray,
      jaw: float,
      *,
      step_rad: float = 0.028,
      hold: int = 2,
      rule: str = "move",
  ) -> Mapping[str, np.ndarray]:
    for _ in range(200):
      diff = q_goal - self.q
      dist = float(np.max(np.abs(diff)))
      if dist <= 1e-4:
        break
      self.q = self.q + min(1.0, step_rad / dist) * diff
      self.act(self.q, jaw)
      self._monitors(jaw, rule)
    for _ in range(max(1, hold)):
      self.act(self.q, jaw)
      self._monitors(jaw, rule)
    return self.meas

  def move_xyz(
      self, xyz: tuple[float, float, float], jaw: float, **kwargs: Any
  ) -> Mapping[str, np.ndarray]:
    q_goal = perc.solve_downward_ik(
        self.model,
        self.data,
        self.site_id,
        self.arm_qpos_ids,
        self.arm_dof_ids,
        self.arm_jnt_ids,
        xyz,
        self.q,
        lock_world_yaw=self.lock_yaw,
    )
    return self.move_q(q_goal, jaw, **kwargs)

  def _cloud(self, cam: str) -> tuple[np.ndarray, np.ndarray]:
    depth_key = f"{cam}_depth"
    depth = self.meas[depth_key] if depth_key in self.meas else None
    return perc.backproject_rgbd(
        self.model,
        self.data,
        self.all_qpos_adrs,
        self.q_robot(),
        cam,
        self.meas[cam],
        depth,
        nominal_table_z_m=self.table_z,
    )

  def perceive_overhead(self, plan: perc.TaskPlan) -> Any:
    pts, cols = self._cloud(perc.OVERHEAD_CAM)
    return perc.localize_overhead_scene(
        pts, cols, plan, nominal_table_z_m=self.table_z
    )

  def perceive_wrist_overview(self, plan: perc.TaskPlan) -> Any:
    """Sweeps two wrist-camera viewpoints when `overhead_cam` is absent."""
    y_ref = 0.02 if self.robot == "yam" else -0.04
    clouds, colors = [], []
    for x_ref in (-0.08, 0.08):
      xy_w = np.array([x_ref, y_ref]) - self.wrist_xy_off
      self.move_xyz(
          (float(xy_w[0]), float(xy_w[1]), self.table_z + 0.22),
          self.jaw_open,
          step_rad=0.032,
          hold=6,
          rule="wrist_scan",
      )
      pts, cols = self._cloud(perc.WRIST_CAM)
      if len(pts):
        clouds.append(pts)
        colors.append(cols)
    if not clouds:
      return (
          None,
          None,
          plan.prior_obj_h,
          plan.prior_support_dz,
          (
              0.0,
              0.0,
              self.table_z,
          ),
      )
    return perc.localize_overhead_scene(
        np.vstack(clouds),
        np.vstack(colors),
        plan,
        nominal_table_z_m=self.table_z,
    )

  def refine_wrist(
      self,
      xy: np.ndarray,
      plane: tuple[float, float, float],
      kind: str,
      color: str = "",
  ) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Hovers the wrist camera 16.5 cm over `xy` to refine xy and the table."""
    z_t = perc.plane_z(plane, float(xy[0]), float(xy[1]))
    xy_w = xy - self.wrist_xy_off
    self.move_xyz(
        (float(xy_w[0]), float(xy_w[1]), z_t + 0.165),
        self.jaw_open,
        step_rad=0.030,
        hold=8,
        rule="wrist_ref",
    )
    pts, cols = self._cloud(perc.WRIST_CAM)
    return perc.refine_wrist_cluster(pts, cols, xy, plane, kind, color)


def run(dl: Any, *, robot: str, prompt: str, variant: str = "v1") -> int:
  """Runs one pick-and-place; returns the tick count or raises on a fault."""
  v0 = variant == "v0"
  plan = perc.parse_task_prompt(prompt)
  arm = _Arm(dl, robot, variant)
  has_overhead = perc.OVERHEAD_CAM in arm.meas
  try:
    if has_overhead:
      arm.move_q(arm.q_scan, arm.jaw_open, step_rad=0.032, hold=8, rule="scan")
      t_xy, d_xy, obj_h, sup_dz, plane = arm.perceive_overhead(plan)
      if t_xy is None or d_xy is None:
        q_alt = arm.q_scan.copy()
        q_alt[0] = -q_alt[0] if abs(q_alt[0]) > 0.2 else 1.15
        arm.move_q(q_alt, arm.jaw_open, step_rad=0.032, hold=8, rule="rescan")
        t2, d2, obj_h, sup_dz, plane = arm.perceive_overhead(plan)
        t_xy = t_xy if t_xy is not None else t2
        d_xy = d_xy if d_xy is not None else d2
    elif perc.WRIST_CAM in arm.meas:
      t_xy, d_xy, obj_h, sup_dz, plane = arm.perceive_wrist_overview(plan)
    else:
      t_xy, d_xy = None, None
      obj_h, sup_dz = plan.prior_obj_h, plan.prior_support_dz
      plane = (0.0, 0.0, arm.table_z)
    if t_xy is None or d_xy is None:
      arm.move_q(arm.q_home, arm.jaw_open, step_rad=0.03, hold=3, rule="home")
      return arm.ticks
    use_wrist = not v0 and f"{perc.WRIST_CAM}_depth" in arm.meas
    if v0:
      d_xy = np.asarray(d_xy) + 0.085
    elif use_wrist:
      d_xy, plane = arm.refine_wrist(
          d_xy, plane, plan.dest_kind, plan.dest_color
      )
    arm.guard_zones = perc.build_guard_zones(t_xy, d_xy, obj_h, sup_dz, plane)
    arm.lock_yaw = arm.n_arm == 5 and plan.dest_kind != "bowl"
    so_peg = arm.n_arm == 5 and plan.family == "peg_in_plate"
    grasp_z = 0.036 if so_peg else max(0.012, obj_h - plan.grasp_below_top_m)
    grasped, z_tbl_t = False, 0.0
    for attempt in range(1 if v0 else 2):
      if attempt:
        if has_overhead:
          arm.move_q(
              arm.q_scan, arm.jaw_open, step_rad=0.032, hold=8, rule="scan"
          )
          t_re, d_re, obj_h, sup_dz, plane = arm.perceive_overhead(plan)
        else:
          t_re, d_re, obj_h, sup_dz, plane = arm.perceive_wrist_overview(plan)
        if t_re is None or d_re is None:
          break
        t_xy, d_xy = t_re, d_re
        grasp_z = (
            0.036 if so_peg else max(0.012, obj_h - plan.grasp_below_top_m)
        )
      if use_wrist:
        key = "peg" if plan.family == "peg_in_plate" else plan.target_color
        t_xy, plane = arm.refine_wrist(t_xy, plane, key, plan.target_color)
      arm.guard_zones = perc.build_guard_zones(t_xy, d_xy, obj_h, sup_dz, plane)
      tx, ty = float(t_xy[0]), float(t_xy[1])
      z_tbl_t = perc.plane_z(plane, tx, ty)
      p_pre = (tx, ty, z_tbl_t + (0.068 if arm.lock_yaw else obj_h + 0.080))
      arm.move_xyz(
          p_pre, arm.jaw_open, step_rad=0.028, hold=2, rule="pre_grasp"
      )
      if not v0:
        arm.probe_open_jaw()
      st_pre = arm.move_xyz(
          (tx, ty, z_tbl_t + grasp_z),
          arm.jaw_open,
          step_rad=0.018,
          hold=3,
          rule="grasp",
      )
      grip = arm.close_on_object()
      if arm.jaw_mm(st_pre) - arm.jaw_mm() < 6.5:
        arm.act(arm.q, arm.jaw_open)
        raise AbortEpisodeError("dead_gripper", already_stopped=True)
      st_lift = arm.move_xyz(p_pre, grip, step_rad=0.024, hold=2, rule="lift")
      if arm.jaw_mm(st_lift) > 8.2:
        grasped = True
        break
      arm.move_q(arm.q, arm.jaw_open, hold=6, rule="reopen_after_miss")
    if grasped:
      dx, dy = float(d_xy[0]), float(d_xy[1])
      z_tbl_d = perc.plane_z(plane, dx, dy)
      z_top = max(z_tbl_d + sup_dz, z_tbl_t)
      p_car = (dx, dy, z_top + (0.066 if arm.lock_yaw else obj_h + 0.065))
      arm.move_xyz(p_car, grip, step_rad=0.026, hold=2, rule="carry")
      z_pl = z_tbl_d + sup_dz + grasp_z + (0.042 if v0 else 0.008)
      st_pl = arm.move_xyz(
          (dx, dy, z_pl), grip, step_rad=0.017, hold=4, rule="descend_place"
      )
      j_rel = arm.jaw_closed + 0.82 * (arm.jaw_open - arm.jaw_closed)
      st_rel = arm.move_q(
          arm.q, j_rel, hold=12 if arm.n_arm == 5 else 6, rule="gentle_release"
      )
      if arm.jaw_mm(st_rel) - arm.jaw_mm(st_pl) < 4.5:
        arm.act(arm.q, arm.jaw_open)
        raise AbortEpisodeError("dead_gripper", already_stopped=True)
      arm.move_q(arm.q, arm.jaw_open, hold=3, rule="gentle_release")
      arm.move_xyz(p_car, arm.jaw_open, step_rad=0.024, hold=2, rule="retract")
    if not v0:  # Swing the base so a frozen overhead camera shows up.
      q_chk = arm.q_home + np.eye(arm.n_arm)[0] * 0.2
      arm.move_q(q_chk, arm.jaw_open, step_rad=0.03, hold=12, rule="home")
    arm.move_q(arm.q_home, arm.jaw_open, step_rad=0.03, hold=12, rule="home")
    if not v0 and arm.t >= arm.last_probe_end_s + 0.12:
      arm.probe_open_jaw()
    return arm.ticks
  except AbortEpisodeError as e:
    if not e.already_stopped:
      arm.act(arm.q, arm.last_jaw)
    raise


def hold(dl: Any, ticks: int) -> None:
  """A blind baseline: holds the measured pose for `ticks` ticks."""
  meas = dl.begin_stepping()
  cmd = {
      k: meas[k.replace("goal_position/", "joint_position/")]
      for k in dl.commands_spec()
  }
  for _ in range(ticks):
    dl.step(cmd)


_ROBOT = flags.DEFINE_enum("robot", "yam", ["yam", "so101"], "Robot.")
_PROMPT = flags.DEFINE_string(
    "prompt", "", "Task prompt (defaults to GAUNTLET_PROMPT)."
)
_TASK_ID = flags.DEFINE_enum(
    "task_id", "cube_in_bowl", list(_FALLBACK_PROMPTS), "Fallback task prompt."
)
_VARIANT = flags.DEFINE_enum("variant", "v1", ["v0", "v1"], "Variant.")
_HOLD_TICKS = flags.DEFINE_integer(
    "hold_ticks", 0, "If > 0, run the blind hold baseline instead."
)


def main(argv: Sequence[str]) -> None:
  del argv
  prompt = (
      _PROMPT.value
      or os.environ.get("GAUNTLET_PROMPT", "")
      or _FALLBACK_PROMPTS[_TASK_ID.value]
  )
  dl = proxy.RemoteDeviceLayer(os.environ["GAUNTLET_URL"])
  doc = {"status": "done", "fault": ""}
  try:
    if _HOLD_TICKS.value > 0:
      hold(dl, _HOLD_TICKS.value)
    else:
      run(dl, robot=_ROBOT.value, prompt=prompt, variant=_VARIANT.value)
  except AbortEpisodeError as e:
    doc = {"status": "abort", "fault": e.fault}
  except proxy.EpisodeOverError:
    pass
  dl.end_stepping()
  print(json.dumps(doc), flush=True)


if __name__ == "__main__":
  app.run(main)
