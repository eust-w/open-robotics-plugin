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
"""Perturbation tiers, CRN seed streams, and the sim-only privileged devices.

Everything that touches the simulator lives here: `privileged_devices` adds
components to the task's `flat_scene` config, and `apply_static` writes the
sampled plant perturbation through `raw.sim.model` once per episode.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import hmac
import math
import os
import re
from typing import Any
import zlib

from dm_env import specs
import mujoco
import numpy as np
from reaf.simulation import torch as reaf_sim
from robosim_gauntlet import task as task_lib
import torch

TIERS = ("T0", "T1", "T2", "T3", "T4")
FAULTS = ("camera_freeze", "dead_gripper")
TRIGGERS = ("arm_motion", "gripper_close")
DISTURBANCES = ("slip", "push", "knockout")
PRIVILEGED_PREFIX = "gauntlet/"
MONITOR = "gauntlet_monitor"
DISTURBER = "gauntlet_disturbance"


@dataclasses.dataclass(frozen=True)
class PSpec:
  """One episode's sampled perturbation; T0 is all defaults."""

  tier: str = "T0"
  idx: int = 0
  friction_scale: float = 1.0
  mass_scale: float = 1.0
  damping_scale: float = 1.0
  gain_scale: float = 1.0
  gravcomp_delta: float = 0.0
  cam_offset_m: tuple[float, ...] = (0.0, 0.0, 0.0)
  cam_tilt_deg: float = 0.0
  table_dz_m: float = 0.0  # Applied as the inverse robot-mount offset.
  table_tilt_deg: float = 0.0
  target_xy_m: tuple[float, ...] = (0.0, 0.0)
  target_yaw_deg: float = 0.0
  dest_xy_m: tuple[float, ...] = (0.0, 0.0)
  proprio_noise: float = 0.0
  proprio_bias: float = 0.0
  hue_deg: float = 0.0
  sat_scale: float = 1.0
  rgb_gain: float = 1.0
  rgb_noise: float = 0.0
  depth_noise_m: float = 0.0
  depth_dropout: float = 0.0
  frame_delay: int = 0
  disturbance: str = ""
  push_dir_rad: float = 0.0
  fault: str = ""
  fault_trigger: str = ""
  fault_delay: int = 0
  fault_restamp: bool = False


_MODERATE = dict(
    proprio_noise=(0.001, 0.001), proprio_bias=(0.002, 0.002),
    rgb_noise=(2.5, 2.5), depth_noise_m=(0.001, 0.001),
    depth_dropout=(0.005, 0.005),
)
# Uniform `(lo, hi)` per field; tuple fields draw one value per component.
TIER_RANGES: Mapping[str, Mapping[str, tuple[float, float]]] = {
    "T0": {},
    "T1": dict(
        _MODERATE, friction_scale=(0.85, 1.15), mass_scale=(0.85, 1.15),
        damping_scale=(0.9, 1.1),
        gain_scale=(0.92, 1.08), table_dz_m=(-0.008, 0.008),
        target_xy_m=(-0.03, 0.03), target_yaw_deg=(-25.0, 25.0),
        dest_xy_m=(-0.02, 0.02), hue_deg=(-4.0, 4.0), sat_scale=(0.9, 1.1),
        rgb_gain=(0.92, 1.08),
    ),
    "T2": dict(
        proprio_noise=(0.002, 0.002), proprio_bias=(0.004, 0.004),
        rgb_noise=(6.0, 6.0), depth_noise_m=(0.002, 0.002),
        depth_dropout=(0.02, 0.02), frame_delay=(0, 1),
        friction_scale=(0.6, 1.4), mass_scale=(0.65, 1.45),
        damping_scale=(0.75, 1.3),
        gain_scale=(0.8, 1.2), table_dz_m=(-0.02, 0.02),
        table_tilt_deg=(-1.5, 1.5), target_xy_m=(-0.06, 0.06),
        target_yaw_deg=(-60.0, 60.0), dest_xy_m=(-0.035, 0.035),
        hue_deg=(-14.0, 14.0), sat_scale=(0.7, 1.3), rgb_gain=(0.8, 1.2),
    ),
    "T3": dict(
        _MODERATE, friction_scale=(0.75, 1.15), table_dz_m=(-0.02, 0.02),
        target_xy_m=(-0.025, 0.025),
    ),
    "T4": dict(
        _MODERATE, mass_scale=(0.85, 1.2), target_xy_m=(-0.02, 0.02),
        gain_scale=(0.85, 1.15), gravcomp_delta=(-0.1, 0.1),
        cam_offset_m=(-0.003, 0.003), cam_tilt_deg=(-0.5, 0.5),
    ),
}
_INT_FIELDS = ("frame_delay",)


@dataclasses.dataclass(frozen=True)
class Streams:
  """Independent child generators so each consumer's draws are isolated."""

  spec: np.random.Generator
  sensor: np.random.Generator


def streams(
    base_seed: int, task_id: str, tier: str, idx: int, *,
    secret: str | None = None,
) -> Streams:
  """Derives the CRN streams for `(base_seed, task_id, tier, idx)`."""
  key = os.environ.get("GAUNTLET_CRN_SECRET", "") if secret is None else secret
  if key:
    msg = f"{int(base_seed)}|{task_id}|{tier}|{idx}".encode("utf-8")
    digest = hmac.new(key.encode("utf-8"), msg, hashlib.sha256).digest()
    entropy: Any = list(np.frombuffer(digest[:16], dtype=np.uint32))
  else:
    entropy = [int(base_seed), zlib.crc32(task_id.encode()), TIERS.index(tier), idx]
  ss = np.random.SeedSequence(entropy)
  spec, sensor = (np.random.default_rng(s) for s in ss.spawn(2))
  return Streams(spec=spec, sensor=sensor)


def sample(tier: str, idx: int, rng: np.random.Generator) -> PSpec:
  """Samples a `PSpec` for `tier`; draws happen in a fixed field order."""
  defaults = PSpec()
  values: dict[str, Any] = {"tier": tier, "idx": idx}
  for field, (lo, hi) in TIER_RANGES[tier].items():
    default = getattr(defaults, field)
    if isinstance(default, tuple):
      values[field] = tuple(float(v) for v in rng.uniform(lo, hi, len(default)))
    elif field in _INT_FIELDS:
      values[field] = int(rng.integers(int(lo), int(hi) + 1))
    else:
      values[field] = float(rng.uniform(lo, hi))
  if tier == "T3":
    values["disturbance"] = DISTURBANCES[idx % 3]
    values["push_dir_rad"] = float(rng.uniform(0.0, 2.0 * math.pi))
  if tier == "T4" and rng.random() < 0.5:
    values["fault"] = FAULTS[int(rng.integers(2))]
    values["fault_trigger"] = TRIGGERS[int(rng.integers(2))]
    hi = 110 if values["fault_trigger"] == "arm_motion" else 67
    values["fault_delay"] = int(rng.integers(3, hi))
    if values["fault"] == "camera_freeze":
      values["fault_restamp"] = bool(rng.integers(2))
  return PSpec(**values)


def privileged_devices(
    task: task_lib.Task, pspec: PSpec
) -> dict[str, reaf_sim.flat_scene.Component]:
  """Returns the sim-only components the gauntlet merges into the scene."""
  devices = {
      MONITOR: reaf_sim.flat_scene.Component(
          MonitorDevice,
          robot_pattern=task.robot_pattern,
          pad_pattern=task.pad_pattern,
          target_body=task.target_body,
      ).mark_privileged()
  }
  if pspec.disturbance:
    devices[DISTURBER] = reaf_sim.flat_scene.Component(
        DisturbanceDevice,
        mode=pspec.disturbance,
        target_body=task.target_body,
        pad_pattern=task.pad_pattern,
        direction_rad=pspec.push_dir_rad,
    ).mark_privileged()
  return devices


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  out = np.zeros(4)
  mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
  return out


def _axis_quat(axis: Sequence[float], deg: float) -> np.ndarray:
  out = np.zeros(4)
  mujoco.mju_axisAngle2Quat(out, np.asarray(axis, float), math.radians(deg))
  return out


def _rot_vec(quat: np.ndarray, vec: Sequence[float]) -> np.ndarray:
  out = np.zeros(3)
  mujoco.mju_rotVecQuat(out, np.asarray(vec, float), np.asarray(quat, float))
  return out


def _write(model: Any, field: str, idx: Any, value: Any) -> None:
  """Writes `model.field[0, idx]` through the tracked `TorchWrapper`."""
  getattr(model, field)[0, idx] = torch.as_tensor(
      np.asarray(value), dtype=getattr(model, field)[0, idx].dtype
  )


def _read(model: Any, field: str, idx: Any) -> np.ndarray:
  return getattr(model, field)[0, idx].cpu().numpy().astype(np.float64)


def apply_static(raw: Any, task: task_lib.Task, pspec: PSpec) -> None:
  """Applies the static plant perturbation to a built sim `DeviceLayer`."""
  sim = raw.sim
  if sim.impl.value != "cpp":
    raise ValueError(f"The gauntlet needs Impl.CPP, got {sim.impl}.")
  mjm, m = sim.mj_model, sim.model
  robot_re = re.compile(task.robot_pattern)
  robot_bodies = [b for b in range(1, mjm.nbody) if robot_re.search(mjm.body(b).name)]
  robot_set = set(robot_bodies)
  tgt = mjm.body(task.target_body).id
  _write(m, "body_mass", tgt, _read(m, "body_mass", tgt) * pspec.mass_scale)
  _write(m, "body_inertia", tgt, _read(m, "body_inertia", tgt) * pspec.mass_scale)
  geoms = [g for g in range(mjm.ngeom) if mjm.geom_bodyid[g] == tgt]
  if geoms:
    fr = _read(m, "geom_friction", geoms)
    fr[:, 0] *= pspec.friction_scale
    _write(m, "geom_friction", geoms, fr)
  dofs = [d for d in range(mjm.nv) if mjm.dof_bodyid[d] in robot_set]
  if dofs:
    _write(m, "dof_damping", dofs, _read(m, "dof_damping", dofs) * pspec.damping_scale)
  acts = [
      a for a in range(mjm.nu)
      if mujoco.mjtTrn(mjm.actuator_trntype[a]) == mujoco.mjtTrn.mjTRN_JOINT
      and mjm.jnt_bodyid[mjm.actuator_trnid[a, 0]] in robot_set
  ]
  if acts:
    gain, bias = _read(m, "actuator_gainprm", acts), _read(m, "actuator_biasprm", acts)
    gain[:, 0] *= pspec.gain_scale
    bias[:, 1:3] *= pspec.gain_scale
    _write(m, "actuator_gainprm", acts, gain)
    _write(m, "actuator_biasprm", acts, bias)
  if robot_bodies:
    gc = np.clip(_read(m, "body_gravcomp", robot_bodies) + pspec.gravcomp_delta, 0, 1)
    _write(m, "body_gravcomp", robot_bodies, gc)
  for cam in range(mjm.ncam):
    _write(m, "cam_pos", cam, _read(m, "cam_pos", cam) + np.asarray(pspec.cam_offset_m))
    tilt = _axis_quat((1.0, 0.0, 0.0), pspec.cam_tilt_deg)
    _write(m, "cam_quat", cam, _quat_mul(_read(m, "cam_quat", cam), tilt))
  # Shift/tilt every root robot mount rigidly as the inverse table offset.
  roots = [b for b in robot_bodies if int(mjm.body_parentid[b]) not in robot_set]
  tilt = _axis_quat((0.0, 1.0, 0.0), -pspec.table_tilt_deg)
  if len(roots) == 1:
    base = roots[0]
    _write(m, "body_pos", base, _read(m, "body_pos", base) - [0, 0, pspec.table_dz_m])
    _write(m, "body_quat", base, _quat_mul(tilt, _read(m, "body_quat", base)))
  elif len(roots) > 1:
    mujoco.mj_kinematics(mjm, sim.mj_data)
    d, p_inv = sim.mj_data, np.zeros(4)
    mounts = [
        (b, d.xpos[p] + _rot_vec(d.xquat[p], _read(m, "body_pos", b)),
         _quat_mul(d.xquat[p], _read(m, "body_quat", b)))
        for b in roots for p in (int(mjm.body_parentid[b]),)
    ]
    pivot = np.mean([wp for _, wp, _ in mounts], axis=0)
    for b, wp, wq in mounts:
      new_wp = pivot + _rot_vec(tilt, wp - pivot) - [0.0, 0.0, pspec.table_dz_m]
      p = int(mjm.body_parentid[b])
      mujoco.mju_negQuat(p_inv, np.asarray(d.xquat[p], float))
      _write(m, "body_pos", b, _rot_vec(p_inv, new_wp - d.xpos[p]))
      _write(m, "body_quat", b, _quat_mul(p_inv, _quat_mul(tilt, wq)))
  sim.set_const()
  moved: set[int] = set()
  _offset_body(sim, task.target_body, pspec.target_xy_m, pspec.target_yaw_deg, robot_set, moved)
  _offset_body(sim, task.support_body, pspec.dest_xy_m, 0.0, robot_set, moved)
  raw.forward()


def _offset_body(
    sim: Any, body: str, xy: Sequence[float], yaw_deg: float,
    robot_set: set[int], moved: set[int],
) -> None:
  """Shifts a free body's qpos, or a static/fixture root's `body_pos`."""
  if not any(xy) and not yaw_deg:
    return
  mjm = sim.mj_model
  bid = mjm.body(body).id
  jnt = int(mjm.body_jntadr[bid])
  yaw = _axis_quat((0.0, 0.0, 1.0), yaw_deg)
  if jnt >= 0 and mujoco.mjtJoint(mjm.jnt_type[jnt]) == mujoco.mjtJoint.mjJNT_FREE:
    adr = mjm.jnt_qposadr[jnt]
    qpos = sim.mj_data.qpos
    qpos[adr : adr + 2] += np.asarray(xy)
    qpos[adr + 3 : adr + 7] = _quat_mul(yaw, qpos[adr + 3 : adr + 7])
    return
  while int(mjm.body_parentid[bid]) > 0:
    bid = int(mjm.body_parentid[bid])
  if bid == 0 or bid in robot_set or bid in moved:
    return
  moved.add(bid)
  pos = _read(sim.model, "body_pos", bid)
  pos[:2] += np.asarray(xy)
  _write(sim.model, "body_pos", bid, pos)
  _write(sim.model, "body_quat", bid, _quat_mul(yaw, _read(sim.model, "body_quat", bid)))
  sim.set_const()


class _SimDevice(reaf_sim.Device):
  """A measurement device that ignores the injected framework kwargs."""

  def __init__(self, name: str):
    self._name = name

  @property
  def name(self) -> str:
    return self._name

  def reset(self, simulator: Any, env_mask: Any = None) -> None:
    del simulator, env_mask


class MonitorDevice(_SimDevice):
  """Privileged per-tick body poses and vectorial 3D contact force reductions.

  `gauntlet/poses` is `(nbody, 7)` over `body_names` (named non-robot bodies).
  `gauntlet/forces` is `[grasp_boxcar, env_boxcar, peak, self, robot_target,
  max_penetration]`, reduced over the substeps of the last control step using
  3D world-frame normal contact vectors (including non-pad target impacts and
  weight-subtracted indirect crushing) and a zero-padded 16 ms causal boxcar.
  """

  def __init__(
      self, *, robot_pattern: str, pad_pattern: str, target_body: str,
      name: str = MONITOR,
  ):
    super().__init__(name)
    self._patterns = (re.compile(robot_pattern), re.compile(pad_pattern))
    self._target_name = target_body
    self.body_names: tuple[str, ...] = ()
    self._free_bodies: frozenset[int] = frozenset()
    self._g_norm = 9.81
    self._step = np.zeros(6)
    self._latched = np.zeros(6)
    self._window: list[np.ndarray] = []

  def initialize(self, simulator: Any) -> None:
    mjm = simulator.mj_model
    robot_re, pad_re = self._patterns
    self._robot = np.array(
        [bool(robot_re.search(mjm.body(b).name)) for b in range(mjm.nbody)]
    )
    pad_body = [bool(pad_re.search(mjm.body(b).name)) for b in range(mjm.nbody)]
    self._pad = np.array(pad_body)[mjm.geom_bodyid]
    self._target = mjm.body(self._target_name).id
    self._bids = [
        b for b in range(1, mjm.nbody) if mjm.body(b).name and not self._robot[b]
    ]
    self._free_bodies = frozenset(
        b for b in range(1, mjm.nbody)
        if not self._robot[b] and mjm.body_jntadr[b] >= 0
        and mjm.jnt_type[mjm.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
    )
    self._g_norm = float(np.linalg.norm(mjm.opt.gravity))
    self.body_names = tuple(mjm.body(b).name for b in self._bids)
    self._width = max(1, round(0.016 / float(mjm.opt.timestep)))
    self._window = [np.zeros(2) for _ in range(self._width)]

  def substep_measurements_spec(self) -> Mapping[str, specs.Array]:
    return {
        "gauntlet/poses": specs.Array((len(self._bids), 7), np.float32),
        "gauntlet/forces": specs.Array((6,), np.float32),
    }

  def _contact_forces(self, simulator: Any) -> np.ndarray:
    """Returns `[grasp, env, peak, self, robot_target, penetration]`."""
    mjm, d = simulator.mj_model, simulator.mj_data
    f6 = np.zeros(6)
    pad_vecs: dict[int, np.ndarray] = {}
    env_body: dict[int, np.ndarray] = {}
    env_pair: dict[tuple[int, int], np.ndarray] = {}
    rob_on_dyn: dict[int, np.ndarray] = {}
    nonrob_on_dyn: dict[int, np.ndarray] = {}
    self_f = robot_target = pen = any_rob = 0.0
    for i in range(d.ncon):
      c = d.contact[i]
      b1, b2 = int(mjm.geom_bodyid[c.geom1]), int(mjm.geom_bodyid[c.geom2])
      if self._target in (b1, b2):
        pen = max(pen, float(-c.dist))
      if mjm.body_weldid[b1] == mjm.body_weldid[b2]:
        continue
      r1, r2 = bool(self._robot[b1]), bool(self._robot[b2])
      if not (r1 or r2) and b1 not in self._free_bodies and b2 not in self._free_bodies:
        continue
      mujoco.mj_contactForce(mjm, d, i, f6)
      mag = abs(float(f6[0]))
      if mag <= 0.0:
        continue
      # `c.frame[0:3]` points from geom1 (b1) to geom2 (b2); `f_w` is b1 on b2.
      f_w = float(f6[0]) * np.asarray(c.frame[:3], float)
      if r1 and r2:
        if not (self._pad[c.geom1] and self._pad[c.geom2]):
          self_f += mag
          any_rob = max(any_rob, mag)
      elif r1 != r2:
        rob_b, rob_g, other_b, sign = (
            (b1, int(c.geom1), b2, 1.0) if r1 else (b2, int(c.geom2), b1, -1.0)
        )
        f_on_other = sign * f_w
        any_rob = max(any_rob, mag)
        if other_b == self._target:
          robot_target = 1.0
        if other_b in self._free_bodies:
          rob_on_dyn[other_b] = rob_on_dyn.get(other_b, 0.0) + f_on_other
        if other_b == self._target and self._pad[rob_g]:
          pad_vecs[rob_b] = pad_vecs.get(rob_b, 0.0) + f_on_other
        else:
          env_body[other_b] = env_body.get(other_b, 0.0) + f_on_other
          pair = (rob_b, other_b)
          env_pair[pair] = env_pair.get(pair, 0.0) + f_on_other
      else:
        if b1 in self._free_bodies:
          nonrob_on_dyn[b1] = nonrob_on_dyn.get(b1, 0.0) - f_w
        if b2 in self._free_bodies:
          nonrob_on_dyn[b2] = nonrob_on_dyn.get(b2, 0.0) + f_w
    grasp_f = max((float(np.linalg.norm(v)) for v in pad_vecs.values()), default=0.0)
    env_f = max(
        (float(np.linalg.norm(v)) for v in (*env_body.values(), *env_pair.values())),
        default=0.0,
    )
    for dyn_b, rob_v in rob_on_dyn.items():
      if dyn_b in nonrob_on_dyn:
        weight = float(mjm.body_subtreemass[dyn_b]) * self._g_norm
        excess = max(0.0, float(np.linalg.norm(nonrob_on_dyn[dyn_b])) - weight)
        env_f = max(env_f, min(float(np.linalg.norm(rob_v)), excess))
    peak = max(grasp_f, env_f, self_f, any_rob)
    return np.array([grasp_f, env_f, peak, self_f, robot_target, pen])

  def on_substep(self, simulator: Any) -> None:
    f = self._contact_forces(simulator)
    self._window = [*self._window, f[:2]][-self._width :]
    box = np.mean(self._window, axis=0)
    self._step = np.maximum(
        self._step, [box[0], box[1], f[2], f[3], f[4], f[5]]
    )

  def on_step(self, simulator: Any) -> None:
    del simulator
    self._latched, self._step = self._step, np.zeros(6)

  def on_begin_stepping(self, simulator: Any, env_mask: Any = None) -> None:
    del env_mask
    f = self._contact_forces(simulator)
    self._window = [np.zeros(2) for _ in range(self._width - 1)] + [f[:2]]
    box = np.mean(self._window, axis=0)
    self._latched = np.array([box[0], box[1], f[2], f[3], f[4], f[5]])
    self._step = np.zeros(6)

  def get_measurements(self, simulator: Any) -> Mapping[str, torch.Tensor]:
    d = simulator.mj_data
    poses = np.concatenate([d.xpos[self._bids], d.xquat[self._bids]], axis=1)
    return {
        "gauntlet/poses": torch.as_tensor(poses, dtype=torch.float32)[None],
        "gauntlet/forces": torch.as_tensor(self._latched, dtype=torch.float32)[None],
    }


class DisturbanceDevice(_SimDevice):
  """T3 online disturbances on the target: slip, mid-approach push, knockout.

  slip: pulls the target down at 3 g for 0.15 s once it is first lifted 2 cm
  while touching the pads. push: kicks it 0.6 m/s sideways when a pad first
  comes within 8 cm of it, before any lift. knockout: kicks it 0.8 m/s sideways
  0.5 s after the pads release it after a lift.
  """

  def __init__(
      self, *, mode: str, target_body: str, pad_pattern: str,
      direction_rad: float, name: str = DISTURBER,
  ):
    super().__init__(name)
    self._mode, self._target_name = mode, target_body
    self._pad_re = re.compile(pad_pattern)
    self._dir = np.array([math.cos(direction_rad), math.sin(direction_rad)])
    self.fired_time: float | None = None
    self._lift_time: float | None = None
    self._release_time: float | None = None
    self._z0: float | None = None

  def initialize(self, simulator: Any) -> None:
    mjm = simulator.mj_model
    self._target = mjm.body(self._target_name).id
    self._qvel = int(mjm.body_dofadr[self._target])
    self._ndof = int(mjm.body_dofnum[self._target])
    self._pads = {
        g for g in range(mjm.ngeom) if self._pad_re.search(mjm.body(mjm.geom_bodyid[g]).name)
    }

  def on_substep(self, simulator: Any) -> None:
    mjm, d = simulator.mj_model, simulator.mj_data
    t, z = float(d.time), float(d.xpos[self._target, 2])
    if self._z0 is None:
      self._z0 = z
    touching = False
    for i in range(d.ncon):
      c = d.contact[i]
      bodies = (mjm.geom_bodyid[c.geom1], mjm.geom_bodyid[c.geom2])
      if self._target in bodies and {c.geom1, c.geom2} & self._pads:
        touching = True
    near = False
    if self._pads:
      gap = d.geom_xpos[sorted(self._pads)] - d.xpos[self._target]
      near = float(np.min(np.linalg.norm(gap, axis=1))) < 0.08
    lifted = z > self._z0 + 0.02
    if lifted and touching and self._lift_time is None:
      self._lift_time = t
    if self._lift_time is not None and not touching and self._release_time is None:
      self._release_time = t
    d.xfrc_applied[self._target] = 0.0
    if self._mode == "slip" and self._lift_time is not None:
      if t - self._lift_time < 0.15:
        d.xfrc_applied[self._target, 2] = -3.0 * 9.81 * mjm.body_mass[self._target]
        self.fired_time = self._lift_time
    elif self.fired_time is None and (
        (self._mode == "push" and near and not touching
         and self._lift_time is None)
        or (
            self._mode == "knockout" and self._release_time is not None
            and t - self._release_time > 0.5
        )
    ):
      speed = 0.6 if self._mode == "push" else 0.8
      if self._ndof >= 2:
        d.qvel[self._qvel : self._qvel + 2] = speed * self._dir
      elif self._ndof == 1:
        d.qvel[self._qvel] = float(speed * self._dir[0])
      self.fired_time = t
