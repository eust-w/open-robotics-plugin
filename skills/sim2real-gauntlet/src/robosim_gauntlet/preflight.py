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
"""Preflight: scene audits that compile/warnings miss, plus goal controls.

Negative controls must fail `task.success`: nc1 the reset scene, nc2 the target
moved away on the table, nc3 the target hovering over the goal, nc4 the target
just outside the goal, nc5 a bystander in the goal. The positive control drops
the target into the goal: it must pass with no robot contact and penetration
<= 5 mm.
"""

from collections.abc import Mapping
import re
from typing import Any

import mujoco
from mujoco_playground import wrapper
import numpy as np
from robosim_gauntlet import grade as grade_lib
from robosim_gauntlet import perturb
from robosim_gauntlet import task as task_lib
from robosim_gauntlet import wrappers


def _audit(mjm: mujoco.MjModel, robot: np.ndarray) -> list[str]:
  """Static audits on the compiled model."""
  issues = []
  free = mjm.jnt_type == mujoco.mjtJoint.mjJNT_FREE
  manip = {mjm.jnt_bodyid[j] for j in np.flatnonzero(free) if not robot[mjm.jnt_bodyid[j]]}
  for e in range(mjm.neq):
    if mjm.eq_type[e] in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
      if {mjm.eq_obj1id[e], mjm.eq_obj2id[e]} & manip:
        issues.append(f"equality {e} constrains a manipuland")
  for a in range(mjm.nu):
    if mjm.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
      if not robot[mjm.jnt_bodyid[mjm.actuator_trnid[a, 0]]]:
        issues.append(f"actuator {mjm.actuator(a).name} drives a non-robot dof")
  for b in sorted(manip):
    if not 0.005 <= mjm.body_mass[b] <= 2.0:
      issues.append(f"manipuland {mjm.body(b).name} mass {mjm.body_mass[b]:.4f}")
    for g in np.flatnonzero(mjm.geom_bodyid == b):
      if mjm.geom_margin[g] > 0 or mjm.geom_gap[g] > 0:
        issues.append(f"manipuland geom {mjm.geom(g).name} has margin/gap")
  if mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE:
    issues.append("contact override is enabled")
  return issues


def _place(sim: Any, body: str, xyz: np.ndarray) -> None:
  """Teleports a free body to `xyz`, upright and at rest."""
  mjm = sim.mj_model
  jnt = mjm.body_jntadr[mjm.body(body).id]
  adr, dof = mjm.jnt_qposadr[jnt], mjm.jnt_dofadr[jnt]
  sim.mj_data.qpos[adr : adr + 7] = [*xyz, 1.0, 0.0, 0.0, 0.0]
  sim.mj_data.qvel[dof : dof + 6] = 0.0


def run(task: task_lib.Task) -> list[str]:
  """Returns the preflight issues for `task`; empty means it is ready."""
  extra = perturb.privileged_devices(task, perturb.PSpec())
  dl, reset, close = task.build_device_layer(mode="sim", extra_devices=extra)
  try:
    reset()
    raw = dl.unwrapped if isinstance(dl, wrapper.DeviceLayerWrapper) else dl
    mjm = raw.sim.mj_model
    names = raw.get_device(perturb.MONITOR).body_names
    robot = np.array([
        bool(re.search(task.robot_pattern, mjm.body(b).name))
        for b in range(mjm.nbody)
    ])
    s = task.safety
    issues = _audit(mjm, robot)
    meas0 = dl.begin_stepping()
    p0 = grade_lib.poses_at(_tick(meas0), names)

    def _hold_for(meas: Mapping[str, Any]) -> dict[str, Any]:
      return {
          k: meas[j] if (j := wrappers.command_to_joint_key(k, s.command_to_joint)) in meas
          else 0.5 * (sp.minimum + sp.maximum)
          for k, sp in dl.commands_spec().items()
      }

    def _is_free(body: str) -> bool:
      jnt = int(mjm.body_jntadr[mjm.body(body).id])
      return jnt >= 0 and mujoco.mjtJoint(mjm.jnt_type[jnt]) == mujoco.mjtJoint.mjJNT_FREE

    hold0 = _hold_for(meas0)
    goal = p0[task.support_body][:3] + [0.0, 0.0, 0.06]

    def trial(
        moves: Mapping[str, Any], seconds: float, *, setter: Any = None
    ) -> tuple[bool, float, float]:
      reset()
      custom_hold = setter(raw.sim, p0) if setter is not None else None
      for body, xyz in moves.items():
        _place(raw.sim, body, np.asarray(xyz, float))
      raw.forward()
      meas = dl.begin_stepping()
      hold = (
          dict(custom_hold) if custom_hold is not None
          else (_hold_for(meas) if setter is not None else hold0)
      )
      forces = [np.array(meas[grade_lib.FORCES])]
      for _ in range(round(seconds / task.control_dt_s)):
        meas = dl.step(hold)
        forces.append(np.array(meas[grade_lib.FORCES]))
      f = np.max(forces, axis=0)
      ok = task.success(grade_lib.poses_at(_tick(meas), names), p0)
      return bool(ok), float(f[4]), float(f[5])

    controls: dict[str, tuple[Mapping[str, Any], float]] = {"nc1_reset": ({}, 0.5)}
    tgt_free = _is_free(task.target_body)
    if tgt_free:
      tgt = p0[task.target_body][:3]
      away = (tgt - goal) * [1.0, 1.0, 0.0]
      norm = float(np.linalg.norm(away))
      unit = away / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
      far = tgt + 0.15 * unit
      controls.update({
          "nc2_far_on_table": ({task.target_body: far}, 1.0),
          "nc3_hover": ({task.target_body: goal + [0.0, 0.0, 0.1]}, 0.0),
          "nc4_outside_wall": ({task.target_body: [goal[0] + 0.12, goal[1], tgt[2]]}, 1.0),
      })
    if task.bystander_bodies and _is_free(task.bystander_bodies[0]):
      controls["nc5_distractor_in_goal"] = ({task.bystander_bodies[0]: goal}, 1.5)
    for name, (moves, seconds) in controls.items():
      if trial(moves, seconds)[0]:
        issues.append(f"{name} passes the success predicate")
    if s.goal_setter is not None or tgt_free:
      ok, touch, pen = (
          trial({}, 1.5, setter=s.goal_setter) if s.goal_setter is not None
          else trial({task.target_body: goal}, 1.5)
      )
      tag = "goal_setter" if s.goal_setter is not None else "drop_into_goal"
      if not ok or (s.require_release and touch > 0) or pen > 0.005:
        issues.append(f"{tag}: success={ok} robot_contact={touch} penetration={pen:.4f}")
    else:
      issues.append("articulated target requires safety.goal_setter for preflight")
    return issues
  finally:
    close()


def _tick(meas: Mapping[str, Any]) -> wrappers.Tick:
  return wrappers.Tick({}, {k: np.array(v) for k, v in meas.items()})
