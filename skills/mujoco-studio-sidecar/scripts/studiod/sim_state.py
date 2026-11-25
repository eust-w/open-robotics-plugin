# Copyright 2026 DeepMind Technologies Limited
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
"""Named joint and actuator state preserved across model reloads."""

import dataclasses

import mujoco
import numpy as np


@dataclasses.dataclass
class PreservedSimState:
  """Named joint and actuator state preserved across model reloads."""

  time: float = 0.0
  joint_qpos: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
  joint_qvel: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
  actuator_ctrl: dict[str, float] = dataclasses.field(default_factory=dict)


def capture_preserved_state(
    model: mujoco.MjModel, data: mujoco.MjData
) -> PreservedSimState:
  """Extracts named joint qpos/qvel and actuator ctrl values from MjData."""
  state = PreservedSimState(time=float(data.time))
  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_JOINT), j)
    if not name:
      continue
    jtype = int(model.jnt_type[j])
    qpos_adr = int(model.jnt_qposadr[j])
    qvel_adr = int(model.jnt_dofadr[j])
    nq = {
        int(mujoco.mjtJoint.mjJNT_FREE): 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 4,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }.get(jtype, 0)
    nv = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6,
        int(mujoco.mjtJoint.mjJNT_BALL): 3,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }.get(jtype, 0)
    if nq > 0:
      state.joint_qpos[name] = np.array(
          data.qpos[qpos_adr : qpos_adr + nq], dtype=np.float64, copy=True
      )
    if nv > 0:
      state.joint_qvel[name] = np.array(
          data.qvel[qvel_adr : qvel_adr + nv], dtype=np.float64, copy=True
      )
  for a in range(model.nu):
    name = mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), a)
    if name:
      state.actuator_ctrl[name] = float(data.ctrl[a])
  return state


def apply_preserved_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    preserved: PreservedSimState | None,
) -> None:
  """Restores compatible named joint qpos/qvel and actuator ctrl into MjData."""
  if preserved is None:
    return
  data.time = preserved.time
  for j in range(model.njnt):
    name = mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_JOINT), j)
    if not name:
      continue
    jtype = int(model.jnt_type[j])
    qpos_adr = int(model.jnt_qposadr[j])
    qvel_adr = int(model.jnt_dofadr[j])
    nq = {
        int(mujoco.mjtJoint.mjJNT_FREE): 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 4,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }.get(jtype, 0)
    nv = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6,
        int(mujoco.mjtJoint.mjJNT_BALL): 3,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }.get(jtype, 0)
    old_qpos = preserved.joint_qpos.get(name)
    if old_qpos is not None and len(old_qpos) == nq:
      data.qpos[qpos_adr : qpos_adr + nq] = old_qpos
    old_qvel = preserved.joint_qvel.get(name)
    if old_qvel is not None and len(old_qvel) == nv:
      data.qvel[qvel_adr : qvel_adr + nv] = old_qvel
  for a in range(model.nu):
    name = mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), a)
    if name and name in preserved.actuator_ctrl:
      data.ctrl[a] = preserved.actuator_ctrl[name]
  mujoco.mj_forward(model, data)
