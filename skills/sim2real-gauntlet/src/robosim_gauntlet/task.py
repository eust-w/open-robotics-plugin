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
"""The consumer contract: a structural `Task` protocol and safety limits.

A task imports nothing from the gauntlet. It builds one `DeviceLayer` per
episode for `mode="sim"` (reaf-sim `flat_scene`, `Impl.CPP`) or `mode="real"`
(the playground hardware stack); that is the only sim/real seam.
"""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, Protocol

import numpy as np

# Body name -> (7,) world pose `[x, y, z, qw, qx, qy, qz]`.
Poses = Mapping[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class SafetyLimits:
  """Physical safety limits and task-level grading/joint-mapping hooks."""

  max_grasp_force_n: float = 35.0  # 16 ms boxcar, pads <-> target.
  max_contact_force_n: float = 60.0  # 16 ms boxcar, robot <-> environment.
  max_impact_force_n: float = 300.0  # One-substep peak, any robot contact.
  max_self_contact_force_n: float = 20.0  # Robot <-> robot, pads excluded.
  max_joint_speed: float = 4.0  # rad/s from measured arm joint positions.
  joint_limit_margin: float = 0.02  # Allowed excursion past command bounds.
  max_command_step_frac: float = 0.25  # Per-tick slew, fraction of key range.
  bystander_tol_m: float = 0.02  # Max bystander displacement.
  drop_z_tol_m: float | None = 0.05  # Max target drop below z0; None disables.
  require_release: bool = True  # Require zero robot-target contact at end.
  arm_prefixes: tuple[str, ...] = ()  # Optional per-arm key prefixes.
  gripper_tokens: tuple[str, ...] = ("gripper", "finger", "jaw")
  command_to_joint: tuple[tuple[str, str], ...] = (("goal_position/", "joint_position/"),)
  goal_setter: Callable[[Any, Poses], Mapping[str, Any] | None] | None = None


class Task(Protocol):
  """A manipulation task the gauntlet can run in sim and on the real robot."""

  task_id: str
  prompt: str
  control_dt_s: float
  episode_s: float
  safety: SafetyLimits
  target_body: str
  support_body: str
  bystander_bodies: tuple[str, ...]
  pad_pattern: str  # Regex over body names of the gripper fingers/pads.
  robot_pattern: str  # Regex over body names of the robot.

  def build_device_layer(
      self, *, mode: str, extra_devices: Mapping[str, Any] = ...
  ) -> tuple[Any, Callable[[], None], Callable[[], None]]:
    """Returns `(numpy DeviceLayer, reset, close)`; sim merges `extra_devices`."""
    ...

  def deployment_keys(self) -> tuple[frozenset[str], frozenset[str]]:
    """Returns the real-robot `(measurement keys, command keys)`."""
    ...

  def success(self, poses: Poses, poses0: Poses) -> bool:
    """Whether the final `poses` solve the task, given the initial `poses0`."""
    ...
