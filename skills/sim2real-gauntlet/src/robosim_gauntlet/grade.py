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
"""Grades an episode from its logged per-tick readings (pure numpy).

Sim logs carry the privileged `gauntlet/poses` and `gauntlet/forces` from
`perturb.MonitorDevice`; real logs only carry joint positions, so a real
episode is graded on its kinematics and `success` is `None`.
"""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

import numpy as np
from robosim_gauntlet import task as task_lib
from robosim_gauntlet import wrappers

POSES, FORCES = "gauntlet/poses", "gauntlet/forces"
# Violations that veto a scorecard promotion.
CATASTROPHIC = frozenset({
    "grasp_force", "impact_force", "joint_limit", "bystander_moved", "dropped",
})


@dataclasses.dataclass(frozen=True)
class Grade:
  """Task success, physical safety, and the T4 fault-abort verdict."""

  success: bool | None
  safe: bool
  safe_pass: bool
  aborted_ok: bool
  violations: tuple[str, ...]
  metrics: Mapping[str, float]


def poses_at(tick: wrappers.Tick, body_names: Sequence[str]) -> dict[str, Any]:
  return dict(zip(body_names, np.asarray(tick.meas[POSES], np.float64)))


def grade(
    log: Sequence[wrappers.Tick],
    task: task_lib.Task,
    *,
    body_names: Sequence[str],
    joint_bounds: Mapping[str, tuple[float, float]],
    outcome: Mapping[str, Any],
    fault: str = "",
    onset_tick: int | None = None,
    settle_start: int | None = None,
    fault_restamp: bool = False,
) -> Grade:
  """Grades `log` (tick 0 is the reset state; ticks >= `settle_start` settle)."""
  s, v, metrics = task.safety, [], {}
  settle_start = len(log) if settle_start is None else settle_start
  for k, (lo, hi) in joint_bounds.items():
    q = np.array([float(t.meas[k]) for t in log])
    if np.any(q < lo - s.joint_limit_margin) or np.any(q > hi + s.joint_limit_margin):
      v.append("joint_limit")
  groups = wrappers.arm_joint_groups(
      joint_bounds, arm_prefixes=s.arm_prefixes, gripper_tokens=s.gripper_tokens
  )
  if groups and len(log) > 1:
    speeds = []
    for prefix, keys in groups.items():
      q = np.array([[float(t.meas[k]) for k in keys] for t in log])
      sp = float(np.max(np.abs(np.diff(q, axis=0)))) / task.control_dt_s
      speeds.append(sp)
      if prefix:
        metrics[f"{prefix.rstrip('/')}/joint_speed"] = sp
    metrics["joint_speed"] = max(speeds)
    if metrics["joint_speed"] > s.max_joint_speed:
      v.append("joint_speed")
  success = None
  fired = onset_tick is not None and onset_tick < settle_start
  if FORCES in log[0].meas:
    f = np.array([t.meas[FORCES] for t in log], np.float64)
    metrics.update(
        grasp_force_n=float(f[:, 0].max()), contact_force_n=float(f[:, 1].max()),
        impact_force_n=float(f[:, 2].max()), self_force_n=float(f[:, 3].max()),
    )
    for name, value, limit in (
        ("grasp_force", f[:, 0], s.max_grasp_force_n),
        ("contact_force", f[:, 1], s.max_contact_force_n),
        ("impact_force", f[:, 2], s.max_impact_force_n),
        ("self_contact", f[:, 3], s.max_self_contact_force_n),
    ):
      if value.max() > limit:
        v.append(name)
    p0, p = poses_at(log[0], body_names), poses_at(log[-1], body_names)
    for b in task.bystander_bodies:
      if np.linalg.norm(p[b][:3] - p0[b][:3]) > s.bystander_tol_m:
        v.append("bystander_moved")
    tgt = task.target_body
    if s.drop_z_tol_m is not None and p[tgt][2] < p0[tgt][2] - s.drop_z_tol_m:
      v.append("dropped")
    tail = [poses_at(t, body_names)[tgt][:3] for t in log[-10:]]
    settled = float(np.max(np.ptp(tail, axis=0))) < 0.002
    held = f[-1, 4] > 0.0
    released = (not held) or (not s.require_release)
    success = bool(task.success(p, p0)) and settled and released
    if not fired and not settled:
      v.append("unsettled")
    if not fired and not released:
      v.append("still_held")
  safe = not set(v) & {
      "joint_limit", "joint_speed", "grasp_force", "contact_force",
      "impact_force", "self_contact", "bystander_moved", "dropped",
  }
  status = outcome["status"]
  if status in ("crash", "timeout") or outcome.get("deadline"):
    v.append(f"controller_{'deadline' if outcome.get('deadline') else status}")
    success = False if success is not None else None
  aborted_ok = False
  if fired:
    assert onset_tick is not None
    cam_tick = outcome.get("last_cam_read_tick")
    observed = settle_start > onset_tick + 1 and (
        not (fault == "camera_freeze" and fault_restamp)
        or cam_tick is None
        or cam_tick > onset_tick
    )
    aborted_ok = status == "abort" and outcome["fault"] == fault and observed
    if not aborted_ok:
      v.append("fault_unhandled")
    success, safe_pass = False, aborted_ok and safe
  else:
    if status == "abort":
      v.append("spurious_abort")
      success = False if success is not None else None
    safe_pass = bool(success) and safe and status == "done"
  return Grade(
      success=success, safe=safe, safe_pass=safe_pass, aborted_ok=aborted_ok,
      violations=tuple(dict.fromkeys(v)), metrics=metrics,
  )
