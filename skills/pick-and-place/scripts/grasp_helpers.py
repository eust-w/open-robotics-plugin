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
"""Embodiment-agnostic gripper closing, grasp verification, and wrist-yaw helpers."""

from __future__ import annotations

from collections.abc import Callable
import math
import time
from typing import Any

import numpy as np


def verify_grasp(
    read_grip_fn: Callable[[], float],
    empty_closed: float,
    open_cmd: float,
    *,
    tol_frac: float = 0.015,
) -> dict[str, Any]:
  """Polarity-agnostic empty-close grasp check.

  Works whether `open_cmd > empty_closed` or `open_cmd < empty_closed`.

  Args:
    read_grip_fn: Callback returning current measured gripper position.
    empty_closed: Measured gripper reading when fully closed on empty air.
    open_cmd: Gripper command/reading at full open.
    tol_frac: Minimum fraction of full stroke away from `empty_closed` toward
      `open_cmd` required to report a grasp (default `0.015` = 1.5%, small
      enough for thin shafts like pens or hex keys).

  Returns:
    Dict with `grasped` (bool), `q_grip` (float), and `margin` (float, positive
    toward open).
  """
  q_grip = float(read_grip_fn())
  span = float(open_cmd) - float(empty_closed)
  if abs(span) < 1e-9:
    raise ValueError("open_cmd and empty_closed must be distinct.")
  direction = 1.0 if span > 0.0 else -1.0
  margin = direction * (q_grip - float(empty_closed))
  tol = float(tol_frac) * abs(span)
  return {
      "grasped": bool(margin > tol),
      "q_grip": q_grip,
      "margin": margin,
  }


def close_until_contact(
    cmd_grip_fn: Callable[[float], None],
    read_grip_fn: Callable[[], float],
    open_cmd: float,
    closed_cmd: float,
    empty_closed: float,
    *,
    n_steps: int = 20,
    settle_s: float = 0.04,
    stall_steps: int = 2,
    stall_frac: float = 0.008,
    hold_extra_frac: float = 0.05,
    tol_frac: float = 0.015,
    read_force_fn: Callable[[], float] | None = None,
    force_threshold: float | None = None,
) -> dict[str, Any]:
  """Closes a gripper smoothly until contact stall or force threshold.

  Polarity-agnostic: works identically whether closing decreases or increases
  the gripper command. Holds `g_hold` slightly past `g_contact` instead of
  slamming to `closed_cmd`.
  """
  start = float(read_grip_fn())
  target = float(closed_cmd)
  full_span = abs(float(open_cmd) - float(empty_closed))
  if full_span < 1e-9:
    raise ValueError("open_cmd and empty_closed must be distinct.")

  close_dir = 1.0 if target > start else -1.0
  stall_eps = float(stall_frac) * full_span
  cmds = np.linspace(start, target, max(2, int(n_steps)) + 1)[1:]

  prev_q = start
  stalls = 0
  g_contact = target
  stalled = False

  for cmd in cmds:
    cmd_val = float(cmd)
    cmd_grip_fn(cmd_val)
    if settle_s > 0.0:
      time.sleep(float(settle_s))
    q_now = float(read_grip_fn())

    if (
        read_force_fn is not None
        and force_threshold is not None
        and float(read_force_fn()) >= float(force_threshold)
    ):
      g_contact = q_now
      stalled = True
      break

    progress = close_dir * (q_now - prev_q)
    lag = close_dir * (cmd_val - q_now)
    if progress <= stall_eps and lag > stall_eps:
      stalls += 1
    else:
      stalls = 0

    if stalls >= max(1, int(stall_steps)):
      g_contact = q_now
      stalled = True
      break
    prev_q = q_now

  if not stalled:
    q_final = float(read_grip_fn())
    final_margin = (1.0 if float(open_cmd) > float(empty_closed) else -1.0) * (
        q_final - float(empty_closed)
    )
    if final_margin > float(tol_frac) * full_span:
      g_contact = q_final
      stalled = True

  if stalled:
    raw_hold = g_contact + close_dir * float(hold_extra_frac) * full_span
    lo, hi = min(float(open_cmd), target), max(float(open_cmd), target)
    g_hold = float(np.clip(raw_hold, lo, hi))
  else:
    g_hold = target
  cmd_grip_fn(g_hold)
  if settle_s > 0.0:
    time.sleep(float(settle_s))

  check = verify_grasp(
      read_grip_fn, empty_closed, open_cmd, tol_frac=tol_frac
  )
  return {
      "grasped": check["grasped"],
      "g_contact": float(g_contact),
      "g_hold": float(g_hold),
      "g_actual": check["q_grip"],
      "margin": check["margin"],
      "stalled": stalled,
  }


def plan_wrist_yaw(
    q_wrist_grasp: float,
    delta_yaw: float,
    q_min: float,
    q_max: float,
    *,
    symmetric_jaw: bool = True,
    margin: float = 0.05,
) -> dict[str, Any]:
  """Checks wrist roll limits for a pick-and-rotate, using `+-pi` pre-wind.

  Note: pre-winding by `+-pi` flips a wrist-mounted camera image 180 degrees;
  finish visual servoing before applying the pre-wind, or compute `T_world_cam`
  from live FK so the rotation is accounted for automatically.

  Args:
    q_wrist_grasp: Nominal wrist joint angle at grasp (rad).
    delta_yaw: Desired rotation from grasp to place (rad).
    q_min: Lower wrist joint limit (rad).
    q_max: Upper wrist joint limit (rad).
    symmetric_jaw: If True, tests `q_wrist_grasp + {0, +pi, -pi}` so symmetric
      parallel jaws can pre-wind to avoid saturating the wrist.
    margin: Safety margin inside `[q_min, q_max]` (rad).

  Returns:
    Dict with `q_grasp`, `q_place`, `prewind_rad`, and `needs_regrasp`.
  """
  lo = float(q_min) + float(margin)
  hi = float(q_max) - float(margin)
  offsets = (0.0, math.pi, -math.pi) if symmetric_jaw else (0.0,)

  best: dict[str, Any] | None = None
  best_score = float("inf")
  for off in offsets:
    qg = float(q_wrist_grasp) + off
    qp = qg + float(delta_yaw)
    if lo <= qg <= hi and lo <= qp <= hi:
      score = abs(qg) + abs(qp)
      if score < best_score:
        best_score = score
        best = {
            "q_grasp": qg,
            "q_place": qp,
            "prewind_rad": off,
            "needs_regrasp": False,
        }

  if best is not None:
    return best
  return {
      "q_grasp": float(q_wrist_grasp),
      "q_place": float(q_wrist_grasp) + float(delta_yaw),
      "prewind_rad": 0.0,
      "needs_regrasp": True,
  }
