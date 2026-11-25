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

"""Stateful multi-phase session wrapper for `Fr3DuoRobot` and planners."""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path  # pylint: disable=g-importing-member
import shutil
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS_DIR))

# pylint: disable=g-import-not-at-top
from fr3_duo_robot import Fr3DuoRobot  # noqa: E402
from fr3_recording import EpisodeRecorder  # noqa: E402
from fr3_safety import AbortWatcher  # noqa: E402

if TYPE_CHECKING:
  from motion_planner import RobotArmPlanner
# pylint: enable=g-import-not-at-top

__all__ = ["Fr3Session", "PhaseResult"]


@dataclass(frozen=True)
class PhaseResult:
  """Immutable outcome record produced by `Fr3Session.run_phase`.

  Attributes:
    name: Phase identifier passed to `run_phase`.
    ok: True when `until` passed (or `until is None`), False otherwise.
    attempts: Number of `step(session)` calls executed (`0` if aborted before
      the first attempt).
    elapsed_s: Wall-clock duration of the phase in seconds.
    value: Return value from the last completed `step(session)` call, or `None`
      if no attempt completed.
    aborted: True when `AbortWatcher.poll()` stopped the phase.
    error: Abort reason or final exception string when `ok` is False.
  """

  name: str
  ok: bool
  attempts: int
  elapsed_s: float
  value: Any = None
  aborted: bool = False
  error: str = ""


class Fr3Session:
  """Keeps `Fr3DuoRobot`, warm planners, and recording open across phases.

  Attributes:
    robot: Wrapped `Fr3DuoRobot` instance kept connected across phases.
    abort_watcher: `AbortWatcher` polled before each `run_phase` attempt.
    recorder: Active `EpisodeRecorder` when `record_dir` was set, else `None`.
    history: Ordered list of `PhaseResult` records from `run_phase` calls.
  """

  def __init__(
      self,
      robot: Fr3DuoRobot,
      *,
      abort_watcher: AbortWatcher | None = None,
      record_dir: str | Path | None = None,
      record_fps: float = 10.0,
  ) -> None:
    """Initializes a multi-phase session around an `Fr3DuoRobot` instance.

    Args:
      robot: `Fr3DuoRobot` to wrap. If `robot` is already connected when
        `connect()` runs, `close()` leaves it connected for the caller;
        otherwise `close()` closes it.
      abort_watcher: Optional external `AbortWatcher`. When omitted, the
        session creates and owns an `AbortWatcher` entered on `connect()` and
        exited on `close()`.
      record_dir: Optional directory for a single session-wide MP4 recording
        started on `connect()` via `robot.record_episode(...)`.
      record_fps: Target recording frame rate in Hz (must be finite and > 0).

    Raises:
      ValueError: If `record_fps` is not finite and > 0.
    """
    fps = float(record_fps)
    if fps <= 0.0 or not math.isfinite(fps):
      raise ValueError(
          f"record_fps must be finite and > 0, got {record_fps!r}."
      )

    self.robot = robot
    self._close_robot_on_exit = False

    if abort_watcher is None:
      self.abort_watcher = AbortWatcher()
      self._owns_abort_watcher = True
    else:
      self.abort_watcher = abort_watcher
      self._owns_abort_watcher = False

    self._abort_entered = False
    self._record_dir = record_dir
    self._record_fps = fps
    self.recorder: EpisodeRecorder | None = None
    self.history: list[PhaseResult] = []
    self._planners: dict[str, RobotArmPlanner] = {}
    self._planner_gripper_cmds: dict[str, float] = {}
    self._mjcf_path: Path | None = None

  def connect(self) -> Fr3Session:
    """Enters owned `AbortWatcher`, connects `robot`, and starts recording.

    Cleans up any partially initialized resources via `close()` if `connect()`
    or `record_episode()` raises.

    Returns:
      This `Fr3Session` instance.
    """
    try:
      if self._owns_abort_watcher and not self._abort_entered:
        self.abort_watcher.__enter__()
        self._abort_entered = True
      was_connected = bool(getattr(self.robot, "_connected", False))
      self.robot.connect()
      if not was_connected:
        self._close_robot_on_exit = True
      if self._record_dir is not None and self.recorder is None:
        self.recorder = self.robot.record_episode(
            self._record_dir, fps=self._record_fps
        )
    except BaseException:
      self.close()
      raise
    return self

  def close(self) -> None:
    """Tears down session resources in deterministic reverse order.

    Order: `recorder.stop()` -> owned `abort_watcher.__exit__` -> temporary
    MJCF directory removal -> owned `robot.close()`.
    """
    with contextlib.ExitStack() as stack:
      if self._close_robot_on_exit:
        self._close_robot_on_exit = False
        stack.callback(self.robot.close)
      if self._mjcf_path is not None:
        tmp_dir = self._mjcf_path.parent
        self._mjcf_path = None
        stack.callback(shutil.rmtree, tmp_dir, ignore_errors=True)
      if self._abort_entered:
        self._abort_entered = False
        stack.callback(self.abort_watcher.__exit__, None, None, None)
      if self.recorder is not None:
        rec = self.recorder
        self.recorder = None
        stack.callback(rec.stop)

  def __enter__(self) -> Fr3Session:
    return self.connect()

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    self.close()

  def _ensure_mjcf_path(self) -> Path:
    if self._mjcf_path is None:
      tmp_dir = Path(tempfile.mkdtemp(prefix="fr3_session_mp_"))
      mjcf_path = tmp_dir / "fr3_duo.xml"
      mjcf_path.write_text(self.robot.mjcf_xml, encoding="utf-8")
      self._mjcf_path = mjcf_path
    return self._mjcf_path

  def planner(self, arm: str, *, gripper_cmd: float = 0.0) -> RobotArmPlanner:
    """Returns a cached warm `RobotArmPlanner` for `arm` (`'left'|'right'`).

    Reuses the cached planner per arm across phases and updates its collision
    geometry via `set_gripper_opening(gripper_cmd)` when `gripper_cmd` changes.

    Args:
      arm: Target arm name (`'left'` or `'right'`).
      gripper_cmd: Normalized gripper opening command passed to
        `RobotArmPlanner`.

    Returns:
      Cached `RobotArmPlanner` instance for `arm`.

    Raises:
      ValueError: If `arm` is not `'left'` or `'right'`, or `gripper_cmd` is
        not finite.
    """
    if arm not in ("left", "right"):
      raise ValueError(f"Invalid arm={arm!r}; expected 'left' or 'right'.")
    cmd = float(gripper_cmd)
    if not math.isfinite(cmd):
      raise ValueError(f"gripper_cmd must be finite, got {gripper_cmd!r}.")
    if arm in self._planners:
      instance = self._planners[arm]
      if self._planner_gripper_cmds[arm] != cmd:
        instance.set_gripper_opening(cmd)
        self._planner_gripper_cmds[arm] = cmd
      return instance

    for candidate in (
        Path(os.path.abspath(__file__)).parents[2]
        / "motion-planning"
        / "scripts",
        _SCRIPTS_DIR.parents[1] / "motion-planning" / "scripts",
    ):
      if candidate.is_dir():
        if str(candidate) not in sys.path:
          sys.path.insert(0, str(candidate))
        break
    motion_planner = importlib.import_module("motion_planner")

    mjcf_path = self._ensure_mjcf_path()
    instance = motion_planner.RobotArmPlanner(
        mjcf_path=str(mjcf_path),
        arm=arm,
        gripper_cmd=cmd,
    )
    self._planners[arm] = instance
    self._planner_gripper_cmds[arm] = cmd
    return instance

  def _sleep_interruptible(self, duration_s: float) -> bool:
    """Sleeps `duration_s` in 50 ms chunks, returning False if aborted."""
    remaining = duration_s
    while remaining > 1e-9:
      if self.abort_watcher.poll():
        return False
      chunk = min(0.05, remaining)
      time.sleep(chunk)
      remaining -= chunk
    return True

  def run_phase(
      self,
      name: str,
      step: Callable[[Fr3Session], Any],
      *,
      until: Callable[[Any], bool] | None = None,
      every_s: float = 0.0,
      max_retries: int = 0,
  ) -> PhaseResult:
    """Runs `step(self)` up to `1 + max_retries` times until `until` passes.

    Polls `self.abort_watcher` before each attempt; if aborted, appends and
    returns `PhaseResult(ok=False, aborted=True, error=abort_watcher.reason)`.
    If `step(self)` raises `Exception`, retries up to `max_retries` times; on
    the final attempt, appends `PhaseResult(ok=False, error=str(exc))` to
    `self.history` and re-raises. If `until` never returns True, appends and
    returns `PhaseResult(ok=False)` with the last `step` return value.

    Args:
      name: Non-empty phase name (also sent to `recorder.set_phase(name)`).
      step: Callable taking this `Fr3Session` and returning a phase output.
      until: Optional completion predicate taking `step`'s return value.
        When `None`, the first non-raising attempt succeeds.
      every_s: Sleep interval in seconds between failed/incomplete attempts.
      max_retries: Non-negative integer number of retries after the first run.

    Returns:
      The recorded `PhaseResult` appended to `self.history`.

    Raises:
      ValueError: If `name` is empty, `every_s` is negative/non-finite, or
        `max_retries` is not a non-negative integer.
    """
    if not isinstance(name, str) or not name.strip():
      raise ValueError("Phase name must be a non-empty string.")
    interval_s = float(every_s)
    if interval_s < 0.0 or not math.isfinite(interval_s):
      raise ValueError(f"every_s must be finite and >= 0, got {every_s!r}.")
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
      raise ValueError(f"max_retries must be >= 0, got {max_retries!r}.")

    if self.recorder is not None:
      self.recorder.set_phase(name)

    t0 = time.monotonic()
    max_attempts = 1 + max_retries
    last_val: Any = None

    def _record(
        ok: bool,
        attempts: int,
        value: Any,
        *,
        aborted: bool = False,
        error: str = "",
    ) -> PhaseResult:
      result = PhaseResult(
          name=name,
          ok=ok,
          attempts=attempts,
          elapsed_s=time.monotonic() - t0,
          value=value,
          aborted=aborted,
          error=error,
      )
      self.history.append(result)
      return result

    for attempt in range(1, max_attempts + 1):
      if self.abort_watcher.poll():
        return _record(
            False,
            attempt - 1,
            last_val,
            aborted=True,
            error=self.abort_watcher.reason,
        )

      try:
        val = step(self)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if self.abort_watcher.poll():
          return _record(
              False,
              attempt,
              last_val,
              aborted=True,
              error=self.abort_watcher.reason or str(exc),
          )
        if attempt == max_attempts:
          _record(False, attempt, last_val, error=str(exc))
          raise
      else:
        last_val = val
        if self.abort_watcher.poll() or getattr(val, "aborted", False):
          return _record(
              False,
              attempt,
              val,
              aborted=True,
              error=self.abort_watcher.reason or getattr(val, "reason", ""),
          )
        if until is None or bool(until(val)):
          return _record(True, attempt, val)

      if attempt < max_attempts and interval_s > 0.0:
        if not self._sleep_interruptible(interval_s):
          return _record(
              False,
              attempt,
              last_val,
              aborted=True,
              error=self.abort_watcher.reason,
          )

    return _record(False, max_attempts, last_val)
