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
"""Runs, replays, and forks gauntlet episodes in sim or on the real robot.

Wrapper stack, bottom to top: the task's `DeviceLayer` -> `Recorder` ->
(optional) `McapWrapper` -> `CommandLimitWrapper` -> `CameraTimestampWrapper`
-> (sim only) `SensorWrapper` -> (sim only) `FaultWrapper` -> (optional)
`LatencyWrapper` -> `proxy.serve` or `AllowlistWrapper`. Sim is pinned to
`Impl.CPP` and every random draw is per tick, so `(seed, per-tick command log)`
reproduces an episode.
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import os
from typing import Any

from dm_env import specs
from mujoco_playground import wrapper
import numpy as np
from robosim_gauntlet import grade as grade_lib
from robosim_gauntlet import perturb
from robosim_gauntlet import proxy
from robosim_gauntlet import task as task_lib
from robosim_gauntlet import wrappers

ControllerSpec = Sequence[str] | Callable[[Any], Mapping[str, Any] | None]


@dataclasses.dataclass
class EpisodeResult:
  """One graded episode; `log[0]` is the reset state."""

  task_id: str
  tier: str
  idx: int
  seed: int
  mode: str
  pspec: perturb.PSpec
  outcome: Mapping[str, Any]
  grade: grade_lib.Grade
  log: list[wrappers.Tick]
  settle_start: int
  onset_tick: int | None
  scene_sha256: str
  body_names: tuple[str, ...]
  mcap_path: str = ""

  @property
  def key(self) -> tuple[str, str, int, int]:
    """The CRN pairing key."""
    return (self.task_id, self.tier, self.idx, self.seed)


class _Resume(wrapper.DeviceLayerWrapper):
  """Serves the replayed state on `/begin` instead of beginning again."""

  def __init__(self, device_layer: Any, meas: Mapping[str, Any]):
    super().__init__(device_layer)
    self._meas = meas

  def begin_stepping(self) -> Any:
    return self._meas


def _hold_command(
    rec: wrappers.Recorder, cspec: Mapping[str, Any], status: str,
    command_to_joint: Sequence[tuple[str, str]] = wrappers.DEFAULT_CMD_TO_JOINT,
) -> dict[str, np.ndarray]:
  """Builds a stationary hold command so post-episode settle stays passive."""
  if status == "done" and len(rec.log) > 1 and rec.log[-1].cmd:
    return dict(rec.log[-1].cmd)
  meas = rec.log[-1].meas
  last_cmd = rec.log[-1].cmd if len(rec.log) > 1 else {}
  cmd: dict[str, np.ndarray] = {}
  for k, s in cspec.items():
    jkey = wrappers.command_to_joint_key(k, command_to_joint)
    if jkey is not None and jkey in meas:
      val = np.asarray(meas[jkey], float)
    elif k in last_cmd:
      val = np.asarray(last_cmd[k], float)
    elif isinstance(s, specs.BoundedArray):
      val = 0.5 * (np.asarray(s.minimum, float) + np.asarray(s.maximum, float))
    else:
      val = np.zeros(s.shape, float)
    if isinstance(s, specs.BoundedArray):
      val = np.clip(val, s.minimum, s.maximum)
    cmd[k] = val.astype(s.dtype)
  return cmd


def settle(
    rec: wrappers.Recorder, cspec: Mapping[str, Any], dt: float, *,
    status: str = "done", min_s: float = 1.0, max_s: float = 2.0,
    command_to_joint: Sequence[tuple[str, str]] = wrappers.DEFAULT_CMD_TO_JOINT,
) -> None:
  """Steps stationary hold for 1.0 s, extending up to 2.0 s while settling."""
  cmd = _hold_command(rec, cspec, status, command_to_joint)
  min_ticks, max_ticks = round(min_s / dt), max(round(min_s / dt), round(max_s / dt))
  chunk = max(1, round(0.1 / dt))
  for _ in range(min_ticks):
    rec.step(cmd)
  done = min_ticks
  while (
      done < max_ticks and len(rec.log) >= 10
      and grade_lib.POSES in rec.log[-1].meas
      and float(np.max(np.ptp([t.meas[grade_lib.POSES][:, :3] for t in rec.log[-10:]], axis=0))) >= 0.002
  ):
    step_n = min(chunk, max_ticks - done)
    for _ in range(step_n):
      rec.step(cmd)
    done += step_n


def _joint_bounds(
    cspec: Mapping[str, Any], pairs: Sequence[tuple[str, str]] = wrappers.DEFAULT_CMD_TO_JOINT,
) -> dict[str, tuple[float, float]]:
  return {
      j: (float(s.minimum), float(s.maximum)) for k, s in cspec.items()
      if isinstance(s, specs.BoundedArray) and (j := wrappers.command_to_joint_key(k, pairs)) is not None
  }


def wrap_episode_layer(
    dl: Any, task: task_lib.Task, pspec: perturb.PSpec,
    sensor_rng: np.random.Generator, *, sim: bool = True,
    mcap_dir: str | os.PathLike[str] | None = None,
    replay: tuple[Sequence[wrappers.Tick], int] | None = None,
    include_latency: bool = False,
) -> tuple[Any, wrappers.Recorder, wrappers.McapWrapper | None, wrappers.FaultWrapper | None]:
  """Stacks episode wrappers onto `dl` (`LatencyWrapper` optional)."""
  s, rec, mcap = task.safety, wrappers.Recorder(dl), None
  base_dl: Any = rec
  if mcap_dir is not None:
    mcap = wrappers.McapWrapper(
        base_dl, mcap_dir, task_id=task.task_id,
        agent_id="sim" if sim else "real", control_dt_s=task.control_dt_s,
    )
    base_dl = mcap
  top: Any = wrappers.CommandLimitWrapper(
      base_dl, s.max_command_step_frac, command_to_joint=s.command_to_joint
  )
  top = wrappers.CameraTimestampWrapper(top, task.control_dt_s, sim=sim)
  fault = None
  if sim:
    top = wrappers.SensorWrapper(top, pspec, sensor_rng, command_to_joint=s.command_to_joint)
    fault = wrappers.FaultWrapper(
        top, pspec, arm_prefixes=s.arm_prefixes,
        gripper_tokens=s.gripper_tokens, command_to_joint=s.command_to_joint,
    )
    top = fault
  if replay is not None:
    log, tick = replay
    meas = top.begin_stepping()
    for t in log[1 : tick + 1]:
      meas = top.step(t.cmd)
    top = _Resume(top, meas)
  if sim and include_latency:
    top = wrappers.LatencyWrapper(top, task.control_dt_s)
  return top, rec, mcap, fault


def _run_inprocess(
    client: wrappers.AllowlistWrapper,
    controller_fn: Callable[[Any], Mapping[str, Any] | None],
) -> dict[str, Any]:
  """Drives `client` in-process and normalizes the episode outcome dict."""
  doc: dict[str, Any] = {"status": "crash", "fault": ""}
  try:
    res = controller_fn(client)
    if isinstance(res, Mapping) and res.get("status") in ("done", "abort"):
      doc = {"status": str(res["status"]), "fault": str(res.get("fault", ""))}
    elif res is None and not client.deadline:
      doc = {"status": "done", "fault": ""}
  except proxy.EpisodeOverError:
    pass
  finally:
    client.end_stepping()
  doc.update(
      returncode=0 if doc["status"] in ("done", "abort") else 1,
      stderr_tail="", ticks=client.ticks, deadline=client.deadline,
      last_cam_read_tick=client.last_cam_read_tick,
  )
  return doc


def run_episode(
    task: task_lib.Task, controller_cmd: ControllerSpec, *,
    tier: str = "T0", idx: int = 0, base_seed: int = 0, mode: str = "sim",
    env: Mapping[str, str] | None = None,
    replay: tuple[Sequence[wrappers.Tick], int] | None = None,
    mcap_dir: str | os.PathLike[str] | None = None,
    include_latency: bool | None = None,
) -> EpisodeResult:
  """Runs one episode of `controller_cmd` on `task` and grades it."""
  sim, s, in_proc = mode == "sim", task.safety, callable(controller_cmd)
  use_lat = (not in_proc) if include_latency is None else include_latency
  rng = perturb.streams(base_seed, task.task_id, tier, idx)
  pspec = perturb.sample(tier, idx, rng.spec) if sim else perturb.PSpec(tier, idx)
  extra = perturb.privileged_devices(task, pspec) if sim else {}
  dl, reset, close = task.build_device_layer(mode=mode, extra_devices=extra)
  mcap: wrappers.McapWrapper | None = None
  try:
    reset()
    sha, names = "", ()
    if sim:
      raw = dl.unwrapped if isinstance(dl, wrapper.DeviceLayerWrapper) else dl
      perturb.apply_static(raw, task, pspec)
      sha = hashlib.sha256(raw.sim.mj_spec.to_xml().encode()).hexdigest()
      names = raw.get_device(perturb.MONITOR).body_names
    top, rec, mcap, fault = wrap_episode_layer(
        dl, task, pspec, rng.sensor, sim=sim, mcap_dir=mcap_dir,
        replay=replay, include_latency=use_lat,
    )
    max_ticks = round(task.episode_s / task.control_dt_s)
    allow = task.deployment_keys()
    tick_fn = lambda: max(0, len(rec.log) - 1)
    outcome = (
        _run_inprocess(
            wrappers.AllowlistWrapper(top, allow, max_ticks=max_ticks, tick_fn=tick_fn),
            controller_cmd,
        )
        if in_proc else proxy.serve(
            top, controller_cmd, allow=allow, max_ticks=max_ticks,
            wall_timeout_s=10.0 * task.episode_s + 60.0, env=env,
            prompt=task.prompt, tick_fn=tick_fn,
        )
    )
    if not rec.log:  # The controller never began the episode.
      rec.begin_stepping()
    settle_start = len(rec.log)
    if sim:
      settle(
          rec, dl.commands_spec(), task.control_dt_s, status=outcome["status"],
          command_to_joint=s.command_to_joint,
      )
    onset = fault.onset_tick if fault is not None else None
    grade = grade_lib.grade(
        rec.log, task, body_names=names,
        joint_bounds=_joint_bounds(dl.commands_spec(), s.command_to_joint),
        outcome=outcome, fault=pspec.fault, onset_tick=onset,
        settle_start=settle_start, fault_restamp=pspec.fault_restamp,
    )
    if mcap is not None:
      if sim:
        mcap.set_outcome("success" if grade.safe_pass else "failure")
      elif outcome["status"] in ("crash", "timeout") or outcome.get("deadline"):
        mcap.set_outcome("invalid")
      else:
        mcap.set_outcome(
            "success" if grade.safe and outcome["status"] == "done" else "failure"
        )
      mcap.close()
  finally:
    if mcap is not None:
      mcap.close()
    close()
  return EpisodeResult(
      task_id=task.task_id, tier=tier, idx=idx, seed=base_seed, mode=mode,
      pspec=pspec, outcome=outcome, grade=grade, log=rec.log,
      settle_start=settle_start, onset_tick=onset, scene_sha256=sha,
      body_names=names,
      mcap_path=str(mcap.last_path) if mcap is not None and mcap.last_path else "",
  )


def fork(
    result: EpisodeResult, task: task_lib.Task, tick: int,
    controller_cmd: ControllerSpec, env: Mapping[str, str] | None = None, *,
    include_latency: bool | None = None,
) -> EpisodeResult:
  """Replays `result` to `tick`, then hands the episode to `controller_cmd`."""
  if not 0 <= tick < result.settle_start:
    raise ValueError(f"tick must be in [0, {result.settle_start}), got {tick}.")
  return run_episode(
      task, controller_cmd, tier=result.tier, idx=result.idx,
      base_seed=result.seed, mode=result.mode, env=env,
      replay=(result.log, tick), include_latency=include_latency,
  )


def run_suite(
    tasks: Sequence[task_lib.Task], controller_cmd: ControllerSpec, *,
    episodes: Mapping[str, int], base_seed: int = 0, mode: str = "sim",
    env: Mapping[str, str] | None = None,
    mcap_dir: str | os.PathLike[str] | None = None,
    include_latency: bool | None = None,
) -> list[EpisodeResult]:
  """Runs `episodes[tier]` episodes per tier for every task."""
  return [
      run_episode(
          t, controller_cmd, tier=tier, idx=i, base_seed=base_seed,
          mode=mode, env=env, mcap_dir=mcap_dir,
          include_latency=include_latency,
      )
      for t in tasks
      for tier, n in episodes.items()
      for i in range(n)
  ]
