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
"""`DeviceLayer` wrappers stacked by the episode runner.

`McapWrapper`, `Recorder`, `CommandLimitWrapper`, and `CameraTimestampWrapper`
run in sim and real. `SensorWrapper`, `FaultWrapper`, and `LatencyWrapper` are
sim-only: on real hardware the sensors are real and `TimeTrigger` paces the
loop.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
import collections
import dataclasses
import datetime
import json
import math
import os
import pathlib
import struct
import time
from typing import Any

import dm_env
from dm_env import specs
from mujoco_playground import real as mp_real
from mujoco_playground import wrapper
import numpy as np
from robosim_gauntlet import perturb
from robosim_gauntlet import proxy

_JOINT = "joint_position/"
_MCAP_MAGIC = b"\x89MCAP0\r\n"
DEFAULT_GRIPPER_TOKENS = ("gripper", "finger", "jaw")
DEFAULT_CMD_TO_JOINT = (("goal_position/", "joint_position/"),)


def command_to_joint_key(
    cmd_key: str, pairs: Sequence[tuple[str, str]] = DEFAULT_CMD_TO_JOINT
) -> str | None:
  """Maps a command key to its joint-position measurement key, or `None`."""
  for src, dst in pairs:
    if src in cmd_key:
      return cmd_key.replace(src, dst, 1)
  return None


def _is_gripper(key: str, tokens: Sequence[str] = DEFAULT_GRIPPER_TOKENS) -> bool:
  return any(t in key for t in tokens)


def arm_joint_groups(
    keys: Iterable[str], *, arm_prefixes: Sequence[str] = (),
    gripper_tokens: Sequence[str] = DEFAULT_GRIPPER_TOKENS,
) -> dict[str, list[str]]:
  """Groups non-gripper joint keys as `{"": [...]}` or per `arm_prefixes`."""
  arm = sorted(k for k in keys if not _is_gripper(k, gripper_tokens))
  if not arm_prefixes:
    return {"": arm} if arm else {}
  groups = {p: [k for k in arm if k.startswith(p)] for p in arm_prefixes}
  if any(not g for g in groups.values()):
    raise ValueError(f"arm_prefixes {arm_prefixes} matched no arm joints in {arm}")
  return groups


def joint_keys(
    mspec: Mapping[str, Any], cspec: Mapping[str, Any],
    pairs: Sequence[tuple[str, str]] = DEFAULT_CMD_TO_JOINT,
) -> list[str]:
  """Returns joint-position measurement keys mapped from `cspec` or `joint_position/`."""
  mapped = {j for k in cspec if (j := command_to_joint_key(k, pairs)) in mspec}
  prefixes = tuple(dst for _, dst in pairs if dst) or (_JOINT,)
  return sorted(
      k for k, s in mspec.items() if k in mapped or (
          len(s.shape) < 2 and not k.startswith(perturb.PRIVILEGED_PREFIX)
          and not k.endswith("_timestamp") and any(p in k for p in prefixes)
      )
  )


def _images(spec: Mapping[str, Any]) -> list[str]:
  """Camera keys: 2-D depth or 3-D RGB, excluding privileged readings."""
  return sorted(
      k for k, s in spec.items()
      if len(s.shape) >= 2 and not k.startswith(perturb.PRIVILEGED_PREFIX)
  )


@dataclasses.dataclass
class Tick:
  """One executed control tick: the command sent and the measurements after."""

  cmd: dict[str, np.ndarray]
  meas: dict[str, np.ndarray]


class Recorder(wrapper.DeviceLayerWrapper):
  """Logs every executed tick's command and its small (non-image) readings."""

  def __init__(self, device_layer: Any, max_size: int = 64):
    super().__init__(device_layer)
    self._keys = [
        k for k, s in device_layer.measurements_spec().items()
        if math.prod(s.shape) <= max_size
    ]
    self.log: list[Tick] = []

  def _snap(self, meas: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {k: np.array(meas[k]) for k in self._keys if k in meas}

  def begin_stepping(self) -> Any:
    meas = self._device_layer.begin_stepping()
    self.log.append(Tick({}, self._snap(meas)))
    return meas

  def step(self, commands: Mapping[str, Any]) -> Any:
    meas = self._device_layer.step(commands)
    cmd = {k: np.array(v) for k, v in commands.items()}
    self.log.append(Tick(cmd, self._snap(meas)))
    return meas


class CommandLimitWrapper(wrapper.DeviceLayerWrapper):
  """Clips commands to their bounds and slews each key per tick."""

  def __init__(
      self, device_layer: Any, max_step_frac: float, *,
      command_to_joint: Sequence[tuple[str, str]] = DEFAULT_CMD_TO_JOINT,
  ):
    super().__init__(device_layer)
    self._specs, self._frac = dict(device_layer.commands_spec()), max_step_frac
    self._pairs, self._last = tuple(command_to_joint), {}

  def begin_stepping(self) -> Any:
    meas = self._device_layer.begin_stepping()
    self._last = {
        k: np.asarray(meas[j], np.float64) for k in self._specs
        if (j := command_to_joint_key(k, self._pairs)) in meas
    }
    return meas

  def step(self, commands: Mapping[str, Any]) -> Any:
    out = {}
    for k, v in commands.items():
      s, v = self._specs[k], np.asarray(v, np.float64)
      if isinstance(s, specs.BoundedArray):
        lo, hi = np.asarray(s.minimum, float), np.asarray(s.maximum, float)
        v = np.clip(v, lo, hi)
        if k in self._last:
          d = self._frac * (hi - lo)
          v = np.clip(v, self._last[k] - d, self._last[k] + d)
      self._last[k] = v
      out[k] = v.astype(s.dtype)
    return self._device_layer.step(out)


class CameraTimestampWrapper(wrapper.DeviceLayerWrapper):
  """Stamps `<camera>_timestamp` (seconds since episode start) on RGB streams.

  In `sim=True`, stamps deterministic tick time `tick * control_dt_s` so CRN
  replays match bit-for-bit. In `sim=False`, advances `<camera>_timestamp` from
  `clock()` only when the RGB frame changes so stalled hardware cameras freeze
  their timestamp.
  """

  def __init__(
      self, device_layer: Any, control_dt_s: float, *, sim: bool = True,
      clock: Callable[[], float] = time.monotonic,
  ):
    super().__init__(device_layer)
    spec = device_layer.measurements_spec()
    self._dt, self._sim, self._clock = float(control_dt_s), sim, clock
    self._cams = [
        k for k, s in spec.items()
        if len(s.shape) == 3 and not k.startswith(perturb.PRIVILEGED_PREFIX)
        and f"{k}_timestamp" not in spec
    ]
    self._mspec = {
        **spec, **{f"{k}_timestamp": specs.Array((), np.float64) for k in self._cams}
    }
    self._tick, self._t0 = 0, 0.0
    self._prev_frame: dict[str, np.ndarray] = {}
    self._prev_ts: dict[str, np.float64] = {}

  def measurements_spec(self) -> Mapping[str, specs.Array]:
    return self._mspec

  def _stamp(self, meas: Mapping[str, Any], *, first: bool) -> dict[str, Any]:
    out = dict(meas)
    if self._sim:
      ts = np.float64(self._tick * self._dt)
      for k in self._cams:
        out[f"{k}_timestamp"] = ts
      return out
    now = max(0.0, self._clock() - self._t0)
    for k in self._cams:
      frame = np.asarray(meas[k])
      if first or not np.array_equal(frame, self._prev_frame[k]):
        prev = -1e-6 if first else float(self._prev_ts[k])
        self._prev_frame[k] = frame.copy()
        self._prev_ts[k] = np.float64(0.0 if first else max(prev + 1e-6, now))
      out[f"{k}_timestamp"] = self._prev_ts[k]
    return out

  def begin_stepping(self) -> Any:
    self._tick = 0
    self._t0 = self._clock()
    self._prev_frame, self._prev_ts = {}, {}
    return self._stamp(self._device_layer.begin_stepping(), first=True)

  def step(self, commands: Mapping[str, Any]) -> Any:
    self._tick += 1
    return self._stamp(self._device_layer.step(commands), first=False)


def _shift_hue(rgb: np.ndarray, p: perturb.PSpec) -> np.ndarray:
  """Rotates chroma by `hue_deg`, scales saturation, applies gain."""
  arr = np.asarray(rgb, np.float32)
  luma = (arr @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
  chroma = (arr - luma) * p.sat_scale
  c, s = math.cos(math.radians(p.hue_deg)), math.sin(math.radians(p.hue_deg))
  u, v = chroma[..., 0].copy(), chroma[..., 2].copy()
  chroma[..., 0], chroma[..., 2] = c * u - s * v, s * u + c * v
  return (luma + chroma) * p.rgb_gain


class SensorWrapper(wrapper.DeviceLayerWrapper):
  """Proprio noise + bias, RGB hue/sat/gain/noise, depth noise, frame delay.

  Every tick draws a fixed amount from `rng`, independent of the commands, so
  `(seed, command log)` replays exactly.
  """

  def __init__(
      self, device_layer: Any, pspec: perturb.PSpec, rng: np.random.Generator,
      *, command_to_joint: Sequence[tuple[str, str]] = DEFAULT_CMD_TO_JOINT,
  ):
    super().__init__(device_layer)
    spec = device_layer.measurements_spec()
    self._proprio = joint_keys(spec, device_layer.commands_spec(), command_to_joint)
    imgs = _images(spec)
    self._rgb = [k for k in imgs if spec[k].dtype == np.uint8]
    self._depth = [k for k in imgs if spec[k].dtype != np.uint8]
    self._cam_ts = [f"{k}_timestamp" for k in imgs if f"{k}_timestamp" in spec]
    self._p, self._rng = pspec, rng
    self._bias: dict[str, float] = {}
    self._frames: collections.deque[dict[str, np.ndarray]] = collections.deque()

  def begin_stepping(self) -> Any:
    self._bias = {
        k: float(self._rng.normal(0.0, self._p.proprio_bias))
        for k in self._proprio
    }
    self._frames = collections.deque(maxlen=self._p.frame_delay + 1)
    return self._corrupt(self._device_layer.begin_stepping())

  def step(self, commands: Mapping[str, Any]) -> Any:
    return self._corrupt(self._device_layer.step(commands))

  def _corrupt(self, meas: Mapping[str, Any]) -> dict[str, Any]:
    p, rng, out = self._p, self._rng, dict(meas)
    for k in self._proprio:
      v = np.asarray(meas[k])
      noise = rng.normal(0.0, p.proprio_noise, v.shape)
      out[k] = (v + self._bias[k] + noise).astype(v.dtype)
    frame = {k: np.array(meas[k]) for k in self._cam_ts}
    for k in self._rgb:
      img = _shift_hue(meas[k], p)
      img = img + rng.normal(0.0, p.rgb_noise, img.shape)
      frame[k] = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    for k in self._depth:
      d = np.asarray(meas[k])
      d = d + rng.normal(0.0, p.depth_noise_m, d.shape)
      frame[k] = np.where(rng.random(d.shape) < p.depth_dropout, 0.0, d).astype(
          meas[k].dtype
      )
    self._frames.append(frame)
    out.update(self._frames[0])
    return out


class FaultWrapper(wrapper.DeviceLayerWrapper):
  """T4 terminal faults, armed by measured arm motion or a gripper close.

  After `pspec.fault_delay` ticks past arming, `camera_freeze` repeats every
  image from the onset tick (freezing `<cam>_timestamp` unless `fault_restamp`
  is set) and `dead_gripper` holds all gripper commands.
  """

  def __init__(
      self, device_layer: Any, pspec: perturb.PSpec, *,
      arm_prefixes: Sequence[str] = (),
      gripper_tokens: Sequence[str] = DEFAULT_GRIPPER_TOKENS,
      command_to_joint: Sequence[tuple[str, str]] = DEFAULT_CMD_TO_JOINT,
  ):
    super().__init__(device_layer)
    spec, cspec = device_layer.measurements_spec(), device_layer.commands_spec()
    self._cspec, self._pairs = dict(cspec), tuple(command_to_joint)
    self._cams = _images(spec)
    self._ts = [f"{k}_timestamp" for k in self._cams if f"{k}_timestamp" in spec]
    groups = arm_joint_groups(
        joint_keys(spec, cspec, self._pairs),
        arm_prefixes=arm_prefixes, gripper_tokens=gripper_tokens,
    )
    self._arm = sorted({k for grp in groups.values() for k in grp})
    self._grips = sorted(k for k in cspec if _is_gripper(k, gripper_tokens))
    self._close_below = {
        k: float(cspec[k].maximum) - 0.3 * float(cspec[k].maximum - cspec[k].minimum)
        for k in self._grips if isinstance(cspec[k], specs.BoundedArray)
    }
    self._p, self.onset_tick = pspec, None

  def _latch_grip(
      self, k: str, meas: Mapping[str, Any], commands: Mapping[str, Any]
  ) -> np.ndarray:
    j, s = command_to_joint_key(k, self._pairs), self._cspec[k]
    v = meas[j] if j is not None and j in meas else commands[k]
    if isinstance(s, specs.BoundedArray):
      v = np.clip(v, s.minimum, s.maximum)
    return np.asarray(v, dtype=s.dtype)

  def begin_stepping(self) -> Any:
    meas = self._device_layer.begin_stepping()
    self._q0 = np.array([float(meas[k]) for k in self._arm])
    self._tick, self._armed, self.onset_tick = 0, None, None
    self._frozen: dict[str, np.ndarray] = {}
    self._frozen_ts: dict[str, np.ndarray] = {}
    self._held: dict[str, np.ndarray] = {}
    return meas

  def step(self, commands: Mapping[str, Any]) -> Any:
    p = self._p
    if self.onset_tick is not None and p.fault == "dead_gripper" and self._held:
      commands = {**commands, **self._held}
    meas = self._device_layer.step(commands)
    self._tick += 1
    if p.fault and self._armed is None:
      q = np.array([float(meas[k]) for k in self._arm])
      moved = bool(self._arm) and float(np.max(np.abs(q - self._q0))) > 0.05
      closing = any(float(commands[k]) < t for k, t in self._close_below.items() if k in commands)
      if (moved and p.fault_trigger == "arm_motion") or (closing and p.fault_trigger == "gripper_close"):
        self._armed = self._tick
    if self._armed is not None and self.onset_tick is None and self._tick >= self._armed + p.fault_delay:
      self.onset_tick = self._tick
      self._held = {k: self._latch_grip(k, meas, commands) for k in self._grips if k in commands}
      self._frozen = {k: np.array(meas[k]) for k in self._cams}
      self._frozen_ts = {k: np.array(meas[k]) for k in self._ts}
    if self.onset_tick is not None and p.fault == "camera_freeze":
      meas = {**meas, **self._frozen}
      if not p.fault_restamp:
        meas.update(self._frozen_ts)
    return meas


class LatencyWrapper(wrapper.DeviceLayerWrapper):
  """Charges controller think time to the sim clock (the sim-only clock rule).

  Think time runs from the return of the last `step()` to the next call, so
  sim and render time are never charged. Beyond `budget_s * slack`, the held
  command is stepped up to 4 extra times first.
  """

  def __init__(
      self, device_layer: Any, budget_s: float, slack: float = 2.5,
      clock: Callable[[], float] = time.monotonic,
  ):
    super().__init__(device_layer)
    self._budget, self._slack, self._clock = budget_s, slack, clock
    self._last: Mapping[str, Any] | None = None
    self._t_ret = 0.0
    self.held_ticks = 0

  def begin_stepping(self) -> Any:
    meas = self._device_layer.begin_stepping()
    self._last, self._t_ret = None, self._clock()
    return meas

  def step(self, commands: Mapping[str, Any]) -> Any:
    think = self._clock() - self._t_ret
    if self._last is not None and think > self._budget * self._slack:
      for _ in range(min(4, int(think // self._budget) - 1)):
        self._device_layer.step(self._last)
        self.held_ticks += 1
    meas = self._device_layer.step(commands)
    self._last, self._t_ret = commands, self._clock()
    return meas


def _mcap_str(text: str) -> bytes:
  raw = text.encode("utf-8")
  return struct.pack("<I", len(raw)) + raw


def _mcap_rec(opcode: int, payload: bytes) -> bytes:
  return struct.pack("<BQ", opcode, len(payload)) + payload


class _FallbackMcapLogger:
  """Minimal binary MCAP v0 writer when optional `safari-sdk` is absent."""

  def __init__(
      self, out_dir: pathlib.Path, task_id: str, agent_id: str,
      dt_s: float, outcome_fn: Callable[[], str | None],
  ):
    self._dir, self._task, self._agent = out_dir, task_id, agent_id
    self._dt, self._outcome_fn = dt_s, outcome_fn
    self._idx = self._steps = self._t0 = self._t1 = 0

  def reset(self, timestep: dm_env.TimeStep) -> None:
    del timestep
    self._steps = 0
    self._t0 = self._t1 = time.time_ns()

  def record_action_and_next_timestep(
      self, action: Any, next_timestep: dm_env.TimeStep, policy_extra: Any = None,
  ) -> None:
    del action, next_timestep, policy_extra
    self._steps += 1
    self._t1 = time.time_ns()

  def write(self) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    day = self._dir / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"
    day.mkdir(parents=True, exist_ok=True)
    tag = self._outcome_fn() or "unlabeled"
    path = day / f"episode_{self._idx:04d}_{tag}.mcap"
    self._idx += 1
    meta = {
        "agent_id": self._agent, "control_timestep_seconds": f"{self._dt:.6f}",
        "first_timestamp_ns": str(self._t0), "last_timestamp_ns": str(self._t1),
        "num_steps": str(self._steps), "outcome": tag, "task_id": self._task,
        "summary_json": json.dumps({
            "agent_id": self._agent, "num_steps": self._steps,
            "outcome": tag, "task_id": self._task,
        }, sort_keys=True),
    }
    kv = b"".join(_mcap_str(k) + _mcap_str(v) for k, v in sorted(meta.items()))
    hdr = _mcap_str("robosim_gauntlet") + _mcap_str("robosim_gauntlet.mcap")
    md = _mcap_str("episode_summary") + struct.pack("<I", len(kv)) + kv
    path.write_bytes(
        _MCAP_MAGIC + _mcap_rec(0x01, hdr) + _mcap_rec(0x0C, md)
        + _mcap_rec(0x02, struct.pack("<QQI", 0, 0, 0)) + _MCAP_MAGIC
    )

  def close(self) -> None:
    pass


class McapWrapper(wrapper.DeviceLayerWrapper):
  """Records `.mcap` episodes via `mujoco_playground.real.create_logger`."""

  def __init__(
      self, device_layer: Any, output_dir: str | os.PathLike[str], *,
      task_id: str, agent_id: str = "robot", control_dt_s: float = 0.02,
      logger_factory: Callable[..., Any] | None = None,
  ):
    super().__init__(device_layer)
    self._dir = pathlib.Path(output_dir)
    self._dir.mkdir(parents=True, exist_ok=True)
    self._outcome: str | None = None
    self._stepping = self._has_episode = self._closed = False
    self.last_path: pathlib.Path | None = None
    obs = {
        k: s for k, s in device_layer.measurements_spec().items()
        if not k.startswith(perturb.PRIVILEGED_PREFIX)
    }
    act = dict(device_layer.commands_spec())
    self._obs_keys = tuple(obs)
    self._cmd_dtypes = {k: np.dtype(s.dtype) for k, s in act.items()}
    imgs = [k for k, s in obs.items() if len(s.shape) == 3 and s.dtype == np.uint8]
    ts_key = (
        "follower/command_timestamp"
        if "follower/command_timestamp" in obs
        else "log_timestamp_ns"
    )
    ts_spec = dm_env.TimeStep(
        step_type=specs.Array((), np.int32),
        reward=specs.Array((), np.float32),
        discount=specs.Array((), np.float32),
        observation=obs,
    )
    create_fn = logger_factory or mp_real.create_logger
    try:
      self._logger = create_fn(
          timestep_spec=ts_spec, action_spec=act, policy_extra_spec={},
          image_observation_keys=imgs, timestamp_key=ts_key, task_id=task_id,
          agent_id=agent_id, output_directory=str(self._dir),
          control_timestep_seconds=float(control_dt_s),
          outcome=lambda: self._outcome,
      )
      self.backend = "mujoco_playground"
    except ImportError:
      self._logger = _FallbackMcapLogger(
          self._dir, task_id, agent_id, float(control_dt_s),
          lambda: self._outcome,
      )
      self.backend = "fallback_mcap"

  def _obs(self, meas: Mapping[str, Any]) -> dict[str, Any]:
    return {k: meas[k] for k in self._obs_keys if k in meas}

  def set_outcome(self, outcome: str | None) -> None:
    self._outcome = outcome

  def begin_stepping(self) -> Any:
    meas = self._device_layer.begin_stepping()
    self._logger.reset(dm_env.restart(self._obs(meas)))
    self._stepping = self._has_episode = True
    return meas

  def step(self, commands: Mapping[str, Any]) -> Any:
    meas = self._device_layer.step(commands)
    if self._stepping:
      cmd = {k: np.asarray(v, self._cmd_dtypes[k]) for k, v in commands.items()}
      self._logger.record_action_and_next_timestep(
          cmd,
          dm_env.transition(np.float32(0.0), self._obs(meas), np.float32(1.0)),
      )
    return meas

  def end_stepping(self) -> None:
    self._stepping = False
    self._device_layer.end_stepping()

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    self._stepping = False
    before = set(self._dir.rglob("*.mcap"))
    if self._has_episode:
      self._logger.write()
      self._has_episode = False
    self._logger.close()
    added = sorted(set(self._dir.rglob("*.mcap")) - before)
    if added:
      self.last_path = added[-1]


AllowlistWrapper = proxy.AllowlistLayer
