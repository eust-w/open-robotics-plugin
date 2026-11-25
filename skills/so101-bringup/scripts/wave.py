#!/usr/bin/env python3
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
"""Standalone SO-101 wave smoke test (LeRobot-first with Feetech SDK fallback).

Uses LeRobot's SO101Follower natively when installed, and falls back seamlessly
to the lightweight Feetech STS3215 driver (feetech-servo-sdk + pyserial) when
running without full LeRobot:
  1. Two-step safe unfold from rest pose via a cleared pre-park waypoint
  2. Symmetric wide greeting wave (base sway, 2x symmetric vertical bounce,
     phase-lagged wrist roll follow-through, and jaw flutter, <= 1.00 rad/s)
  3. Two-step safe fold back to rest pose before disabling motor torque
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import math
import pathlib
import signal
import sys
import time
from typing import Any

BODY_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ALL_MOTORS = BODY_JOINTS + ("gripper",)
MOTOR_IDS = {name: idx for idx, name in enumerate(ALL_MOTORS, start=1)}

DEFAULT_MAX_SPEED_RAD_S = 1.00  # ~57.3 deg/s
HARD_MAX_SPEED_DEG_S = math.degrees(1.00)
DEFAULT_MAX_STEP_DEG = 20.0
JOINT_MARGIN_DEG = 5.0
REST_TOLERANCE_DEG = 5.0

# Feetech STS3215 registers (Protocol 0, 1 Mbps).
_ADDR_MIN_POS = 9
_ADDR_MAX_POS = 11
_ADDR_MAX_TORQUE = 16
_ADDR_P_COEFF = 21
_ADDR_D_COEFF = 22
_ADDR_I_COEFF = 23
_ADDR_PROT_CURRENT = 28
_ADDR_OP_MODE = 33
_ADDR_OVERLOAD_TORQUE = 36
_ADDR_TORQUE_ENABLE = 40
_ADDR_GOAL_POS = 42
_ADDR_LOCK = 55
_ADDR_PRESENT_POS = 56
_ADDR_PRESENT_VOLT = 62


def _detect_serial_port(config_path: pathlib.Path | None) -> str:
  """Detect SO-101 USB serial port cross-platform (macOS, Linux, Windows)."""
  if config_path and config_path.is_file():
    try:
      cfg = json.loads(config_path.read_text(encoding="utf-8"))
      port = cfg.get("port") or cfg.get("serial_port")
      if port:
        return str(port)
    except (OSError, ValueError):
      pass

  try:
    from serial.tools import list_ports  # pylint: disable=import-outside-toplevel,g-import-not-at-top
  except ImportError as exc:
    raise RuntimeError(
        "pyserial is required for --port auto; pass --port explicitly."
    ) from exc

  candidates: list[str] = []
  for p in list_ports.comports():
    dev = str(p.device)
    desc = f"{p.description or ''} {p.hwid or ''}".lower()
    if "bluetooth" in dev.lower() or "debug-console" in dev.lower():
      continue
    if p.vid is not None or any(
        tok in dev.lower()
        for tok in (
            "usbmodem",
            "usbserial",
            "wchusbserial",
            "ttyacm",
            "ttyusb",
            "com",
        )
    ):
      if "bluetooth" not in desc:
        candidates.append(dev)

  if len(candidates) == 1:
    return candidates[0]
  if not candidates:
    raise RuntimeError("No USB serial port found. Check USB cable.")
  raise RuntimeError(
      f"Multiple serial ports found ({candidates}). Pass --port explicitly."
  )


def _calibrated_bounds_deg(
    ranges: dict[str, tuple[int, int]],
) -> dict[str, tuple[float, float]]:
  bounds: dict[str, tuple[float, float]] = {}
  for j in BODY_JOINTS:
    r_min, r_max = ranges[j]
    half = ((r_max - r_min) / 2.0) * 360.0 / 4095.0
    bounds[j] = (-half, half)
  bounds["gripper"] = (0.0, 100.0)
  return bounds


def _validate_eeprom_ranges(
    ranges: dict[str, tuple[int, int]], raw_ticks: dict[str, int]
) -> bool:
  """Return True when all motor EEPROM ranges and current ticks are valid."""
  for m in ALL_MOTORS:
    if m not in ranges:
      return False
    r_min, r_max = ranges[m]
    if not 0 <= r_min < r_max <= 4095:
      return False
    if m != "wrist_roll" and (r_min, r_max) == (0, 4095):
      return False
    if m != "wrist_roll" and m in raw_ticks:
      if raw_ticks[m] < r_min - 80 or raw_ticks[m] > r_max + 80:
        return False
  return True


def _filter_kwargs(
    callable_or_cls: Any, kwargs: dict[str, Any]
) -> dict[str, Any]:
  """Keep only keyword arguments supported by the installed LeRobot version."""
  if dataclasses.is_dataclass(callable_or_cls):
    valid = {f.name for f in dataclasses.fields(callable_or_cls)}
  else:
    sig = inspect.signature(callable_or_cls)
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    ):
      return dict(kwargs)
    valid = set(sig.parameters.keys())
  return {k: v for k, v in kwargs.items() if k in valid}


class _LeRobotDriver:
  """Primary driver using LeRobot's SO101Follower when installed."""

  def __init__(self, port: str, robot_id: str) -> None:
    follower_cls = config_cls = None
    for mod_name in (
        "lerobot.robots.so_follower",
        "lerobot.robots.so101_follower",
    ):
      try:
        mod = importlib.import_module(mod_name)
        follower_cls = getattr(mod, "SO101Follower")
        config_cls = getattr(mod, "SO101FollowerConfig")
        break
      except Exception:  # pylint: disable=broad-exception-caught
        continue
    if follower_cls is None or config_cls is None:
      raise ImportError("LeRobot SO101Follower not importable")

    cfg_kwargs = _filter_kwargs(
        config_cls,
        {
            "id": robot_id,
            "port": port,
            "use_degrees": True,
            "max_relative_target": DEFAULT_MAX_STEP_DEG,
            "disable_torque_on_disconnect": False,
        },
    )
    self.robot = follower_cls(config_cls(**cfg_kwargs))
    self.bounds: dict[str, tuple[float, float]] = {}

  def connect(self) -> None:
    """Sync valid EEPROM calibration into LeRobot cache and connect."""
    bus = self.robot.bus
    bus.connect(handshake=True)
    try:
      raw_ticks = {
          k: int(v)
          for k, v in bus.sync_read(
              "Present_Position", normalize=False, num_retry=2
          ).items()
      }
      eeprom_cal = bus.read_calibration()
    finally:
      bus.disconnect(disable_torque=False)

    ranges = {
        m: (int(eeprom_cal[m].range_min), int(eeprom_cal[m].range_max))
        for m in ALL_MOTORS
        if m in eeprom_cal
    }
    if _validate_eeprom_ranges(ranges, raw_ticks) and hasattr(
        self.robot, "calibration_fpath"
    ):
      cal_path = pathlib.Path(self.robot.calibration_fpath)
      if (
          not cal_path.is_file()
          or getattr(self.robot, "calibration", None) != eeprom_cal
      ):
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: dataclasses.asdict(v) for k, v in eeprom_cal.items()}
        cal_path.write_text(
            json.dumps(payload, indent=4) + "\n", encoding="utf-8"
        )
        self.robot.calibration = eeprom_cal
        bus.calibration = eeprom_cal
        print(f"Synced onboard EEPROM calibration -> {cal_path}")
      self.bounds = _calibrated_bounds_deg(ranges)
    elif getattr(self.robot, "calibration", None):
      cal = self.robot.calibration
      self.bounds = _calibrated_bounds_deg({
          m: (int(cal[m].range_min), int(cal[m].range_max))
          for m in ALL_MOTORS
      })
    else:
      raise RuntimeError(
          "No valid calibration found in motor EEPROM or LeRobot cache. "
          "Run lerobot-calibrate or scripts/calibrate.py first."
      )

    connect_kwargs = _filter_kwargs(self.robot.connect, {"calibrate": False})
    self.robot.connect(**connect_kwargs)

  def read_pose(self) -> dict[str, float]:
    """Read current normalized joint pose from LeRobot."""
    obs = self.robot.get_observation()
    return {j: float(obs[f"{j}.pos"]) for j in ALL_MOTORS}

  def send_pose(self, pose: dict[str, float]) -> None:
    """Send normalized joint targets to LeRobot."""
    self.robot.send_action({f"{j}.pos": float(pose[j]) for j in ALL_MOTORS})

  def disconnect(self, disable_torque: bool) -> None:
    """Disconnect from LeRobot, optionally disabling servo torque."""
    if hasattr(self.robot, "config") and hasattr(
        self.robot.config, "disable_torque_on_disconnect"
    ):
      self.robot.config.disable_torque_on_disconnect = disable_torque
    self.robot.disconnect()


class _FeetechDriver:
  """Lightweight direct driver for 6x Feetech STS3215 servos via scservo_sdk."""

  def __init__(self, port: str) -> None:
    import scservo_sdk as scs  # pylint: disable=import-outside-toplevel,g-import-not-at-top

    self._scs = scs
    self.port = port
    self.ph = scs.PortHandler(port)
    self.pk = scs.PacketHandler(0)
    self.ranges: dict[str, tuple[int, int]] = {}
    self.bounds: dict[str, tuple[float, float]] = {}
    self._last_pose: dict[str, float] | None = None

  def _read1(self, mid: int, addr: int) -> int:
    for _ in range(5):
      val, res, _ = self.pk.read1ByteTxRx(self.ph, mid, addr)
      if res == self._scs.COMM_SUCCESS:
        return int(val)
      time.sleep(0.005)
    raise RuntimeError(f"Motor ID {mid} failed to reply (addr={addr}).")

  def _read2(self, mid: int, addr: int) -> int:
    for _ in range(5):
      val, res, _ = self.pk.read2ByteTxRx(self.ph, mid, addr)
      if res == self._scs.COMM_SUCCESS:
        return int(val)
      time.sleep(0.005)
    raise RuntimeError(f"Motor ID {mid} failed to reply (addr={addr}).")

  def _write1(self, mid: int, addr: int, val: int) -> None:
    for _ in range(3):
      res, _ = self.pk.write1ByteTxRx(self.ph, mid, addr, int(val))
      if res == self._scs.COMM_SUCCESS:
        if addr < 40:
          time.sleep(0.012)
        return
      time.sleep(0.015)
    raise RuntimeError(f"Write1 failed on motor ID {mid} (addr={addr}).")

  def _write2(self, mid: int, addr: int, val: int) -> None:
    for _ in range(3):
      res, _ = self.pk.write2ByteTxRx(self.ph, mid, addr, int(val))
      if res == self._scs.COMM_SUCCESS:
        if addr < 40:
          time.sleep(0.012)
        return
      time.sleep(0.015)
    raise RuntimeError(f"Write2 failed on motor ID {mid} (addr={addr}).")

  def _ensure_eeprom(self, mid: int, items: list[tuple[int, int, int]]) -> None:
    """Write only EEPROM registers (addr, width_bytes, desired) that differ."""
    pending: list[tuple[int, int, int]] = []
    for addr, width, desired in items:
      curr = self._read1(mid, addr) if width == 1 else self._read2(mid, addr)
      if curr != desired:
        pending.append((addr, width, desired))
    if not pending:
      return
    self._write1(mid, _ADDR_LOCK, 0)
    try:
      for addr, width, desired in pending:
        if width == 1:
          self._write1(mid, addr, desired)
        else:
          self._write2(mid, addr, desired)
    finally:
      self._write1(mid, _ADDR_LOCK, 1)

  def _ticks_to_pose(self, ticks: dict[str, int]) -> dict[str, float]:
    pose: dict[str, float] = {}
    for j in BODY_JOINTS:
      r_min, r_max = self.ranges[j]
      mid = (r_min + r_max) / 2.0
      pose[j] = (float(ticks[j]) - mid) * 360.0 / 4095.0
    g_min, g_max = self.ranges["gripper"]
    g_val = ((float(ticks["gripper"]) - g_min) / max(g_max - g_min, 1)) * 100.0
    pose["gripper"] = max(0.0, min(100.0, g_val))
    return pose

  def _pose_to_ticks(self, pose: dict[str, float]) -> dict[str, int]:
    """Convert normalized joint degrees and gripper percent to raw ticks."""
    ticks: dict[str, int] = {}
    for j in BODY_JOINTS:
      r_min, r_max = self.ranges[j]
      mid = (r_min + r_max) / 2.0
      raw = int(round(mid + float(pose[j]) * 4095.0 / 360.0))
      ticks[j] = max(0, min(4095, raw))
    g_min, g_max = self.ranges["gripper"]
    frac = max(0.0, min(100.0, float(pose["gripper"]))) / 100.0
    ticks["gripper"] = max(
        0, min(4095, int(round(g_min + frac * (g_max - g_min))))
    )
    return ticks

  def connect(self) -> None:
    """Open the bus, validate EEPROM ranges, and enable servo torque."""
    if not self.ph.openPort():
      raise RuntimeError(f"Failed to open serial port {self.port}")
    if not self.ph.setBaudRate(1_000_000):
      self.ph.closePort()
      raise RuntimeError(f"Failed to set 1,000,000 baud on {self.port}")

    raw_ticks: dict[str, int] = {}
    voltages: dict[str, float] = {}
    for name, mid in MOTOR_IDS.items():
      raw_ticks[name] = self._read2(mid, _ADDR_PRESENT_POS)
      voltages[name] = self._read1(mid, _ADDR_PRESENT_VOLT) / 10.0
      self.ranges[name] = (
          self._read2(mid, _ADDR_MIN_POS),
          self._read2(mid, _ADDR_MAX_POS),
      )

    print(
        f"Connected to 6 STS3215 servos on {self.port} "
        f"(voltage {min(voltages.values()):.1f}-{max(voltages.values()):.1f}V)."
    )
    if not _validate_eeprom_ranges(self.ranges, raw_ticks):
      self.ph.closePort()
      raise RuntimeError(
          "Motor EEPROM does not hold valid calibration ranges. "
          "Run scripts/calibrate.py first."
      )
    self.bounds = _calibrated_bounds_deg(self.ranges)

    # Configure safe PID gains and gripper protection (only if changed).
    for name, mid in MOTOR_IDS.items():
      self._write1(mid, _ADDR_TORQUE_ENABLE, 0)
      eeprom_targets = [
          (_ADDR_OP_MODE, 1, 0),
          (_ADDR_P_COEFF, 1, 16),
          (_ADDR_I_COEFF, 1, 0),
          (_ADDR_D_COEFF, 1, 32),
      ]
      if name == "gripper":
        eeprom_targets.extend([
            (_ADDR_MAX_TORQUE, 2, 500),
            (_ADDR_PROT_CURRENT, 2, 250),
            (_ADDR_OVERLOAD_TORQUE, 1, 25),
        ])
      self._ensure_eeprom(mid, eeprom_targets)
      self._write2(mid, _ADDR_GOAL_POS, raw_ticks[name])
      self._write1(mid, _ADDR_TORQUE_ENABLE, 1)

    self._last_pose = self._ticks_to_pose(raw_ticks)

  def read_pose(self) -> dict[str, float]:
    raw = {
        name: self._read2(mid, _ADDR_PRESENT_POS)
        for name, mid in MOTOR_IDS.items()
    }
    pose = self._ticks_to_pose(raw)
    self._last_pose = pose
    return pose

  def send_pose(self, pose: dict[str, float]) -> None:
    """Send a step-clamped joint pose to all six servos."""
    safe_pose = dict(pose)
    if self._last_pose is not None:
      for j in ALL_MOTORS:
        delta = safe_pose[j] - self._last_pose[j]
        if abs(delta) > DEFAULT_MAX_STEP_DEG:
          safe_pose[j] = self._last_pose[j] + math.copysign(
              DEFAULT_MAX_STEP_DEG, delta
          )
    ticks = self._pose_to_ticks(safe_pose)
    for name, mid in MOTOR_IDS.items():
      self._write2(mid, _ADDR_GOAL_POS, ticks[name])
    self._last_pose = safe_pose

  def disconnect(self, disable_torque: bool) -> None:
    try:
      if disable_torque:
        for mid in MOTOR_IDS.values():
          try:
            self._write1(mid, _ADDR_TORQUE_ENABLE, 0)
          except Exception:  # pylint: disable=broad-exception-caught
            pass
    finally:
      self.ph.closePort()


def _clamp_pose(
    pose: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    margin: float = JOINT_MARGIN_DEG,
) -> dict[str, float]:
  """Clamp body joints inside calibrated bounds with a safety margin."""
  out: dict[str, float] = {}
  for j in BODY_JOINTS:
    lo, hi = bounds[j]
    safe_lo = min(lo + margin, 0.0)
    safe_hi = max(hi - margin, 0.0)
    out[j] = max(safe_lo, min(safe_hi, float(pose[j])))
  out["gripper"] = max(0.0, min(100.0, float(pose["gripper"])))
  return out


def _cosine_segment(
    start: dict[str, float],
    end: dict[str, float],
    max_speed_deg_s: float,
    hz: float,
    min_duration: float = 0.45,
) -> list[dict[str, float]]:
  """Cosine-eased zero-endpoint-velocity segment bounded by max_speed_deg_s."""
  max_delta = max(abs(end[j] - start[j]) for j in ALL_MOTORS)
  if max_delta < 1e-3:
    return []
  min_dur = (math.pi / 2.0) * max_delta / max(max_speed_deg_s * 0.92, 1.0)
  dur = max(min_duration, min_dur)
  min_steps = int(
      math.ceil((math.pi / 2.0) * max_delta / (DEFAULT_MAX_STEP_DEG * 0.9))
  )
  steps = max(2, int(round(dur * hz)), min_steps)
  waypoints: list[dict[str, float]] = []
  for k in range(1, steps + 1):
    ease = 0.5 * (1.0 - math.cos(math.pi * (k / steps)))
    waypoints.append({
        j: start[j] + (end[j] - start[j]) * ease
        for j in ALL_MOTORS
    })
  return waypoints


def _smooth_plateau_env(u: float, ramp: float = 0.20) -> float:
  """Zero-velocity ramp-in, full 1.0 amplitude plateau, zero-velocity ramp-out."""
  if u <= 0.0 or u >= 1.0:
    return 0.0
  if u < ramp:
    return 0.5 * (1.0 - math.cos(math.pi * (u / ramp)))
  if u > 1.0 - ramp:
    return 0.5 * (1.0 - math.cos(math.pi * ((1.0 - u) / ramp)))
  return 1.0


def _pre_park_pose(
    rest_pose: dict[str, float], bounds: dict[str, tuple[float, float]]
) -> dict[str, float]:
  """Cleared elevated waypoint so folding/unfolding never clips the arm or desk."""
  return _clamp_pose(
      {
          "shoulder_pan": rest_pose["shoulder_pan"],
          "shoulder_lift": -48.0,
          "elbow_flex": 48.0,
          "wrist_flex": 20.0,
          "wrist_roll": 0.0,
          "gripper": rest_pose["gripper"],
      },
      bounds,
  )


def build_cute_wave_plan(
    rest_pose: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    cycles: float,
    max_speed_deg_s: float,
    hz: float,
) -> list[dict[str, float]]:
  """Build symmetric wide wave with 2-step collision-free unfold and park."""
  grip_closed = max(0.0, min(100.0, float(rest_pose["gripper"])))
  grip_open = min(88.0, max(grip_closed + 38.0, 68.0))
  grip_mid = 0.5 * (grip_closed + grip_open)
  grip_amp = 0.45 * (grip_open - grip_closed)

  pre_park = _pre_park_pose(rest_pose, bounds)
  wave_ready = _clamp_pose(
      {
          "shoulder_pan": 0.0,
          "shoulder_lift": -22.0,
          "elbow_flex": 8.0,
          "wrist_flex": -14.0,
          "wrist_roll": 0.0,
          "gripper": grip_mid,
      },
      bounds,
  )

  # 1. Two-step safe unfold: rest -> cleared pre-park -> upright wave pose
  waypoints: list[dict[str, float]] = []
  waypoints.extend(
      _cosine_segment(
          rest_pose, pre_park, max_speed_deg_s * 0.85, hz, min_duration=1.2
      )
  )
  waypoints.extend(
      _cosine_segment(
          pre_park, wave_ready, max_speed_deg_s, hz, min_duration=1.2
      )
  )

  # 2. Symmetric wide wave: 1x horizontal sway + 2x symmetric vertical bounce
  pan_amp = 24.0
  elbow_bounce = 4.0  # 2x freq -> same height on both left and right peaks
  flex_bounce = 6.0  # 2x freq -> symmetric cheerful wrist bob on both sides
  roll_amp = 28.0
  max_eff_amp = max(
      pan_amp,
      2.0 * elbow_bounce,
      2.0 * flex_bounce,
      roll_amp,
      grip_amp,
  )
  period_sec = max(
      1.6, (2.0 * math.pi * max_eff_amp) / max(max_speed_deg_s * 0.92, 1.0)
  )
  wave_dur = max(1.0, cycles * period_sec)
  wave_steps = max(12, int(round(wave_dur * hz)))

  for k in range(1, wave_steps + 1):
    u = k / wave_steps
    env = _smooth_plateau_env(u, ramp=0.18)
    phase = 2.0 * math.pi * cycles * u
    bounce = 1.0 - math.cos(2.0 * phase)
    target = {
        "shoulder_pan": (
            wave_ready["shoulder_pan"] + env * pan_amp * math.sin(phase)
        ),
        "shoulder_lift": wave_ready["shoulder_lift"],
        "elbow_flex": wave_ready["elbow_flex"] - env * elbow_bounce * bounce,
        "wrist_flex": wave_ready["wrist_flex"] - env * flex_bounce * bounce,
        "wrist_roll": (
            wave_ready["wrist_roll"] + env * roll_amp * math.sin(phase - 0.35)
        ),
        "gripper": grip_mid + env * grip_amp * math.sin(phase),
    }
    waypoints.append(_clamp_pose(target, bounds))

  # 3. Two-step safe fold: wave -> cleared pre-park -> rest_pose
  waypoints.extend(
      _cosine_segment(
          waypoints[-1], pre_park, max_speed_deg_s * 0.9, hz, min_duration=1.2
      )
  )
  waypoints.extend(
      _cosine_segment(
          pre_park, rest_pose, max_speed_deg_s * 0.8, hz, min_duration=1.4
      )
  )
  return waypoints


def _summarize_plan(
    start: dict[str, float], waypoints: list[dict[str, float]], hz: float
) -> tuple[float, float, float]:
  prev = start
  max_step_body = 0.0
  for wp in waypoints:
    step = max(abs(wp[j] - prev[j]) for j in BODY_JOINTS)
    max_step_body = max(max_step_body, step)
    prev = wp
  return max_step_body, max_step_body * hz, len(waypoints) / hz


def _create_driver(port: str, robot_id: str) -> Any:
  """Create LeRobot driver when available, falling back to Feetech SDK."""
  try:
    drv = _LeRobotDriver(port=port, robot_id=robot_id)
    print("Driver: LeRobot SO101Follower")
    return drv
  except Exception:  # pylint: disable=broad-exception-caught
    print("Driver: lightweight Feetech STS3215 fallback (scservo_sdk)")
    return _FeetechDriver(port=port)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Standalone SO-101 wave smoke test"
          " (LeRobot-first with Feetech fallback)."
      )
  )
  parser.add_argument(
      "--port",
      default="auto",
      help="Serial port path (or 'auto' for cross-platform USB detection).",
  )
  parser.add_argument(
      "--robot-id",
      default=None,
      help="LeRobot calibration ID (default: from config or 'my_follower').",
  )
  parser.add_argument(
      "--config",
      type=pathlib.Path,
      default=pathlib.Path("config/robot_config.json"),
      help="Optional workspace config JSON (used only if present).",
  )
  parser.add_argument(
      "--cycles", type=float, default=3.0, help="Wave cycles (default: 3.0)."
  )
  parser.add_argument(
      "--speed-rad-s",
      type=float,
      default=DEFAULT_MAX_SPEED_RAD_S,
      help=(
          f"Max joint speed in rad/s (default: {DEFAULT_MAX_SPEED_RAD_S:.2f}"
          " rad/s)."
      ),
  )
  parser.add_argument(
      "--hz", type=float, default=30.0, help="Control rate in Hz (default: 30)."
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Plan and validate the wave trajectory without touching hardware.",
  )
  args = parser.parse_args()

  max_speed_deg_s = max(
      5.0, min(math.degrees(float(args.speed_rad_s)), HARD_MAX_SPEED_DEG_S)
  )
  hz = max(10.0, min(float(args.hz), 50.0))
  dt = 1.0 / hz

  robot_id = args.robot_id
  if robot_id is None and args.config.is_file():
    try:
      robot_id = json.loads(args.config.read_text(encoding="utf-8")).get(
          "robot_id"
      )
    except (OSError, ValueError):
      pass
  robot_id = str(robot_id or "my_follower")

  if args.dry_run:
    sample_rest = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -95.0,
        "elbow_flex": 95.0,
        "wrist_flex": 20.0,
        "wrist_roll": 0.0,
        "gripper": 35.0,
    }
    sample_bounds = {j: (-110.0, 110.0) for j in BODY_JOINTS}
    sample_bounds["gripper"] = (0.0, 100.0)
    plan = build_cute_wave_plan(
        sample_rest, sample_bounds, args.cycles, max_speed_deg_s, hz
    )
    max_step, peak_deg_s, dur = _summarize_plan(sample_rest, plan, hz)
    print(
        f"[DRY-RUN] Planned {len(plan)} waypoints ({dur:.1f}s @ {hz:.0f} Hz) |"
        f" max_step={max_step:.2f} deg | peak_speed={peak_deg_s:.1f} deg/s"
        f" ({math.radians(peak_deg_s):.2f} rad/s)"
    )
    return

  port = (
      _detect_serial_port(args.config)
      if args.port == "auto"
      else str(args.port)
  )
  driver = _create_driver(port=port, robot_id=robot_id)

  aborted = False

  def _handle_signal(signum: int, frame: Any) -> None:
    del frame
    nonlocal aborted
    aborted = True
    print(f"\nSignal {signum} received — parking smoothly to rest pose...")

  prev_sigint = signal.signal(signal.SIGINT, _handle_signal)
  prev_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

  rest_pose: dict[str, float] | None = None
  driver.connect()

  disable_torque = False
  try:
    rest_pose = driver.read_pose()
    rounded_rest = {k: round(v, 1) for k, v in rest_pose.items()}
    print(f"Recorded rest_pose: {rounded_rest}")
    plan = build_cute_wave_plan(
        rest_pose, driver.bounds, args.cycles, max_speed_deg_s, hz
    )
    max_step, peak_deg_s, dur = _summarize_plan(rest_pose, plan, hz)
    print(
        f"Executing cute wave: {len(plan)} steps ({dur:.1f}s @ {hz:.0f} Hz), "
        f"max_step={max_step:.2f} deg, peak_speed={peak_deg_s:.1f} deg/s "
        f"({math.radians(peak_deg_s):.2f} rad/s)..."
    )

    for wp in plan:
      if aborted:
        break
      t0 = time.monotonic()
      driver.send_pose(wp)
      rem = dt - (time.monotonic() - t0)
      if rem > 0:
        time.sleep(rem)
  finally:
    try:
      if rest_pose is not None:
        curr = driver.read_pose()
        err = max(abs(curr[j] - rest_pose[j]) for j in BODY_JOINTS)
        if err > 2.0:
          print(
              "Parking safely via pre-park back to rest_pose "
              f"(current max delta={err:.1f} deg)..."
          )
          pre_park = _pre_park_pose(rest_pose, driver.bounds)
          park_steps = _cosine_segment(
              curr, pre_park, min(max_speed_deg_s, 35.0), hz, min_duration=1.2
          )
          park_steps.extend(
              _cosine_segment(
                  pre_park,
                  rest_pose,
                  min(max_speed_deg_s, 30.0),
                  hz,
                  min_duration=1.4,
              )
          )
          for wp in park_steps:
            t0 = time.monotonic()
            driver.send_pose(wp)
            rem = dt - (time.monotonic() - t0)
            if rem > 0:
              time.sleep(rem)
          time.sleep(0.25)

        final_pose = driver.read_pose()
        final_err = max(abs(final_pose[j] - rest_pose[j]) for j in BODY_JOINTS)
        if final_err <= REST_TOLERANCE_DEG:
          print(
              f"Parked cleanly at rest_pose (max error={final_err:.2f} deg)."
              " Disabling torque."
          )
          disable_torque = True
        else:
          print(
              f"WARNING: Arm is {final_err:.1f} deg away from rest_pose; "
              "leaving torque ON. Please support the arm before cutting power."
          )
    finally:
      try:
        driver.disconnect(disable_torque=disable_torque)
      finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(130)
