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
"""Lightweight SO-101 read-only check & LeRobot-compatible joint calibration.

Uses only feetech-servo-sdk + pyserial (zero ML/LeRobot dependencies), while
writing the exact same STS3215 EEPROM registers and LeRobot JSON cache format
(~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<id>.json).

Usage:
  # Read-only motor, voltage, EEPROM calibration & span check (Torque OFF):
  python3 scripts/calibrate.py --check

  # Interactive joint calibration (Torque OFF; run in your own terminal):
  python3 scripts/calibrate.py --port auto --robot-id my_follower
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
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

# Nominal SO-101 URDF joint travel spans in degrees (so101_new_calib.urdf).
URDF_NOMINAL_SPAN_DEG = {
    "shoulder_pan": 220.0,
    "shoulder_lift": 200.0,
    "elbow_flex": 194.0,
    "wrist_flex": 190.0,
    "wrist_roll": 320.0,
    "gripper": 110.0,
}

# Feetech STS3215 registers (Protocol 0, 1 Mbps).
_ADDR_MIN_POS = 9
_ADDR_MAX_POS = 11
_ADDR_HOMING_OFFSET = 31
_ADDR_OP_MODE = 33
_ADDR_TORQUE_ENABLE = 40
_ADDR_LOCK = 55
_ADDR_PRESENT_POS = 56
_ADDR_PRESENT_VOLT = 62
_SIGN_BIT_HOMING = 11


def _encode_sign_magnitude(value: int, sign_bit: int) -> int:
  max_mag = (1 << sign_bit) - 1
  mag = min(abs(int(value)), max_mag)
  return ((1 << sign_bit) | mag) if value < 0 else mag


def _decode_sign_magnitude(raw: int, sign_bit: int) -> int:
  mag = int(raw) & ((1 << sign_bit) - 1)
  return -mag if (int(raw) & (1 << sign_bit)) else mag


def detect_serial_port() -> str:
  """Auto-detect a single USB serial port cross-platform (macOS/Linux/Windows)."""
  from serial.tools import list_ports  # pylint: disable=import-outside-toplevel,g-import-not-at-top

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


def default_cache_path(robot_id: str) -> pathlib.Path:
  env_home = os.environ.get("HF_LEROBOT_HOME")
  base = (
      pathlib.Path(env_home).expanduser()
      if env_home
      else pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"
  )
  return (
      base / "calibration" / "robots" / "so101_follower" / f"{robot_id}.json"
  )


def _ticks_to_rest_pose(
    raw_ticks: dict[str, int], cal: dict[str, dict[str, int]]
) -> dict[str, float]:
  pose: dict[str, float] = {}
  for j in BODY_JOINTS:
    r_min, r_max = cal[j]["range_min"], cal[j]["range_max"]
    mid = (r_min + r_max) / 2.0
    pose[j] = round((raw_ticks[j] - mid) * 360.0 / 4095.0, 2)
  g_min, g_max = cal["gripper"]["range_min"], cal["gripper"]["range_max"]
  g_pct = ((raw_ticks["gripper"] - g_min) / max(g_max - g_min, 1)) * 100.0
  pose["gripper"] = round(max(0.0, min(100.0, g_pct)), 1)
  return pose


def evaluate_calibration(
    cal: dict[str, dict[str, int]], raw_ticks: dict[str, int]
) -> tuple[bool, list[str], dict[str, float]]:
  """Check validity and flag joints outside ~70%-130% of nominal URDF span."""
  valid = True
  warnings: list[str] = []
  spans_deg: dict[str, float] = {}

  for name in ALL_MOTORS:
    r_min = int(cal[name]["range_min"])
    r_max = int(cal[name]["range_max"])
    span = round((r_max - r_min) * 360.0 / 4095.0, 1)
    spans_deg[name] = span

    if not 0 <= r_min < r_max <= 4095:
      valid = False
      warnings.append(f"{name}: invalid bounds [{r_min}, {r_max}]")
      continue
    if name != "wrist_roll" and (r_min, r_max) == (0, 4095):
      valid = False
      warnings.append(f"{name}: factory uncalibrated range [0, 4095]")
      continue
    if name != "wrist_roll" and name in raw_ticks:
      if raw_ticks[name] < r_min - 80 or raw_ticks[name] > r_max + 80:
        valid = False
        warnings.append(
            f"{name}: current tick {raw_ticks[name]} outside [{r_min}, {r_max}]"
        )

    nominal = URDF_NOMINAL_SPAN_DEG[name]
    ratio = span / nominal
    if ratio < 0.70 or ratio > 1.30:
      warnings.append(
          f"{name}: span {span:.1f} deg is {ratio * 100:.0f}% of nominal "
          f"{nominal:.0f} deg URDF span (outside 70%-130% heads-up band)"
      )

  return valid, warnings, spans_deg


def _open_bus(port: str) -> tuple[Any, Any, Any]:
  import scservo_sdk as scs  # pylint: disable=import-outside-toplevel,g-import-not-at-top

  ph = scs.PortHandler(port)
  pk = scs.PacketHandler(0)
  if not ph.openPort():
    raise RuntimeError(f"Failed to open serial port {port}")
  if not ph.setBaudRate(1_000_000):
    ph.closePort()
    raise RuntimeError(f"Failed to set 1,000,000 baud on {port}")
  return scs, ph, pk


def _read1(scs: Any, ph: Any, pk: Any, mid: int, addr: int) -> int:
  for _ in range(5):
    val, res, _ = pk.read1ByteTxRx(ph, mid, addr)
    if res == scs.COMM_SUCCESS:
      return int(val)
    time.sleep(0.005)
  raise RuntimeError(f"Motor ID {mid} did not respond (addr={addr})")


def _read2(scs: Any, ph: Any, pk: Any, mid: int, addr: int) -> int:
  for _ in range(5):
    val, res, _ = pk.read2ByteTxRx(ph, mid, addr)
    if res == scs.COMM_SUCCESS:
      return int(val)
    time.sleep(0.005)
  raise RuntimeError(f"Motor ID {mid} did not respond (addr={addr})")


def _write1(scs: Any, ph: Any, pk: Any, mid: int, addr: int, val: int) -> None:
  for _ in range(3):
    res, _ = pk.write1ByteTxRx(ph, mid, addr, int(val))
    if res == scs.COMM_SUCCESS:
      if addr < 40:
        time.sleep(0.012)
      return
    time.sleep(0.015)
  raise RuntimeError(f"Write1 failed on motor ID {mid} (addr={addr})")


def _write2(scs: Any, ph: Any, pk: Any, mid: int, addr: int, val: int) -> None:
  for _ in range(3):
    res, _ = pk.write2ByteTxRx(ph, mid, addr, int(val))
    if res == scs.COMM_SUCCESS:
      if addr < 40:
        time.sleep(0.012)
      return
    time.sleep(0.015)
  raise RuntimeError(f"Write2 failed on motor ID {mid} (addr={addr})")


def _save_cache(cal: dict[str, dict[str, int]], path: pathlib.Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(cal, indent=4) + "\n", encoding="utf-8")


def run_check(port: str, robot_id: str, sync_cache: bool) -> None:
  """Read-only motor, voltage, torque, and EEPROM calibration check."""
  scs, ph, pk = _open_bus(port)
  raw_ticks: dict[str, int] = {}
  torque: dict[str, int] = {}
  volts: dict[str, float] = {}
  cal: dict[str, dict[str, int]] = {}
  missing: list[str] = []

  try:
    for name, mid in MOTOR_IDS.items():
      try:
        raw_ticks[name] = _read2(scs, ph, pk, mid, _ADDR_PRESENT_POS)
        torque[name] = _read1(scs, ph, pk, mid, _ADDR_TORQUE_ENABLE)
        volts[name] = round(
            _read1(scs, ph, pk, mid, _ADDR_PRESENT_VOLT) / 10.0, 1
        )
        r_min = _read2(scs, ph, pk, mid, _ADDR_MIN_POS)
        r_max = _read2(scs, ph, pk, mid, _ADDR_MAX_POS)
        hoff_raw = _read2(scs, ph, pk, mid, _ADDR_HOMING_OFFSET)
        cal[name] = {
            "id": mid,
            "drive_mode": 0,
            "homing_offset": _decode_sign_magnitude(hoff_raw, _SIGN_BIT_HOMING),
            "range_min": r_min,
            "range_max": r_max,
        }
      except RuntimeError:
        missing.append(f"ID {mid} ({name})")
  finally:
    ph.closePort()

  print(f"Port: {port}")
  if missing:
    raise RuntimeError(
        f"Missing motor responses: {', '.join(missing)}. "
        "Check servo power supply and daisy-chain cables."
    )

  print(f"Voltages (V):  {volts}")
  print(f"Torque_Enable: {torque}")
  if any(v != 0 for v in torque.values()):
    print("WARNING: One or more servos have Torque_Enable != 0!")

  valid, warnings, spans = evaluate_calibration(cal, raw_ticks)
  print(f"Spans (deg):   {spans}")
  for w in warnings:
    print(f"HEADS-UP: {w}")

  if valid:
    rest_pose = _ticks_to_rest_pose(raw_ticks, cal)
    print("Calibration:   VALID (onboard EEPROM)")
    print(f"Rest pose:     {rest_pose}")
    if abs(rest_pose["shoulder_pan"]) > 20.0:
      print(
          f"HEADS-UP: arm base is currently turned"
          f" {rest_pose['shoulder_pan']:+.1f} deg off-center — straighten it to"
          " its centered folded resting spot before saving rest_pose."
      )
    if sync_cache:
      cache_path = default_cache_path(robot_id)
      _save_cache(cal, cache_path)
      print(f"Synced cache -> {cache_path}")
  else:
    print(
        "Calibration:   INVALID/UNCALIBRATED — run calibrate.py interactively."
    )
    sys.exit(2)


def run_interactive_calibrate(port: str, robot_id: str) -> None:
  """Interactive Torque-OFF joint calibration compatible with LeRobot."""
  scs, ph, pk = _open_bus(port)
  try:
    # Torque OFF, unlock EEPROM, position mode, reset offsets/limits to raw.
    for mid in MOTOR_IDS.values():
      _write1(scs, ph, pk, mid, _ADDR_TORQUE_ENABLE, 0)
      _write1(scs, ph, pk, mid, _ADDR_LOCK, 0)
      _write1(scs, ph, pk, mid, _ADDR_OP_MODE, 0)
      _write2(scs, ph, pk, mid, _ADDR_HOMING_OFFSET, 0)
      _write2(scs, ph, pk, mid, _ADDR_MIN_POS, 0)
      _write2(scs, ph, pk, mid, _ADDR_MAX_POS, 4095)

    input(
        "Torque is OFF. Hold the SO-101 arm at the middle of every joint's "
        "range of motion, then press ENTER..."
    )
    homings: dict[str, int] = {}
    for name, mid in MOTOR_IDS.items():
      pos = _read2(scs, ph, pk, mid, _ADDR_PRESENT_POS)
      offset = pos - 2047
      homings[name] = offset
      _write2(
          scs, ph, pk, mid, _ADDR_HOMING_OFFSET,
          _encode_sign_magnitude(offset, _SIGN_BIT_HOMING),
      )
    time.sleep(0.05)

    sweep_joints = [j for j in ALL_MOTORS if j != "wrist_roll"]
    mins = {
        j: _read2(scs, ph, pk, MOTOR_IDS[j], _ADDR_PRESENT_POS)
        for j in sweep_joints
    }
    maxes = dict(mins)

    print(
        "\nSlowly sweep all joints (except wrist_roll) through their full "
        "min-to-max range of motion.\nPress ENTER when finished..."
    )
    stop_event = threading.Event()

    def _wait_enter() -> None:
      try:
        sys.stdin.readline()
      finally:
        stop_event.set()

    threading.Thread(target=_wait_enter, daemon=True).start()
    consecutive_errors = 0
    while not stop_event.is_set():
      read_ok = False
      for j in sweep_joints:
        try:
          pos = _read2(scs, ph, pk, MOTOR_IDS[j], _ADDR_PRESENT_POS)
          mins[j] = min(mins[j], pos)
          maxes[j] = max(maxes[j], pos)
          read_ok = True
        except RuntimeError:
          # Ignore brief bus glitches while the user is flexing cables by hand.
          continue
      if read_ok:
        consecutive_errors = 0
      else:
        consecutive_errors += 1
        if consecutive_errors >= 30:
          print()
          raise RuntimeError(
              "Lost communication with all servos during sweep. Check USB/power"
              " cable."
          )
      status = " | ".join(f"{j}:[{mins[j]},{maxes[j]}]" for j in sweep_joints)
      print(f"\r{status}   ", end="", flush=True)
      time.sleep(0.03)
    print()

    mins["wrist_roll"] = 0
    maxes["wrist_roll"] = 4095

    cal: dict[str, dict[str, int]] = {}
    raw_ticks: dict[str, int] = {}
    for name, mid in MOTOR_IDS.items():
      _write2(scs, ph, pk, mid, _ADDR_MIN_POS, mins[name])
      _write2(scs, ph, pk, mid, _ADDR_MAX_POS, maxes[name])
      _write1(scs, ph, pk, mid, _ADDR_LOCK, 1)
      raw_ticks[name] = _read2(scs, ph, pk, mid, _ADDR_PRESENT_POS)
      cal[name] = {
          "id": mid,
          "drive_mode": 0,
          "homing_offset": homings[name],
          "range_min": mins[name],
          "range_max": maxes[name],
      }
  finally:
    for mid in MOTOR_IDS.values():
      try:
        _write1(scs, ph, pk, mid, _ADDR_LOCK, 1)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    ph.closePort()

  cache_path = default_cache_path(robot_id)
  _save_cache(cal, cache_path)
  valid, warnings, spans = evaluate_calibration(cal, raw_ticks)
  print(f"\nSaved calibration to motor EEPROM and {cache_path}")
  print(f"Calibrated spans (deg): {spans}")
  for w in warnings:
    print(f"HEADS-UP: {w}")
  print(
      "\nNext: set the arm back straight/centered in its folded resting spot "
      "(torque is still OFF) before running powered motions."
  )
  if not valid:
    print("WARNING: Calibration had invalid bounds; please re-run.")
    sys.exit(2)


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Lightweight SO-101 read-only check & LeRobot-compatible calibration."
      )
  )
  parser.add_argument("--port", default="auto", help="Serial port or 'auto'.")
  parser.add_argument(
      "--robot-id", default="my_follower", help="Robot ID for cache."
  )
  parser.add_argument(
      "--check",
      action="store_true",
      help=(
          "Read-only check of servos, voltages, EEPROM calibration, and spans"
          " (Torque OFF)."
      ),
  )
  parser.add_argument(
      "--no-sync-cache",
      action="store_true",
      help=(
          "With --check, do not write valid EEPROM calibration to LeRobot"
          " cache."
      ),
  )
  args = parser.parse_args()

  port = detect_serial_port() if args.port == "auto" else str(args.port)
  if args.check:
    run_check(
        port=port, robot_id=args.robot_id, sync_cache=not args.no_sync_cache
    )
  else:
    run_interactive_calibrate(port=port, robot_id=args.robot_id)


if __name__ == "__main__":
  main()
