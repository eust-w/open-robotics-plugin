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

"""Modbus RTU driver for Robotiq 2F-85/2F-140/Hand-E grippers.

Communicates over RS-485 USB-serial adapters at 115200 8N1 (default slave ID 9)
using standard Modbus RTU function codes 0x10 (Preset Multiple Registers at
0x03E8) and 0x04 (Read Input Registers at 0x07D0).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import glob
import math
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any

try:
  import serial  # pytype: disable=import-error
except ImportError:
  serial: Any = None

DEFAULT_SLAVE_ID = 9
DEFAULT_BAUDRATE = 115200
DEADTIME_S = 0.160
EMPTY_CLOSE_MIN_RATIO = 0.80

ACTION_REG_START = 0x03E8
STATUS_REG_START = 0x07D0


def crc16_modbus(data: bytes) -> int:
  """Computes the 16-bit Modbus RTU CRC (reflected polynomial 0xA001)."""
  crc = 0xFFFF
  for byte in bytes(data):
    crc ^= byte
    for _ in range(8):
      if crc & 0x0001:
        crc = (crc >> 1) ^ 0xA001
      else:
        crc >>= 1
  return crc & 0xFFFF


def closure_to_raw(closure_0_to_1: float) -> int:
  """Maps normalized closure in [0.0, 1.0] (0=open, 1=closed) to [0, 255]."""
  val = float(closure_0_to_1)
  if not math.isfinite(val):
    raise ValueError(f"closure_0_to_1 must be finite, got {closure_0_to_1!r}")
  clamped = max(0.0, min(1.0, val))
  return int(round(clamped * 255.0))


def raw_to_closure(raw_0_to_255: int) -> float:
  """Maps raw register value in [0, 255] to normalized closure in [0.0, 1.0]."""
  clamped = max(0, min(255, int(raw_0_to_255)))
  return float(clamped) / 255.0


def build_write_registers_frame(
    slave_id: int, start_reg: int, values: Sequence[int]
) -> bytes:
  """Builds a Modbus RTU function 0x10 (Preset Multiple Registers) frame."""
  if not 0 <= int(slave_id) <= 247:
    raise ValueError(f"Invalid Modbus slave_id: {slave_id}")
  if not 0 <= int(start_reg) <= 0xFFFF:
    raise ValueError(f"Invalid start_reg: {start_reg:#x}")
  reg_values = [int(v) & 0xFFFF for v in values]
  if not 1 <= len(reg_values) <= 123:
    raise ValueError(f"Invalid register count: {len(reg_values)}")

  count = len(reg_values)
  byte_count = count * 2
  header = struct.pack(
      ">BBHHB", int(slave_id), 0x10, int(start_reg), count, byte_count
  )
  body = b"".join(struct.pack(">H", reg) for reg in reg_values)
  payload = header + body
  crc = crc16_modbus(payload)
  return payload + struct.pack("<H", crc)


def build_read_input_registers_frame(
    slave_id: int, start_reg: int, count: int
) -> bytes:
  """Builds a Modbus RTU function 0x04 (Read Input Registers) frame."""
  if not 0 <= int(slave_id) <= 247:
    raise ValueError(f"Invalid Modbus slave_id: {slave_id}")
  if not 0 <= int(start_reg) <= 0xFFFF:
    raise ValueError(f"Invalid start_reg: {start_reg:#x}")
  if not 1 <= int(count) <= 125:
    raise ValueError(f"Invalid register count: {count}")

  payload = struct.pack(
      ">BBHH", int(slave_id), 0x04, int(start_reg), int(count)
  )
  crc = crc16_modbus(payload)
  return payload + struct.pack("<H", crc)


def _read_sysfs_text(path: Path) -> str:
  try:
    return path.read_text(encoding="utf-8", errors="ignore").strip()
  except OSError:
    return ""


def _read_sysfs_int(path: Path) -> int | None:
  if not path.is_file():
    return None
  try:
    return int(_read_sysfs_text(path))
  except ValueError:
    return None


def _side_hint_from_name(name: str, *, require_robotiq: bool = False) -> str:
  lower = name.lower()
  if require_robotiq and "robotiq" not in lower:
    return ""
  if "left" in lower:
    return "left"
  if "right" in lower:
    return "right"
  return ""


def _inspect_tty_sysfs(
    resolved_tty_path: str, *, sys_tty_root: Path = Path("/sys/class/tty")
) -> dict[str, Any]:
  """Inspects sysfs attributes for a resolved TTY device node."""
  info: dict[str, Any] = {
      "vendor_id": "",
      "product_id": "",
      "serial": "",
      "product": "",
      "latency_timer_ms": None,
      "device_role_hint": "unknown_serial",
  }
  tty_name = Path(resolved_tty_path).name
  dev_dir = sys_tty_root / tty_name / "device"
  if not dev_dir.is_dir():
    target = sys_tty_root / tty_name
    if target.is_symlink():
      try:
        dev_dir = target.resolve() / "device"
      except OSError:
        # Broken or unreadable sysfs symlink; continue with defaults.
        pass

  for lat_cand in (
      dev_dir / "latency_timer",
      sys_tty_root / tty_name / "latency_timer",
  ):
    lat_val = _read_sysfs_int(lat_cand)
    if lat_val is not None:
      info["latency_timer_ms"] = lat_val
      break

  cur: Path | None = None
  try:
    if dev_dir.is_dir():
      cur = dev_dir.resolve()
  except OSError:
    cur = None

  if cur is not None:
    for _ in range(6):
      found_usb_dev = (cur / "idVendor").is_file() or (
          cur / "idProduct"
      ).is_file()
      if not info["vendor_id"] and (cur / "idVendor").is_file():
        info["vendor_id"] = _read_sysfs_text(cur / "idVendor").lower()
      if not info["product_id"] and (cur / "idProduct").is_file():
        info["product_id"] = _read_sysfs_text(cur / "idProduct").lower()
      if not info["serial"] and (cur / "serial").is_file():
        raw_ser = _read_sysfs_text(cur / "serial")
        if ":" not in raw_ser and raw_ser.lower() not in (
            "unknown",
            "none",
            "0",
        ):
          info["serial"] = raw_ser
      if not info["product"] and (cur / "product").is_file():
        info["product"] = _read_sysfs_text(cur / "product")
      if info["latency_timer_ms"] is None:
        info["latency_timer_ms"] = _read_sysfs_int(cur / "latency_timer")
      if found_usb_dev or cur.parent == cur:
        break
      cur = cur.parent

  vid = str(info["vendor_id"]).lower()
  pid = str(info["product_id"]).lower()
  prod_lower = str(info["product"]).lower()
  path_lower = str(resolved_tty_path).lower()

  if (vid == "0403" and pid == "6015") or any(
      tok in prod_lower or tok in path_lower
      for tok in ("robotiq", "rs485", "rs-485")
  ):
    info["device_role_hint"] = "robotiq_rs485"
  elif (vid == "0403" and pid == "6014") or any(
      tok in prod_lower or tok in path_lower
      for tok in ("ft232h", "dynamixel", "u2d2", "teleop")
  ):
    info["device_role_hint"] = "teleop_or_ft232h"
  elif vid == "0403":
    info["device_role_hint"] = "ftdi_serial"
  else:
    info["device_role_hint"] = "unknown_serial"

  return info


def resolve_serial_ports(
    *,
    well_known_symlinks: Sequence[str] = (
        "/dev/robotiq_gripper_left",
        "/dev/robotiq_gripper_right",
    ),
    by_id_dir: Path = Path("/dev/serial/by-id"),
    dev_patterns: Sequence[str] = (
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ),
    sys_tty_root: Path = Path("/sys/class/tty"),
) -> list[dict[str, Any]]:
  """Discovers persistent serial ports with role, latency, and side hints."""
  entries_by_resolved: dict[str, dict[str, Any]] = {}

  for symlink_path in well_known_symlinks:
    p = Path(symlink_path)
    if p.exists():
      try:
        resolved = os.path.realpath(str(p))
      except OSError:
        continue
      entries_by_resolved[resolved] = {
          "path": str(p),
          "resolved_path": resolved,
          "name": p.name,
          "source": "udev_symlink",
          "side_hint": _side_hint_from_name(p.name),
      }

  if by_id_dir.is_dir():
    for entry in sorted(by_id_dir.iterdir(), key=lambda p: p.name):
      try:
        resolved = os.path.realpath(str(entry))
      except OSError:
        continue
      if (
          resolved in entries_by_resolved
          and entries_by_resolved[resolved]["source"] == "udev_symlink"
      ):
        continue
      entries_by_resolved[resolved] = {
          "path": str(entry),
          "resolved_path": resolved,
          "name": entry.name,
          "source": "by-id",
          "side_hint": _side_hint_from_name(entry.name, require_robotiq=True),
      }

  for pattern in dev_patterns:
    for raw_path in sorted(glob.glob(pattern)):
      try:
        resolved = os.path.realpath(raw_path)
      except OSError:
        continue
      if resolved in entries_by_resolved:
        continue
      entries_by_resolved[resolved] = {
          "path": raw_path,
          "resolved_path": resolved,
          "name": Path(raw_path).name,
          "source": "dev",
          "side_hint": "",
      }

  results: list[dict[str, Any]] = []
  for resolved, item in entries_by_resolved.items():
    sys_info = _inspect_tty_sysfs(resolved, sys_tty_root=sys_tty_root)
    role_hint = sys_info["device_role_hint"]
    if "robotiq" in item["name"].lower() or "robotiq" in item["path"].lower():
      role_hint = "robotiq_rs485"

    serial_rw_ok = (
        os.access(resolved, os.R_OK | os.W_OK)
        if Path(resolved).exists()
        else None
    )
    results.append({
        "path": item["path"],
        "resolved_path": resolved,
        "name": item["name"],
        "source": item["source"],
        "vendor_id": sys_info["vendor_id"],
        "product_id": sys_info["product_id"],
        "serial": sys_info["serial"],
        "latency_timer_ms": sys_info["latency_timer_ms"],
        "device_role_hint": role_hint,
        "side_hint": item["side_hint"],
        "serial_rw_ok": serial_rw_ok,
    })

  role_order = {
      "robotiq_rs485": 0,
      "ftdi_serial": 1,
      "unknown_serial": 1,
      "teleop_or_ft232h": 2,
  }
  results.sort(
      key=lambda x: (
          role_order.get(x["device_role_hint"], 1),
          x["resolved_path"],
      )
  )
  return results


def select_gripper_ports(
    serial_ports: Sequence[dict[str, Any]],
    configured_ports: dict[str, str] | None = None,
) -> dict[str, Any]:
  """Selects active gripper ports, warns on mismatches, and adds advisories."""
  configured = configured_ports or {}
  chosen = {
      side: str(configured.get(side, "") or "") for side in ("left", "right")
  }
  robotiq_candidates = [
      p["path"]
      for p in serial_ports
      if p.get("device_role_hint") == "robotiq_rs485"
  ]
  ignored_teleop_ports = [
      p["path"]
      for p in serial_ports
      if p.get("device_role_hint") == "teleop_or_ft232h"
  ]
  warnings: list[str] = []
  advisories: list[str] = []

  for side, cfg_path in chosen.items():
    if cfg_path:
      resolved_cfg = os.path.realpath(cfg_path)
      matched_sp = next(
          (
              sp
              for sp in serial_ports
              if sp.get("path") == cfg_path
              or sp.get("resolved_path") == resolved_cfg
          ),
          None,
      )
      if (
          matched_sp
          and matched_sp.get("device_role_hint") == "teleop_or_ft232h"
      ):
        warnings.append(
            f"Configured {side} gripper port '{cfg_path}' resolves to FT232H / "
            "teleop adapter (0403:6014), not a Robotiq RS-485 adapter "
            "(0403:6015)."
        )

  for side in ("left", "right"):
    if not chosen[side]:
      side_cands = [
          p["path"]
          for p in serial_ports
          if p.get("device_role_hint") == "robotiq_rs485"
          and p.get("side_hint") == side
      ]
      if len(side_cands) == 1:
        chosen[side] = side_cands[0]

  if not chosen["left"] or not chosen["right"]:
    fallback_roles = (
        ("robotiq_rs485",)
        if robotiq_candidates
        else ("robotiq_rs485", "ftdi_serial")
    )
    used_paths = {p for p in chosen.values() if p}
    used_resolved = {os.path.realpath(p) for p in used_paths}
    unassigned = [
        p
        for p in serial_ports
        if p.get("device_role_hint") in fallback_roles
        and p.get("path") not in used_paths
        and os.path.realpath(str(p.get("resolved_path") or p.get("path", "")))
        not in used_resolved
    ]
    unassigned.sort(
        key=lambda p: (
            str(p.get("serial") or ""),
            str(p.get("resolved_path") or p.get("path") or ""),
        )
    )
    auto_assigned: list[str] = []
    for side in ("left", "right"):
      if not chosen[side] and unassigned:
        entry = unassigned.pop(0)
        chosen[side] = str(entry["path"])
        auto_assigned.append(
            f"{side}={chosen[side]} (serial={entry.get('serial') or 'n/a'})"
        )
    if auto_assigned:
      advisories.append(
          "Auto-assigned Robotiq RS-485 gripper port(s) without udev "
          "side_hint: "
          + ", ".join(auto_assigned)
          + ". Verify left/right gripper mapping in Stage 2 or add "
          "/dev/robotiq_gripper_{left,right} udev symlinks."
      )

  left, right = chosen["left"], chosen["right"]
  if (not left or not right) and robotiq_candidates and ignored_teleop_ports:
    advisories.append(
        f"Detected teleop / FT232H serial port(s) {ignored_teleop_ports} "
        "(0403:6014) ignored for gripper control; available Robotiq "
        f"candidate(s): {robotiq_candidates}."
    )

  selected_resolved = {os.path.realpath(p) for p in (left, right) if p}
  for p in serial_ports:
    lat = p.get("latency_timer_ms")
    if lat is not None and lat > 10:
      is_candidate = p.get("device_role_hint") == "robotiq_rs485"
      is_selected = (
          p.get("path") in (left, right)
          or p.get("resolved_path") in selected_resolved
      )
      if is_candidate or is_selected:
        tty_name = Path(p.get("resolved_path") or p["path"]).name
        advisories.append(
            f"Serial port {p['path']} latency_timer is {lat} ms (> 10 ms). "
            "Recommend lowering to 10 ms (or 1 ms) to reduce Modbus polling "
            f"delay: echo 10 | sudo tee "
            f"/sys/bus/usb-serial/devices/{tty_name}/latency_timer"
        )

  return {
      "left": left,
      "right": right,
      "robotiq_candidates": robotiq_candidates,
      "ignored_teleop_ports": ignored_teleop_ports,
      "warnings": warnings,
      "advisories": advisories,
  }


@dataclass
class RobotiqStatus:
  """Decoded 6-byte status block from Robotiq registers 0x07D0..0x07D2."""

  activated: bool
  goto_active: bool
  obj_status: int
  fault: int
  requested_pos_raw: int
  measured_pos_raw: int
  current_ma: float

  @property
  def gOBJ(self) -> int:
    """Modbus `gOBJ` alias for `obj_status` (0=moving, 1=open, 2=close)."""
    return self.obj_status

  @property
  def position_ratio(self) -> float:
    """Measured closure ratio in [0.0, 1.0] (0.0 = open, 1.0 = closed)."""
    return max(0.0, min(1.0, float(self.measured_pos_raw) / 255.0))

  @property
  def is_object_grasped(self) -> bool:
    """True when jaws stalled on an object during closing before full stroke."""
    return bool(
        self.obj_status == 2 and self.position_ratio < EMPTY_CLOSE_MIN_RATIO
    )

  @property
  def is_empty_closed(self) -> bool:
    """True when jaws closed without grasping an object (>= 0.80 stroke)."""
    return bool(
        self.position_ratio >= EMPTY_CLOSE_MIN_RATIO
        and self.obj_status in (2, 3)
    )

  @classmethod
  def from_status_bytes(cls, status_bytes: bytes) -> RobotiqStatus:
    """Decodes 6 register data bytes (0x07D0..0x07D2) into a RobotiqStatus."""
    if len(status_bytes) < 6:
      raise ValueError(
          f"Expected 6 Robotiq status bytes, got {len(status_bytes)}"
      )
    b0 = status_bytes[0]
    b2, b3, b4, b5 = status_bytes[2:6]
    g_act = b0 & 0x01
    g_gto = (b0 >> 3) & 0x01
    g_sta = (b0 >> 4) & 0x03
    g_obj = (b0 >> 6) & 0x03
    # Per Robotiq 2F-85 Modbus spec: gSTA==0 is reset, gSTA==1 is activation in
    # progress, and gSTA==3 indicates completed activation.
    activated = bool(g_act == 1 and g_sta == 3)
    return cls(
        activated=activated,
        goto_active=bool(g_gto == 1),
        obj_status=int(g_obj),
        fault=int(b2 & 0x0F),
        requested_pos_raw=int(b3),
        measured_pos_raw=int(b4),
        current_ma=float(b5) * 10.0,
    )

  def to_dict(self) -> dict[str, Any]:
    """Serializes status fields and derived grasp indicators to a plain dict."""
    return {
        "activated": self.activated,
        "goto_active": self.goto_active,
        "obj_status": self.obj_status,
        "gOBJ": self.obj_status,
        "fault": self.fault,
        "requested_pos_raw": self.requested_pos_raw,
        "measured_pos_raw": self.measured_pos_raw,
        "current_ma": self.current_ma,
        "position_ratio": self.position_ratio,
        "is_object_grasped": self.is_object_grasped,
        "is_empty_closed": self.is_empty_closed,
    }


def _open_default_serial(port: str, baudrate: int, timeout_s: float) -> Any:
  """Opens an 8N1 RS-485 connection via pyserial."""
  if serial is None:
    raise RuntimeError(
        "pyserial is required for Robotiq RS-485 hardware communication. "
        "Install it with `pip install pyserial`."
    )
  return serial.Serial(
      port=port,
      baudrate=baudrate,
      bytesize=8,
      parity="N",
      stopbits=1,
      timeout=timeout_s,
      write_timeout=timeout_s,
  )


class RobotiqGripper:
  """Thread-safe Modbus RTU driver for a single Robotiq 2F-85/2F-140/Hand-E."""

  def __init__(
      self,
      port: str,
      *,
      baudrate: int = DEFAULT_BAUDRATE,
      slave_id: int = DEFAULT_SLAVE_ID,
      timeout_s: float = 0.2,
      serial_factory: Any = None,
      dry_run: bool = False,
  ) -> None:
    """Initializes Modbus RTU connection settings for a Robotiq gripper."""
    self.port = str(port)
    self.baudrate = int(baudrate)
    self.slave_id = int(slave_id)
    self.timeout_s = float(timeout_s)
    self._serial_factory = serial_factory
    self.dry_run = bool(dry_run)
    self._serial: Any = None
    self._lock = threading.Lock()
    self._custom_transport = serial_factory is not None

  def connect(self) -> None:
    """Opens the underlying serial transport unless in dry-run mode."""
    with self._lock:
      if self.dry_run or self._serial is not None:
        return
      if self._serial_factory is None:
        self._serial = _open_default_serial(
            self.port, self.baudrate, self.timeout_s
        )
        return
      if hasattr(self._serial_factory, "write") and hasattr(
          self._serial_factory, "read"
      ):
        self._serial = self._serial_factory
        return
      self._serial = self._serial_factory(
          self.port, baudrate=self.baudrate, timeout=self.timeout_s
      )

  def close(self) -> None:
    """Closes the serial port if open."""
    with self._lock:
      if self._serial is not None:
        close_fn = getattr(self._serial, "close", None)
        if callable(close_fn):
          close_fn()
        self._serial = None

  def __enter__(self) -> RobotiqGripper:
    """Connects to the gripper when entering a context manager."""
    self.connect()
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    """Closes the serial port when exiting a context manager."""
    self.close()

  def _ensure_connected_unlocked(self) -> Any:
    if self._serial is None:
      raise RuntimeError(
          f"Robotiq gripper on port {self.port!r} is not connected; "
          "call connect() first."
      )
    return self._serial

  def _read_exact_unlocked(self, ser: Any, expected_len: int) -> bytes:
    buf = bytearray()
    deadline = time.monotonic() + max(self.timeout_s, 0.05)
    while len(buf) < expected_len:
      chunk = ser.read(expected_len - len(buf))
      if chunk:
        buf.extend(chunk)
        continue
      if time.monotonic() >= deadline:
        break
    return bytes(buf)

  def _transact_unlocked(self, request: bytes, expected_len: int) -> bytes:
    ser = self._ensure_connected_unlocked()
    reset_in = getattr(ser, "reset_input_buffer", None)
    if callable(reset_in):
      reset_in()
    ser.write(request)
    flush_fn = getattr(ser, "flush", None)
    if callable(flush_fn):
      flush_fn()
    response = self._read_exact_unlocked(ser, expected_len)
    if len(response) < expected_len:
      raise TimeoutError(
          f"Short Modbus RTU response on {self.port}: expected "
          f"{expected_len} bytes, received {len(response)} bytes "
          f"({response.hex()})"
      )
    if response[0] != self.slave_id:
      raise RuntimeError(
          f"Unexpected Modbus slave ID {response[0]} "
          f"(expected {self.slave_id})"
      )
    if response[1] & 0x80:
      exc_code = response[2] if len(response) > 2 else -1
      raise RuntimeError(
          f"Modbus exception response fn={response[1]:#x} code={exc_code:#x}"
      )
    payload, rx_crc_bytes = response[:-2], response[-2:]
    rx_crc = struct.unpack("<H", rx_crc_bytes)[0]
    calc_crc = crc16_modbus(payload)
    if rx_crc != calc_crc:
      raise RuntimeError(
          f"Modbus CRC mismatch on {self.port}: received {rx_crc:#06x}, "
          f"computed {calc_crc:#06x}"
      )
    return response

  def _require_live_connection(self, action: str) -> None:
    if self.dry_run:
      raise RuntimeError(f"Robotiq {action} is blocked in dry_run mode.")
    if self._serial is None:
      self.connect()

  def _write_action_registers(
      self, reg0: int, reg1: int, reg2: int
  ) -> None:
    self._require_live_connection("actuation")
    frame = build_write_registers_frame(
        self.slave_id, ACTION_REG_START, (reg0, reg1, reg2)
    )
    with self._lock:
      resp = self._transact_unlocked(frame, expected_len=8)
    if resp[1] != 0x10:
      raise RuntimeError(
          f"Unexpected Modbus write function code {resp[1]:#x} "
          "(expected 0x10)"
      )

  def read_status(self) -> RobotiqStatus:
    """Reads 3 input registers starting at 0x07D0 and returns RobotiqStatus."""
    if self.dry_run:
      return RobotiqStatus(
          activated=False,
          goto_active=False,
          obj_status=0,
          fault=0,
          requested_pos_raw=0,
          measured_pos_raw=0,
          current_ma=0.0,
      )
    if self._serial is None:
      self.connect()
    frame = build_read_input_registers_frame(
        self.slave_id, STATUS_REG_START, count=3
    )
    with self._lock:
      resp = self._transact_unlocked(frame, expected_len=11)
    if resp[1] != 0x04:
      raise RuntimeError(
          f"Unexpected Modbus read function code {resp[1]:#x} (expected 0x04)"
      )
    if resp[2] != 6:
      raise RuntimeError(
          f"Unexpected Modbus byte count {resp[2]} (expected 6)"
      )
    return RobotiqStatus.from_status_bytes(resp[3:9])

  def activate(
      self, *, reset_first: bool = False, timeout_s: float = 3.0
  ) -> RobotiqStatus:
    """Activates the Robotiq gripper (triggers jaw calibration if needed)."""
    self._require_live_connection("activation")

    if reset_first:
      self._write_action_registers(0x0000, 0x0000, 0x0000)
      if not self._custom_transport:
        time.sleep(0.05)

    self._write_action_registers(0x0100, 0x0000, 0x0000)
    deadline = time.monotonic() + max(float(timeout_s), 0.05)
    poll_dt = 0.005 if self._custom_transport else 0.05
    status = self.read_status()
    while not status.activated and time.monotonic() < deadline:
      time.sleep(poll_dt)
      status = self.read_status()
    if not status.activated:
      raise TimeoutError(
          f"Robotiq gripper on {self.port} did not finish activation within "
          f"{timeout_s:.2f} s (last status={status.to_dict()})"
      )
    return status

  def move_normalized(
      self,
      closure_0_to_1: float,
      *,
      speed: float = 0.5,
      force: float = 0.5,
      wait: bool = True,
      timeout_s: float = 1.5,
  ) -> RobotiqStatus:
    """Commands normalized jaw closure in [0.0, 1.0] (0.0=open, 1.0=closed)."""
    self._require_live_connection("actuation")

    pre_status = self.read_status()
    if not pre_status.activated:
      pre_status = self.activate()

    pos_raw = closure_to_raw(closure_0_to_1)
    speed_raw = closure_to_raw(speed)
    force_raw = closure_to_raw(force)

    reg0 = 0x0900  # rACT=1, rGTO=1
    reg1 = pos_raw & 0xFF
    reg2 = ((speed_raw & 0xFF) << 8) | (force_raw & 0xFF)
    write_ts = time.monotonic()
    self._write_action_registers(reg0, reg1, reg2)

    if not wait:
      return self.read_status()

    if not self._custom_transport:
      time.sleep(DEADTIME_S)

    status = self.read_status()
    deadline = time.monotonic() + max(float(timeout_s), DEADTIME_S + 0.05)
    poll_dt = 0.005 if self._custom_transport else 0.03
    while time.monotonic() < deadline:
      in_deadtime = (time.monotonic() - write_ts) < DEADTIME_S
      stale_pre_motion = (
          in_deadtime
          and abs(pos_raw - pre_status.measured_pos_raw) > 3
          and abs(status.measured_pos_raw - pre_status.measured_pos_raw) <= 1
      )
      if (
          status.requested_pos_raw == pos_raw
          and status.obj_status in (1, 2, 3)
          and not stale_pre_motion
      ):
        return status
      if (
          status.requested_pos_raw == pos_raw
          and abs(status.measured_pos_raw - pos_raw) <= 3
          and not status.goto_active
      ):
        return status
      time.sleep(poll_dt)
      status = self.read_status()
    return status

  def open(
      self, *, speed: float = 0.5, force: float = 0.5, wait: bool = True
  ) -> RobotiqStatus:
    """Opens the gripper jaws to normalized position 0.0."""
    return self.move_normalized(0.0, speed=speed, force=force, wait=wait)

  def close_jaws(
      self, *, speed: float = 0.5, force: float = 0.5, wait: bool = True
  ) -> RobotiqStatus:
    """Closes the gripper jaws toward normalized position 1.0."""
    return self.move_normalized(1.0, speed=speed, force=force, wait=wait)
