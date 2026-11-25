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

"""Configuration schema, PREEMPT_RT detection, and output helpers for FR3 Duo.

Provides:
- Config dataclasses (`ArmConfig`, `ArmsConfig`, `CamerasConfig`, `RobotConfig`)
- Geometry constants & helpers (`BASE_WORLD_Z_M`, `BRACKET_UNDERSIDE_FRONT_X_M`,
  `table_box_bounds`, `validate_robot_config_complete`)
- Config helpers (`DEFAULT_CONFIG_PATH`, `default_robot_config`,
  `normalize_robot_config`, `load_robot_config`, `save_robot_config`)
- PREEMPT_RT auto-detection and `pylibfranka` `RealtimeConfig` resolution
- Stage output path resolution and RGB PNG / MP4 writer utilities
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

import fr3_camera_controller
import fr3_mjcf
import robotiq_gripper_controller

DEFAULT_CONFIG_PATH = "config/robot_config.json"
MIN_INTER_ARM_CLEARANCE_M = 0.030
BASE_WORLD_Z_M = 0.163887
BRACKET_UNDERSIDE_FRONT_X_M = fr3_mjcf.BRACKET_UNDERSIDE_FRONT_X_M


@dataclass
class ArmConfig:
  """Configuration for one 7-DOF FR3 arm and its Robotiq RS-485 gripper."""

  ip: str = "172.16.0.2"
  enabled: bool = True
  rest_pose_rad: list[float] = field(
      default_factory=lambda: list(fr3_mjcf.DEFAULT_REST_POSE_RAD["left"])
  )
  gripper_port: str = ""
  gripper_slave_id: int = robotiq_gripper_controller.DEFAULT_SLAVE_ID
  gripper_baudrate: int = robotiq_gripper_controller.DEFAULT_BAUDRATE


@dataclass
class ArmsConfig:
  """Pair of left and right arm configurations."""

  left: ArmConfig = field(
      default_factory=lambda: ArmConfig(
          ip="172.16.0.2",
          rest_pose_rad=list(fr3_mjcf.DEFAULT_REST_POSE_RAD["left"]),
      )
  )
  right: ArmConfig = field(
      default_factory=lambda: ArmConfig(
          ip="172.16.1.2",
          rest_pose_rad=list(fr3_mjcf.DEFAULT_REST_POSE_RAD["right"]),
      )
  )

  def __getitem__(self, side: str) -> ArmConfig:
    """Returns the `ArmConfig` for `'left'` or `'right'`."""
    if side == "left":
      return self.left
    if side == "right":
      return self.right
    raise KeyError(side)

  def items(self) -> tuple[tuple[str, ArmConfig], ...]:
    """Returns `((side, ArmConfig), ...)` for both arms."""
    return (("left", self.left), ("right", self.right))


@dataclass
class CamerasConfig:
  """Physical camera configurations (1 stereo head ZED + 2 wrist RealSense)."""

  head_stereo: fr3_camera_controller.CameraStreamSpec = field(
      default_factory=lambda: fr3_camera_controller.CameraStreamSpec(
          role="head_stereo", width=2560, height=720
      )
  )
  wrist_camera_left: fr3_camera_controller.CameraStreamSpec = field(
      default_factory=lambda: fr3_camera_controller.CameraStreamSpec(
          role="wrist_camera_left"
      )
  )
  wrist_camera_right: fr3_camera_controller.CameraStreamSpec = field(
      default_factory=lambda: fr3_camera_controller.CameraStreamSpec(
          role="wrist_camera_right"
      )
  )

  def as_spec_map(self) -> dict[str, fr3_camera_controller.CameraStreamSpec]:
    """Returns a role-keyed dictionary of camera stream specifications."""
    return {
        "head_stereo": self.head_stereo,
        "wrist_camera_left": self.wrist_camera_left,
        "wrist_camera_right": self.wrist_camera_right,
    }

  def __getitem__(self, role: str) -> fr3_camera_controller.CameraStreamSpec:
    """Returns the `CameraStreamSpec` for a canonical camera role."""
    if role in ("head_stereo", "wrist_camera_left", "wrist_camera_right"):
      return getattr(self, role)
    raise KeyError(role)


@dataclass
class RobotConfig:
  """Canonical configuration schema for a Franka Research 3 Duo cell."""

  platform: str = "franka_research_3_duo"
  base_height_m: float = 0.913887
  table_height_m: float = 0.75
  table_front_offset_m: float = 0.25
  table_width_m: float = 1.22
  table_depth_m: float = 0.74
  geometry_confirmed: bool = False
  min_inter_arm_clearance_m: float = MIN_INTER_ARM_CLEARANCE_M
  tcp_offset_z_m: float = 0.145
  realtime_mode: str = "auto"
  max_joint_speed_deg_s: float | None = None
  arms: ArmsConfig = field(default_factory=ArmsConfig)
  cameras: CamerasConfig = field(default_factory=CamerasConfig)

  @property
  def z_table_m(self) -> float:
    """Derived tabletop Z in Duo `base` frame from floor heights."""
    delta_h = float(self.base_height_m) - float(self.table_height_m)
    return round(BASE_WORLD_Z_M - delta_h, 6)

  def table_box_bounds(self) -> tuple[list[float], list[float]]:
    """Returns `(center, half_extents)` for the configured tabletop box."""
    table_thickness = 0.04
    x_front = BRACKET_UNDERSIDE_FRONT_X_M + float(self.table_front_offset_m)
    center = [
        round(x_front + 0.5 * float(self.table_depth_m), 6),
        0.0,
        round(float(self.z_table_m) - 0.5 * table_thickness, 6),
    ]
    half_extents = [
        round(0.5 * float(self.table_depth_m), 6),
        round(0.5 * float(self.table_width_m), 6),
        round(0.5 * table_thickness, 6),
    ]
    return center, half_extents

  def to_dict(self) -> dict[str, Any]:
    """Serializes to a clean JSON-ready dictionary without redundant nulls."""
    data = asdict(self)
    if data.get("max_joint_speed_deg_s") is None:
      data.pop("max_joint_speed_deg_s", None)
    return data


def _overlay_arm_config(arm: ArmConfig, raw: Mapping[str, Any]) -> None:
  if "ip" in raw:
    arm.ip = "" if raw["ip"] is None else str(raw["ip"]).strip()
  if "enabled" in raw and raw["enabled"] is not None:
    arm.enabled = bool(raw["enabled"])
  raw_rest = raw.get("rest_pose_rad")
  if (
      isinstance(raw_rest, (Sequence, np.ndarray))
      and not isinstance(raw_rest, (str, bytes))
      and len(raw_rest) == 7
  ):
    arm.rest_pose_rad = [float(x) for x in raw_rest]
  if "gripper_port" in raw and raw["gripper_port"] is not None:
    arm.gripper_port = str(raw["gripper_port"])
  if "gripper_slave_id" in raw and raw["gripper_slave_id"] is not None:
    arm.gripper_slave_id = int(raw["gripper_slave_id"])
  if "gripper_baudrate" in raw and raw["gripper_baudrate"] is not None:
    arm.gripper_baudrate = int(raw["gripper_baudrate"])


def _overlay_camera_spec(
    base: fr3_camera_controller.CameraStreamSpec, raw: Mapping[str, Any]
) -> fr3_camera_controller.CameraStreamSpec:
  merged = asdict(base)
  merged.update({k: v for k, v in raw.items() if v is not None})
  return fr3_camera_controller.CameraStreamSpec.from_mapping(
      base.role, merged
  )


def default_robot_config() -> RobotConfig:
  """Returns the canonical default `RobotConfig` for an FR3 Duo cell."""
  return RobotConfig()


def normalize_robot_config(
    updates: RobotConfig | Mapping[str, Any] | None = None,
) -> RobotConfig:
  """Merges partial `updates` (or copies `RobotConfig`) onto defaults.

  `z_table_m` is derived from `base_height_m` and `table_height_m`. Legacy
  mappings supplying only `z_table_m` (without raw floor heights) adjust
  `base_height_m` so the derived `z_table_m` property matches cleanly.
  """
  if isinstance(updates, RobotConfig):
    return copy.deepcopy(updates)

  raw = dict(updates) if isinstance(updates, Mapping) else {}
  cfg = default_robot_config()

  for key, cast in (
      ("platform", str),
      ("realtime_mode", str),
      ("base_height_m", float),
      ("table_height_m", float),
      ("table_front_offset_m", float),
      ("table_width_m", float),
      ("table_depth_m", float),
      ("min_inter_arm_clearance_m", float),
      ("tcp_offset_z_m", float),
      ("geometry_confirmed", bool),
      ("max_joint_speed_deg_s", float),
  ):
    if raw.get(key) is not None:
      setattr(cfg, key, cast(raw[key]))

  has_raw_heights = (
      raw.get("base_height_m") is not None
      or raw.get("table_height_m") is not None
  )
  if raw.get("z_table_m") is not None and not has_raw_heights:
    cfg.base_height_m = round(
        BASE_WORLD_Z_M + float(cfg.table_height_m) - float(raw["z_table_m"]), 6
    )

  raw_arms = raw.get("arms")
  if isinstance(raw_arms, Mapping):
    for side in ("left", "right"):
      raw_arm = raw_arms.get(side)
      if isinstance(raw_arm, Mapping):
        _overlay_arm_config(cfg.arms[side], raw_arm)

  raw_cams = raw.get("cameras")
  if isinstance(raw_cams, Mapping):
    for role in ("head_stereo", "wrist_camera_left", "wrist_camera_right"):
      raw_cam = raw_cams.get(role)
      if isinstance(raw_cam, fr3_camera_controller.CameraStreamSpec):
        setattr(cfg.cameras, role, copy.deepcopy(raw_cam))
      elif isinstance(raw_cam, Mapping):
        setattr(
            cfg.cameras,
            role,
            _overlay_camera_spec(cfg.cameras[role], raw_cam),
        )

  return cfg


def table_box_bounds(
    config: RobotConfig | Mapping[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
  """Returns `(center, half_extents)` for the configured tabletop box."""
  cfg = (
      config
      if isinstance(config, RobotConfig)
      else normalize_robot_config(config)
  )
  return cfg.table_box_bounds()


def _is_camera_bound(cam: fr3_camera_controller.CameraStreamSpec) -> bool:
  return bool((cam.device_path or "").strip() or (cam.serial or "").strip())


def _is_valid_geom_number(value: Any, *, allow_zero: bool) -> bool:
  try:
    fval = float(value)
  except (TypeError, ValueError):
    return False
  if not math.isfinite(fval):
    return False
  return fval >= 0.0 if allow_zero else fval > 0.0


def validate_robot_config_complete(
    config: RobotConfig | Mapping[str, Any],
) -> list[str]:
  """Returns missing or unconfirmed config fields required for live hardware."""
  cfg = normalize_robot_config(config)
  missing: list[str] = []

  for side in ("left", "right"):
    arm_cfg = cfg.arms[side]
    if not str(arm_cfg.ip or "").strip():
      missing.append(f"arms.{side}.ip")
    if not str(arm_cfg.gripper_port or "").strip():
      missing.append(f"arms.{side}.gripper_port")

  for role in ("head_stereo", "wrist_camera_left", "wrist_camera_right"):
    if not _is_camera_bound(cfg.cameras[role]):
      missing.append(f"cameras.{role}")

  geom_rules = (
      ("base_height_m", False),
      ("table_height_m", True),
      ("table_front_offset_m", True),
      ("table_width_m", False),
      ("table_depth_m", False),
      ("tcp_offset_z_m", False),
  )
  for field_name, allow_zero in geom_rules:
    if not _is_valid_geom_number(
        getattr(cfg, field_name), allow_zero=allow_zero
    ):
      missing.append(field_name)

  if not bool(cfg.geometry_confirmed):
    missing.append("geometry_confirmed")

  return missing


def load_robot_config(path: str | Path | None = None) -> RobotConfig:
  """Loads `robot_config.json`, merging values onto `default_robot_config()`."""
  target = Path(DEFAULT_CONFIG_PATH) if path is None else Path(path)
  if not target.is_file():
    if path is None or target.as_posix().endswith(DEFAULT_CONFIG_PATH):
      return default_robot_config()
    raise FileNotFoundError(f"Robot config file not found: {target}")
  loaded = json.loads(target.read_text(encoding="utf-8"))
  if not isinstance(loaded, dict):
    raise ValueError(f"Expected JSON object in {target}, got {type(loaded)!r}")
  return normalize_robot_config(loaded)


def save_robot_config(
    config: RobotConfig | Mapping[str, Any], path: str | Path
) -> None:
  """Writes `config` as formatted JSON."""
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  norm = normalize_robot_config(config)
  serialized = json.dumps(norm.to_dict(), indent=2, sort_keys=False) + "\n"
  target.write_text(serialized, encoding="utf-8")


def resolve_stage_output_dir(
    stage_name: str, explicit_dir: Path | None, *, dry_run: bool
) -> Path | None:
  """Resolves stage log directory while keeping source trees clean."""
  if explicit_dir is not None:
    return explicit_dir
  if dry_run:
    return None
  cwd = Path.cwd()
  if os.environ.get("TEST_TMPDIR") or any(
      (p / "WORKSPACE").is_file() for p in (cwd, *cwd.parents)
  ):
    return Path(tempfile.gettempdir()) / "fr3_duo_logs" / stage_name
  return Path("logs") / stage_name


def save_rgb_png(rgb: np.ndarray, out_path: Path) -> None:
  """Saves an `(H, W, 3)` uint8 RGB image to `out_path` via OpenCV."""
  cv2 = fr3_camera_controller.require_cv2()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  arr = np.ascontiguousarray(rgb, dtype=np.uint8)
  bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
  if not cv2.imwrite(str(out_path), bgr):
    raise RuntimeError(f"cv2.imwrite failed for {out_path}")


def open_mp4_writer(
    mp4_path: Path, width: int, height: int, fps: float
) -> Any:
  """Opens an OpenCV `cv2.VideoWriter` for `.mp4` output."""
  cv2 = fr3_camera_controller.require_cv2()
  mp4_path.parent.mkdir(parents=True, exist_ok=True)
  for fourcc_str in ("mp4v", "avc1"):
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*fourcc_str),
        float(fps),
        (int(width), int(height)),
    )
    if writer.isOpened():
      return writer
    writer.release()
  raise RuntimeError(f"Failed to open cv2.VideoWriter for {mp4_path}")


def detect_preempt_rt() -> bool:
  """Returns True when running on a Linux `PREEMPT_RT` real-time kernel."""
  rt_sysfs = Path("/sys/kernel/realtime")
  try:
    if (
        rt_sysfs.is_file()
        and rt_sysfs.read_text(encoding="utf-8").strip() == "1"
    ):
      return True
  except OSError:
    # Non-Linux or restricted sysfs environments fall back to uname inspection.
    pass
  try:
    uname_ver = os.uname().version.upper()
    if "PREEMPT_RT" in uname_ver or "PREEMPT RT" in uname_ver:
      return True
  except AttributeError:
    # `os.uname` is absent on non-POSIX platforms.
    pass
  return False


def resolve_realtime_config(
    mode: str = "auto", *, is_preempt_rt: bool | None = None
) -> dict[str, Any]:
  """Resolves `pylibfranka` `RealtimeConfig` and safe joint speed/accel caps."""
  normalized = mode.strip().lower()
  if normalized not in ("auto", "enforce", "ignore"):
    raise ValueError(
        f"Invalid realtime mode={mode!r}; "
        "expected 'auto', 'enforce', or 'ignore'"
    )
  rt_active = (
      detect_preempt_rt() if is_preempt_rt is None else bool(is_preempt_rt)
  )

  if normalized == "enforce" and not rt_active:
    raise RuntimeError(
        "PREEMPT_RT kernel is required for `--realtime enforce`, but the "
        "current kernel is not real-time patched. Boot a PREEMPT_RT kernel "
        "or pass `--realtime auto`."
    )
  if normalized == "enforce":
    rt_cfg, max_speed, max_acc = "kEnforce", 45.0, 5.0
  elif normalized == "auto" and rt_active:
    rt_cfg, max_speed, max_acc = "kEnforce", 30.0, 3.5
  else:
    rt_cfg, max_speed, max_acc = "kIgnore", 15.0, 2.0
  return {
      "mode": normalized,
      "is_preempt_rt": rt_active,
      "realtime_config": rt_cfg,
      "max_joint_speed_deg_s": max_speed,
      "max_joint_acc_rad_s2": max_acc,
  }
