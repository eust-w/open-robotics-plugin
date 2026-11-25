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

"""Linux V4L2 / OpenCV camera rig layer for the Franka Research 3 Duo platform.

Supports:
- ZED Mini side-by-side stereo UVC capture (`2560x720` -> `head_camera_left` &
  `head_camera_right` `1280x720` RGB views) + factory `.conf` rectification.
- Dual wrist RealSense D405 RGB UVC capture (`wrist_camera_left` &
  `wrist_camera_right`) resolved by USB serial or `/dev/v4l/by-id/`.
- In-process MuJoCo rendering via `Fr3DuoRobot.render_sim_cameras(...)`
  and `--dry-run` synthetic frames.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import configparser
from dataclasses import asdict, dataclass, field, fields
import os
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np

try:
  import cv2  # pytype: disable=import-error
except ImportError:
  cv2: Any = None

try:
  import grp  # pytype: disable=import-error
except ImportError:
  grp: Any = None

CANONICAL_CAMERAS = (
    "head_camera_left",
    "head_camera_right",
    "wrist_camera_left",
    "wrist_camera_right",
)

ZED_VENDOR_ID = "2b03"
INTEL_VENDOR_ID = "8086"


def require_cv2() -> Any:
  """Returns the imported `cv2` module or raises `RuntimeError` if absent."""
  if cv2 is None:
    raise RuntimeError(
        "opencv-python is required for camera/video operations. "
        "Install with: pip install opencv-python"
    )
  return cv2


@dataclass
class CameraFrame:
  """Single RGB camera frame with monotonic capture timestamp."""

  name: str
  rgb: np.ndarray
  timestamp_s: float
  width: int
  height: int
  backend: str

  def age_s(self, now_s: float | None = None) -> float:
    """Returns frame age in seconds relative to `now_s` or monotonic time."""
    ref_s = time.monotonic() if now_s is None else float(now_s)
    return float(ref_s - float(self.timestamp_s))

  @property
  def brightness_mean(self) -> float:
    """Mean pixel intensity in [0.0, 255.0]."""
    return float(np.mean(self.rgb)) if self.rgb.size else 0.0

  @property
  def pixel_std(self) -> float:
    """Standard deviation of pixel intensities across the RGB frame."""
    return float(np.std(self.rgb)) if self.rgb.size else 0.0


@dataclass
class CameraStreamSpec:
  """Configuration for one logical or stereo camera stream."""

  role: str
  serial: str = ""
  device_path: str = ""
  width: int = 1280
  height: int = 720
  fps: int = 30
  zed_conf_path: str = ""

  @classmethod
  def from_mapping(
      cls, role: str, data: CameraStreamSpec | Mapping[str, Any]
  ) -> CameraStreamSpec:
    """Builds a `CameraStreamSpec` from an existing spec or mapping."""
    if isinstance(data, CameraStreamSpec):
      return data
    return cls(
        role=str(data.get("role", role)),
        serial=str(data.get("serial", "")),
        device_path=str(data.get("device_path", "")),
        width=int(data.get("width", 1280)),
        height=int(data.get("height", 720)),
        fps=int(data.get("fps", 30)),
        zed_conf_path=str(data.get("zed_conf_path", "")),
    )


class _DictCompatMixin:
  """Read-only mapping accessors delegating to fields or `to_dict()`."""

  _dict_delegate_to_dict: bool = False

  def to_dict(self) -> dict[str, Any]:
    """Returns a dictionary representation of this dataclass."""
    return asdict(self)  # type: ignore[arg-type]

  def _field_names(self) -> set[str]:
    return {f.name for f in fields(self)}  # type: ignore[arg-type]

  def __getitem__(self, key: str) -> Any:
    if not self._dict_delegate_to_dict and key in self._field_names():
      val = getattr(self, key)
      if key == "error" and val is None:
        raise KeyError(key)
      return val
    return self.to_dict()[key]

  def get(self, key: str, default: Any = None) -> Any:
    """Returns `self[key]` if present, otherwise `default`."""
    try:
      return self[key]
    except KeyError:
      return default

  def __contains__(self, key: object) -> bool:
    if not isinstance(key, str):
      return False
    if not self._dict_delegate_to_dict:
      if key == "error":
        return getattr(self, "error", None) is not None
      if key in self._field_names():
        return True
    return key in self.to_dict()

  def keys(self) -> Any:
    return self.to_dict().keys()

  def items(self) -> Any:
    return self.to_dict().items()

  def values(self) -> Any:
    return self.to_dict().values()

  def __iter__(self) -> Iterator[str]:
    return iter(self.to_dict())

  def __len__(self) -> int:
    return len(self.to_dict())


@dataclass
class CameraCalibration(_DictCompatMixin):
  """OpenCV intrinsics, base/EE extrinsics, and optional ZED rectification."""

  _dict_delegate_to_dict = True

  width: int
  height: int
  K: np.ndarray
  intrinsics_tuple: tuple[float, float, float, float]
  mp_intrinsics: dict[str, list[float]]
  T_base_cam: np.ndarray = field(
      default_factory=lambda: np.eye(4, dtype=np.float64)
  )
  T_ee_cam: np.ndarray | None = None
  mj_pose: dict[str, np.ndarray] = field(default_factory=dict)
  P: np.ndarray | None = None
  R_rect: np.ndarray | None = None
  baseline_m: float | None = None
  rectified: bool = False
  conf_path: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Returns a dictionary view omitting unset rectification fields."""
    out: dict[str, Any] = {
        "width": int(self.width),
        "height": int(self.height),
        "K": self.K,
        "intrinsics_tuple": self.intrinsics_tuple,
        "mp_intrinsics": self.mp_intrinsics,
        "T_base_cam": self.T_base_cam,
        "T_ee_cam": self.T_ee_cam,
        "mj_pose": self.mj_pose,
    }
    if (
        self.rectified
        or self.P is not None
        or self.R_rect is not None
        or self.baseline_m is not None
        or self.conf_path is not None
    ):
      if self.P is not None:
        out["P"] = self.P
      if self.R_rect is not None:
        out["R_rect"] = self.R_rect
      if self.baseline_m is not None:
        out["baseline_m"] = self.baseline_m
      out["rectified"] = bool(self.rectified)
      if self.conf_path is not None:
        out["conf_path"] = self.conf_path
    return out


@dataclass
class UvcDeviceInfo(_DictCompatMixin):
  """Discovered Linux V4L2/UVC camera device metadata."""

  name: str
  family: str
  device_path: str
  video_node: str
  serial: str
  vendor_id: str
  product_id: str
  busnum: str
  usb_speed_mbps: float
  is_usb3: bool
  interface: str
  power_control: str
  udev_aliases: list[str] = field(default_factory=list)
  side_hint: str = ""
  v4l2_rw_ok: bool | None = None


@dataclass
class UsbPermissionsReport(_DictCompatMixin):
  """Linux `video`/`dialout` group and device R/W permission report."""

  groups: list[str]
  in_video_group: bool
  in_dialout_group: bool
  v4l2_access_issues: list[UvcDeviceInfo | dict[str, Any]] = field(
      default_factory=list
  )
  serial_access_issues: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ZedEyeIntrinsics(_DictCompatMixin):
  """Single-eye intrinsics and distortion coefficients from ZED `.conf`."""

  fx: float
  fy: float
  cx: float
  cy: float
  k1: float
  k2: float
  p1: float
  p2: float
  k3: float
  K: list[list[float]]
  dist_coeffs: list[float]


@dataclass
class ZedStereoExtrinsics(_DictCompatMixin):
  """Stereo baseline and relative rotation/translation from ZED `.conf`."""

  baseline_mm: float
  baseline_m: float
  ty_mm: float
  tz_mm: float
  rx: float
  cv: float
  rz: float


@dataclass
class ZedCalibration(_DictCompatMixin):
  """Parsed Stereolabs ZED factory calibration for one resolution."""

  resolution: str
  left: ZedEyeIntrinsics
  right: ZedEyeIntrinsics
  stereo: ZedStereoExtrinsics


@dataclass
class ZedRectificationEntry:
  """Cached OpenCV stereo rectification remap tables and matrices."""

  m1_l: np.ndarray
  m2_l: np.ndarray
  m1_r: np.ndarray
  m2_r: np.ndarray
  r1: np.ndarray
  r2: np.ndarray
  p1: np.ndarray
  p2: np.ndarray
  width: int
  height: int
  conf_path: str


@dataclass
class CameraFreshnessEntry(_DictCompatMixin):
  """Freshness and pixel-statistics summary for one camera stream."""

  ok: bool
  max_age_s: float
  timestamps_advancing: bool
  pixels_changing: bool
  brightness_mean: float
  pixel_std: float
  width: int
  height: int
  backend: str


@dataclass
class CameraFreshnessReport(_DictCompatMixin):
  """Aggregate camera freshness report across all configured streams."""

  ok: bool
  cameras: dict[str, CameraFreshnessEntry] = field(default_factory=dict)
  errors: list[str] = field(default_factory=list)


def _read_sysfs_text(path: Path) -> str:
  try:
    return path.read_text(encoding="utf-8", errors="ignore").strip()
  except OSError:
    return ""


def _valid_usb_serial(raw: str) -> str:
  if ":" in raw or raw.lower() in ("", "unknown", "none", "null", "0", "ov580"):
    return ""
  return raw


def _inspect_usb_ancestors(sys_video_dir: Path) -> dict[str, str]:
  """Reads USB attributes from sysfs ancestors of a video4linux node."""
  info = {
      "vendor_id": "",
      "product_id": "",
      "serial": "",
      "speed_mbps": "",
      "interface": "",
      "busnum": "",
      "power_control": "",
  }
  try:
    dev_dir = sys_video_dir / "device"
    info["interface"] = _read_sysfs_text(dev_dir / "interface")
    info["power_control"] = _read_sysfs_text(dev_dir / "power" / "control")
    cur = dev_dir.resolve()
    info["interface"] = info["interface"] or _read_sysfs_text(cur / "interface")
    info["power_control"] = info["power_control"] or _read_sysfs_text(
        cur / "power" / "control"
    )
  except OSError:
    return info

  for _ in range(6):
    found_usb = (cur / "idVendor").is_file() or (cur / "idProduct").is_file()
    if not info["vendor_id"] and (cur / "idVendor").is_file():
      info["vendor_id"] = _read_sysfs_text(cur / "idVendor").lower()
    if not info["product_id"] and (cur / "idProduct").is_file():
      info["product_id"] = _read_sysfs_text(cur / "idProduct").lower()
    if not info["serial"] and (cur / "serial").is_file():
      info["serial"] = _valid_usb_serial(_read_sysfs_text(cur / "serial"))
    if not info["speed_mbps"] and (cur / "speed").is_file():
      info["speed_mbps"] = _read_sysfs_text(cur / "speed")
    if not info["busnum"] and (cur / "busnum").is_file():
      info["busnum"] = _read_sysfs_text(cur / "busnum")
    if not info["power_control"] and (cur / "power" / "control").is_file():
      info["power_control"] = _read_sysfs_text(cur / "power" / "control")
    if found_usb:
      # ZED Mini OV580 video interfaces often omit iSerialNumber while the
      # companion ZED HID MCU (2b03) exposes the factory serial on the same bus.
      if info["vendor_id"] == ZED_VENDOR_ID and not info["serial"]:
        search_dirs: list[Path] = []
        if cur.parent != cur:
          search_dirs.extend(sorted(cur.parent.iterdir(), key=lambda p: p.name))
          if cur.parent.parent != cur.parent:
            search_dirs.extend(
                sorted(cur.parent.parent.glob("usb*/*"), key=lambda p: p.name)
            )
        if (
            str(sys_video_dir).startswith("/sys")
            and Path("/sys/bus/usb/devices").is_dir()
        ):
          search_dirs.extend(
              sorted(
                  Path("/sys/bus/usb/devices").iterdir(), key=lambda p: p.name
              )
          )
        for sib in search_dirs:
          if sib == cur or not sib.is_dir():
            continue
          if _read_sysfs_text(sib / "idVendor").lower() == ZED_VENDOR_ID:
            sib_ser = _valid_usb_serial(_read_sysfs_text(sib / "serial"))
            if sib_ser:
              info["serial"] = sib_ser
              break
      break
    if cur.parent == cur:
      break
    cur = cur.parent

  return info


def _classify_camera_family(name: str, vendor_id: str) -> str | None:
  lowered = name.lower()
  vid = vendor_id.lower().removeprefix("0x")
  if vid == ZED_VENDOR_ID or (not vid and "zed" in lowered):
    return "zed"
  if vid == INTEL_VENDOR_ID or (not vid and "realsense" in lowered):
    return "realsense"
  return None


def _collect_symlink_map(symlink_dir: Path) -> dict[str, str]:
  mapping: dict[str, str] = {}
  if not symlink_dir.is_dir():
    return mapping
  for symlink in sorted(symlink_dir.iterdir(), key=lambda p: p.name):
    try:
      resolved = os.path.realpath(str(symlink))
    except OSError:
      continue
    if resolved not in mapping or "video-index0" in symlink.name:
      mapping[resolved] = str(symlink)
  return mapping


def discover_uvc_devices(
    *,
    sys_v4l: Path = Path("/sys/class/video4linux"),
    v4l_by_id: Path = Path("/dev/v4l/by-id"),
    v4l_by_path: Path = Path("/dev/v4l/by-path"),
    dev_root: Path = Path("/dev"),
) -> list[UvcDeviceInfo]:
  """Discovers ZED Mini and RealSense RGB V4L2 camera nodes on Linux."""
  if not sys_v4l.is_dir():
    return []

  by_id_map = _collect_symlink_map(v4l_by_id)
  by_path_map = _collect_symlink_map(v4l_by_path)

  udev_aliases_by_resolved: dict[str, list[str]] = {}
  if dev_root.is_dir():
    for entry in sorted(dev_root.iterdir(), key=lambda p: p.name):
      try:
        if entry.is_symlink() and entry.name.lower().startswith(
            ("realsense", "zed")
        ):
          udev_aliases_by_resolved.setdefault(
              os.path.realpath(str(entry)), []
          ).append(str(entry))
      except OSError:
        continue

  raw_v4l: list[dict[str, Any]] = []
  rs_group_indices: dict[str, set[str]] = {}
  rs_group_explicit_rgb: dict[str, bool] = {}

  for vdir in sorted(sys_v4l.iterdir(), key=lambda p: p.name):
    node_index = _read_sysfs_text(vdir / "index")
    if node_index and (not node_index.isdigit() or int(node_index) % 2 != 0):
      continue
    name = _read_sysfs_text(vdir / "name") or vdir.name
    usb_info = _inspect_usb_ancestors(vdir)
    family = _classify_camera_family(name, usb_info["vendor_id"])
    if family is None or (
        family != "realsense" and node_index not in ("", "0")
    ):
      continue

    cand_node = dev_root / vdir.name
    dev_node = str(cand_node) if cand_node.exists() else f"/dev/{vdir.name}"
    resolved_dev = os.path.realpath(dev_node)
    try:
      speed_val = (
          float(usb_info["speed_mbps"]) if usb_info["speed_mbps"] else 0.0
      )
    except ValueError:
      speed_val = 0.0

    id_symlink = by_id_map.get(resolved_dev)
    if id_symlink and os.path.realpath(id_symlink) == resolved_dev:
      device_path = id_symlink
    else:
      device_path = by_path_map.get(resolved_dev, dev_node)

    group_key = usb_info["serial"] or resolved_dev
    if family == "realsense":
      if node_index:
        rs_group_indices.setdefault(group_key, set()).add(node_index)
      combined_lower = f"{name} {usb_info['interface']}".lower()
      if "rgb" in combined_lower or "color" in combined_lower:
        rs_group_explicit_rgb[group_key] = True

    raw_v4l.append({
        "name": name,
        "node_index": node_index,
        "group_key": group_key,
        "dev_node": dev_node,
        "resolved_dev": resolved_dev,
        "device_path": device_path,
        "usb_info": usb_info,
        "speed_val": speed_val,
        "family": family,
    })

  devices: list[UvcDeviceInfo] = []
  for item in raw_v4l:
    usb_info = item["usb_info"]
    ser = usb_info["serial"]
    g_key = item["group_key"]
    if item["family"] == "realsense":
      # RGB-only: drop RealSense depth and IR nodes (D405 exposes index=0
      # depth, index=2 RGB, index=4 IR).
      combined = f"{item['name']} {usb_info['interface']}".lower()
      has_rgb = "rgb" in combined or "color" in combined
      if rs_group_explicit_rgb.get(g_key, False) or has_rgb:
        if not has_rgb:
          continue
      elif "2" in rs_group_indices.get(g_key, set()):
        if item["node_index"] != "2":
          continue
      elif any(
          tok in usb_info["interface"].lower()
          for tok in ("depth", "ir", "infra")
      ):
        continue

    item_aliases = list(udev_aliases_by_resolved.get(item["resolved_dev"], []))
    side_hint = ""
    if item["family"] == "realsense":
      for alias in item_aliases:
        alias_lower = Path(alias).name.lower()
        if "left" in alias_lower:
          side_hint = "left"
          break
        if "right" in alias_lower:
          side_hint = "right"
          break

    v4l2_rw_ok = (
        os.access(item["dev_node"], os.R_OK | os.W_OK)
        if Path(item["dev_node"]).exists()
        else None
    )
    devices.append(
        UvcDeviceInfo(
            name=item["name"],
            family=item["family"],
            device_path=item["device_path"],
            video_node=item["dev_node"],
            serial=ser,
            vendor_id=usb_info["vendor_id"],
            product_id=usb_info["product_id"],
            busnum=usb_info["busnum"],
            usb_speed_mbps=item["speed_val"],
            is_usb3=bool(item["speed_val"] >= 5000.0),
            interface=usb_info["interface"],
            power_control=usb_info["power_control"],
            udev_aliases=item_aliases,
            side_hint=side_hint,
            v4l2_rw_ok=v4l2_rw_ok,
        )
    )

  return devices


def check_linux_usb_permissions(
    uvc_devices: Sequence[UvcDeviceInfo | Mapping[str, Any]] | None = None,
    *,
    serial_ports: Sequence[Mapping[str, Any]] | None = None,
    group_names_override: set[str] | None = None,
) -> UsbPermissionsReport:
  """Checks Linux `video`/`dialout` group membership and device R/W access."""
  if group_names_override is not None:
    group_set = set(group_names_override)
  else:
    group_set = set()
    if grp is not None:
      try:
        gids = set(os.getgroups()) | {os.getegid(), os.getgid()}
        for gid in gids:
          try:
            group_set.add(grp.getgrgid(gid).gr_name)
          except (KeyError, OverflowError):
            # Ignore numeric GIDs without a local /etc/group entry.
            pass
      except OSError:
        # Process group inspection may be restricted in sandboxed envs.
        pass

  if uvc_devices is None:
    uvc_devices = discover_uvc_devices()

  v4l2_issues = [
      d
      for d in uvc_devices
      if str(d.get("family", "")).lower() in ("zed", "realsense")
      and d.get("v4l2_rw_ok") is False
  ]
  serial_issues = [
      s
      for s in (serial_ports or [])
      if str(s.get("device_role_hint", "")) != "teleop_or_ft232h"
      and s.get("serial_rw_ok") is False
  ]
  return UsbPermissionsReport(
      groups=sorted(group_set),
      in_video_group="video" in group_set,
      in_dialout_group="dialout" in group_set,
      v4l2_access_issues=v4l2_issues,
      serial_access_issues=serial_issues,
  )


def _iter_glob_zed_dirs() -> list[Path]:
  """Finds vendor/site ZED settings directories via shallow prefix globs."""
  dirs: list[Path] = []
  for root in (Path("/usr/local"), Path("/opt")):
    try:
      dirs.extend(sorted(root.glob("*/zed/settings")))
    except OSError:
      pass
  return dirs


def _safe_iterdir(directory: Path) -> list[Path]:
  """Returns sorted entries of `directory`, or `[]` if unreadable."""
  try:
    return sorted(directory.iterdir(), key=lambda p: p.name)
  except OSError:
    return []


def _find_zed_conf_in_dirs(
    dirs: Sequence[Path], clean_serial: str
) -> Path | None:
  existing_dirs = []
  for d in dirs:
    try:
      if d.is_dir():
        existing_dirs.append(d)
    except OSError:
      continue
  if clean_serial:
    target_name = f"sn{clean_serial.lower()}.conf"
    for d in existing_dirs:
      for f in _safe_iterdir(d):
        if f.is_file() and f.name.lower() == target_name:
          return f
    if clean_serial.isdigit():
      return None

  for d in existing_dirs:
    confs = [
        f
        for f in _safe_iterdir(d)
        if f.is_file()
        and f.name.lower().startswith("sn")
        and f.name.lower().endswith(".conf")
    ]
    if len(confs) == 1:
      return confs[0]
  return None


def resolve_zed_conf_path(
    serial: str = "",
    explicit_path: str | Path = "",
    search_dirs: Sequence[str | Path] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
  """Locates factory ZED `SN<serial>.conf` across file, dir, env, and globs."""
  candidate_dirs: list[Path] = []
  if explicit_path:
    p = Path(explicit_path)
    if p.is_file():
      return p
    if p.is_dir():
      candidate_dirs.append(p)

  env_map = os.environ if env is None else env
  candidate_dirs.extend(
      Path(part.strip())
      for part in env_map.get("ZED_SETTINGS_DIR", "").split(os.pathsep)
      if part.strip()
  )
  if search_dirs:
    candidate_dirs.extend(
        Path(sdir) for sdir in search_dirs if str(sdir).strip()
    )
  candidate_dirs.extend([
      Path("/usr/local/zed/settings"),
      Path.home() / "zed" / "settings",
      Path.home() / ".config" / "zed" / "settings",
  ])

  clean_serial = str(serial).strip()
  if clean_serial.lower().startswith("sn") and len(clean_serial) > 2:
    clean_serial = clean_serial[2:].strip()

  found = _find_zed_conf_in_dirs(candidate_dirs, clean_serial)
  if found is not None:
    return found
  return _find_zed_conf_in_dirs(_iter_glob_zed_dirs(), clean_serial)


def split_zed_sbs_frame(
    frame: np.ndarray, *, is_bgr: bool = False
) -> tuple[np.ndarray, np.ndarray]:
  """Splits a ZED side-by-side stereo frame into `(left, right)` RGB."""
  arr = np.asarray(frame)
  if arr.ndim != 3 or arr.shape[2] != 3:
    raise ValueError(
        f"Expected 3-channel (H, W, 3) stereo frame, got shape {arr.shape}"
    )
  height, width, _ = arr.shape
  if height <= 0 or width < 2 or (width % 2) != 0:
    raise ValueError(
        "Expected positive even stereo width for side-by-side split, got"
        f" {arr.shape}"
    )
  if arr.dtype != np.uint8:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
  if is_bgr:
    arr = arr[:, :, ::-1]
  mid = width // 2
  return (
      np.ascontiguousarray(arr[:, :mid, :]),
      np.ascontiguousarray(arr[:, mid:, :]),
  )


def _parse_zed_cam_section(
    sec: configparser.SectionProxy,
) -> ZedEyeIntrinsics:
  fx = float(sec.get("fx", 0.0))
  fy = float(sec.get("fy", 0.0))
  cx = float(sec.get("cx", 0.0))
  cy = float(sec.get("cy", 0.0))
  k1 = float(sec.get("k1", 0.0))
  k2 = float(sec.get("k2", 0.0))
  p1 = float(sec.get("p1", 0.0))
  p2 = float(sec.get("p2", 0.0))
  k3 = float(sec.get("k3", 0.0))
  return ZedEyeIntrinsics(
      fx=fx,
      fy=fy,
      cx=cx,
      cy=cy,
      k1=k1,
      k2=k2,
      p1=p1,
      p2=p2,
      k3=k3,
      K=[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
      dist_coeffs=[k1, k2, p1, p2, k3],
  )


_ZED_RESOLUTION_ALIASES: dict[str, tuple[str, ...]] = {
    "HD720": ("HD720", "HD"),
    "HD": ("HD", "HD720"),
    "HD1080": ("HD1080", "FHD"),
    "FHD": ("FHD", "HD1080"),
    "HD2K": ("HD2K", "2K"),
    "2K": ("2K", "HD2K"),
    "VGA": ("VGA", "WVGA"),
    "WVGA": ("WVGA", "VGA"),
}

_ZED_NATIVE_EYE_SIZES: frozenset[tuple[int, int]] = frozenset({
    (2208, 1242),
    (1920, 1080),
    (1280, 720),
    (672, 376),
})


def parse_zed_conf(
    conf_path: str | Path, resolution: str = "HD720"
) -> ZedCalibration:
  """Parses a Stereolabs ZED factory calibration `.conf` INI file."""
  path = Path(conf_path)
  if not path.is_file():
    raise FileNotFoundError(f"ZED calibration file not found: {path}")

  res = resolution.strip().upper()
  parser = configparser.ConfigParser(strict=False)
  parser.read(path, encoding="utf-8")

  tags = _ZED_RESOLUTION_ALIASES.get(res, (res,))
  matched_tag: str | None = None
  for tag in tags:
    if f"LEFT_CAM_{tag}" in parser and f"RIGHT_CAM_{tag}" in parser:
      matched_tag = tag
      break
  if matched_tag is None:
    left_sec_name = f"LEFT_CAM_{res}"
    right_sec_name = f"RIGHT_CAM_{res}"
    raise ValueError(
        f"Sections [{left_sec_name}] / [{right_sec_name}] missing in {path} "
        f"(available: {parser.sections()})"
    )

  left_sec_name = f"LEFT_CAM_{matched_tag}"
  right_sec_name = f"RIGHT_CAM_{matched_tag}"
  stereo_sec = parser["STEREO"] if "STEREO" in parser else {}
  baseline_mm = float(
      stereo_sec.get("baseline", stereo_sec.get("Baseline", 63.0))
      if stereo_sec
      else 63.0
  )
  stereo_tags = [matched_tag] + [t for t in tags if t != matched_tag]

  def _stereo_angle(prefix: str) -> float:
    if not stereo_sec:
      return 0.0
    for tag in stereo_tags:
      key = f"{prefix}_{tag.lower()}"
      if key in stereo_sec:
        return float(stereo_sec[key])
    return 0.0

  return ZedCalibration(
      resolution=res,
      left=_parse_zed_cam_section(parser[left_sec_name]),
      right=_parse_zed_cam_section(parser[right_sec_name]),
      stereo=ZedStereoExtrinsics(
          baseline_mm=baseline_mm,
          baseline_m=baseline_mm / 1000.0,
          ty_mm=float(stereo_sec.get("ty", 0.0)) if stereo_sec else 0.0,
          tz_mm=float(stereo_sec.get("tz", 0.0)) if stereo_sec else 0.0,
          rx=_stereo_angle("rx"),
          cv=_stereo_angle("cv"),
          rz=_stereo_angle("rz"),
      ),
  )


def _is_sbs_stereo_spec(spec: CameraStreamSpec) -> bool:
  """Returns True when `spec` dimensions encode a side-by-side stereo frame."""
  return spec.height > 0 and (float(spec.width) / float(spec.height)) >= 2.2


def _zed_res_name_for_size(width: int, height: int) -> str:
  if height >= 1200 or width >= 1920:
    return "HD1080" if height <= 1080 else "HD2K"
  if height >= 720:
    return "HD720"
  return "VGA"


class V4L2Camera:
  """OpenCV V4L2 (`cv2.CAP_V4L2`) capture handle."""

  def __init__(self, spec: CameraStreamSpec, width: int, height: int) -> None:
    """Opens a V4L2 video stream matching `spec` at `(width, height)`."""
    cv2_mod = require_cv2()
    candidates: list[str] = []
    if spec.device_path and os.path.exists(spec.device_path):
      candidates.append(spec.device_path)

    if not candidates:
      is_zed_role = spec.role == "head_stereo"
      matching = [
          d
          for d in discover_uvc_devices()
          if d.get("device_path")
          and (
              (spec.serial and d.get("serial") == spec.serial)
              or (is_zed_role and d.get("family") == "zed")
          )
      ]
      matching.sort(key=lambda d: str(d.get("video_node", "")))
      for d in matching:
        cand = str(d["device_path"])
        if cand not in candidates:
          candidates.append(cand)

    if not candidates:
      raise RuntimeError(
          f"No V4L2 device_path resolved for camera role={spec.role!r} "
          f"(serial={spec.serial!r})"
      )

    self._cv2 = cv2_mod
    self._cap = None
    for cand_path in candidates:
      real_dev = os.path.realpath(cand_path)
      cap = cv2_mod.VideoCapture(real_dev, cv2_mod.CAP_V4L2)
      if cap.isOpened():
        self._cap = cap
        break
      cap.release()

    if self._cap is None or not self._cap.isOpened():
      raise RuntimeError(
          f"Failed to open V4L2 camera at {candidates!r} "
          f"for role={spec.role!r}"
      )

    self._cap.set(cv2_mod.CAP_PROP_BUFFERSIZE, 1)
    self._cap.set(cv2_mod.CAP_PROP_FRAME_WIDTH, int(width))
    self._cap.set(cv2_mod.CAP_PROP_FRAME_HEIGHT, int(height))
    self._cap.set(cv2_mod.CAP_PROP_FPS, int(spec.fps))

  def capture_rgb(self) -> tuple[np.ndarray, float]:
    """Reads one frame from V4L2 and returns `(rgb_uint8, monotonic_ts)`."""
    ok, bgr = self._cap.read()
    ts = time.monotonic()
    if not ok or bgr is None:
      raise RuntimeError("V4L2 frame read failed")
    rgb = self._cv2.cvtColor(
        np.asarray(bgr, dtype=np.uint8), self._cv2.COLOR_BGR2RGB
    )
    return np.ascontiguousarray(rgb, dtype=np.uint8), ts

  def close(self) -> None:
    """Releases the underlying OpenCV `VideoCapture` handle."""
    if self._cap is not None:
      self._cap.release()
      self._cap = None


def _read_override_frame(source: Any) -> tuple[np.ndarray, float, str]:
  """Reads `(rgb, timestamp_s, backend)` from a test override handle."""
  raw = (
      source.capture()
      if hasattr(source, "capture")
      else (source.read() if hasattr(source, "read") else source())
  )
  if isinstance(raw, CameraFrame):
    return (
        np.ascontiguousarray(raw.rgb, dtype=np.uint8),
        float(raw.timestamp_s),
        str(raw.backend or "override"),
    )
  if (
      isinstance(raw, tuple)
      and len(raw) == 2
      and isinstance(raw[0], (bool, np.bool_))
  ):
    if not raw[0] or raw[1] is None:
      raise RuntimeError("Camera override returned failed read")
    return (
        np.ascontiguousarray(np.asarray(raw[1], dtype=np.uint8)[:, :, ::-1]),
        time.monotonic(),
        "override",
    )
  if (
      isinstance(raw, tuple)
      and len(raw) >= 2
      and isinstance(raw[0], np.ndarray)
  ):
    return (
        np.ascontiguousarray(raw[0], dtype=np.uint8),
        float(raw[1]),
        str(raw[2]) if len(raw) >= 3 else "override",
    )
  raise TypeError(f"Unsupported camera override payload: {type(raw)!r}")


def build_sim_camera_frames(
    rendered: Mapping[str, np.ndarray],
    timestamp_s: float,
) -> dict[str, CameraFrame]:
  """Wraps rendered MuJoCo `(H, W, 3)` RGB arrays into `CameraFrame`s."""
  frames: dict[str, CameraFrame] = {}
  for cam_name in CANONICAL_CAMERAS:
    rgb = np.ascontiguousarray(rendered[cam_name], dtype=np.uint8)
    frames[cam_name] = CameraFrame(
        name=cam_name,
        rgb=rgb,
        timestamp_s=float(timestamp_s),
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        backend="mujoco_sim",
    )
  return frames


def _cal_get(
    cal: CameraCalibration | Mapping[str, Any], key: str, default: Any = None
) -> Any:
  return (
      getattr(cal, key, default)
      if isinstance(cal, CameraCalibration)
      else cal.get(key, default)
  )


def _cal_set(
    cal: CameraCalibration | dict[str, Any], key: str, value: Any
) -> None:
  if isinstance(cal, CameraCalibration):
    setattr(cal, key, value)
  else:
    cal[key] = value


def scale_camera_calibration(
    cam_cal: CameraCalibration | dict[str, Any],
    width: int,
    height: int,
) -> None:
  """Scales OpenCV intrinsics in `cam_cal` in-place to `(width, height)`."""
  cal_w = int(_cal_get(cam_cal, "width", width))
  cal_h = int(_cal_get(cam_cal, "height", height))
  if width <= 0 or height <= 0 or cal_w <= 0 or cal_h <= 0:
    return
  if width == cal_w and height == cal_h:
    return
  sx = float(width) / float(cal_w)
  sy = float(height) / float(cal_h)
  k_mat = np.array(_cal_get(cam_cal, "K"), dtype=np.float64, copy=True)
  k_mat[0, 0] *= sx
  k_mat[0, 2] *= sx
  k_mat[1, 1] *= sy
  k_mat[1, 2] *= sy
  fx, fy = float(k_mat[0, 0]), float(k_mat[1, 1])
  cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])
  updates = {
      "width": int(width),
      "height": int(height),
      "K": k_mat,
      "intrinsics_tuple": (fx, fy, cx, cy),
      "mp_intrinsics": {
          "focal_px": [fx, fy],
          "principal_point_px": [cx, cy],
      },
  }
  for key, val in updates.items():
    _cal_set(cam_cal, key, val)


def apply_rectified_camera_calibration(
    calibration: dict[str, Any],
    rectified_calibration: (
        Mapping[str, CameraCalibration | Mapping[str, Any]] | None
    ),
) -> dict[str, Any]:
  """Overrides `calibration` in-place with active rectified ZED intrinsics."""
  if not rectified_calibration:
    return calibration
  for cam_name, rect_entry in rectified_calibration.items():
    target = calibration.get(cam_name)
    if target is None:
      continue
    rect_map = (
        rect_entry.to_dict()
        if isinstance(rect_entry, CameraCalibration)
        else rect_entry
    )
    k_mat = np.asarray(rect_map["K"], dtype=np.float64).copy()
    intr_vals = tuple(float(v) for v in rect_map["intrinsics_tuple"])
    was_rectified = bool(_cal_get(target, "rectified", False))
    _cal_set(target, "width", int(rect_map["width"]))
    _cal_set(target, "height", int(rect_map["height"]))
    _cal_set(target, "K", k_mat)
    _cal_set(
        target,
        "intrinsics_tuple",
        (intr_vals[0], intr_vals[1], intr_vals[2], intr_vals[3]),
    )
    _cal_set(
        target,
        "mp_intrinsics",
        {
            "focal_px": [float(k_mat[0, 0]), float(k_mat[1, 1])],
            "principal_point_px": [float(k_mat[0, 2]), float(k_mat[1, 2])],
        },
    )
    for extra_key in ("P", "R_rect", "baseline_m", "rectified", "conf_path"):
      if extra_key in rect_map:
        val = rect_map[extra_key]
        _cal_set(
            target,
            extra_key,
            np.asarray(val, dtype=np.float64).copy()
            if isinstance(val, np.ndarray)
            else val,
        )
    if not was_rectified:
      raw_r = _cal_get(target, "R_rect")
      raw_t = _cal_get(target, "T_base_cam")
      if raw_r is not None and raw_t is not None:
        r_rect = np.asarray(raw_r, dtype=np.float64)
        if r_rect.shape == (3, 3):
          t_bc = np.asarray(raw_t, dtype=np.float64).copy()
          t_bc[:3, :3] = t_bc[:3, :3] @ r_rect.T
          _cal_set(target, "T_base_cam", t_bc)
  return calibration


class Fr3DuoCameraController:
  """Unified 4-camera capture controller for Franka Research 3 Duo."""

  def __init__(
      self,
      specs: Any = None,
      *,
      sim_robot: Any = None,
      backend_overrides: dict[str, Any] | None = None,
      dry_run: bool = False,
  ) -> None:
    """Initializes stream specs and optional sim, override, or dry-run mode."""
    self.specs: dict[str, CameraStreamSpec] = {}
    if specs is None:
      self.specs["head_stereo"] = CameraStreamSpec(
          role="head_stereo", width=2560, height=720
      )
      for cam_name in ("wrist_camera_left", "wrist_camera_right"):
        self.specs[cam_name] = CameraStreamSpec(role=cam_name)
    else:
      spec_map = specs.as_spec_map() if hasattr(specs, "as_spec_map") else specs
      for key, val in spec_map.items():
        self.specs[key] = CameraStreamSpec.from_mapping(key, val)

    self.sim_robot = sim_robot
    self.backend_overrides = (
        dict(backend_overrides) if backend_overrides else {}
    )
    self.dry_run = bool(dry_run)
    self._opened = False
    self._handles: dict[str, tuple[Any, str, tuple[str, ...]]] = {}
    self._zed_rectify_cache: dict[
        tuple[str, int, int], ZedRectificationEntry | None
    ] = {}
    self._last_zed_rectification: ZedRectificationEntry | None = None
    self._zed_rectify_warned: set[tuple[str, int, int]] = set()
    self._zed_conf_fallback_cache: dict[tuple[str, str], str] = {}
    self._frame_seq = 0

  def _warn_zed_rectify_once(
      self, cache_key: tuple[str, int, int], msg: str
  ) -> None:
    if cache_key not in self._zed_rectify_warned:
      self._zed_rectify_warned.add(cache_key)
      warnings.warn(msg, RuntimeWarning, stacklevel=2)

  def _resolve_zed_conf_for_pair(self) -> str:
    sp = self.specs.get("head_stereo")
    if sp is None:
      return ""
    cache_key = (sp.serial, sp.zed_conf_path)
    if cache_key not in self._zed_conf_fallback_cache:
      found = resolve_zed_conf_path(
          serial=sp.serial, explicit_path=sp.zed_conf_path
      )
      self._zed_conf_fallback_cache[cache_key] = (
          str(found) if found is not None else ""
      )
    return self._zed_conf_fallback_cache[cache_key]

  def _maybe_rectify_zed_pair(
      self, left_rgb: np.ndarray, right_rgb: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    conf_path = self._resolve_zed_conf_for_pair()
    if not conf_path:
      self._last_zed_rectification = None
      return left_rgb, right_rgb
    h, w = int(left_rgb.shape[0]), int(left_rgb.shape[1])
    cache_key = (conf_path, w, h)
    if cache_key not in self._zed_rectify_cache:
      if not Path(conf_path).is_file():
        self._zed_rectify_cache[cache_key] = None
      elif (w, h) not in _ZED_NATIVE_EYE_SIZES:
        self._warn_zed_rectify_once(
            cache_key,
            f"Skipping ZED stereo rectification for non-native eye size "
            f"{w}x{h} (supported: {sorted(_ZED_NATIVE_EYE_SIZES)}).",
        )
        self._zed_rectify_cache[cache_key] = None
      else:
        try:
          parsed = parse_zed_conf(conf_path, _zed_res_name_for_size(w, h))
        except (ValueError, configparser.Error, OSError) as exc:
          self._warn_zed_rectify_once(
              cache_key,
              f"Skipping ZED stereo rectification for {w}x{h}: {exc}",
          )
          self._zed_rectify_cache[cache_key] = None
        else:
          cv2_mod = require_cv2()
          k_l = np.asarray(parsed.left.K, dtype=np.float64)
          d_l = np.asarray(parsed.left.dist_coeffs, dtype=np.float64)
          k_r = np.asarray(parsed.right.K, dtype=np.float64)
          d_r = np.asarray(parsed.right.dist_coeffs, dtype=np.float64)
          st = parsed.stereo
          r_mat, _ = cv2_mod.Rodrigues(
              np.array([st.rx, st.cv, st.rz], dtype=np.float64)
          )
          t_vec = np.array(
              [-st.baseline_m, st.ty_mm / 1000.0, st.tz_mm / 1000.0],
              dtype=np.float64,
          ).reshape(3, 1)
          r1, r2, p1, p2, _, _, _ = cv2_mod.stereoRectify(
              k_l, d_l, k_r, d_r, (w, h), r_mat, t_vec, alpha=0
          )
          m1_l, m2_l = cv2_mod.initUndistortRectifyMap(
              k_l, d_l, r1, p1, (w, h), cv2_mod.CV_16SC2
          )
          m1_r, m2_r = cv2_mod.initUndistortRectifyMap(
              k_r, d_r, r2, p2, (w, h), cv2_mod.CV_16SC2
          )
          self._zed_rectify_cache[cache_key] = ZedRectificationEntry(
              m1_l=m1_l,
              m2_l=m2_l,
              m1_r=m1_r,
              m2_r=m2_r,
              r1=np.asarray(r1, dtype=np.float64).copy(),
              r2=np.asarray(r2, dtype=np.float64).copy(),
              p1=np.asarray(p1, dtype=np.float64).copy(),
              p2=np.asarray(p2, dtype=np.float64).copy(),
              width=w,
              height=h,
              conf_path=conf_path,
          )

    entry = self._zed_rectify_cache[cache_key]
    if entry is None:
      self._last_zed_rectification = None
      return left_rgb, right_rgb

    cv2_mod = require_cv2()
    self._last_zed_rectification = entry
    return (
        np.ascontiguousarray(
            cv2_mod.remap(
                left_rgb, entry.m1_l, entry.m2_l, cv2_mod.INTER_LINEAR
            )
        ),
        np.ascontiguousarray(
            cv2_mod.remap(
                right_rgb, entry.m1_r, entry.m2_r, cv2_mod.INTER_LINEAR
            )
        ),
    )

  def get_rectified_calibration(self) -> dict[str, CameraCalibration]:
    """Returns post-rectification intrinsics for ZED head cameras if active."""
    entry = self._last_zed_rectification
    if entry is None:
      return {}
    p1 = np.asarray(entry.p1, dtype=np.float64)
    p2 = np.asarray(entry.p2, dtype=np.float64)
    fx_r = float(p2[0, 0])
    baseline_m = abs(float(p2[0, 3])) / fx_r if fx_r > 1e-9 else 0.0
    out: dict[str, CameraCalibration] = {}
    for cam_name, p_mat, r_mat in (
        ("head_camera_left", p1, entry.r1),
        ("head_camera_right", p2, entry.r2),
    ):
      k_rect = p_mat[:3, :3].copy()
      fx = float(k_rect[0, 0])
      fy = float(k_rect[1, 1])
      cx = float(k_rect[0, 2])
      cy = float(k_rect[1, 2])
      out[cam_name] = CameraCalibration(
          width=int(entry.width),
          height=int(entry.height),
          K=k_rect,
          intrinsics_tuple=(fx, fy, cx, cy),
          mp_intrinsics={
              "focal_px": [fx, fy],
              "principal_point_px": [cx, cy],
          },
          P=p_mat.copy(),
          R_rect=np.asarray(r_mat, dtype=np.float64).copy(),
          baseline_m=baseline_m,
          rectified=True,
          conf_path=str(entry.conf_path),
      )
    return out

  def open(self) -> None:
    """Opens configured camera streams."""
    if self._opened:
      return
    self._handles.clear()

    if self.sim_robot is not None or (
        self.dry_run and not self.backend_overrides
    ):
      self._opened = True
      return

    if self.backend_overrides:
      for key, source in self.backend_overrides.items():
        targets = (
            ("head_camera_left", "head_camera_right")
            if key == "head_stereo"
            else (key,)
        )
        self._handles[key] = (source, "override", targets)
      self._opened = True
      return

    try:
      for role, sp in self.specs.items():
        if role == "head_stereo":
          cap_w = sp.width if _is_sbs_stereo_spec(sp) else sp.width * 2
          self._handles[role] = (
              V4L2Camera(sp, cap_w, sp.height),
              "v4l2",
              ("head_camera_left", "head_camera_right"),
          )
        else:
          self._handles[role] = (
              V4L2Camera(sp, sp.width, sp.height),
              "v4l2",
              (role,),
          )
    except Exception:
      self.close()
      raise

    self._opened = True

  def close(self) -> None:
    """Releases any opened physical or mock capture handles."""
    for handle, _, _ in self._handles.values():
      close_fn = getattr(handle, "close", None) or getattr(
          handle, "release", None
      )
      if callable(close_fn):
        close_fn()
    self._handles.clear()
    self._last_zed_rectification = None
    self._opened = False

  def __enter__(self) -> Fr3DuoCameraController:
    """Opens camera streams when entering a context manager."""
    self.open()
    return self

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    """Closes camera streams when exiting a context manager."""
    self.close()

  def _synthesize_dry_run_frame(
      self, cam_idx: int, role: str, ts: float
  ) -> CameraFrame:
    spec = self.specs.get(role, CameraStreamSpec(role=role))
    w, h = int(spec.width), int(spec.height)
    xs = np.linspace(24, 220, w, dtype=np.uint8)
    ys = np.linspace(32, 210, h, dtype=np.uint8)
    shift = (self._frame_seq * 7 + cam_idx * 37) % 64
    r = np.broadcast_to(
        ((ys[:, None].astype(np.uint16) + shift) % 256).astype(np.uint8),
        (h, w),
    )
    g = np.broadcast_to(xs[None, :], (h, w))
    b = np.full((h, w), 60 + cam_idx * 40, dtype=np.uint8)
    rgb = np.ascontiguousarray(np.stack([r, g, b], axis=-1), dtype=np.uint8)
    return CameraFrame(
        name=role,
        rgb=rgb,
        timestamp_s=ts,
        width=w,
        height=h,
        backend="dry_run",
    )

  def capture_all(self) -> dict[str, CameraFrame]:
    """Captures RGB frames across all configured cameras."""
    if not self._opened:
      self.open()
    self._frame_seq += 1
    self._last_zed_rectification = None

    if self.sim_robot is not None:
      ref_spec = (
          self.specs.get("wrist_camera_left")
          or self.specs.get("wrist_camera_right")
          or next(iter(self.specs.values()), CameraStreamSpec(role="sim"))
      )
      sim_w = (
          int(ref_spec.width) // 2
          if ref_spec.role == "head_stereo" and _is_sbs_stereo_spec(ref_spec)
          else int(ref_spec.width)
      )
      rendered = self.sim_robot.render_sim_cameras(
          width=sim_w, height=int(ref_spec.height)
      )
      return build_sim_camera_frames(rendered, time.monotonic())

    if self.dry_run and not self._handles:
      ts = time.monotonic()
      return {
          cam_name: self._synthesize_dry_run_frame(idx, cam_name, ts)
          for idx, cam_name in enumerate(CANONICAL_CAMERAS)
      }

    frames: dict[str, CameraFrame] = {}
    for handle, default_backend, targets in self._handles.values():
      if hasattr(handle, "capture_rgb"):
        rgb_raw, ts = handle.capture_rgb()
        backend = default_backend
      else:
        rgb_raw, ts, backend = _read_override_frame(handle)

      if len(targets) == 2:
        t_left, t_right = targets
        stereo_spec = self.specs.get("head_stereo")
        raw_h, raw_w = int(rgb_raw.shape[0]), int(rgb_raw.shape[1])
        expected_sbs_w: int | None = None
        if stereo_spec is not None:
          expected_sbs_w = (
              int(stereo_spec.width)
              if _is_sbs_stereo_spec(stereo_spec)
              else int(stereo_spec.width) * 2
          )
        is_sbs_frame = (
            raw_w >= 2
            and (raw_w % 2) == 0
            and (
                (expected_sbs_w is not None and raw_w == expected_sbs_w)
                or (raw_h > 0 and (float(raw_w) / float(raw_h)) >= 2.2)
            )
        )
        if is_sbs_frame:
          left_rgb, right_rgb = self._maybe_rectify_zed_pair(
              *split_zed_sbs_frame(rgb_raw, is_bgr=False)
          )
        else:
          left_rgb, right_rgb = rgb_raw, rgb_raw.copy()
        views = ((t_left, left_rgb), (t_right, right_rgb))
      else:
        views = ((targets[0], rgb_raw),)
      for t_name, view_rgb in views:
        oriented = np.ascontiguousarray(view_rgb, dtype=np.uint8)
        frames[t_name] = CameraFrame(
            name=t_name,
            rgb=oriented,
            timestamp_s=ts,
            width=int(oriented.shape[1]),
            height=int(oriented.shape[0]),
            backend=backend,
        )
    return frames

  def check_freshness(
      self,
      *,
      max_age_s: float = 0.30,
      num_reads: int = 3,
      interval_s: float = 0.05,
  ) -> CameraFreshnessReport:
    """Verifies frame age (`<= max_age_s`), timestamps, and pixel variance."""
    rounds = max(1, int(num_reads))
    history: dict[str, list[tuple[CameraFrame, float]]] = {}

    if (
        not self.dry_run
        and self.sim_robot is None
        and not self.backend_overrides
    ):
      self.capture_all()
      if interval_s > 0.0:
        time.sleep(float(interval_s))

    for idx in range(rounds):
      if idx > 0 and interval_s > 0.0:
        time.sleep(float(interval_s))
      batch = self.capture_all()
      now_s = time.monotonic()
      for name, frame in batch.items():
        history.setdefault(name, []).append((frame, frame.age_s(now_s)))

    errors: list[str] = []
    cam_reports: dict[str, CameraFreshnessEntry] = {}
    if not history:
      return CameraFreshnessReport(
          ok=False,
          cameras={},
          errors=["No camera frames captured."],
      )

    has_full_physical_rig = {
        "head_stereo",
        "wrist_camera_left",
        "wrist_camera_right",
    }.issubset(set(self.specs.keys()))
    if has_full_physical_rig:
      for req_cam in CANONICAL_CAMERAS:
        if req_cam not in history:
          errors.append(f"Missing canonical camera stream: {req_cam}")

    for name, samples in history.items():
      ages = [age for _f, age in samples]
      timestamps = [f.timestamp_s for f, _age in samples]
      last_frame = samples[-1][0]
      max_observed_age = max(ages)
      ts_advancing = len(timestamps) < 2 or timestamps[-1] > timestamps[0]
      pixels_changing = True
      if len(samples) >= 2 and last_frame.backend not in (
          "mujoco_sim",
          "dry_run",
      ):
        first_rgb = samples[0][0].rgb
        pixels_changing = any(
            not np.array_equal(first_rgb, frm.rgb) for frm, _ in samples[1:]
        )
      mean_val = last_frame.brightness_mean
      std_val = last_frame.pixel_std

      cam_ok = True
      if max_observed_age > float(max_age_s):
        cam_ok = False
        errors.append(
            f"{name}: stale frame age {max_observed_age:.3f}s >"
            f" {max_age_s:.3f}s"
        )
      if not ts_advancing:
        cam_ok = False
        errors.append(f"{name}: frozen timestamp ({timestamps[-1]:.6f})")
      if not pixels_changing:
        cam_ok = False
        errors.append(
            f"{name}: frozen pixel buffer across {len(samples)} reads"
        )
      if std_val <= 0.5 or mean_val <= 1.0 or mean_val >= 254.5:
        cam_ok = False
        errors.append(
            f"{name}: degenerate image statistics "
            f"(mean={mean_val:.2f}, std={std_val:.2f})"
        )

      cam_reports[name] = CameraFreshnessEntry(
          ok=cam_ok,
          max_age_s=float(max_observed_age),
          timestamps_advancing=ts_advancing,
          pixels_changing=pixels_changing,
          brightness_mean=mean_val,
          pixel_std=std_val,
          width=last_frame.width,
          height=last_frame.height,
          backend=last_frame.backend,
      )

    return CameraFreshnessReport(
        ok=len(errors) == 0,
        cameras=cam_reports,
        errors=errors,
    )


def render_contact_sheet(
    frames: Mapping[str, CameraFrame | np.ndarray],
) -> np.ndarray:
  """Renders a labeled 2x2 RGB mosaic of the 4 canonical Duo cameras."""
  cv2_mod = require_cv2()
  tile_w, tile_h = 640, 360

  def _unpack(
      item: CameraFrame | np.ndarray | None,
  ) -> tuple[np.ndarray | None, int, int, str]:
    if item is None:
      return None, 0, 0, ""
    if isinstance(item, np.ndarray):
      h = int(item.shape[0]) if item.ndim >= 2 else 0
      w = int(item.shape[1]) if item.ndim >= 2 else 0
      return item, w, h, "rgb"
    return item.rgb, int(item.width), int(item.height), str(item.backend)

  for candidate in frames.values():
    _, w0, h0, _ = _unpack(candidate)
    if w0 > 0 and h0 > 0:
      tile_w = min(640, w0)
      tile_h = max(120, int(round(tile_w * h0 / w0)))
      break

  tiles: list[np.ndarray] = []
  for cam_name in CANONICAL_CAMERAS:
    rgb_arr, width, height, backend = _unpack(frames.get(cam_name))
    if rgb_arr is None or rgb_arr.size == 0:
      tile = np.full((tile_h, tile_w, 3), 28, dtype=np.uint8)
      label = f"{cam_name}: MISSING"
    else:
      if rgb_arr.shape[0] != tile_h or rgb_arr.shape[1] != tile_w:
        tile = cv2_mod.resize(
            rgb_arr, (tile_w, tile_h), interpolation=cv2_mod.INTER_AREA
        )
      else:
        tile = rgb_arr.copy()
      label = f"{cam_name} {width}x{height} {backend}".strip()

    banner_h = min(tile_h, 26)
    tile[:banner_h] //= 4
    cv2_mod.putText(
        tile,
        label,
        (10, banner_h - 6),
        cv2_mod.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 248, 255),
        1,
        cv2_mod.LINE_AA,
    )
    tiles.append(tile)

  top_row = np.concatenate([tiles[0], tiles[1]], axis=1)
  bottom_row = np.concatenate([tiles[2], tiles[3]], axis=1)
  return np.ascontiguousarray(np.concatenate([top_row, bottom_row], axis=0))
