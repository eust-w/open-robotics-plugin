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

"""Stage 1 read-only hardware discovery and FK cross-check for FR3 Duo."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from fr3_arm_controller import probe_arm_readonly
from fr3_arm_controller import tcp_port_reachable
from fr3_camera_controller import (
    UsbPermissionsReport,
    UvcDeviceInfo,
    V4L2Camera,
    _DictCompatMixin,
    check_linux_usb_permissions,
    discover_uvc_devices,
    require_cv2,
    resolve_zed_conf_path,
)
from fr3_config import (
    RobotConfig,
    detect_preempt_rt,
    normalize_robot_config,
    resolve_realtime_config,
    resolve_stage_output_dir,
    save_rgb_png,
    save_robot_config,
)
from fr3_kinematics import compute_arm_fk
from fr3_mjcf import (
    build_fr3_duo_mjcf,
    find_menagerie_dir,
)
from robotiq_gripper_controller import (
    resolve_serial_ports,
    select_gripper_ports,
)


@dataclass
class ConflictingProcess(_DictCompatMixin):
  """Process holding an open file descriptor on a robot device node."""

  pid: int
  process: str
  device_path: str
  device_label: str

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary of process conflict info."""
    return {
        "pid": self.pid,
        "process": self.process,
        "device_path": self.device_path,
        "device_label": self.device_label,
    }


@dataclass
class NominalArmFk(_DictCompatMixin):
  """Nominal forward kinematics summary at an arm's configured rest pose."""

  rest_pose_rad: list[float]
  tcp_pos_base_m: list[float]
  tcp_z_axis_base: list[float]

  def to_dict(self) -> dict[str, list[float]]:
    """Returns a JSON-serializable dictionary of nominal arm FK."""
    return {
        "rest_pose_rad": list(self.rest_pose_rad),
        "tcp_pos_base_m": list(self.tcp_pos_base_m),
        "tcp_z_axis_base": list(self.tcp_z_axis_base),
    }


@dataclass
class FciTargetStatus(_DictCompatMixin):
  """TCP 1337 reachability status for one FR3 FCI controller IP."""

  ip: str
  tcp_1337_reachable: bool

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary of FCI target status."""
    return {
        "ip": self.ip,
        "tcp_1337_reachable": self.tcp_1337_reachable,
    }


@dataclass
class DiscoveryReport(_DictCompatMixin):
  """Structured Stage 1 hardware discovery and preflight report."""

  timestamp_utc: str
  stage: str
  mode: str
  dry_run: bool
  read_only: bool
  menagerie_found: bool
  menagerie_dir: str | None
  preempt_rt_detected: bool
  realtime_profile: dict[str, Any]
  realtime_error: str | None
  fci_targets: dict[str, FciTargetStatus]
  serial_ports: list[dict[str, Any]]
  gripper_selection: dict[str, Any]
  uvc_devices: list[UvcDeviceInfo | dict[str, Any]]
  usb_permissions: UsbPermissionsReport | dict[str, Any]
  conflicting_processes: list[ConflictingProcess]
  nominal_fk: dict[str, NominalArmFk]
  arm_readout: dict[str, Any]
  warnings: list[str]
  advisories: list[str]
  wrote_config: str | None
  wrist_mapping_confirmed: bool
  discovered_config: dict[str, Any]
  report_path: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary of the discovery report."""
    out: dict[str, Any] = {
        "timestamp_utc": self.timestamp_utc,
        "stage": self.stage,
        "mode": self.mode,
        "dry_run": self.dry_run,
        "read_only": self.read_only,
        "menagerie_found": self.menagerie_found,
        "menagerie_dir": self.menagerie_dir,
        "preempt_rt_detected": self.preempt_rt_detected,
        "realtime_profile": self.realtime_profile,
        "realtime_error": self.realtime_error,
        "fci_targets": {
            k: v.to_dict() for k, v in self.fci_targets.items()
        },
        "serial_ports": list(self.serial_ports),
        "gripper_selection": self.gripper_selection,
        "uvc_devices": [
            d.to_dict() if hasattr(d, "to_dict") else d
            for d in self.uvc_devices
        ],
        "usb_permissions": (
            self.usb_permissions.to_dict()
            if hasattr(self.usb_permissions, "to_dict")
            else self.usb_permissions
        ),
        "conflicting_processes": [
            c.to_dict() for c in self.conflicting_processes
        ],
        "nominal_fk": {k: v.to_dict() for k, v in self.nominal_fk.items()},
        "arm_readout": self.arm_readout,
        "warnings": list(self.warnings),
        "advisories": list(self.advisories),
        "wrote_config": self.wrote_config,
        "wrist_mapping_confirmed": self.wrist_mapping_confirmed,
        "discovered_config": self.discovered_config,
    }
    if self.report_path is not None:
      out["report_path"] = self.report_path
    return out


def _capture_wrist_preview_png(
    cfg: RobotConfig, preview_path: Path
) -> Path | None:
  """Grabs a side-by-side 1-frame preview of wrist_camera_left/right."""
  tiles: list[np.ndarray] = []
  try:
    cv2 = require_cv2()
  except RuntimeError:
    return None

  for role, label_prefix in (
      ("wrist_camera_left", "A / wrist_camera_left"),
      ("wrist_camera_right", "B / wrist_camera_right"),
  ):
    spec = cfg.cameras[role]
    frame_rgb: np.ndarray | None = None
    try:
      cam = V4L2Camera(spec, spec.width, spec.height)
      try:
        frame_rgb, _ = cam.capture_rgb()
      finally:
        cam.close()
    except (RuntimeError, OSError, ValueError, KeyError):
      frame_rgb = None
    if frame_rgb is None:
      return None

    tile = cv2.resize(frame_rgb, (640, 360), interpolation=cv2.INTER_AREA)
    banner_h = 28
    tile[:banner_h] = (tile[:banner_h].astype(np.uint16) // 4).astype(np.uint8)
    ser_tag = f"serial={spec.serial}" if spec.serial else spec.device_path
    cv2.putText(
        tile,
        f"{label_prefix} ({ser_tag})",
        (10, banner_h - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 248, 255),
        1,
        cv2.LINE_AA,
    )
    tiles.append(tile)

  mosaic = np.ascontiguousarray(np.concatenate(tiles, axis=1))
  save_rgb_png(mosaic, preview_path)
  return preview_path


def _confirm_wrist_camera_mapping(
    cfg: RobotConfig,
    *,
    preview_dir: Path,
    input_fn: Callable[[str], str],
) -> bool:
  """Prompts operator to confirm or swap unhinted left/right wrist cameras."""
  wl = cfg.cameras.wrist_camera_left
  wr = cfg.cameras.wrist_camera_right
  preview_file = _capture_wrist_preview_png(
      cfg, preview_dir / "wrist_camera_preview.png"
  )

  print("\nAuto-bound RealSense wrist cameras without udev/config mapping:")
  print(
      f"  [A] wrist_camera_left : serial={wl.serial or 'n/a'} "
      f"({wl.device_path or 'n/a'})"
  )
  print(
      f"  [B] wrist_camera_right: serial={wr.serial or 'n/a'} "
      f"({wr.device_path or 'n/a'})"
  )
  if preview_file is not None:
    print(f"  Preview image: {preview_file}")

  reply = input_fn(
      "Confirm left/right wrist mapping ([K]eep A=left/B=right, or [s]wap): "
  )
  if reply.strip().lower() in ("s", "swap"):
    wl.serial, wr.serial = wr.serial, wl.serial
    wl.device_path, wr.device_path = wr.device_path, wl.device_path
    print(
        "  Swapped wrist cameras -> "
        f"left={wl.serial or wl.device_path}, "
        f"right={wr.serial or wr.device_path}"
    )
  return True


def _discover_grippers(
    cfg: RobotConfig, *, rediscover: bool
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
  """Resolves Robotiq serial ports or reuses valid configured paths."""
  has_configured = bool(
      not rediscover
      and cfg.arms.left.gripper_port
      and cfg.arms.right.gripper_port
      and os.path.exists(cfg.arms.left.gripper_port)
      and os.path.exists(cfg.arms.right.gripper_port)
  )
  if not has_configured:
    serial_ports = resolve_serial_ports()
    cfg_grippers = {
        "left": cfg.arms.left.gripper_port,
        "right": cfg.arms.right.gripper_port,
    }
    return serial_ports, select_gripper_ports(serial_ports, cfg_grippers), (
        serial_ports
    )

  selection: dict[str, Any] = {
      "left": cfg.arms.left.gripper_port,
      "right": cfg.arms.right.gripper_port,
      "warnings": [],
      "advisories": [],
  }
  perm_nodes = [
      {
          "path": port,
          "resolved_path": os.path.realpath(port),
          "device_role_hint": "robotiq_rs485",
          "serial_rw_ok": os.access(port, os.R_OK | os.W_OK),
      }
      for port in (cfg.arms.left.gripper_port, cfg.arms.right.gripper_port)
      if port
  ]
  return [], selection, perm_nodes


def _discover_cameras(
    cfg: RobotConfig, *, rediscover: bool
) -> list[UvcDeviceInfo]:
  """Discovers UVC devices or reuses valid configured camera paths."""
  configured_cams = cfg.cameras.as_spec_map()
  has_configured = bool(
      not rediscover
      and all(
          sp.device_path and os.path.exists(sp.device_path)
          for sp in configured_cams.values()
      )
  )
  if not has_configured:
    return discover_uvc_devices()

  return [
      UvcDeviceInfo(
          name=role,
          family="zed" if role == "head_stereo" else "realsense",
          device_path=sp.device_path,
          video_node=os.path.realpath(sp.device_path),
          serial=sp.serial,
          vendor_id="",
          product_id="",
          busnum="",
          usb_speed_mbps=0.0,
          is_usb3=False,
          interface="",
          power_control="",
          v4l2_rw_ok=os.access(sp.device_path, os.R_OK | os.W_OK),
      )
      for role, sp in configured_cams.items()
  ]


def _collect_hardware_device_labels(
    serial_nodes: Sequence[Mapping[str, Any]],
    uvc_devices: Sequence[UvcDeviceInfo | Mapping[str, Any]],
) -> dict[str, str]:
  """Maps resolved `/dev/...` paths for discovered hardware to display labels."""
  dev_labels: dict[str, str] = {}
  for sp in serial_nodes:
    raw = str(sp.get("path") or "").strip()
    resolved = str(sp.get("resolved_path") or raw).strip()
    role = str(sp.get("device_role_hint") or "serial_port")
    for cand in (raw, resolved):
      if cand:
        key = os.path.realpath(cand)
        dev_labels.setdefault(key, f"serial port {raw or resolved} ({role})")

  for dev in uvc_devices:
    raw = str(dev.get("device_path") or "").strip()
    vnode = str(dev.get("video_node") or raw).strip()
    fam = str(dev.get("family") or "uvc").upper()
    for cand in (raw, vnode):
      if cand:
        key = os.path.realpath(cand)
        dev_labels.setdefault(key, f"{fam} camera {vnode or raw}")
  return dev_labels


def find_conflicting_hardware_processes(
    serial_nodes: Sequence[Mapping[str, Any]],
    uvc_devices: Sequence[UvcDeviceInfo | Mapping[str, Any]],
    *,
    proc_root: Path = Path("/proc"),
    own_pid: int | None = None,
) -> list[ConflictingProcess]:
  """Finds other Linux processes holding open FDs on robot serial/V4L2 nodes."""
  dev_labels = _collect_hardware_device_labels(serial_nodes, uvc_devices)
  if not dev_labels or not proc_root.is_dir():
    return []

  self_pid = os.getpid() if own_pid is None else int(own_pid)
  conflicts: list[ConflictingProcess] = []
  seen: set[tuple[int, str]] = set()
  try:
    proc_entries = sorted(proc_root.iterdir(), key=lambda p: p.name)
  except OSError:
    return []

  for pid_dir in proc_entries:
    if not pid_dir.name.isdigit():
      continue
    pid = int(pid_dir.name)
    if pid == self_pid:
      continue
    fd_dir = pid_dir / "fd"
    try:
      fd_entries = list(fd_dir.iterdir())
    except OSError:
      continue

    comm: str | None = None
    for fd_link in fd_entries:
      try:
        target = os.readlink(fd_link)
      except OSError:
        continue
      if not target.startswith("/dev/"):
        continue
      norm_target = os.path.realpath(target)
      label = dev_labels.get(norm_target) or dev_labels.get(target)
      if label is None:
        continue
      key = (pid, norm_target)
      if key in seen:
        continue
      seen.add(key)
      if comm is None:
        try:
          comm = (
              (pid_dir / "comm").read_text(encoding="utf-8").strip()
              or "unknown"
          )
        except OSError:
          comm = "unknown"
      conflicts.append(
          ConflictingProcess(
              pid=pid,
              process=comm,
              device_path=norm_target,
              device_label=label,
          )
      )
  return conflicts


def _append_usb_permission_warnings(
    usb_permissions: UsbPermissionsReport | Mapping[str, Any],
    uvc_devices: Sequence[UvcDeviceInfo | Mapping[str, Any]],
    warnings_list: list[str],
) -> None:
  """Appends human-actionable dialout/video R/W permission warnings."""
  serial_issues = usb_permissions.get("serial_access_issues") or []
  if serial_issues:
    issue_paths = ", ".join(
        str(s.get("path") or s.get("resolved_path") or "unknown")
        for s in serial_issues
    )
    if not usb_permissions.get("in_dialout_group", False):
      warnings_list.append(
          f"Serial port(s) ({issue_paths}) lack R/W permission and user is "
          "not in 'dialout' group. Run: "
          'sudo usermod -aG video,dialout "$USER"'
      )
    else:
      warnings_list.append(
          f"Serial port(s) ({issue_paths}) lack R/W permission."
      )

  v4l2_issues = usb_permissions.get("v4l2_access_issues") or []
  if uvc_devices and v4l2_issues:
    issue_paths = ", ".join(
        str(d.get("video_node") or d.get("device_path") or "unknown")
        for d in v4l2_issues
    )
    if not usb_permissions.get("in_video_group", False):
      warnings_list.append(
          f"V4L2 camera node(s) ({issue_paths}) lack R/W permission and user "
          "is not in 'video' group. Run: "
          'sudo usermod -aG video,dialout "$USER"'
      )
    else:
      warnings_list.append(
          f"V4L2 camera node(s) ({issue_paths}) lack R/W permission."
      )


def _auto_bind_devices(
    cfg: RobotConfig,
    gripper_selection: Mapping[str, Any],
    uvc_devices: Sequence[UvcDeviceInfo | Mapping[str, Any]],
) -> list[str]:
  """Auto-binds discovered gripper ports, ZED stereo, and D405 wrists."""
  for side in ("left", "right"):
    arm_cfg = cfg.arms[side]
    if not arm_cfg.gripper_port and gripper_selection.get(side):
      arm_cfg.gripper_port = str(gripper_selection[side])

  zed_cams = [d for d in uvc_devices if d.get("family") == "zed"]
  if zed_cams:
    zed = zed_cams[0]
    z_ser = str(zed.get("serial") or "")
    z_dev = str(zed.get("device_path") or zed.get("video_node") or "")
    head_cfg = cfg.cameras.head_stereo
    found_conf = resolve_zed_conf_path(
        serial=z_ser, explicit_path=head_cfg.zed_conf_path
    )
    if found_conf is not None and (not z_ser or z_ser.upper() == "OV580"):
      conf_stem = found_conf.stem
      if conf_stem.lower().startswith("sn") and len(conf_stem) > 2:
        z_ser = conf_stem[2:]
        for d in uvc_devices:
          if d.get("family") == "zed" and not d.get("serial"):
            d["serial"] = z_ser
    if not head_cfg.serial and z_ser:
      head_cfg.serial = z_ser
    if not head_cfg.device_path and z_dev:
      head_cfg.device_path = z_dev
    if found_conf is not None and (
        not head_cfg.zed_conf_path or Path(head_cfg.zed_conf_path).is_dir()
    ):
      head_cfg.zed_conf_path = str(found_conf)

  unique_rs = sorted(
      (d for d in uvc_devices if d.get("family") == "realsense"),
      key=lambda d: (
          str(d.get("serial") or ""),
          str(d.get("device_path") or d.get("video_node") or ""),
      ),
  )
  assigned_rs_indices: set[int] = set()
  for side in ("left", "right"):
    w_cfg = cfg.cameras[f"wrist_camera_{side}"]
    matched_idx: int | None = None
    if w_cfg.serial:
      for idx, d in enumerate(unique_rs):
        if idx not in assigned_rs_indices and d.get("serial") == w_cfg.serial:
          matched_idx = idx
          break
    if matched_idx is None:
      hint_matches = [
          idx
          for idx, d in enumerate(unique_rs)
          if idx not in assigned_rs_indices and d.get("side_hint") == side
      ]
      if len(hint_matches) == 1:
        matched_idx = hint_matches[0]
    if matched_idx is not None:
      assigned_rs_indices.add(matched_idx)
      rs = unique_rs[matched_idx]
      if not w_cfg.serial and rs.get("serial"):
        w_cfg.serial = str(rs["serial"])
      dev_cand = rs.get("device_path") or rs.get("video_node")
      if not w_cfg.device_path and dev_cand:
        w_cfg.device_path = str(dev_cand)

  auto_bound_wrists: list[str] = []
  for side in ("left", "right"):
    w_cfg = cfg.cameras[f"wrist_camera_{side}"]
    if w_cfg.serial or w_cfg.device_path:
      continue
    next_idx = next(
        (
            idx
            for idx in range(len(unique_rs))
            if idx not in assigned_rs_indices
        ),
        None,
    )
    if next_idx is not None:
      assigned_rs_indices.add(next_idx)
      rs = unique_rs[next_idx]
      if rs.get("serial"):
        w_cfg.serial = str(rs["serial"])
      dev_cand = rs.get("device_path") or rs.get("video_node")
      if dev_cand:
        w_cfg.device_path = str(dev_cand)
      auto_bound_wrists.append(
          f"wrist_camera_{side}=serial:{w_cfg.serial or 'n/a'} "
          f"({w_cfg.device_path or 'n/a'})"
      )
  return auto_bound_wrists


def _print_discovery_summary(
    *,
    cfg: RobotConfig,
    report: DiscoveryReport | Mapping[str, Any],
    menagerie_path: Path | None,
) -> None:
  """Prints concise human-readable Stage 1 discovery telemetry."""
  is_preempt_rt = bool(report["preempt_rt_detected"])
  rt_profile = report["realtime_profile"]
  rt_summary = (
      f"realtime={rt_profile.get('realtime_config', 'kEnforce')} "
      "(PREEMPT_RT=True)"
      if is_preempt_rt
      else "realtime=kIgnore (PREEMPT_RT=False, non-RT OK)"
  )
  model_src = "menagerie"
  serial_ports = report["serial_ports"]
  uvc_devices = report["uvc_devices"]
  print(
      f"[{str(report['mode']).upper()}] Stage 1 FR3 Duo Discovery | "
      f"model={model_src} | {rt_summary} | "
      f"serial_ports={len(serial_ports)} | uvc_devices={len(uvc_devices)} | "
      f"menagerie={menagerie_path is not None}"
  )

  arm_readout = report["arm_readout"]
  fci_reachability = report["fci_targets"]
  for arm in ("left", "right"):
    rd = arm_readout[arm]
    fci_info = fci_reachability.get(arm, {})
    fci_reach = bool(fci_info.get("tcp_1337_reachable"))
    fci_str = (
        f"ip={fci_info.get('ip')} "
        f"(tcp_1337={'reachable' if fci_reach else 'UNREACHABLE'})"
    )
    if rd.get("connected") is False:
      print(
          f"  {arm:5s}: {fci_str} | status=NOT_CONNECTED | "
          f"nominal_tcp_base={rd['tcp_pos_base_m']} m"
      )
      continue
    da = rd.get("desk_audit") or {}
    da_str = ""
    if da.get("checked"):
      da_str = (
          f" | mode={da.get('robot_mode')} | m_total={da.get('m_total_kg')} "
          f"kg | grav_err={da.get('gravity_angle_err_deg')} deg | "
          f"fci_f_ext={da.get('fci_ext_force_norm_n')} N"
      )
    print(
        f"  {arm:5s}: {fci_str} | tcp_base={rd['tcp_pos_base_m']} m | "
        f"fk_err={rd['fk_pos_error_m'] * 1000.0:.2f} mm | "
        f"|f_ext|={rd['free_air_force_norm_n']:.2f} N{da_str}"
    )

  for idx, sp in enumerate(serial_ports):
    role = str(sp.get("device_role_hint", "unknown_serial"))
    path = str(sp.get("path", ""))
    rpath = str(sp.get("resolved_path", path))
    ser = str(sp.get("serial") or "n/a")
    vid_pid = (
        f"{sp.get('vendor_id')}:{sp.get('product_id')}"
        if sp.get("vendor_id") and sp.get("product_id")
        else "n/a"
    )
    side_str = f" side={sp['side_hint']}" if sp.get("side_hint") else ""
    lat_str = (
        f" latency={sp['latency_timer_ms']}ms"
        if sp.get("latency_timer_ms") is not None
        else ""
    )
    target_str = f" -> {rpath}" if rpath and rpath != path else ""
    print(
        f"  serial[{idx}]: [{role}] {path}{target_str} | "
        f"usb={vid_pid} serial={ser}{side_str}{lat_str}"
    )

  for idx, dev in enumerate(uvc_devices):
    fam = str(dev.get("family", "uvc")).upper()
    name = str(dev.get("name", "unknown"))
    ser = str(dev.get("serial") or "n/a")
    vnode = str(dev.get("video_node") or dev.get("device_path") or "n/a")
    dpath = str(dev.get("device_path") or vnode)
    speed = float(dev.get("usb_speed_mbps") or 0.0)
    side_str = f" side={dev['side_hint']}" if dev.get("side_hint") else ""
    speed_str = f" {speed:.0f}Mbps" if speed > 0 else ""
    path_str = f" ({dpath})" if dpath and dpath != vnode else ""
    print(
        f"  camera[{idx}]: [{fam}] {name} | serial={ser} | "
        f"node={vnode}{path_str}{speed_str}{side_str}"
    )

  wrote_cfg_path = report.get("wrote_config")
  if wrote_cfg_path is not None:
    hs = cfg.cameras.head_stereo
    wl = cfg.cameras.wrist_camera_left
    wr = cfg.cameras.wrist_camera_right
    gl = cfg.arms.left.gripper_port or "n/a"
    gr = cfg.arms.right.gripper_port or "n/a"
    print(
        f"  config : wrote {wrote_cfg_path} | "
        f"head_stereo={hs.serial or 'n/a'} ({hs.device_path or 'n/a'}) | "
        f"wrist_left={wl.serial or 'n/a'} ({wl.device_path or 'n/a'}) | "
        f"wrist_right={wr.serial or 'n/a'} ({wr.device_path or 'n/a'}) | "
        f"grippers=left:{gl}, right:{gr}"
    )
  for warning in report["warnings"]:
    print(f"  WARNING: {warning}")
  for advisory in report["advisories"]:
    print(f"  ADVISORY: {advisory}")


def run_discovery(
    config: RobotConfig | Mapping[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
    output_json: Path | None = None,
    write_config: Path | None = None,
    read_only: bool = False,
    realtime: str = "auto",
    ping_timeout_sec: float = 0.25,
    strict_preflight: bool = False,
    rediscover: bool = False,
    yes: bool = False,
    input_fn: Callable[[str], str] | None = None,
) -> tuple[int, DiscoveryReport]:
  """Runs Stage 1 read-only hardware discovery and returns `(rc, report)`."""
  cfg = normalize_robot_config(config)
  is_preempt_rt = detect_preempt_rt()
  realtime_error: str | None = None
  try:
    rt_profile = resolve_realtime_config(realtime, is_preempt_rt=is_preempt_rt)
  except RuntimeError as exc:
    realtime_error = str(exc)
    rt_profile = resolve_realtime_config("auto", is_preempt_rt=is_preempt_rt)

  serial_ports, gripper_selection, serial_perm_nodes = _discover_grippers(
      cfg, rediscover=rediscover
  )
  uvc_devices = _discover_cameras(cfg, rediscover=rediscover)
  usb_permissions = check_linux_usb_permissions(
      uvc_devices, serial_ports=serial_perm_nodes
  )
  menagerie_path = find_menagerie_dir()
  mj_model, _ = build_fr3_duo_mjcf(cfg, menagerie_dir=menagerie_path)
  if menagerie_path is None:
    menagerie_path = find_menagerie_dir()

  nominal_fk: dict[str, NominalArmFk] = {}
  fci_reachability: dict[str, FciTargetStatus] = {}
  for arm in ("left", "right"):
    q_rest = cfg.arms[arm].rest_pose_rad
    pos_b, rot_b = compute_arm_fk(mj_model, arm, q_rest)
    nominal_fk[arm] = NominalArmFk(
        rest_pose_rad=[round(float(v), 6) for v in q_rest],
        tcp_pos_base_m=[round(float(v), 5) for v in pos_b],
        tcp_z_axis_base=[round(float(v), 5) for v in rot_b[:, 2]],
    )
    ip = cfg.arms[arm].ip
    fci_reachability[arm] = FciTargetStatus(
        ip=ip,
        tcp_1337_reachable=tcp_port_reachable(
            ip, timeout_s=ping_timeout_sec
        ),
    )

  warnings_list: list[str] = list(gripper_selection.get("warnings", []))
  advisories_list: list[str] = list(gripper_selection.get("advisories", []))
  _append_usb_permission_warnings(usb_permissions, uvc_devices, warnings_list)
  conflicting_processes = find_conflicting_hardware_processes(
      serial_perm_nodes, uvc_devices
  )
  for proc_conflict in conflicting_processes:
    warnings_list.append(
        f"Process '{proc_conflict.process}' (PID {proc_conflict.pid}) "
        f"already holds {proc_conflict.device_label}; kill conflicting "
        "camera/serial processes before continuing."
    )

  arm_readout: dict[str, Any] = {}
  rt_cfg_name = str(rt_profile.get("realtime_config", "kIgnore"))
  for arm in ("left", "right"):
    arm_cfg = cfg.arms[arm]
    arm_res = probe_arm_readonly(
        arm,
        arm_cfg.ip,
        mj_model,
        rest_pose_rad=arm_cfg.rest_pose_rad,
        tcp_offset_z_m=cfg.tcp_offset_z_m,
        realtime_cfg_name=rt_cfg_name,
        timeout_s=ping_timeout_sec,
        tcp_reachable=bool(fci_reachability[arm]["tcp_1337_reachable"]),
    )
    for w in arm_res.get("warnings", []):
      if w not in warnings_list:
        warnings_list.append(w)
    if (
        fci_reachability[arm]["tcp_1337_reachable"]
        and not arm_res["connected"]
    ):
      fci_conflict_hint = (
          f"FR3 {arm} ({arm_cfg.ip}): TCP 1337 reachable but FCI probe "
          "failed; kill any other process controlling the arm and verify "
          "Desk FCI mode is active."
      )
      if fci_conflict_hint not in warnings_list:
        warnings_list.append(fci_conflict_hint)
    for adv in arm_res.get("advisories", []):
      if adv not in advisories_list:
        advisories_list.append(adv)
    arm_readout[arm] = {
        "connected": arm_res["connected"],
        "connection_error": arm_res["connection_error"],
        "q_rad": arm_res["q_rad"],
        "tcp_pos_base_m": arm_res["tcp_pos_base_m"],
        "fk_pos_error_m": arm_res["fk_pos_error_m"],
        "free_air_force_norm_n": arm_res["free_air_force_norm_n"],
        "gripper_ratio": 0.0,
        "desk_audit": arm_res["desk_audit"],
    }

  auto_bound_wrists = _auto_bind_devices(cfg, gripper_selection, uvc_devices)
  resolved_out_dir = resolve_stage_output_dir(
      "stage1", output_dir, dry_run=False
  )

  wrist_confirmed = False
  if auto_bound_wrists:
    can_prompt = bool(
        not yes
        and write_config is not None
        and not read_only
        and (input_fn is not None or sys.stdin.isatty())
    )
    if can_prompt:
      assert write_config is not None
      preview_dir = (
          resolved_out_dir
          if resolved_out_dir is not None
          else write_config.parent
      )
      wrist_confirmed = _confirm_wrist_camera_mapping(
          cfg, preview_dir=preview_dir, input_fn=input_fn or input
      )
    if not wrist_confirmed:
      adv_msg = (
          "Auto-bound RealSense wrist camera(s) without udev side_hint: "
          + ", ".join(auto_bound_wrists)
          + ". Verify physical left/right wrist views in Stage 2 "
          "(uv run scripts/bringup.py --step cameras) and swap in config if "
          "reversed."
      )
      if adv_msg not in advisories_list:
        advisories_list.append(adv_msg)

  cfg.realtime_mode = "auto" if realtime_error else realtime
  wrote_cfg_path: str | None = None
  if write_config is not None and not read_only:
    save_robot_config(cfg, write_config)
    wrote_cfg_path = str(write_config)

  report = DiscoveryReport(
      timestamp_utc=datetime.now(timezone.utc).isoformat(),
      stage="stage1_discover_hardware",
      mode="hardware_readonly",
      dry_run=False,
      read_only=bool(read_only or write_config is None),
      menagerie_found=menagerie_path is not None,
      menagerie_dir=(
          str(menagerie_path) if menagerie_path is not None else None
      ),
      preempt_rt_detected=is_preempt_rt,
      realtime_profile=rt_profile,
      realtime_error=realtime_error,
      fci_targets=fci_reachability,
      serial_ports=serial_ports,
      gripper_selection=gripper_selection,
      uvc_devices=uvc_devices,
      usb_permissions=usb_permissions,
      conflicting_processes=conflicting_processes,
      nominal_fk=nominal_fk,
      arm_readout=arm_readout,
      warnings=warnings_list,
      advisories=advisories_list,
      wrote_config=wrote_cfg_path,
      wrist_mapping_confirmed=wrist_confirmed,
      discovered_config=cfg.to_dict(),
  )
  json_path = output_json or (
      resolved_out_dir / "discovery_report.json"
      if resolved_out_dir is not None
      else None
  )
  if json_path is not None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    report.report_path = str(json_path)

  _print_discovery_summary(
      cfg=cfg, report=report, menagerie_path=menagerie_path
  )
  if realtime_error is not None:
    print(f"ERROR: {realtime_error}", file=sys.stderr)
    return 1, report
  if strict_preflight and warnings_list:
    print(
        f"ERROR: --strict-preflight failed with {len(warnings_list)} "
        "warning(s).",
        file=sys.stderr,
    )
    return 2, report
  return 0, report
