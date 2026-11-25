#!/usr/bin/env python3
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
#
# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#     "mujoco",
#     "numpy",
#     "opencv-python",
#     "pylibfranka; sys_platform == 'linux' and platform_machine == 'x86_64'",
#     "pyserial",
# ]
# ///

"""Interactive terminal bring-up CLI runner for Franka Research 3 Duo."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import check_cameras
import check_robot
import discover_hardware
import fr3_arm_controller
from fr3_camera_controller import _DictCompatMixin
from fr3_duo_robot import Fr3DuoRobot
from fr3_safety import AbortWatcher
import robotiq_gripper_controller
from fr3_config import (
    BASE_WORLD_Z_M,
    BRACKET_UNDERSIDE_FRONT_X_M,
    DEFAULT_CONFIG_PATH,
    RobotConfig,
    load_robot_config,
    normalize_robot_config,
    resolve_stage_output_dir,
    save_robot_config,
    validate_robot_config_complete,
)
from fr3_mjcf import (
    build_fr3_duo_mjcf,
    find_menagerie_dir,
)

STAGE_ORDER: tuple[str, ...] = (
    "env",
    "preflight",
    "geometry",
    "cameras",
    "gripper",
    "nudge",
    "abort-test",
    "hello-world",
    "all",
)
STEP_CHOICES: tuple[str, ...] = (
    "env",
    "preflight",
    "geometry",
    "cameras",
    "gripper",
    "nudge",
    "abort-test",
    "hello-world",
    "home",
)

_GEOMETRY_CLI_ALLOW_ZERO: dict[str, bool] = {
    "base_height_m": False,
    "table_height_m": True,
    "table_front_offset_m": True,
    "table_width_m": False,
    "table_depth_m": False,
    "tcp_offset_z_m": False,
}
_COMPLETE_CONFIG_REQUIRED_STEPS: frozenset[str] = frozenset({
    "gripper",
    "nudge",
    "abort-test",
    "hello-world",
})

_ACTUATION_DESCRIPTIONS: dict[str, str] = {
    "gripper": (
        "Step 3a: Robotiq 2F-85 Gripper Activation & Cycling"
        " (open -> close -> open)"
    ),
    "nudge": (
        "Step 3b: Single-Joint J1 Nudge (5 deg slow stroke on left & right"
        " arms)"
    ),
    "abort-test": (
        "Step 3c: Abort-and-Unwind Test (10 deg stroke interrupted to test"
        " Hermite C^2 stop & rewind)"
    ),
    "hello-world": (
        "Step 4: Bimanual Hello World (synchronized wave + gripper cycle +"
        " 4-camera video)"
    ),
    "home": "Step 4h: Smooth Bimanual Homing to Canonical Rest Pose",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments for the FR3 Duo bring-up runner."""
  parser = argparse.ArgumentParser(
      description=(
          "Interactive terminal bring-up runner for Franka Research 3 Duo"
          " across Stages 0-4 (env check, preflight discovery, geometry"
          " calibration, camera verification, staged actuation, and homing)."
      )
  )
  parser.add_argument(
      "--config",
      type=Path,
      default=Path(DEFAULT_CONFIG_PATH),
      help="Path to robot configuration JSON.",
  )
  parser.add_argument(
      "--no-write-config",
      action="store_true",
      help="If set, runs preflight without saving robot_config.json.",
  )
  parser.add_argument(
      "--read-only",
      action="store_true",
      help=(
          "Run in read-only mode without writing config or commanding"
          " physical actuation."
      ),
  )
  parser.add_argument(
      "--left-ip", type=str, default=None, help="Left arm FCI IP."
  )
  parser.add_argument(
      "--right-ip", type=str, default=None, help="Right arm FCI IP."
  )
  parser.add_argument(
      "--zed-conf-path",
      type=str,
      default=None,
      help=(
          "Optional ZED SN<serial>.conf file or settings directory stored on"
          " cameras.head_stereo.zed_conf_path."
      ),
  )
  parser.add_argument(
      "--base-height-m",
      type=float,
      default=None,
      help="Floor to flat center bracket underside height in meters.",
  )
  parser.add_argument(
      "--table-height-m",
      type=float,
      default=None,
      help="Floor to tabletop height in meters.",
  )
  parser.add_argument(
      "--table-front-offset-m",
      type=float,
      default=None,
      help=(
          "Horizontal gap from bracket underside front lip to table front [m]."
      ),
  )
  parser.add_argument(
      "--table-width-m",
      type=float,
      default=None,
      help="Tabletop width along Y in meters.",
  )
  parser.add_argument(
      "--table-depth-m",
      type=float,
      default=None,
      help="Tabletop depth along X in meters.",
  )
  parser.add_argument(
      "--tcp-offset-z-m",
      type=float,
      default=None,
      help="Robotiq fingertip Z offset from flange in meters (0.145 or 0.185).",
  )
  parser.add_argument(
      "--confirm-geometry",
      action="store_true",
      help="Mark physical cell geometry & fingertip measurements confirmed.",
  )
  parser.add_argument(
      "--touch-geometry",
      action="store_true",
      help=(
          "Calibrate z_table_m (and optionally table_front_offset_m) via"
          " read-only FCI joint state + FK while closed gripper(s) rest on"
          " the tabletop via the FR3 cuff button (zero arm motion)."
      ),
  )
  parser.add_argument(
      "--touch-arm",
      choices=("auto", "left", "right", "both"),
      default="auto",
      help=(
          "Which enabled arm(s) rest on the tabletop during"
          " --touch-geometry (default: auto -> left if enabled else right;"
          " pass 'both' to average both arms)."
      ),
  )
  parser.add_argument(
      "--close-grippers",
      action="store_true",
      help=(
          "Activate and close enabled Robotiq grippers before cuff touch"
          " during --step geometry (skipped under --dry-run / --read-only)."
      ),
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=None,
      help="Base directory for bring-up step outputs and summary JSON.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help=(
          "Verify MJCF kinematics and trajectory limits without opening live"
          " hardware or commanding motion."
      ),
  )
  parser.add_argument(
      "--realtime",
      choices=("auto", "enforce", "ignore"),
      default="auto",
      help="FCI realtime scheduling policy.",
  )
  target_group = parser.add_mutually_exclusive_group()
  target_group.add_argument(
      "--up-to",
      choices=STAGE_ORDER,
      default="all",
      help="Run progressively up to and including this stage.",
  )
  target_group.add_argument(
      "--step",
      choices=STEP_CHOICES,
      default=None,
      help="Run only this single bring-up step (or 'home').",
  )
  parser.add_argument(
      "--yes",
      "-y",
      action="store_true",
      help="Auto-confirm interactive actuation prompts without prompting.",
  )
  return parser.parse_args(argv)


@dataclass
class StepStatus(_DictCompatMixin):
  """Per-step outcome recorded in `bringup_summary.json`."""

  status: str
  exit_code: int | None = None
  error: str | None = None
  warnings: list[str] = field(default_factory=list)
  warning: str | None = None
  read_only: bool | None = None
  extras: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Returns a compact JSON-serializable dictionary of step status."""
    out: dict[str, Any] = {"status": self.status}
    if self.exit_code is not None:
      out["exit_code"] = self.exit_code
    if self.error:
      out["error"] = self.error
    if self.warnings:
      out["warnings"] = list(self.warnings)
    if self.warning:
      out["warning"] = self.warning
    if self.read_only is not None:
      out["read_only"] = self.read_only
    out.update(self.extras)
    return out


@dataclass
class BringupSummary(_DictCompatMixin):
  """Top-level `bringup_summary.json` schema."""

  timestamp_utc: str
  mode: str
  up_to: str | None
  step: str | None
  read_only: bool
  status: str = "in_progress"
  steps: dict[str, StepStatus | dict[str, Any]] = field(default_factory=dict)
  artifacts: dict[str, str] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Returns a JSON-serializable dictionary of the bring-up summary."""
    return {
        "timestamp_utc": self.timestamp_utc,
        "mode": self.mode,
        "up_to": self.up_to,
        "step": self.step,
        "read_only": self.read_only,
        "status": self.status,
        "steps": {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in self.steps.items()
        },
        "artifacts": dict(self.artifacts),
    }


def _write_summary(
    out_dir: Path, summary: BringupSummary | Mapping[str, Any]
) -> None:
  try:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "bringup_summary.json"
    payload = summary.to_dict() if hasattr(summary, "to_dict") else summary
    summary_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
  except OSError as exc:
    print(
        f"Warning: Failed to write bringup_summary.json: {exc}",
        file=sys.stderr,
    )


def _call_stage(
    fn: Callable[..., tuple[int, Any]], *args: Any, **kwargs: Any
) -> tuple[int, Any, str | None]:
  """Calls a stage `run_*` function and traps unexpected runtime errors."""
  try:
    rc, report = fn(*args, **kwargs)
    return int(rc), report, None
  except Exception as exc:  # pylint: disable=broad-exception-caught
    err_msg = f"{type(exc).__name__}: {exc}"
    print(f"  ERROR: {err_msg}", file=sys.stderr)
    return 1, {}, err_msg


def _fmt_opt(val: Any) -> Any:
  return "N/A" if val is None else val


def _fail_step(
    out_dir: Path,
    summary: BringupSummary,
    step_key: str,
    rc: int,
    error: str | None = None,
) -> int:
  summary.status = "failed"
  summary.steps[step_key] = StepStatus(
      status="failed",
      exit_code=rc,
      error=error,
  )
  _write_summary(out_dir, summary)
  return rc


def _resolve_steps_to_run(args: argparse.Namespace) -> list[str]:
  if args.step is not None:
    return [str(args.step)]
  up_to = str(args.up_to or "all")
  progressive = (
      "env",
      "preflight",
      "geometry",
      "cameras",
      "gripper",
      "nudge",
      "abort-test",
      "hello-world",
  )
  if up_to == "all":
    return list(progressive)
  idx = progressive.index(up_to)
  return list(progressive[: idx + 1])


def _run_env_step(
    cfg: RobotConfig,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    is_hw: bool,
    mode_label: str,
) -> int:
  """Executes Step 0 module, MJCF, and dry-run trajectory pre-checks."""
  print("\n--- Step 0: Environment & Pre-Check ---")
  optional_notes = {
      "serial": "pyserial hardware driver; only required for live hardware",
      "pylibfranka": "Franka FCI binding; only required for live hardware",
  }
  missing_required: list[str] = []
  for mod_name, is_req in (
      ("numpy", True),
      ("mujoco", True),
      ("cv2", True),
      ("serial", is_hw),
      ("pylibfranka", is_hw),
  ):
    try:
      mod = importlib.import_module(mod_name)
      ver = getattr(mod, "__version__", "")
      print(
          f"  [PASS] Python module '{mod_name}'" + (f" ({ver})" if ver else "")
      )
    except ImportError:
      if is_req:
        print(
            f"  [FAIL] Python module '{mod_name}' missing (required for"
            f" mode={mode_label})"
        )
        missing_required.append(mod_name)
      else:
        print(
            f"  [WARN] Python module '{mod_name}' missing"
            f" ({optional_notes.get(mod_name, 'optional dependency')})"
        )

  menagerie_path = find_menagerie_dir()
  if missing_required:
    print(
        f"\n[FAIL] Step 0 environment check failed: missing {missing_required}",
        file=sys.stderr,
    )
    summary.status = "failed"
    summary.steps["env"] = StepStatus(
        status="failed",
        extras={"missing_modules": missing_required},
    )
    _write_summary(out_dir, summary)
    return 1

  step0_dir = out_dir / "step0_precheck"
  step0_dir.mkdir(parents=True, exist_ok=True)
  mjcf_err: str | None = None
  try:
    build_fr3_duo_mjcf(cfg)
    if menagerie_path is None:
      menagerie_path = find_menagerie_dir()
    if menagerie_path is not None:
      print(f"  [PASS] Canonical Duo MJCF ready (Menagerie: {menagerie_path})")
    else:
      print("  [PASS] Canonical Duo MJCF ready (Menagerie)")
  except (RuntimeError, OSError, ValueError) as exc:
    mjcf_err = f"{type(exc).__name__}: {exc}"
    print(f"  ERROR: MJCF compile check failed: {mjcf_err}", file=sys.stderr)

  dry_check_rc, _, dry_check_err = _call_stage(
      check_robot.run_robot_check,
      cfg,
      mode="nudge",
      dry_run=True,
      output_dir=step0_dir,
      output_json=step0_dir / "dry_run_check.json",
  )
  if mjcf_err is not None or dry_check_rc != 0:
    print(
        "\n[FAIL] Step 0 MuJoCo & trajectory pre-check failed.", file=sys.stderr
    )
    summary.status = "failed"
    summary.steps["env"] = StepStatus(
        status="failed",
        error=mjcf_err or dry_check_err,
        extras={
            "mjcf_ok": mjcf_err is None,
            "dry_check_rc": dry_check_rc,
        },
    )
    _write_summary(out_dir, summary)
    return 1

  print(
      "  [PASS] MuJoCo & dry-run pre-check succeeded"
      " (kinematics & safety gates ok)"
  )
  summary.steps["env"] = StepStatus(
      status="ok",
      extras={"menagerie": bool(menagerie_path)},
  )
  return 0


def _run_preflight_step(
    cfg: RobotConfig,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    effective_dry_run: bool,
    is_read_only: bool,
) -> tuple[int, RobotConfig]:
  """Runs Step 1 read-only hardware discovery and updates `cfg` if saved."""
  print("\n--- Step 1: Hardware Discovery & Preflight Audit ---")
  step1_dir = out_dir / "step1_preflight"
  step1_dir.mkdir(parents=True, exist_ok=True)
  s1_json = step1_dir / "discovery_report.json"
  write_cfg_in_place = bool(
      not effective_dry_run and not args.no_write_config and not is_read_only
  )
  rc1, d_rep, err1 = _call_stage(
      discover_hardware.run_discovery,
      cfg,
      realtime=args.realtime,
      output_dir=step1_dir,
      output_json=s1_json,
      write_config=args.config if write_cfg_in_place else None,
      read_only=not (write_cfg_in_place and args.config is not None),
      strict_preflight=not (effective_dry_run or is_read_only),
      yes=args.yes,
  )
  if rc1 != 0:
    print(
        f"\n[FAIL] Step 1 preflight blocked (exit code {rc1}) — fix warnings"
        " above before continuing.",
        file=sys.stderr,
    )
    return _fail_step(out_dir, summary, "preflight", rc1, err1), cfg

  if write_cfg_in_place and args.config is not None and args.config.is_file():
    cfg = load_robot_config(args.config)
  elif d_rep and isinstance(d_rep.get("discovered_config"), Mapping):
    cfg = normalize_robot_config(d_rep["discovered_config"])
  elif d_rep:
    grip_sel = d_rep.get("gripper_selection") or {}
    for side in ("left", "right"):
      if not cfg.arms[side].gripper_port and grip_sel.get(side):
        cfg.arms[side].gripper_port = str(grip_sel[side])
    uvc_devs = d_rep.get("uvc_devices") or []
    if uvc_devs:
      discover_hardware._auto_bind_devices(cfg, grip_sel, list(uvc_devs))
  if args.left_ip:
    cfg.arms.left.ip = str(args.left_ip)
  if args.right_ip:
    cfg.arms.right.ip = str(args.right_ip)
  if args.zed_conf_path is not None:
    cfg.cameras.head_stereo.zed_conf_path = str(args.zed_conf_path).strip()

  if d_rep:
    if d_rep.get("report_path"):
      summary.artifacts["discovery_report"] = d_rep["report_path"]
    elif s1_json.is_file():
      summary.artifacts["discovery_report"] = str(s1_json)
    rt_cfg = (d_rep.get("realtime_profile") or {}).get(
        "realtime_config", "unknown"
    )
    is_rt = d_rep.get("preempt_rt_detected", False)
    print(f"  Realtime Profile: {rt_cfg} (PREEMPT_RT={is_rt})")
    arm_ro = d_rep.get("arm_readout") or {}
    fci_targets = d_rep.get("fci_targets") or {}
    for arm in ("left", "right"):
      rd = arm_ro.get(arm, {})
      da = rd.get("desk_audit") or {}
      reach = (fci_targets.get(arm) or {}).get("tcp_1337_reachable")
      fk_err_mm = float(rd.get("fk_pos_error_m", 0.0)) * 1000.0
      f_norm = float(rd.get("free_air_force_norm_n", 0.0))
      print(
          f"  {arm.capitalize()} Arm: tcp_1337={reach} |"
          f" mode={_fmt_opt(da.get('robot_mode'))} |"
          f" m_total={_fmt_opt(da.get('m_total_kg'))} kg |"
          f" grav_err={_fmt_opt(da.get('gravity_angle_err_deg'))} deg |"
          f" fk_err={fk_err_mm:.2f} mm | free_air={f_norm:.2f} N"
      )
    grippers = d_rep.get("gripper_selection") or {}
    print(
        f"  Robotiq Grippers: left={grippers.get('left')} |"
        f" right={grippers.get('right')}"
    )
    print(
        f"  UVC Cameras: {len(d_rep.get('uvc_devices') or [])} device(s) found"
    )
    notes = [
        *list(d_rep.get("warnings") or []),
        *list(d_rep.get("advisories") or []),
    ]
    if notes:
      print(f"  Advisories/Warnings ({len(notes)}):")
      for note in notes:
        print(f"    - {note}")

  summary.steps["preflight"] = StepStatus(status="ok", exit_code=0)
  return 0, cfg


def _skill_asset_path(filename: str) -> Path:
  """Resolves `assets/<filename>` relative to the `fr3-duo-control` skill."""
  rel = Path("assets") / filename
  for base in (os.path.abspath(__file__), Path(__file__).resolve()):
    candidate = Path(base).parents[1] / rel
    if candidate.is_file():
      return candidate
  return Path(__file__).resolve().parents[1] / rel


def _geometry_reference_image_path() -> Path:
  """Resolves `assets/fr3_duo_dimensions_reference.png` relative to skill."""
  return _skill_asset_path("fr3_duo_dimensions_reference.png")


def _validate_geom_value(
    name: str, val: float, *, allow_zero: bool = False
) -> float:
  if not math.isfinite(val):
    raise ValueError(f"{name} must be finite, got {val!r}")
  if (allow_zero and val < 0.0) or (not allow_zero and val <= 0.0):
    op = ">=" if allow_zero else ">"
    raise ValueError(f"{name} must be {op} 0, got {val!r}")
  return float(val)


def _extract_geometry_cli_updates(
    args: argparse.Namespace,
) -> tuple[dict[str, float], str | None]:
  """Extracts and validates geometry CLI overrides from parsed `args`."""
  updates: dict[str, float] = {}
  for field_name, allow_zero in _GEOMETRY_CLI_ALLOW_ZERO.items():
    raw_val = getattr(args, field_name, None)
    if raw_val is None:
      continue
    try:
      updates[field_name] = _validate_geom_value(
          field_name, float(raw_val), allow_zero=allow_zero
      )
    except ValueError as exc:
      return {}, str(exc)
  return updates, None


_DEFAULT_TOUCH_ARM = "auto"
_ALOHA_TABLE_WIDTH_M = 1.22
_ALOHA_TABLE_DEPTH_M = 0.74
_TOUCH_MAX_Z_SPREAD_M = 0.025
_TOUCH_MAX_ABS_Z_M = 0.15
_TOUCH_MAX_FK_ERR_M = 0.005
_ARM_PERSPECTIVE_DESC: dict[str, str] = {
    "left": "LEFT (+Y, operator's RIGHT facing pedestal)",
    "right": "RIGHT (-Y, operator's LEFT facing pedestal)",
}


def _is_non_interactive_tty(input_fn: Callable[[str], str]) -> bool:
  """Returns True when stdin is non-interactive and no test hook is passed."""
  return input_fn is input and not sys.stdin.isatty()


def _save_geometry_if_enabled(
    cfg: RobotConfig,
    args: argparse.Namespace,
    *,
    write_cfg: bool,
    confirmed: bool = True,
) -> None:
  """Persists `cfg` when writing is enabled, or logs why it stayed in memory."""
  if write_cfg and args.config is not None:
    save_robot_config(cfg, args.config)
    label = "confirmed" if confirmed else "partial (unconfirmed)"
    print(f"  Saved {label} geometry to {args.config}")
  elif not args.dry_run:
    print(
        "  [INFO] Config file not written (--no-write-config/--read-only);"
        " calibrated geometry kept in memory."
    )


def _print_human_table_bounds(
    cfg: RobotConfig, *, touch_mode: bool = False
) -> None:
  """Prints operator-readable tabletop bounds, Z, and raw box extents."""
  table_center, table_half_extents = cfg.table_box_bounds()
  x_front = BRACKET_UNDERSIDE_FRONT_X_M + float(cfg.table_front_offset_m)
  x_back = x_front + float(cfg.table_depth_m)
  y_half = 0.5 * float(cfg.table_width_m)
  print(
      f"  -> Computed z_table_m     : {cfg.z_table_m:+.6f} m"
      f" ({cfg.z_table_m * 1000.0:+.1f} mm vs Duo base)"
  )
  if touch_mode:
    print(
        f"  -> base_height_m (synth.) : {cfg.base_height_m:.6f} m"
        f" (for table_height_m={cfg.table_height_m:.3f} m -> z_table_m)"
    )
  print(
      f"  -> table_front_offset_m   : {cfg.table_front_offset_m:.6f} m"
      f" (x_front={x_front:+.4f} m)"
  )
  print(
      f"  -> Tabletop footprint     : {cfg.table_width_m:.3f} m W"
      f" (Y [{-y_half:+.3f}, {+y_half:+.3f}]) x {cfg.table_depth_m:.3f} m D"
      f" (X [{x_front:+.3f}, {x_back:+.3f}])"
  )
  print(f"  -> tcp_offset_z_m         : {cfg.tcp_offset_z_m:.3f} m")
  print(
      f"  -> Table box bounds       : center={table_center},"
      f" half_extents={table_half_extents}"
  )


def _print_geometry_summary(cfg: RobotConfig, ref_img: Path) -> None:
  """Prints current physical geometry, derived `z_table_m`, and table box."""
  print(f"  Reference diagram : {ref_img}")
  print(
      f"  base_height_m     : {cfg.base_height_m:.6f} m"
      " (floor -> flat center bracket underside)"
  )
  print(f"  table_height_m    : {cfg.table_height_m:.6f} m (floor -> tabletop)")
  _print_human_table_bounds(cfg, touch_mode=False)
  print(f"  geometry_confirmed: {bool(cfg.geometry_confirmed)}")


def _parse_tcp_offset_input(raw: str, default_val: float) -> float:
  """Parses interactive fingertip offset (`1/stock`, `2/extended`, float)."""
  token = raw.strip().lower()
  if not token:
    return float(default_val)
  if token in ("1", "stock", "s", "pads", "2f85", "2f-85"):
    return 0.145
  if token in ("2", "extended", "ext", "aloha"):
    return 0.185
  return _validate_geom_value("tcp_offset_z_m", float(token))


def _prompt_tcp_offset(
    default_val: float,
    input_fn: Callable[[str], str],
    *,
    prefix: str = "5)",
    closed_note: bool = False,
) -> float:
  """Prompts once for Robotiq fingertip Z offset (`1/stock`, `2/extended`)."""
  if closed_note:
    print(
        "  NOTE: Ensure Robotiq gripper(s) are CLOSED before touching the"
        " tabletop so tcp_offset_z_m matches physical closed tips"
        " (e.g. 0.185 m closed extended fingers vs shorter open jaws)."
    )
  tag = " (CLOSED)" if closed_note else ""
  raw = input_fn(
      f"  {prefix} Fingertips{tag}: [1/stock] 2F-85 pads (0.145) |"
      f" [2/extended] ALOHA (0.185) | float [m] (default {default_val:.3f}): "
  )
  return _parse_tcp_offset_input(raw, default_val)


def _stop_geometry_step(
    out_dir: Path, summary: BringupSummary, cfg: RobotConfig
) -> tuple[int, RobotConfig]:
  """Records operator-stopped geometry calibration without mutating `cfg`."""
  print("Geometry calibration stopped; geometry remains unconfirmed.")
  summary.status = "stopped_by_operator"
  summary.steps["geometry"] = StepStatus(
      status="stopped_by_operator",
      extras={"geometry_confirmed": False},
  )
  _write_summary(out_dir, summary)
  return 0, cfg


def _touch_sample_arms(
    cfg: RobotConfig,
    arms: Sequence[str],
) -> dict[str, dict[str, Any]]:
  """Samples read-only FCI joint state and FK for `arms` (zero arm motion)."""
  mj_model, _ = build_fr3_duo_mjcf(cfg)
  samples: dict[str, dict[str, Any]] = {}
  for side in arms:
    arm_cfg = cfg.arms[side]
    samples[side] = fr3_arm_controller.probe_arm_readonly(
        side,
        arm_cfg.ip,
        mj_model,
        rest_pose_rad=arm_cfg.rest_pose_rad,
        tcp_offset_z_m=cfg.tcp_offset_z_m,
        realtime_cfg_name="kIgnore",
    )
  return samples


def _resolve_touch_arms(
    cfg: RobotConfig, touch_arm: str
) -> tuple[list[str], str | None]:
  """Resolves enabled arm side(s) requested via `--touch-arm`."""
  if touch_arm == "auto":
    for side in ("left", "right"):
      if cfg.arms[side].enabled:
        return [side], None
    return [], "no_enabled_arms_for_touch_geometry"
  if touch_arm == "both":
    arms = [side for side in ("left", "right") if cfg.arms[side].enabled]
    if not arms:
      return [], "no_enabled_arms_for_touch_geometry"
    return arms, None
  if not cfg.arms[touch_arm].enabled:
    return [], f"touch_arm_{touch_arm}_disabled"
  return [touch_arm], None


def _prompt_optional_geom_float(
    field_name: str,
    prompt_str: str,
    current_val: float,
    input_fn: Callable[[str], str],
) -> float:
  raw = input_fn(prompt_str).strip()
  if not raw:
    return float(current_val)
  return _validate_geom_value(field_name, float(raw))


def _prompt_touch_setup(
    cfg: RobotConfig,
    args: argparse.Namespace,
    input_fn: Callable[[str], str],
) -> tuple[str, RobotConfig, str | None]:
  """Prompts interactively for closed fingertips and tabletop footprint."""
  updated = normalize_robot_config(cfg)
  try:
    if args.tcp_offset_z_m is None:
      updated.tcp_offset_z_m = _prompt_tcp_offset(
          updated.tcp_offset_z_m, input_fn, prefix="1)", closed_note=True
      )
    if args.table_width_m is None and args.table_depth_m is None:
      preset_raw = (
          input_fn(
              "  2) Tabletop footprint: [1/aloha] 1.220 x 0.740 m |"
              " [2] Custom W x D"
              f" (default {updated.table_width_m:.3f} x"
              f" {updated.table_depth_m:.3f}): "
          )
          .strip()
          .lower()
      )
      if preset_raw in ("1", "aloha"):
        updated.table_width_m = _ALOHA_TABLE_WIDTH_M
        updated.table_depth_m = _ALOHA_TABLE_DEPTH_M
      elif preset_raw in ("2", "custom"):
        updated.table_width_m = _prompt_optional_geom_float(
            "table_width_m",
            f"     table_width_m [m] (default {updated.table_width_m:.3f}): ",
            updated.table_width_m,
            input_fn,
        )
        updated.table_depth_m = _prompt_optional_geom_float(
            "table_depth_m",
            f"     table_depth_m [m] (default {updated.table_depth_m:.3f}): ",
            updated.table_depth_m,
            input_fn,
        )
      elif preset_raw:
        raise ValueError(
            f"Invalid table footprint choice {preset_raw!r}; expected 1 or 2"
        )
    else:
      if args.table_width_m is None:
        updated.table_width_m = _prompt_optional_geom_float(
            "table_width_m",
            f"  2) table_width_m [m] (default {updated.table_width_m:.3f}): ",
            updated.table_width_m,
            input_fn,
        )
      if args.table_depth_m is None:
        updated.table_depth_m = _prompt_optional_geom_float(
            "table_depth_m",
            f"  2) table_depth_m [m] (default {updated.table_depth_m:.3f}): ",
            updated.table_depth_m,
            input_fn,
        )
  except (EOFError, KeyboardInterrupt):
    return "stopped", cfg, "geometry_prompt_aborted"
  except ValueError as exc:
    return "invalid", cfg, str(exc)
  return "ok", updated, None


def _extract_touch_coords(
    samples: Mapping[str, Mapping[str, Any]],
    candidate: RobotConfig,
    arms_to_probe: Sequence[str],
) -> tuple[dict[str, list[float]], str | None]:
  """Validates probed arm connections and extracts finite `[x, y, z]` TCP."""
  coords: dict[str, list[float]] = {}
  for side in arms_to_probe:
    rd = samples.get(side) or {}
    if not bool(rd.get("connected")):
      err_detail = rd.get("connection_error") or "unreachable"
      return (
          {},
          f"FR3 {side} arm ({candidate.arms[side].ip}) unreachable during"
          f" --touch-geometry: {err_detail}",
      )
    pos = rd.get("tcp_pos_base_m")
    if (
        not isinstance(pos, Sequence)
        or isinstance(pos, (str, bytes))
        or len(pos) < 3
    ):
      return {}, f"FR3 {side} arm returned invalid tcp_pos_base_m={pos!r}"
    xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
    if not all(math.isfinite(v) for v in xyz):
      return {}, f"FR3 {side} arm returned non-finite tcp_pos_base_m={pos!r}"
    coords[side] = xyz
  return coords, None


def _collect_touch_sample_warnings(
    samples: Mapping[str, Mapping[str, Any]],
    arms_to_probe: Sequence[str],
) -> list[str]:
  """Collects live Desk `robot_mode` and FK residual warnings."""
  warnings: list[str] = []
  for side in arms_to_probe:
    rd = samples.get(side) or {}
    robot_mode = (rd.get("desk_audit") or {}).get("robot_mode")
    if robot_mode is not None and str(robot_mode) != "kIdle":
      warnings.append(
          f"FR3 {side} Desk mode is {robot_mode!r} (expected 'kIdle';"
          " release the cuff button / clear Desk before sampling)."
      )
    fk_err = rd.get("fk_pos_error_m")
    if (
        fk_err is not None
        and math.isfinite(float(fk_err))
        and float(fk_err) > _TOUCH_MAX_FK_ERR_M
    ):
      warnings.append(
          f"FR3 {side} FK mismatch {float(fk_err) * 1000.0:.2f} mm >"
          f" {_TOUCH_MAX_FK_ERR_M * 1000.0:.1f} mm (verify closed gripper &"
          " tcp_offset_z_m)."
      )
  return warnings


def _compute_touch_geometry(
    candidate: RobotConfig,
    coords: Mapping[str, Sequence[float]],
    *,
    apply_front_offset: bool,
) -> tuple[RobotConfig | None, list[str], str | None, str | None]:
  """Derives tabletop height/front offset and sanity warnings from touch FK."""
  if "left" in coords and "right" in coords:
    z_l = float(coords["left"][2])
    z_r = float(coords["right"][2])
    z_spread = abs(z_l - z_r)
    if z_spread > _TOUCH_MAX_Z_SPREAD_M:
      msg = (
          f"Left/right sampled Z mismatch {z_spread * 1000.0:.1f} mm >"
          f" {_TOUCH_MAX_Z_SPREAD_M * 1000.0:.1f} mm"
          f" (left={z_l:+.5f} m, right={z_r:+.5f} m). One arm may be"
          " parked; rest both closed grippers on the tabletop or pass"
          " --touch-arm left / --touch-arm right."
      )
      return None, [], f"touch_arm_z_mismatch:{z_spread:.6f}", msg

  xs = [float(v[0]) for v in coords.values()]
  zs = [float(v[2]) for v in coords.values()]
  x_meas_m = sum(xs) / len(xs)
  z_meas_m = sum(zs) / len(zs)

  new_base_height_m = round(
      float(candidate.table_height_m) + BASE_WORLD_Z_M - z_meas_m, 6
  )
  try:
    _validate_geom_value("base_height_m", new_base_height_m, allow_zero=False)
  except ValueError as exc:
    err_msg = f"invalid_touch_base_height: {exc}"
    return None, [], err_msg, err_msg

  warnings: list[str] = []
  if abs(z_meas_m) > _TOUCH_MAX_ABS_Z_M:
    warnings.append(
        f"Derived z_table_m={z_meas_m:+.4f} m is >"
        f" {_TOUCH_MAX_ABS_Z_M * 100.0:.0f} cm from Duo base origin;"
        f" confirm gripper was CLOSED and tcp_offset_z_m="
        f"{candidate.tcp_offset_z_m:.3f} m is right."
    )

  new_front_offset_m = float(candidate.table_front_offset_m)
  if apply_front_offset:
    if x_meas_m <= BRACKET_UNDERSIDE_FRONT_X_M:
      warnings.append(
          f"Sampled X ({x_meas_m:+.4f} m) <= bracket front lip"
          f" ({BRACKET_UNDERSIDE_FRONT_X_M:+.4f} m); clamping"
          " table_front_offset_m=0.0 m. Verify tip touched table FRONT edge."
      )
    new_front_offset_m = round(
        max(0.0, x_meas_m - BRACKET_UNDERSIDE_FRONT_X_M), 6
    )

  calibrated = normalize_robot_config({
      **candidate.to_dict(),
      "base_height_m": new_base_height_m,
      "table_front_offset_m": new_front_offset_m,
      "geometry_confirmed": True,
  })
  return calibrated, warnings, None, None


def _sample_touch_with_retry(
    candidate: RobotConfig,
    arms_to_probe: Sequence[str],
    *,
    auto_yes: bool,
    prompt_instruction: str,
    prompt_line: str,
    input_fn: Callable[[str], str],
) -> tuple[str, dict[str, list[float]], list[str], str | None]:
  """Prompts operator to position arm(s), probes FK, and retries on warnings."""
  while True:
    if not auto_yes:
      print(prompt_instruction)
      try:
        ans = input_fn(prompt_line).strip().lower()
      except (EOFError, KeyboardInterrupt):
        return "stopped", {}, [], "geometry_prompt_aborted"
      if ans in ("q", "quit"):
        return "stopped", {}, [], "geometry_prompt_aborted"

    try:
      samples = _touch_sample_arms(candidate, arms_to_probe)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      return (
          "fail",
          {},
          [],
          f"touch_sample_failed: {type(exc).__name__}: {exc}",
      )

    coords, coord_err = _extract_touch_coords(
        samples, candidate, arms_to_probe
    )
    if coord_err is not None:
      return "fail", {}, [], coord_err

    for side in arms_to_probe:
      x_v, y_v, z_v = coords[side]
      print(
          f"  Sampled {side:<5s} TCP (base): x={x_v:+.5f} m,"
          f" y={y_v:+.5f} m, z={z_v:+.5f} m"
      )

    sample_warnings = _collect_touch_sample_warnings(samples, arms_to_probe)
    for warn_msg in sample_warnings:
      print(f"  [WARN] {warn_msg}")

    if sample_warnings and not auto_yes:
      try:
        retry_ans = (
            input_fn(
                "  Retry sample after releasing cuff / fixing Desk? [Y/n/q]: "
            )
            .strip()
            .lower()
        )
      except (EOFError, KeyboardInterrupt):
        return "stopped", {}, sample_warnings, None
      if retry_ans in ("q", "quit"):
        return "stopped", {}, sample_warnings, None
      if retry_ans in ("", "y", "yes", "r", "retry"):
        continue

    return "ok", coords, sample_warnings, None


def _fail_invalid_geometry(
    out_dir: Path, summary: dict[str, Any], err_msg: str | None
) -> int:
  print(
      f"\n[FAIL] Invalid geometry input ({err_msg}); geometry remains"
      " unconfirmed.",
      file=sys.stderr,
  )
  return _fail_step(
      out_dir, summary, "geometry", 2, f"invalid_geometry_input: {err_msg}"
  )


def _confirm_touch_geometry(
    out_dir: Path,
    summary: dict[str, Any],
    orig_cfg: RobotConfig,
    *,
    input_fn: Callable[[str], str],
) -> tuple[str, int]:
  """Prompts operator to confirm and save calibrated touch geometry."""
  try:
    confirm = input_fn("  Confirm & save? [y/N]: ").strip().lower()
  except (EOFError, KeyboardInterrupt):
    confirm = "n"
  if confirm not in ("y", "yes"):
    rc, _ = _stop_geometry_step(out_dir, summary, orig_cfg)
    return "stopped", rc
  return "ok", 0


def _preview_touch_geometry_dry_run(
    cfg: RobotConfig,
    candidate: RobotConfig,
    summary: dict[str, Any],
    ref_img: Path,
    touch_gif: Path,
    touch_arm: str,
    arms_to_probe: Sequence[str],
    *,
    close_grippers: bool = False,
) -> tuple[int, RobotConfig]:
  print(
      "  [DRY-RUN] FR3 cuff touch calibration preview (no FCI sockets"
      " opened):"
  )
  print(f"    touch_arm         : {touch_arm} (arms={arms_to_probe})")
  print(f"    Touch guide GIF   : {touch_gif}")
  print(f"    tcp_offset_z_m    : {candidate.tcp_offset_z_m:.3f} m")
  print(f"    current z_table_m : {candidate.z_table_m:+.6f} m")
  if close_grippers:
    print(
        "    [DRY-RUN] --close-grippers: skipping physical Robotiq"
        " activation/closing."
    )
  _print_geometry_summary(candidate, ref_img)
  missing = validate_robot_config_complete(cfg)
  print(
      "  Dry-run completeness preview missing:"
      f" {missing or 'NONE (complete)'}"
  )
  summary.steps["geometry"] = StepStatus(
      status="ok",
      exit_code=0,
      extras={
          "touch_geometry": True,
          "touch_arm": touch_arm,
          "arms_probed": arms_to_probe,
          "tcp_offset_z_m": candidate.tcp_offset_z_m,
          "z_table_m": candidate.z_table_m,
          "geometry_confirmed": bool(cfg.geometry_confirmed),
          "close_grippers": bool(close_grippers),
          "missing_for_live": missing,
      },
  )
  return 0, cfg


def _close_enabled_grippers_for_touch(cfg: RobotConfig) -> list[str]:
  """Activates and closes enabled Robotiq grippers before cuff touch."""
  closed_arms: list[str] = []
  for side in ("left", "right"):
    arm_cfg = cfg.arms[side]
    if not arm_cfg.enabled or not arm_cfg.gripper_port:
      continue
    with robotiq_gripper_controller.RobotiqGripper(
        port=arm_cfg.gripper_port, dry_run=False
    ) as gripper:
      gripper.activate()
      st = gripper.move_normalized(1.0, wait=True)
    print(
        f"  Closed {side} Robotiq gripper ({arm_cfg.gripper_port}):"
        f" ratio={st.position_ratio:.3f}, gOBJ={st.gOBJ}"
    )
    closed_arms.append(side)
  if not closed_arms:
    raise RuntimeError(
        "No enabled Robotiq gripper_port configured for --close-grippers."
    )
  return closed_arms


def _run_touch_geometry_step(
    cfg: RobotConfig,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    ref_img: Path,
    *,
    effective_dry_run: bool,
    is_read_only: bool,
    write_cfg: bool,
    input_fn: Callable[[str], str],
) -> tuple[int, RobotConfig]:
  """Runs FR3 cuff hand-guiding zero-motion FK tabletop calibration."""
  touch_arm = str(getattr(args, "touch_arm", None) or _DEFAULT_TOUCH_ARM)
  arms_to_probe, arm_err = _resolve_touch_arms(cfg, touch_arm)
  if arm_err is not None:
    print(f"\n[FAIL] {arm_err}", file=sys.stderr)
    return _fail_step(out_dir, summary, "geometry", 2, arm_err), cfg

  if (
      not effective_dry_run
      and not args.yes
      and _is_non_interactive_tty(input_fn)
  ):
    print(
        "[FAIL] --touch-geometry requires an interactive TTY or --yes.",
        file=sys.stderr,
    )
    return _fail_step(
        out_dir, summary, "geometry", 2, "non_interactive_without_yes"
    ), cfg

  candidate = cfg
  touch_gif = _skill_asset_path("cuff_touch_explainer.gif")
  if not effective_dry_run and not args.yes:
    print(f"  Touch guide GIF   : {touch_gif}")
    print(f"  Reference diagram : {ref_img}")
    setup_status, candidate, setup_err = _prompt_touch_setup(
        cfg, args, input_fn
    )
    if setup_status == "invalid":
      return _fail_invalid_geometry(out_dir, summary, setup_err), cfg
    if setup_status == "stopped":
      return _stop_geometry_step(out_dir, summary, cfg)

  if effective_dry_run:
    return _preview_touch_geometry_dry_run(
        cfg,
        candidate,
        summary,
        ref_img,
        touch_gif,
        touch_arm,
        arms_to_probe,
        close_grippers=bool(getattr(args, "close_grippers", False)),
    )

  if getattr(args, "close_grippers", False):
    if is_read_only:
      print(
          "  [READ-ONLY] Skipping physical Robotiq activation/closing"
          " (--close-grippers)."
      )
    else:
      try:
        _close_enabled_grippers_for_touch(candidate)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        err_msg = f"close_grippers_failed: {exc}"
        print(f"\n[FAIL] {err_msg}", file=sys.stderr)
        return _fail_step(out_dir, summary, "geometry", 2, err_msg), cfg

  arms_label = ", ".join(
      _ARM_PERSPECTIVE_DESC.get(s, s) for s in arms_to_probe
  )
  apply_front_offset = args.table_front_offset_m is None
  prompt_line = "  and press ENTER to sample FCI FK ('q' = quit): "
  sample_status, coords, sample_warnings, sample_err = (
      _sample_touch_with_retry(
          candidate,
          arms_to_probe,
          auto_yes=bool(args.yes),
          prompt_instruction=(
              f"  Hold the {arms_label} FR3 cuff hand-guiding button, rest"
              " CLOSED fingertip(s) on the table FRONT EDGE, release the cuff"
              " button so Desk returns to Idle (zero software motion),"
          ),
          prompt_line=prompt_line,
          input_fn=input_fn,
      )
  )
  if sample_status == "stopped":
    return _stop_geometry_step(out_dir, summary, cfg)
  if sample_status != "ok":
    print(f"\n[FAIL] {sample_err}", file=sys.stderr)
    return _fail_step(out_dir, summary, "geometry", 2, sample_err), cfg

  calibrated, sanity_warnings, calc_code, calc_msg = _compute_touch_geometry(
      candidate,
      coords,
      apply_front_offset=apply_front_offset,
  )
  if calibrated is None:
    print(f"\n[FAIL] {calc_msg}", file=sys.stderr)
    return _fail_step(out_dir, summary, "geometry", 2, calc_code), cfg

  all_warnings = [*sample_warnings, *sanity_warnings]
  for warn_msg in sanity_warnings:
    print(f"  [WARN] {warn_msg}")

  _print_human_table_bounds(calibrated, touch_mode=True)

  if not args.yes:
    c_outcome, c_rc = _confirm_touch_geometry(
        out_dir,
        summary,
        cfg,
        input_fn=input_fn,
    )
    if c_outcome != "ok":
      return c_rc, cfg

  cfg = calibrated
  _save_geometry_if_enabled(cfg, args, write_cfg=write_cfg, confirmed=True)
  summary.steps["geometry"] = StepStatus(
      status="ok",
      exit_code=0,
      warnings=all_warnings,
      read_only=True if is_read_only else None,
      extras={
          "touch_geometry": True,
          "touch_arm": touch_arm,
          "arms_probed": arms_to_probe,
          "geometry_confirmed": True,
          "z_table_m": cfg.z_table_m,
          "table_front_offset_m": cfg.table_front_offset_m,
          "table_width_m": cfg.table_width_m,
          "table_depth_m": cfg.table_depth_m,
          "tcp_offset_z_m": cfg.tcp_offset_z_m,
      },
  )
  return 0, cfg


def _record_non_interactive_geometry(
    cfg: RobotConfig,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    ref_img: Path,
    *,
    has_cli_updates: bool,
    confirmed_via_cli: bool,
    effective_dry_run: bool,
    is_read_only: bool,
    write_cfg: bool,
) -> tuple[int, RobotConfig]:
  """Summarizes and optionally saves geometry when prompting is skipped."""
  _print_geometry_summary(cfg, ref_img)
  step_entry = StepStatus(
      status="ok",
      exit_code=0,
      extras={
          "geometry_confirmed": bool(cfg.geometry_confirmed),
          "z_table_m": cfg.z_table_m,
      },
  )
  if has_cli_updates or confirmed_via_cli:
    _save_geometry_if_enabled(
        cfg,
        args,
        write_cfg=write_cfg,
        confirmed=bool(cfg.geometry_confirmed),
    )
  elif effective_dry_run:
    missing = validate_robot_config_complete(cfg)
    print(
        "  Dry-run completeness preview missing:"
        f" {missing or 'NONE (complete)'}"
    )
    if not confirmed_via_cli:
      step_entry.extras["missing_for_live"] = missing
  elif cfg.geometry_confirmed:
    print("  [PASS] Cell geometry & fingertip dimensions already confirmed.")
  elif is_read_only or args.no_write_config:
    print(
        "  [WARN] geometry_confirmed is False (skipping interactive prompt in"
        " read-only/--no-write-config mode). Run '--step geometry' or pass"
        " explicit measurement flags + '--confirm-geometry' before live"
        " actuation."
    )
    step_entry.read_only = is_read_only
  elif args.yes:
    print(
        "  [WARN] geometry_confirmed is False; --yes will NOT auto-confirm"
        " unverified default dimensions. Run '--step geometry' interactively"
        " or pass explicit measurement flags + '--confirm-geometry'."
    )
    step_entry.warning = "geometry_unconfirmed_with_yes"
  else:
    print(
        "  [WARN] geometry_confirmed is False and stdin is not interactive."
        " Run '--step geometry' in an interactive TTY or pass explicit"
        " measurement flags + '--confirm-geometry'."
    )
    step_entry.warning = "geometry_unconfirmed_non_tty"
  summary.steps["geometry"] = step_entry
  return 0, cfg


def _run_geometry_step(
    cfg: RobotConfig,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    effective_dry_run: bool,
    is_read_only: bool,
    input_fn: Callable[[str], str],
    cli_updates: Mapping[str, float] | None = None,
) -> tuple[int, RobotConfig]:
  """Runs Step 1b physical cell geometry & fingertip calibration."""
  print("\n--- Step 1b: Physical Cell Geometry & Fingertip Calibration ---")
  ref_img = _geometry_reference_image_path()
  if cli_updates is None:
    updates, cli_err = _extract_geometry_cli_updates(args)
    if cli_err is not None:
      print(f"\n[FAIL] Invalid geometry CLI flag: {cli_err}", file=sys.stderr)
      return _fail_step(out_dir, summary, "geometry", 2, cli_err), cfg
    if updates:
      cfg = normalize_robot_config({**cfg.to_dict(), **updates})
  else:
    updates = cli_updates

  write_cfg = (
      not effective_dry_run and not args.no_write_config and not is_read_only
  )
  if getattr(args, "touch_geometry", False):
    return _run_touch_geometry_step(
        cfg,
        args,
        out_dir,
        summary,
        ref_img,
        effective_dry_run=effective_dry_run,
        is_read_only=is_read_only,
        write_cfg=write_cfg,
        input_fn=input_fn,
    )

  confirmed_via_cli = bool(args.confirm_geometry) or (
      len(updates) == len(_GEOMETRY_CLI_ALLOW_ZERO)
  )
  if confirmed_via_cli:
    cfg = normalize_robot_config({**cfg.to_dict(), "geometry_confirmed": True})

  if (
      confirmed_via_cli
      or effective_dry_run
      or is_read_only
      or args.no_write_config
      or args.yes
      or _is_non_interactive_tty(input_fn)
  ):
    return _record_non_interactive_geometry(
        cfg,
        args,
        out_dir,
        summary,
        ref_img,
        has_cli_updates=bool(updates),
        confirmed_via_cli=confirmed_via_cli,
        effective_dry_run=effective_dry_run,
        is_read_only=is_read_only,
        write_cfg=write_cfg,
    )

  config_on_disk = args.config is not None and args.config.is_file()
  if config_on_disk and cfg.geometry_confirmed and not updates:
    _print_geometry_summary(cfg, ref_img)
    try:
      reuse_raw = (
          input_fn(
              "  Existing config found: [1] Reuse existing config |"
              " [2] Recalibrate (start fresh) [1]: "
          )
          .strip()
          .lower()
      )
    except (EOFError, KeyboardInterrupt):
      return _stop_geometry_step(out_dir, summary, cfg)
    if reuse_raw in ("", "1", "reuse"):
      print("  Reusing existing calibrated config.")
      summary.steps["geometry"] = StepStatus(
          status="ok",
          exit_code=0,
          extras={
              "geometry_confirmed": True,
              "z_table_m": cfg.z_table_m,
              "reused_existing": True,
          },
      )
      return 0, cfg
    if reuse_raw not in ("2", "recalibrate", "fresh"):
      err_msg = f"Invalid config choice {reuse_raw!r}; expected 1 or 2"
      return _fail_invalid_geometry(out_dir, summary, err_msg), cfg

  return _run_touch_geometry_step(
      cfg,
      args,
      out_dir,
      summary,
      ref_img,
      effective_dry_run=effective_dry_run,
      is_read_only=is_read_only,
      write_cfg=write_cfg,
      input_fn=input_fn,
  )


def _run_cameras_step(
    cfg: RobotConfig,
    args: argparse.Namespace,
    out_dir: Path,
    summary: BringupSummary,
    *,
    effective_dry_run: bool,
) -> int:
  """Runs Step 2 4-camera freshness check, contact sheet, and MP4 recording."""
  print("\n--- Step 2: Camera Verification & Contact Sheet ---")
  step2_dir = out_dir / "step2_cameras"
  step2_dir.mkdir(parents=True, exist_ok=True)
  rc2, c_rep, err2 = _call_stage(
      check_cameras.run_camera_check,
      cfg,
      realtime=args.realtime,
      dry_run=effective_dry_run,
      output_dir=step2_dir,
      output_json=step2_dir / "camera_check_report.json",
  )
  if rc2 != 0:
    print(
        f"\n[FAIL] Step 2 camera check failed (exit code {rc2}).",
        file=sys.stderr,
    )
    return _fail_step(out_dir, summary, "cameras", rc2, err2)

  for key, art_key, label in (
      ("contact_sheet_path", "contact_sheet_2x2", "Contact Sheet"),
      ("fk_projection_overlay_path", "fk_projection_overlay_2x2", "FK Overlay"),
  ):
    if c_rep.get(key):
      summary.artifacts[art_key] = c_rep[key]
      print(f"  {label:<14}: {c_rep[key]}")
  mp4_path = (c_rep.get("recording") or {}).get("mp4_path")
  if mp4_path:
    summary.artifacts["cameras_recording_mp4"] = mp4_path
    print(f"  MP4 Recording : {mp4_path}")

  summary.steps["cameras"] = StepStatus(status="ok", exit_code=0)
  return 0


def _confirm_actuation(
    step_key: str,
    args: argparse.Namespace,
    out_dir: Path,
    summary: dict[str, Any],
    *,
    will_actuate: bool,
    input_fn: Callable[[str], str],
) -> tuple[str, int]:
  """Handles interactive `[E-STOP READY]` prompt before live actuation."""
  if not will_actuate:
    return "run", 0

  if args.yes:
    if step_key == "abort-test" and sys.stdin.isatty():
      print(
          "  NOTE: Type 'q' + ENTER (or press Ctrl+C) while the left arm"
          " slowly nudges (10 deg @ 4 deg/s) to trigger smooth C^2"
          " abort-and-unwind."
      )
    return "run", 0

  if _is_non_interactive_tty(input_fn):
    print(
        f"[FAIL] Physical actuation for '{step_key}' requires an"
        " interactive TTY or --yes.",
        file=sys.stderr,
    )
    return "fail", _fail_step(
        out_dir, summary, step_key, 2, "non_interactive_without_yes"
    )

  print(f"[E-STOP READY] Operator confirmation required for '{step_key}':")
  if step_key == "abort-test":
    print(
        "  NOTE: Type 'q' + ENTER (or press Ctrl+C) while the left arm"
        " slowly nudges (10 deg @ 4 deg/s) to trigger smooth C^2"
        " abort-and-unwind."
    )

  try:
    ans = (
        input_fn("Proceed with actuation? [y = run / s = skip / N,q = quit]: ")
        .strip()
        .lower()
    )
  except (EOFError, KeyboardInterrupt):
    ans = "q"

  if ans in ("s", "skip"):
    print(f"Skipping '{step_key}'.")
    summary.steps[step_key] = StepStatus(status="skipped")
    return "skip", 0
  if ans not in ("y", "yes"):
    msg = (
        f"Bring-up stopped by operator before '{step_key}'."
        if ans in ("q", "quit", "n", "no", "")
        else f"Unrecognized response {ans!r}; stopping bring-up."
    )
    print(msg)
    summary.status = "stopped_by_operator"
    summary.steps[step_key] = StepStatus(status="stopped_by_operator")
    _write_summary(out_dir, summary)
    return "stop", 0
  return "run", 0


def _validate_actuation_result(
    step_key: str,
    sub_rep: Mapping[str, Any],
    sub_dir: Path,
    out_dir: Path,
    summary: BringupSummary,
) -> tuple[str, int]:
  """Checks post-step success/abort invariants for live actuation steps."""
  if step_key == "gripper":
    bad_arms = [
        arm_name
        for arm_name, arm_res in (sub_rep.get("results") or {}).items()
        if isinstance(arm_res, dict)
        and not bool(
            (arm_res.get("gripper_cycle") or {}).get("stroke_ok", True)
        )
    ]
    if bad_arms:
      print(
          f"\n[FAIL] Step 3a gripper stroke check failed on {bad_arms}.",
          file=sys.stderr,
      )
      return "fail", _fail_step(
          out_dir, summary, step_key, 1, "gripper_stroke_failed"
      )
    return "ok", 0

  if step_key in ("nudge", "home"):
    if any(
        isinstance(arm_res, dict)
        and bool((arm_res.get(step_key) or {}).get("aborted", False))
        for arm_res in (sub_rep.get("results") or {}).values()
    ):
      print(f"Bring-up stopped by operator abort during '{step_key}'.")
      summary.status = "stopped_by_operator"
      summary.steps[step_key] = StepStatus(
          status="stopped_by_operator",
          extras={"aborted": True},
      )
      _write_summary(out_dir, summary)
      return "stop", 0
    return "ok", 0

  if step_key == "abort-test":
    abort_info = next(
        (
            arm_res.get("abort_test")
            for arm_res in (sub_rep.get("results") or {}).values()
            if isinstance(arm_res, dict)
            and isinstance(arm_res.get("abort_test"), dict)
        ),
        None,
    )
    if abort_info is not None:
      aborted_ok = bool(abort_info.get("aborted", False))
      unwound_ok = bool(abort_info.get("unwound_to_start", False))
      grip_ok = bool(abort_info.get("gripper_preserved", True))
      if not (aborted_ok and unwound_ok and grip_ok):
        print(
            "\n[FAIL] Step 3c abort-test did not verify smooth"
            f" abort-and-unwind (aborted={aborted_ok},"
            f" unwound_to_start={unwound_ok},"
            f" gripper_preserved={grip_ok}). Re-run and type 'q' +"
            " ENTER while the left arm moves.",
            file=sys.stderr,
        )
        return "fail", _fail_step(
            out_dir, summary, step_key, 1, "abort_unwind_not_verified"
        )
    return "ok", 0

  if step_key == "hello-world":
    mp4_cand = sub_rep.get("mp4_path") or str(
        sub_dir / "stage4_hello_world.mp4"
    )
    if Path(mp4_cand).is_file():
      summary.artifacts["stage4_hello_world_mp4"] = str(mp4_cand)
      print(f"  Stage 4 MP4 Recording: {mp4_cand}")
    if bool(sub_rep.get("aborted", False)):
      print("Bring-up stopped by operator abort during 'hello-world'.")
      summary.status = "stopped_by_operator"
      summary.steps[step_key] = StepStatus(
          status="stopped_by_operator",
          extras={"aborted": True},
      )
      _write_summary(out_dir, summary)
      return "stop", 0
  return "ok", 0


def _run_home_step(
    cfg: RobotConfig,
    *,
    realtime: str,
    output_dir: Path,
) -> tuple[int, dict[str, Any], str | None]:
  """Runs live bimanual homing with upward-only initial tabletop lift-off."""
  try:
    with Fr3DuoRobot(
        cfg, mode="hardware", dry_run=False, realtime_mode=realtime
    ) as robot:
      with AbortWatcher() as watcher:
        move_res = robot.home(
            abort_watcher=watcher,
            allow_initial_table_lift=True,
        )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return 1, {}, f"{type(exc).__name__}: {exc}"

  res_dict = (
      move_res.to_dict() if hasattr(move_res, "to_dict") else dict(move_res)
  )
  report = {
      "stage": "stage3_home",
      "mode": "hardware",
      "allow_initial_table_lift": True,
      "results": {
          "left": {
              "home": {
                  "ok": bool(res_dict.get("ok", True)),
                  "aborted": bool(res_dict.get("aborted", False)),
                  "final_q_rad": list(res_dict.get("final_left_q_rad") or []),
              }
          },
          "right": {
              "home": {
                  "ok": bool(res_dict.get("ok", True)),
                  "aborted": bool(res_dict.get("aborted", False)),
                  "final_q_rad": list(res_dict.get("final_right_q_rad") or []),
              }
          },
      },
  }
  (output_dir / "check_robot_home.json").write_text(
      json.dumps(report, indent=2) + "\n", encoding="utf-8"
  )
  ok_or_aborted = bool(res_dict.get("ok", True)) or bool(
      res_dict.get("aborted", False)
  )
  return (0 if ok_or_aborted else 1), report, None


def main(
    argv: Sequence[str] | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
  """Executes the progressive bring-up pipeline or a single bring-up step."""
  args = parse_args(argv)
  effective_dry_run = bool(args.dry_run)
  is_read_only = bool(args.read_only)
  mode_label = (
      "dry_run"
      if effective_dry_run
      else ("read_only" if is_read_only else "hardware")
  )

  resolved_out_dir = resolve_stage_output_dir(
      "bringup", args.output_dir, dry_run=effective_dry_run
  )
  out_dir = (
      resolved_out_dir
      if resolved_out_dir is not None
      else Path(tempfile.mkdtemp(prefix="fr3_bringup_"))
  )
  out_dir.mkdir(parents=True, exist_ok=True)

  steps_to_run = _resolve_steps_to_run(args)
  target_desc = (
      f"step={args.step}" if args.step is not None else f"up_to={args.up_to}"
  )
  summary = BringupSummary(
      timestamp_utc=datetime.now(timezone.utc).isoformat(),
      mode=mode_label,
      up_to=None if args.step is not None else args.up_to,
      step=args.step,
      read_only=is_read_only,
  )

  print(
      f"=== Franka Research 3 Duo Bring-Up (mode={mode_label},"
      f" {target_desc}) ==="
  )
  print(f"Output directory: {out_dir}")

  try:
    cfg = load_robot_config(args.config)
    if args.left_ip:
      cfg.arms.left.ip = str(args.left_ip)
    if args.right_ip:
      cfg.arms.right.ip = str(args.right_ip)
    if args.zed_conf_path is not None:
      cfg.cameras.head_stereo.zed_conf_path = str(args.zed_conf_path).strip()
  except (OSError, ValueError, TypeError) as exc:
    err_msg = f"{type(exc).__name__}: {exc}"
    print(f"\n[FAIL] Invalid robot configuration: {err_msg}", file=sys.stderr)
    return _fail_step(out_dir, summary, steps_to_run[0], 1, err_msg)

  cli_geom_updates, cli_geom_err = _extract_geometry_cli_updates(args)
  if cli_geom_err is not None:
    print(
        f"\n[FAIL] Invalid geometry CLI flag: {cli_geom_err}", file=sys.stderr
    )
    err_step = "geometry" if "geometry" in steps_to_run else steps_to_run[0]
    return _fail_step(out_dir, summary, err_step, 2, cli_geom_err)
  if cli_geom_updates:
    cfg = normalize_robot_config({**cfg.to_dict(), **cli_geom_updates})

  if "env" in steps_to_run:
    rc0 = _run_env_step(
        cfg,
        out_dir,
        summary,
        is_hw=not effective_dry_run,
        mode_label=mode_label,
    )
    if rc0 != 0:
      return rc0

  if "preflight" in steps_to_run:
    rc1, cfg = _run_preflight_step(
        cfg,
        args,
        out_dir,
        summary,
        effective_dry_run=effective_dry_run,
        is_read_only=is_read_only,
    )
    if rc1 != 0:
      return rc1

  if "geometry" in steps_to_run:
    rc_geom, cfg = _run_geometry_step(
        cfg,
        args,
        out_dir,
        summary,
        effective_dry_run=effective_dry_run,
        is_read_only=is_read_only,
        input_fn=input_fn,
        cli_updates=cli_geom_updates,
    )
    if rc_geom != 0 or summary.status == "stopped_by_operator":
      return rc_geom
  elif cli_geom_updates or args.confirm_geometry:
    confirmed_via_cli = bool(args.confirm_geometry) or (
        len(cli_geom_updates) == len(_GEOMETRY_CLI_ALLOW_ZERO)
    )
    cfg = normalize_robot_config({
        **cfg.to_dict(),
        **cli_geom_updates,
        "geometry_confirmed": bool(cfg.geometry_confirmed) or confirmed_via_cli,
    })
    if (
        not effective_dry_run
        and not args.no_write_config
        and not is_read_only
        and args.config is not None
    ):
      save_robot_config(cfg, args.config)

  if "cameras" in steps_to_run:
    rc2 = _run_cameras_step(
        cfg,
        args,
        out_dir,
        summary,
        effective_dry_run=effective_dry_run,
    )
    if rc2 != 0:
      return rc2

  for step_key in steps_to_run:
    desc = _ACTUATION_DESCRIPTIONS.get(step_key)
    if desc is None:
      continue
    print(f"\n--- {desc} ---")

    will_actuate = not effective_dry_run and not is_read_only
    if will_actuate and step_key in _COMPLETE_CONFIG_REQUIRED_STEPS:
      missing_fields = validate_robot_config_complete(cfg)
      if missing_fields:
        missing_str = ", ".join(missing_fields)
        print(
            f"\n[FAIL] Cannot run live actuation '{step_key}': incomplete"
            f" robot configuration. Missing/unconfirmed: {missing_str}."
            " Run '--step preflight' and '--step geometry' (or pass"
            " measurement flags + '--confirm-geometry') first.",
            file=sys.stderr,
        )
        return _fail_step(
            out_dir,
            summary,
            step_key,
            2,
            f"incomplete_config: {missing_str}",
        )

    action, prompt_rc = _confirm_actuation(
        step_key,
        args,
        out_dir,
        summary,
        will_actuate=will_actuate,
        input_fn=input_fn,
    )
    if action == "skip":
      continue
    if action in ("fail", "stop"):
      return prompt_rc

    sub_dir = out_dir / f"step3_{step_key}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    if step_key == "home" and will_actuate:
      rc, sub_rep, sub_err = _run_home_step(
          cfg, realtime=args.realtime, output_dir=sub_dir
      )
    else:
      rc, sub_rep, sub_err = _call_stage(
          check_robot.run_robot_check,
          cfg,
          mode=step_key,
          read_only=is_read_only,
          dry_run=effective_dry_run,
          realtime=args.realtime,
          output_dir=sub_dir,
      )
    if rc != 0:
      print(
          f"\n[FAIL] {desc} returned non-zero exit code {rc}.", file=sys.stderr
      )
      return _fail_step(out_dir, summary, step_key, rc, sub_err)

    if will_actuate:
      outcome, out_rc = _validate_actuation_result(
          step_key, sub_rep, sub_dir, out_dir, summary
      )
      if outcome in ("fail", "stop"):
        return out_rc

    summary.steps[step_key] = StepStatus(
        status="ok",
        exit_code=0,
        read_only=True if is_read_only else None,
    )

  summary.status = "ok"
  _write_summary(out_dir, summary)

  print("\n=======================================================")
  print("  FR3 DUO BRING-UP COMPLETED SUCCESSFULLY!")
  print(f"  Mode     : {mode_label}")
  print(f"  Target   : {target_desc}")
  print(f"  Summary  : {out_dir / 'bringup_summary.json'}")
  for art_name, art_p in summary.artifacts.items():
    print(f"  Artifact : [{art_name}] -> {art_p}")
  print("=======================================================")
  return 0


if __name__ == "__main__":
  sys.exit(main())
