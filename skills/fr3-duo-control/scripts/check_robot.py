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

"""Stage 3-4 robot verification, abort-unwind test, homing, and Hello World."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

from fr3_arm_controller import validate_arm_readiness
from fr3_camera_controller import Fr3DuoCameraController
from fr3_config import (
    RobotConfig,
    normalize_robot_config,
    resolve_realtime_config,
    resolve_stage_output_dir,
)
from fr3_duo_robot import Fr3DuoRobot, RobotState
from fr3_kinematics import compute_arm_fk
from fr3_mjcf import (
    FR3_IK_MARGINS_RAD,
    FR3_JOINT_LIMITS_RAD,
)
from fr3_recording import MosaicVideoWriter, _DictCompatMixin
from fr3_safety import (
    AbortWatcher,
    Fr3DuoSafetyChecker,
    clamp_q_to_padded_limits,
    generate_quintic_trajectory,
)
from robotiq_gripper_controller import EMPTY_CLOSE_MIN_RATIO

GREETING_DELTA_LEFT_RAD = (0.08, -0.08, 0.0, 0.12, 0.0, 0.10, 0.0)
GREETING_DELTA_RIGHT_RAD = (-0.08, -0.08, 0.0, 0.12, 0.0, 0.10, 0.0)
_MOTION_PHASES = frozenset(
    {"outbound_greeting", "gripper_cycle", "return_to_rest"}
)


@dataclass
class HelloWorldReport(_DictCompatMixin):
  """Structured Stage 4 Hello World report written to JSON."""

  timestamp_utc: str
  stage: str
  mode: str
  dry_run: bool
  min_tcp_z_base_m: float | None = None
  min_inter_arm_tcp_dist_m: float | None = None
  outbound_steps_1khz: int | None = None
  max_speed_deg_s: float | None = None
  aborted: bool | None = None
  frames_written: int | None = None
  motion_frames_written: int | None = None
  mp4_path: str | None = None
  jsonl_path: str | None = None
  contact_sheet_path: str | None = None
  gripper_summary: dict[str, Any] | None = None
  preflight_blocked: bool | None = None
  warnings: list[str] | None = None
  report_path: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Converts the Hello World report to a JSON-serializable dict."""
    out: dict[str, Any] = {
        "timestamp_utc": self.timestamp_utc,
        "stage": self.stage,
        "mode": self.mode,
        "dry_run": self.dry_run,
    }
    optional_fields = (
        "min_tcp_z_base_m",
        "min_inter_arm_tcp_dist_m",
        "outbound_steps_1khz",
        "max_speed_deg_s",
        "aborted",
        "frames_written",
        "motion_frames_written",
        "mp4_path",
        "jsonl_path",
        "contact_sheet_path",
        "gripper_summary",
        "preflight_blocked",
        "warnings",
        "report_path",
    )
    for name in optional_fields:
      val = getattr(self, name)
      if val is not None:
        out[name] = val
    return out


@dataclass
class RobotCheckReport(_DictCompatMixin):
  """Structured Stage 3 robot check report written to JSON."""

  timestamp_utc: str
  stage: str
  check_mode: str
  execution_mode: str
  dry_run: bool
  preflight_blocked: bool
  realtime_profile: dict[str, Any]
  preview: dict[str, Any]
  results: dict[str, Any] = field(default_factory=dict)
  warnings: list[str] = field(default_factory=list)
  report_path: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Converts the robot check report to a JSON-serializable dict."""
    out: dict[str, Any] = {
        "timestamp_utc": self.timestamp_utc,
        "stage": self.stage,
        "check_mode": self.check_mode,
        "execution_mode": self.execution_mode,
        "dry_run": self.dry_run,
        "preflight_blocked": self.preflight_blocked,
        "realtime_profile": dict(self.realtime_profile),
        "preview": dict(self.preview),
        "results": dict(self.results),
        "warnings": list(self.warnings),
    }
    if self.report_path is not None:
      out["report_path"] = self.report_path
    return out


def _build_nudge_target(
    q_start: list[float], delta_deg: float, arm: str
) -> list[float]:
  """Computes a limit-padded J1 nudge target pose for `arm`."""
  q_target = list(q_start)
  sign = 1.0 if arm == "left" else -1.0
  delta_rad = math.radians(abs(float(delta_deg))) * sign
  low = float(FR3_JOINT_LIMITS_RAD[0, 0]) + float(FR3_IK_MARGINS_RAD[0])
  high = float(FR3_JOINT_LIMITS_RAD[0, 1]) - float(FR3_IK_MARGINS_RAD[0])
  clamped = max(low, min(high, float(q_start[0]) + delta_rad))
  if abs(clamped - q_start[0]) < 1e-3:
    clamped = max(low, min(high, float(q_start[0]) - delta_rad))
  q_target[0] = clamped
  return q_target


def _create_robot(
    cfg: RobotConfig,
    *,
    realtime_mode: str,
    robot_factory: Callable[..., Fr3DuoRobot] | None,
) -> Fr3DuoRobot:
  """Constructs a live `Fr3DuoRobot` or invokes `robot_factory`."""
  if robot_factory is not None:
    return robot_factory(cfg, realtime_mode=realtime_mode)
  return Fr3DuoRobot(cfg, mode="hardware", realtime_mode=realtime_mode)


def _preview_greeting_safety(
    cfg: RobotConfig, *, max_vel_rad_s: float
) -> tuple[float, float, int]:
  """Validates Stage 4 greeting trajectory against table/arm clearance."""
  safety = Fr3DuoSafetyChecker(cfg)
  mj_model = safety.mj_model
  q_left_rest = clamp_q_to_padded_limits(cfg.arms.left.rest_pose_rad).tolist()
  q_right_rest = clamp_q_to_padded_limits(cfg.arms.right.rest_pose_rad).tolist()
  q_left_greet = clamp_q_to_padded_limits(
      [q + dq for q, dq in zip(q_left_rest, GREETING_DELTA_LEFT_RAD)]
  ).tolist()
  q_right_greet = clamp_q_to_padded_limits(
      [q + dq for q, dq in zip(q_right_rest, GREETING_DELTA_RIGHT_RAD)]
  ).tolist()

  traj_l_out = generate_quintic_trajectory(
      [q_left_rest, q_left_greet], dt_s=0.001, max_vel_rad_s=max_vel_rad_s
  )
  traj_r_out = generate_quintic_trajectory(
      [q_right_rest, q_right_greet], dt_s=0.001, max_vel_rad_s=max_vel_rad_s
  )
  n_preview = max(len(traj_l_out), len(traj_r_out))
  sample_idx = np.unique(
      np.linspace(0, n_preview - 1, min(16, n_preview), dtype=np.intp)
  )
  min_tcp_z = float("inf")
  min_tcp_dist = float("inf")
  for idx in sample_idx:
    ql = traj_l_out[min(idx, len(traj_l_out) - 1)]
    qr = traj_r_out[min(idx, len(traj_r_out) - 1)]
    pos_l, _ = compute_arm_fk(mj_model, "left", ql)
    pos_r, _ = compute_arm_fk(mj_model, "right", qr)
    min_tcp_z = min(min_tcp_z, float(pos_l[2]), float(pos_r[2]))
    min_tcp_dist = min(min_tcp_dist, float(np.linalg.norm(pos_l - pos_r)))

  z_table_m = float(cfg.z_table_m)
  min_clearance_m = float(cfg.min_inter_arm_clearance_m)
  if min_tcp_z < z_table_m:
    raise RuntimeError(
        "Stage 4 greeting trajectory dips below z_table_m "
        f"({min_tcp_z:.4f} < {z_table_m:.4f})"
    )
  if min_tcp_dist < min_clearance_m:
    raise RuntimeError(
        f"Stage 4 inter-arm TCP distance {min_tcp_dist:.4f} m < "
        f"{min_clearance_m:.4f} m"
    )
  safety.verify_bimanual_trajectory(traj_l_out, traj_r_out)
  return min_tcp_z, min_tcp_dist, n_preview


def _execute_greeting_with_recording(
    robot: Fr3DuoRobot,
    rig: Fr3DuoCameraController,
    *,
    mp4_path: Path,
    jsonl_path: Path,
    sheet_path: Path,
    fps: float,
    max_speed_deg_s: float,
) -> tuple[bool, int, int, dict[str, Any]]:
  """Runs Stage 4 greeting + gripper cycle while recording 4-camera MP4."""
  writer = MosaicVideoWriter(
      mp4_path,
      fps=fps,
      first_frame_path=sheet_path,
  )
  jsonl_path.parent.mkdir(parents=True, exist_ok=True)
  record_lock = threading.Lock()
  stop_rec = threading.Event()
  current_phase = "pre_motion"
  rec_errors: list[Exception] = []
  motion_frames_written = 0
  aborted = False
  gripper_summary: dict[str, Any] = {}

  try:
    with jsonl_path.open("w", encoding="utf-8") as log_f:

      def _write_one_frame(phase_label: str) -> None:
        nonlocal motion_frames_written
        batch = rig.capture_all()
        with record_lock:
          idx_now, _ = writer.write_batch(batch)
          if phase_label in _MOTION_PHASES:
            motion_frames_written += 1
          event = {
              "frame_index": idx_now,
              "phase": phase_label,
              "monotonic_s": round(time.monotonic(), 6),
          }
          log_f.write(json.dumps(event) + "\n")

      def _recorder_worker() -> None:
        dt_frame = 1.0 / max(1.0, float(fps))
        while not stop_rec.is_set():
          t0 = time.monotonic()
          try:
            _write_one_frame(current_phase)
          except Exception as exc:  # pylint: disable=broad-exception-caught
            rec_errors.append(exc)
            break
          wait_s = max(0.0, dt_frame - (time.monotonic() - t0))
          stop_rec.wait(min(wait_s, 0.05))

      _write_one_frame("pre_motion")
      rec_thread = threading.Thread(
          target=_recorder_worker, name="fr3_stage4_recorder", daemon=True
      )
      rec_thread.start()

      try:
        st_init = robot.read_state()
        ql_start = clamp_q_to_padded_limits(st_init.left.q_rad).tolist()
        qr_start = clamp_q_to_padded_limits(st_init.right.q_rad).tolist()
        ql_mid = clamp_q_to_padded_limits(
            [q + dq for q, dq in zip(ql_start, GREETING_DELTA_LEFT_RAD)]
        ).tolist()
        qr_mid = clamp_q_to_padded_limits(
            [q + dq for q, dq in zip(qr_start, GREETING_DELTA_RIGHT_RAD)]
        ).tolist()

        with AbortWatcher() as watcher:
          current_phase = "outbound_greeting"
          out_res = robot.move_bimanual_waypoints(
              [ql_start, ql_mid],
              [qr_start, qr_mid],
              abort_watcher=watcher,
              speed_deg_s=min(max_speed_deg_s, 20.0),
          )
          aborted = bool(out_res.aborted)

          if not aborted:
            current_phase = "gripper_cycle"
            robot.activate_grippers()
            for arm in ("left", "right"):
              g_open_0 = robot.move_gripper(arm, 0.0)
              g_closed = robot.move_gripper(arm, 1.0)
              g_open_1 = robot.move_gripper(arm, 0.0)
              gripper_summary[arm] = {
                  "open_initial": round(float(g_open_0.position_ratio), 4),
                  "closed_ratio": round(float(g_closed.position_ratio), 4),
                  "open_final": round(float(g_open_1.position_ratio), 4),
              }

            current_phase = "return_to_rest"
            ret_res = robot.move_bimanual_waypoints(
                [ql_mid, ql_start],
                [qr_mid, qr_start],
                abort_watcher=watcher,
                speed_deg_s=min(max_speed_deg_s, 20.0),
            )
            aborted = aborted or bool(ret_res.aborted)
      finally:
        stop_rec.set()
        rec_thread.join(timeout=2.0)

      if rec_errors:
        raise RuntimeError(
            "Background camera capture failed during Stage 4: "
            f"{rec_errors[0]}"
        ) from rec_errors[0]
      _write_one_frame("returned_to_rest")
  finally:
    summary = writer.close()

  return (
      aborted,
      int(summary.frame_count),
      motion_frames_written,
      gripper_summary,
  )


def run_hello_world(
    config: RobotConfig | Mapping[str, Any] | None = None,
    *,
    fps: float = 15.0,
    dry_run: bool = False,
    realtime: str = "auto",
    output_dir: Path | None = None,
    output_mp4: Path | None = None,
    output_json: Path | None = None,
    robot_factory: Callable[..., Fr3DuoRobot] | None = None,
) -> tuple[int, HelloWorldReport]:
  """Runs Stage 4 offline trajectory preview or live bimanual Hello World."""
  cfg = normalize_robot_config(config)
  effective_dry_run = bool(dry_run)
  rt_info = resolve_realtime_config(
      "auto" if effective_dry_run else realtime
  )
  max_speed_deg_s = float(rt_info.get("max_joint_speed_deg_s", 15.0))
  min_tcp_z, min_tcp_dist, n_preview = _preview_greeting_safety(
      cfg, max_vel_rad_s=math.radians(max_speed_deg_s)
  )

  resolved_out_dir = resolve_stage_output_dir(
      "stage4", output_dir, dry_run=effective_dry_run
  )
  mode_label = "dry_run" if effective_dry_run else "hardware"

  if effective_dry_run:
    report = HelloWorldReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        stage="stage4_hello_world",
        mode=mode_label,
        dry_run=True,
        min_tcp_z_base_m=round(min_tcp_z, 5),
        min_inter_arm_tcp_dist_m=round(min_tcp_dist, 5),
        outbound_steps_1khz=int(n_preview),
        max_speed_deg_s=max_speed_deg_s,
    )
    json_path = output_json or (
        resolved_out_dir / "stage4_hello_world.json"
        if resolved_out_dir is not None
        else None
    )
    if json_path is not None:
      json_path.parent.mkdir(parents=True, exist_ok=True)
      json_path.write_text(
          json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
      )
      report.report_path = str(json_path)
    print(
        f"[DRY-RUN] Stage 4 FR3 Duo Hello World | "
        f"min_tcp_z={min_tcp_z:.4f} m | min_tcp_dist={min_tcp_dist:.4f} m | "
        f"steps={n_preview}"
    )
    return 0, report

  if resolved_out_dir is not None:
    base_dir = resolved_out_dir
  elif output_mp4 is not None:
    base_dir = output_mp4.parent
  else:
    base_dir = Path(tempfile.gettempdir()) / "fr3_duo_logs" / "stage4"

  mp4_path = output_mp4 or (base_dir / "stage4_hello_world.mp4")
  jsonl_path = base_dir / "stage4_hello_world_events.jsonl"
  sheet_path = base_dir / "stage4_contact_sheet.png"
  json_path = output_json or (base_dir / "stage4_hello_world.json")

  robot = _create_robot(
      cfg, realtime_mode=realtime, robot_factory=robot_factory
  )
  with robot:
    st_init = robot.read_state()
    readiness_warnings: list[str] = []
    for side in ("left", "right"):
      for w in validate_arm_readiness(side, st_init[side]):
        if w not in readiness_warnings:
          readiness_warnings.append(w)
    if readiness_warnings:
      for w in readiness_warnings:
        print(f"  WARNING: {w}")
      print(
          "ERROR: Refusing Stage 4 Hello World due to "
          f"{len(readiness_warnings)} preflight warning(s).",
          file=sys.stderr,
      )
      return 2, HelloWorldReport(
          timestamp_utc=datetime.now(timezone.utc).isoformat(),
          stage="stage4_hello_world",
          mode=mode_label,
          dry_run=False,
          preflight_blocked=True,
          warnings=readiness_warnings,
      )
    with Fr3DuoCameraController(specs=cfg.cameras, dry_run=False) as rig:
      freshness = rig.check_freshness(
          max_age_s=0.30, num_reads=2, interval_s=0.05
      )
      fresh_ok = bool(
          freshness["ok"]
          if isinstance(freshness, Mapping)
          else freshness.ok
      )
      fresh_errs = (
          freshness["errors"]
          if isinstance(freshness, Mapping)
          else freshness.errors
      )
      if not fresh_ok:
        raise RuntimeError(f"Camera pre-flight failed: {fresh_errs}")
      aborted, frames_written, motion_frames_written, gripper_summary = (
          _execute_greeting_with_recording(
              robot,
              rig,
              mp4_path=mp4_path,
              jsonl_path=jsonl_path,
              sheet_path=sheet_path,
              fps=fps,
              max_speed_deg_s=max_speed_deg_s,
          )
      )

  if not mp4_path.is_file() or mp4_path.stat().st_size == 0:
    raise RuntimeError(
        f"Failed to encode valid Stage 4 MP4 video at {mp4_path}"
    )

  report = HelloWorldReport(
      timestamp_utc=datetime.now(timezone.utc).isoformat(),
      stage="stage4_hello_world",
      mode=mode_label,
      dry_run=False,
      aborted=aborted,
      frames_written=frames_written,
      motion_frames_written=motion_frames_written,
      mp4_path=str(mp4_path),
      jsonl_path=str(jsonl_path),
      contact_sheet_path=str(sheet_path),
      min_tcp_z_base_m=round(min_tcp_z, 5),
      min_inter_arm_tcp_dist_m=round(min_tcp_dist, 5),
      gripper_summary=gripper_summary,
  )
  json_path.parent.mkdir(parents=True, exist_ok=True)
  json_path.write_text(
      json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
  )
  report.report_path = str(json_path)

  print(
      f"[{mode_label.upper()}] Stage 4 FR3 Duo Hello World | aborted={aborted}"
      f" | frames={frames_written} (motion={motion_frames_written}) | "
      f"mp4={mp4_path}"
  )
  return 0, report


def _preview_robot_trajectories(
    cfg: RobotConfig,
    *,
    selected_arms: tuple[str, ...],
    eff_mode: str,
    eff_delta_deg: float,
    max_vel_rad_s: float,
) -> dict[str, Any]:
  """Runs offline MJCF FK and collision preview for `selected_arms`."""
  safety = Fr3DuoSafetyChecker(cfg)
  mj_model = safety.mj_model
  z_table_m = float(cfg.z_table_m)
  preview: dict[str, Any] = {}
  rest_trajs = {
      side: np.asarray(
          [clamp_q_to_padded_limits(cfg.arms[side].rest_pose_rad).tolist()],
          dtype=np.float64,
      )
      for side in ("left", "right")
  }
  for side in selected_arms:
    q0 = clamp_q_to_padded_limits(cfg.arms[side].rest_pose_rad).tolist()
    q_nudge = _build_nudge_target(q0, eff_delta_deg, side)
    wps = (
        [q0, q_nudge, q0] if eff_mode in ("nudge", "abort-test") else [q0, q0]
    )
    traj = generate_quintic_trajectory(
        wps, dt_s=0.001, max_vel_rad_s=max_vel_rad_s
    )
    sample_idx = np.unique(
        np.linspace(0, len(traj) - 1, min(16, len(traj)), dtype=np.intp)
    )
    min_z = min(
        float(compute_arm_fk(mj_model, side, traj[idx])[0][2])
        for idx in sample_idx
    )
    if min_z < z_table_m:
      raise RuntimeError(
          f"{side} preview trajectory dips below z_table_m "
          f"({min_z:.4f} < {z_table_m:.4f})"
      )
    if side == "left":
      safety.verify_bimanual_trajectory(traj, rest_trajs["right"])
    else:
      safety.verify_bimanual_trajectory(rest_trajs["left"], traj)
    preview[side] = {
        "q_start_rad": [round(float(v), 6) for v in q0],
        "q_peak_rad": [round(float(v), 6) for v in q_nudge],
        "trajectory_steps_1khz": int(len(traj)),
        "min_tcp_z_base_m": round(min_z, 5),
    }
  return preview


def _run_live_robot_mode(
    cfg: RobotConfig,
    robot: Fr3DuoRobot,
    *,
    eff_mode: str,
    arm: str,
    selected_arms: tuple[str, ...],
    eff_delta_deg: float,
    max_speed_deg_s: float,
    st0: RobotState,
    results: dict[str, Any],
    warnings_list: list[str],
) -> None:
  """Executes one live Stage 3 actuation/homing mode using `robot`."""
  if eff_mode == "gripper":
    robot.activate_grippers()
    for side in selected_arms:
      st_open_1 = robot.move_gripper(side, 0.0)
      st_closed = robot.move_gripper(side, 1.0)
      st_open_2 = robot.move_gripper(side, 0.0)
      excursion = float(st_closed.position_ratio - st_open_1.position_ratio)
      ok_stroke = (
          st_closed.position_ratio >= EMPTY_CLOSE_MIN_RATIO
          and excursion >= 0.50
      )
      if not ok_stroke:
        warnings_list.append(
            f"{side}: gripper stroke below threshold "
            f"(closed={st_closed.position_ratio:.3f})"
        )
      results[side]["gripper_cycle"] = {
          "open_initial": round(float(st_open_1.position_ratio), 4),
          "closed_ratio": round(float(st_closed.position_ratio), 4),
          "open_final": round(float(st_open_2.position_ratio), 4),
          "is_empty_closed": bool(st_closed.is_empty_closed),
          "stroke_ok": ok_stroke,
      }
    return

  if eff_mode == "nudge":
    with AbortWatcher() as watcher:
      for side in selected_arms:
        q_now = clamp_q_to_padded_limits(
            robot.read_state()[side]["q_rad"]
        ).tolist()
        q_peak = _build_nudge_target(q_now, eff_delta_deg, side)
        move_res = robot.move_arm_waypoints(
            side,
            [q_now, q_peak, q_now],
            abort_watcher=watcher,
            speed_deg_s=min(max_speed_deg_s, 15.0),
        )
        results[side]["nudge"] = (
            move_res.to_dict() if hasattr(move_res, "to_dict") else move_res
        )
        if watcher.aborted:
          break
    return

  if eff_mode == "abort-test":
    target_arm = selected_arms[0]
    q_now = clamp_q_to_padded_limits(
        robot.read_state()[target_arm]["q_rad"]
    ).tolist()
    grip_before = float(robot.read_state()[target_arm]["gripper_ratio"])
    q_far = _build_nudge_target(q_now, eff_delta_deg, target_arm)
    timer: threading.Timer | None = None
    with AbortWatcher() as watcher:
      if not sys.stdin.isatty():
        timer = threading.Timer(
            0.015, watcher.trigger, kwargs={"reason": "auto_abort_test"}
        )
        timer.daemon = True
        timer.start()
      try:
        if sys.stdin.isatty():
          print(
              f"  [ACTIVE] Moving {target_arm} arm slowly (10 deg @ 4 deg/s)"
              " — type 'q' + ENTER (or Ctrl+C) now to trigger"
              " abort-and-unwind.",
              flush=True,
          )
        move_res = robot.move_arm_waypoints(
            target_arm,
            [q_now, q_far],
            abort_watcher=watcher,
            speed_deg_s=4.0,
        )
      finally:
        if timer is not None:
          timer.cancel()
    grip_after = float(robot.read_state()[target_arm]["gripper_ratio"])
    move_dict = (
        move_res.to_dict() if hasattr(move_res, "to_dict") else dict(move_res)
    )
    results[target_arm]["abort_test"] = {
        **move_dict,
        "gripper_preserved": abs(grip_after - grip_before) < 1e-3,
    }
    return

  if eff_mode == "home":
    speed_cap = min(max_speed_deg_s, 15.0)
    with AbortWatcher() as watcher:
      if arm == "both":
        q_l_now = clamp_q_to_padded_limits(st0["left"]["q_rad"]).tolist()
        q_r_now = clamp_q_to_padded_limits(st0["right"]["q_rad"]).tolist()
        q_l_home = clamp_q_to_padded_limits(
            cfg.arms.left.rest_pose_rad
        ).tolist()
        q_r_home = clamp_q_to_padded_limits(
            cfg.arms.right.rest_pose_rad
        ).tolist()
        move_res = robot.move_bimanual_waypoints(
            [q_l_now, q_l_home],
            [q_r_now, q_r_home],
            abort_watcher=watcher,
            speed_deg_s=speed_cap,
        )
        results["left"]["home"] = {
            "ok": move_res["ok"],
            "aborted": move_res["aborted"],
            "final_q_rad": list(move_res["final_left_q_rad"]),
        }
        results["right"]["home"] = {
            "ok": move_res["ok"],
            "aborted": move_res["aborted"],
            "final_q_rad": list(move_res["final_right_q_rad"]),
        }
      else:
        side = selected_arms[0]
        q_now = clamp_q_to_padded_limits(st0[side]["q_rad"]).tolist()
        q_home = clamp_q_to_padded_limits(
            cfg.arms[side].rest_pose_rad
        ).tolist()
        home_res = robot.move_arm_waypoints(
            side, [q_now, q_home], abort_watcher=watcher, speed_deg_s=speed_cap
        )
        results[side]["home"] = (
            home_res.to_dict() if hasattr(home_res, "to_dict") else home_res
        )


def run_robot_check(
    config: RobotConfig | Mapping[str, Any] | None = None,
    *,
    mode: str = "check",
    arm: str = "both",
    delta_deg: float | None = None,
    read_only: bool = False,
    dry_run: bool = False,
    realtime: str = "auto",
    fps: float = 15.0,
    output_dir: Path | None = None,
    output_mp4: Path | None = None,
    output_json: Path | None = None,
    robot_factory: Callable[..., Fr3DuoRobot] | None = None,
) -> tuple[int, RobotCheckReport | HelloWorldReport]:
  """Runs Stage 3-4 offline trajectory check or live robot verification."""
  valid_modes = (
      "check",
      "gripper",
      "nudge",
      "abort-test",
      "home",
      "hello-world",
  )
  if mode not in valid_modes:
    raise ValueError(
        f"Invalid check_robot mode={mode!r}; expected one of {valid_modes}"
    )
  if arm not in ("both", "left", "right"):
    raise ValueError(
        f"Invalid arm={arm!r}; expected 'both', 'left', or 'right'"
    )

  cfg = normalize_robot_config(config)
  eff_mode = "check" if read_only else mode
  effective_dry_run = bool(dry_run or read_only)

  if eff_mode == "hello-world":
    return run_hello_world(
        cfg,
        fps=fps,
        dry_run=effective_dry_run,
        realtime=realtime,
        output_dir=output_dir,
        output_mp4=output_mp4,
        output_json=output_json,
        robot_factory=robot_factory,
    )

  rt_info = resolve_realtime_config(
      "auto" if effective_dry_run else realtime
  )
  max_speed_deg_s = float(rt_info.get("max_joint_speed_deg_s", 15.0))
  selected_arms = ("left", "right") if arm == "both" else (arm,)
  eff_delta_deg = (
      float(delta_deg)
      if delta_deg is not None
      else (10.0 if eff_mode == "abort-test" else 5.0)
  )

  preview = _preview_robot_trajectories(
      cfg,
      selected_arms=selected_arms,
      eff_mode=eff_mode,
      eff_delta_deg=eff_delta_deg,
      max_vel_rad_s=math.radians(max_speed_deg_s),
  )

  results: dict[str, Any] = {}
  warnings_list: list[str] = []
  preflight_blocked = False

  if not effective_dry_run:
    robot = _create_robot(
        cfg, realtime_mode=realtime, robot_factory=robot_factory
    )
    with robot:
      st0 = robot.read_state()
      for side in ("left", "right"):
        arm_st = st0[side]
        fk_err_m = float(arm_st.get("fk_pos_error_m", 0.0))
        f_ext_n = float(
            np.linalg.norm(
                np.asarray(arm_st["tau_ext_base_6d"], dtype=np.float64)[:3]
            )
        )
        for w in validate_arm_readiness(side, arm_st):
          if w not in warnings_list:
            warnings_list.append(w)
        results[side] = {
            "initial_fk_pos_error_m": round(fk_err_m, 6),
            "initial_force_norm_n": round(f_ext_n, 3),
            "initial_gripper_ratio": round(float(arm_st["gripper_ratio"]), 4),
            "desk_audit": arm_st.get("desk_audit"),
        }

      if eff_mode != "check" and warnings_list:
        preflight_blocked = True
      else:
        _run_live_robot_mode(
            cfg,
            robot,
            eff_mode=eff_mode,
            arm=arm,
            selected_arms=selected_arms,
            eff_delta_deg=eff_delta_deg,
            max_speed_deg_s=max_speed_deg_s,
            st0=st0,
            results=results,
            warnings_list=warnings_list,
        )

  mode_label = "dry_run" if effective_dry_run else "hardware"
  report = RobotCheckReport(
      timestamp_utc=datetime.now(timezone.utc).isoformat(),
      stage="stage3_check_robot",
      check_mode=eff_mode,
      execution_mode=mode_label,
      dry_run=effective_dry_run,
      preflight_blocked=preflight_blocked,
      realtime_profile=dict(rt_info),
      preview=preview,
      results=results,
      warnings=warnings_list,
  )

  resolved_out_dir = resolve_stage_output_dir(
      "stage3", output_dir, dry_run=effective_dry_run
  )
  json_path = output_json or (
      resolved_out_dir / f"check_robot_{eff_mode}.json"
      if resolved_out_dir is not None
      else None
  )
  if json_path is not None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    report.report_path = str(json_path)

  tag = "[DRY-RUN]" if effective_dry_run else "[HARDWARE]"
  print(
      f"{tag} Stage 3 Robot Check (mode={eff_mode}, arms={list(selected_arms)})"
      f" | max_speed={max_speed_deg_s:.1f} deg/s | "
      f"warnings={len(warnings_list)}"
  )
  for warning in warnings_list:
    print(f"  WARNING: {warning}")
  if preflight_blocked:
    print(
        f"ERROR: Refusing Stage 3 hardware actuation ({eff_mode}) due to "
        f"{len(warnings_list)} preflight warning(s).",
        file=sys.stderr,
    )
    return 2, report
  if eff_mode == "gripper" and any(
      not (results.get(side, {}).get("gripper_cycle") or {}).get(
          "stroke_ok", True
      )
      for side in selected_arms
  ):
    return 1, report
  return 0, report
