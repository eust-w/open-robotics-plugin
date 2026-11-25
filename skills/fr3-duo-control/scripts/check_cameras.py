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

"""Stage 2 4-camera freshness check, 2x2 contact sheet, and MP4 recorder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from fr3_arm_controller import read_arm_joints_or_rest
from fr3_camera_controller import (
    CANONICAL_CAMERAS,
    CameraFrame,
    Fr3DuoCameraController,
    apply_rectified_camera_calibration,
    render_contact_sheet,
    scale_camera_calibration,
)
from fr3_config import (
    RobotConfig,
    normalize_robot_config,
    resolve_stage_output_dir,
    save_rgb_png,
)
import fr3_kinematics
import fr3_mjcf
from fr3_recording import _DictCompatMixin, record_mosaic_clip

if TYPE_CHECKING:
  import fr3_duo_robot


@dataclass
class CameraCheckReport(_DictCompatMixin):
  """Structured Stage 2 camera check report written to JSON."""

  timestamp_utc: str
  stage: str
  mode: str
  dry_run: bool
  freshness: Any
  contact_sheet_shape: list[int]
  contact_sheet_path: str | None
  fk_projection_overlay_path: str | None
  recording: dict[str, Any] | None
  report_path: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Converts the camera check report to a JSON-serializable dict."""
    freshness_val = (
        self.freshness.to_dict()
        if hasattr(self.freshness, "to_dict")
        else self.freshness
    )
    out: dict[str, Any] = {
        "timestamp_utc": self.timestamp_utc,
        "stage": self.stage,
        "mode": self.mode,
        "dry_run": self.dry_run,
        "freshness": freshness_val,
        "contact_sheet_shape": list(self.contact_sheet_shape),
        "contact_sheet_path": self.contact_sheet_path,
        "fk_projection_overlay_path": self.fk_projection_overlay_path,
        "recording": (
            dict(self.recording) if self.recording is not None else None
        ),
    }
    if self.report_path is not None:
      out["report_path"] = self.report_path
    return out


def _record_mosaic_mp4(
    rig: Fr3DuoCameraController,
    mp4_path: Path,
    jsonl_path: Path,
    *,
    duration_s: float,
    target_fps: float,
) -> dict[str, Any]:
  """Records 2x2 contact-sheet frames to MP4 and per-frame logs to JSONL."""
  summary = record_mosaic_clip(
      rig.capture_all,
      mp4_path,
      jsonl_path,
      duration_s=duration_s,
      fps=target_fps,
  )
  if not mp4_path.is_file() or mp4_path.stat().st_size == 0:
    raise RuntimeError(f"Failed to encode valid MP4 video at {mp4_path}")

  return {
      "mp4_path": summary.mp4_path or str(mp4_path),
      "jsonl_path": summary.jsonl_path or str(jsonl_path),
      "frames_written": int(summary.frame_count),
      "resolution": list(summary.resolution),
  }


def _render_fk_overlay(
    cfg: RobotConfig,
    snapshot: Mapping[str, CameraFrame],
    fk_file: Path,
    *,
    realtime: str,
    rectified_calibration: Mapping[str, Mapping[str, Any]] | None = None,
) -> str | None:
  """Projects left/right pinch TCPs onto camera frames and writes mosaic."""
  try:
    left_q, right_q, fk_warns = read_arm_joints_or_rest(
        cfg, realtime_mode=realtime
    )
    for fw in fk_warns:
      print(f"Warning: {fw}", file=sys.stderr)
    mj_model, _ = fr3_mjcf.build_fr3_duo_mjcf(cfg)
    cal = fr3_kinematics.get_camera_calibration_opencv(
        mj_model, left_q, right_q
    )
    apply_rectified_camera_calibration(cal, rectified_calibration)
    p_left, _ = fr3_kinematics.compute_arm_fk(mj_model, "left", left_q)
    p_right, _ = fr3_kinematics.compute_arm_fk(mj_model, "right", right_q)
    pinch_sites = {"L pinch": p_left, "R pinch": p_right}
    colors = {
        "L pinch": np.array([0, 255, 255], dtype=np.uint8),
        "R pinch": np.array([255, 128, 0], dtype=np.uint8),
    }

    overlay_snapshot: dict[str, CameraFrame] = {}
    for c_name in CANONICAL_CAMERAS:
      frm = snapshot.get(c_name)
      if frm is None:
        continue
      tile = frm.rgb.copy()
      h_tile, w_tile = tile.shape[:2]
      if c_name in cal:
        cam_cal = cal[c_name]
        scale_camera_calibration(cam_cal, w_tile, h_tile)
        k_val = cam_cal["K"] if isinstance(cam_cal, Mapping) else cam_cal.K
        k_mat = np.asarray(k_val, dtype=np.float64)
        t_base_cam = (
            cam_cal["T_base_cam"]
            if isinstance(cam_cal, Mapping)
            else cam_cal.T_base_cam
        )
        for label, p_base in pinch_sites.items():
          (u, v), z_depth = fr3_kinematics.project_base_points_opencv(
              p_base, k_mat, t_base_cam
          )
          if z_depth > 0.04:
            ui, vi = int(round(u)), int(round(v))
            col = colors[label]
            r_c = 18
            if -r_c <= ui < w_tile + r_c and -r_c <= vi < h_tile + r_c:
              tile[
                  max(0, vi - 2) : min(h_tile, vi + 3),
                  max(0, ui - r_c) : min(w_tile, ui + r_c + 1),
              ] = col
              tile[
                  max(0, vi - r_c) : min(h_tile, vi + r_c + 1),
                  max(0, ui - 2) : min(w_tile, ui + 3),
              ] = col
      overlay_snapshot[c_name] = CameraFrame(
          name=frm.name,
          rgb=tile,
          timestamp_s=frm.timestamp_s,
          width=frm.width,
          height=frm.height,
          backend=frm.backend,
      )
    save_rgb_png(render_contact_sheet(overlay_snapshot), fk_file)
    return str(fk_file)
  except (RuntimeError, OSError, ValueError, KeyError) as exc:
    print(
        f"Warning: Failed to generate fk_projection_overlay_2x2.png: {exc}",
        file=sys.stderr,
    )
    return None


def run_camera_check(
    config: RobotConfig | Mapping[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
    output_mp4: Path | None = None,
    output_json: Path | None = None,
    duration_sec: float | None = None,
    fps: float = 15.0,
    max_age_sec: float = 0.30,
    num_reads: int = 3,
    save_references: bool = False,
    dry_run: bool = False,
    realtime: str = "auto",
    sim_robot: fr3_duo_robot.Fr3DuoRobot | None = None,
) -> tuple[int, CameraCheckReport]:
  """Runs Stage 2 4-camera freshness check, contact sheet, and MP4 recording."""
  cfg = normalize_robot_config(config)
  effective_dry_run = bool(dry_run)

  resolved_out_dir = resolve_stage_output_dir(
      "stage2_cameras", output_dir, dry_run=effective_dry_run
  )
  duration_s = (
      float(duration_sec)
      if duration_sec is not None
      else (0.25 if effective_dry_run else 2.0)
  )

  with Fr3DuoCameraController(
      specs=cfg.cameras,
      dry_run=effective_dry_run,
      sim_robot=sim_robot,
  ) as rig:
    freshness = rig.check_freshness(
        max_age_s=max_age_sec,
        num_reads=num_reads,
        interval_s=0.01 if effective_dry_run else 0.05,
    )
    snapshot = rig.capture_all()
    mosaic_rgb = render_contact_sheet(snapshot)

    contact_sheet_path: str | None = None
    fk_overlay_path: str | None = None
    recording_info: dict[str, Any] | None = None

    if resolved_out_dir is not None or output_mp4 is not None:
      if resolved_out_dir is not None:
        base_dir = resolved_out_dir
      else:
        assert output_mp4 is not None
        base_dir = output_mp4.parent
      sheet_file = base_dir / "contact_sheet_2x2.png"
      save_rgb_png(mosaic_rgb, sheet_file)
      contact_sheet_path = str(sheet_file)

      if not effective_dry_run:
        rect_getter = getattr(rig, "get_rectified_calibration", None)
        rect_cal = rect_getter() if callable(rect_getter) else None
        fk_overlay_path = _render_fk_overlay(
            cfg,
            snapshot,
            base_dir / "fk_projection_overlay_2x2.png",
            realtime=realtime,
            rectified_calibration=rect_cal,
        )

      if save_references:
        for cam_name in CANONICAL_CAMERAS:
          if cam_name in snapshot:
            save_rgb_png(
                snapshot[cam_name].rgb, base_dir / f"ref_{cam_name}.png"
            )

      mp4_file = output_mp4 or (base_dir / "cameras_recording.mp4")
      jsonl_file = base_dir / "camera_frames.jsonl"
      recording_info = _record_mosaic_mp4(
          rig,
          mp4_file,
          jsonl_file,
          duration_s=duration_s,
          target_fps=fps,
      )

  mode_label = "dry_run" if effective_dry_run else "hardware"
  report = CameraCheckReport(
      timestamp_utc=datetime.now(timezone.utc).isoformat(),
      stage="stage2_check_cameras",
      mode=mode_label,
      dry_run=effective_dry_run,
      freshness=freshness,
      contact_sheet_shape=list(mosaic_rgb.shape),
      contact_sheet_path=contact_sheet_path,
      fk_projection_overlay_path=fk_overlay_path,
      recording=recording_info,
  )

  json_path = output_json or (
      resolved_out_dir / "camera_check_report.json"
      if resolved_out_dir is not None
      else None
  )
  if json_path is not None:
    report.report_path = str(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

  fresh_ok = bool(
      freshness["ok"] if isinstance(freshness, Mapping) else freshness.ok
  )
  fresh_cams = (
      freshness["cameras"]
      if isinstance(freshness, Mapping)
      else freshness.cameras
  )
  fresh_errs = (
      freshness.get("errors", [])
      if isinstance(freshness, Mapping)
      else freshness.errors
  )
  tag = "[DRY-RUN]" if effective_dry_run else "[HARDWARE]"
  print(
      f"{tag} Stage 2 Camera Check | ok={fresh_ok} | "
      f"streams={sorted(fresh_cams.keys())} | "
      f"mosaic={list(mosaic_rgb.shape)}"
  )
  if contact_sheet_path is not None:
    print(f"  contact_sheet: {contact_sheet_path}")
  if fk_overlay_path is not None:
    print(f"  fk_overlay   : {fk_overlay_path}")
  if recording_info is not None and recording_info.get("mp4_path"):
    print(
        f"  recording_mp4: {recording_info['mp4_path']} "
        f"({recording_info.get('frames_written', 0)} frames)"
    )
  if json_path is not None:
    print(f"  report_json  : {json_path}")
  for err in fresh_errs:
    print(f"  ERROR: {err}", file=sys.stderr)

  return (0 if fresh_ok else 1), report
