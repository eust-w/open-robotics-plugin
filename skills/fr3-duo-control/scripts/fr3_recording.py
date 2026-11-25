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

"""Episode recording, inspection, and MuJoCo replay utilities for FR3 Duo."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Any

from fr3_camera_controller import (
    CameraFrame,
    _DictCompatMixin,
    build_sim_camera_frames,
    render_contact_sheet,
)
from fr3_config import (
    RobotConfig,
    default_robot_config,
    normalize_robot_config,
    open_mp4_writer,
    save_rgb_png,
)
import fr3_kinematics
import numpy as np

__all__ = [
    "EpisodeRecorder",
    "EpisodeSummary",
    "MosaicVideoSummary",
    "MosaicVideoWriter",
    "PhaseSlice",
    "RecordedArmTelemetry",
    "RecordedEpisode",
    "RecordedFrame",
    "ReplayResult",
    "load_recorded_episode",
    "record_mosaic_clip",
    "replay_episode_in_sim",
]


@dataclass
class MosaicVideoSummary(_DictCompatMixin):
  """Summary of an encoded 2x2 contact-sheet MP4 and boundary PNGs."""

  mp4_path: str | None
  frame_count: int
  resolution: list[int]
  first_frame_path: str | None
  last_frame_path: str | None
  jsonl_path: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Converts the mosaic video summary to a plain dictionary."""
    return {
        "mp4_path": self.mp4_path,
        "frame_count": int(self.frame_count),
        "resolution": list(self.resolution),
        "first_frame_path": self.first_frame_path,
        "last_frame_path": self.last_frame_path,
        "jsonl_path": self.jsonl_path,
    }


@dataclass
class RecordedArmTelemetry(_DictCompatMixin):
  """Per-arm telemetry captured in a recorded episode frame."""

  q_rad: list[float]
  tcp_pos_base_m: list[float]
  gripper_ratio: float
  f_ext_norm_n: float

  def to_dict(self) -> dict[str, Any]:
    """Converts the arm telemetry to a plain dictionary."""
    return {
        "q_rad": [float(x) for x in self.q_rad],
        "tcp_pos_base_m": [float(x) for x in self.tcp_pos_base_m],
        "gripper_ratio": float(self.gripper_ratio),
        "f_ext_norm_n": float(self.f_ext_norm_n),
    }

  @classmethod
  def from_mapping(
      cls, raw: RecordedArmTelemetry | Mapping[str, Any]
  ) -> RecordedArmTelemetry:
    """Constructs `RecordedArmTelemetry` from an existing mapping or copy."""
    if isinstance(raw, cls):
      return copy.deepcopy(raw)
    return cls(
        q_rad=[float(x) for x in raw.get("q_rad", [])],
        tcp_pos_base_m=[float(x) for x in raw.get("tcp_pos_base_m", [])],
        gripper_ratio=float(raw.get("gripper_ratio", 0.0)),
        f_ext_norm_n=float(raw.get("f_ext_norm_n", 0.0)),
    )


@dataclass
class RecordedFrame(_DictCompatMixin):
  """Single synchronized JSONL frame/event row in `episode_frames.jsonl`."""

  frame_index: int | None
  elapsed_s: float
  timestamp_s: float
  phase: str
  event: str
  state_16d: list[float]
  arms: dict[str, RecordedArmTelemetry]
  camera_timestamps: dict[str, float] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    """Converts the recorded frame to a JSON-serializable dictionary."""
    return {
        "frame_index": self.frame_index,
        "elapsed_s": float(self.elapsed_s),
        "timestamp_s": float(self.timestamp_s),
        "phase": str(self.phase),
        "event": str(self.event),
        "state_16d": [float(x) for x in self.state_16d],
        "arms": {k: v.to_dict() for k, v in self.arms.items()},
        "camera_timestamps": {
            str(k): float(v) for k, v in self.camera_timestamps.items()
        },
    }

  @classmethod
  def from_mapping(
      cls, raw: RecordedFrame | Mapping[str, Any]
  ) -> RecordedFrame:
    """Constructs a `RecordedFrame` from a parsed JSONL row or mapping."""
    if isinstance(raw, cls):
      return copy.deepcopy(raw)
    raw_idx = raw.get("frame_index")
    raw_arms = raw.get("arms")
    arms_out: dict[str, RecordedArmTelemetry] = {}
    if isinstance(raw_arms, Mapping):
      for side, entry in raw_arms.items():
        if isinstance(entry, (RecordedArmTelemetry, Mapping)):
          arms_out[str(side)] = RecordedArmTelemetry.from_mapping(entry)
    raw_cam_ts = raw.get("camera_timestamps")
    cam_ts = (
        {str(k): float(v) for k, v in raw_cam_ts.items()}
        if isinstance(raw_cam_ts, Mapping)
        else {}
    )
    return cls(
        frame_index=int(raw_idx) if raw_idx is not None else None,
        elapsed_s=float(raw.get("elapsed_s", 0.0)),
        timestamp_s=float(raw.get("timestamp_s", 0.0)),
        phase=str(raw.get("phase", "")),
        event=str(raw.get("event", "")),
        state_16d=[float(x) for x in raw.get("state_16d", [])],
        arms=arms_out,
        camera_timestamps=cam_ts,
    )


@dataclass
class EpisodeSummary(_DictCompatMixin):
  """Manifest written to `episode_summary.json` when recording stops."""

  mode: str
  dry_run: bool
  fps: float
  resolution: list[int]
  frame_count: int
  state_sample_count: int
  duration_s: float
  phases: list[str]
  z_table_m: float
  tcp_offset_z_m: float
  mp4_path: str | None
  jsonl_path: str
  first_frame_path: str | None
  last_frame_path: str | None
  error: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Converts the episode summary to a JSON-serializable dictionary."""
    out: dict[str, Any] = {
        "mode": str(self.mode),
        "dry_run": bool(self.dry_run),
        "fps": float(self.fps),
        "resolution": list(self.resolution),
        "frame_count": int(self.frame_count),
        "state_sample_count": int(self.state_sample_count),
        "duration_s": float(self.duration_s),
        "phases": list(self.phases),
        "z_table_m": float(self.z_table_m),
        "tcp_offset_z_m": float(self.tcp_offset_z_m),
        "mp4_path": self.mp4_path,
        "jsonl_path": str(self.jsonl_path),
        "first_frame_path": self.first_frame_path,
        "last_frame_path": self.last_frame_path,
    }
    if self.error is not None:
      out["error"] = str(self.error)
    return out

  @classmethod
  def from_mapping(
      cls, raw: EpisodeSummary | Mapping[str, Any]
  ) -> EpisodeSummary:
    """Constructs an `EpisodeSummary` from a parsed JSON mapping."""
    if isinstance(raw, cls):
      return copy.deepcopy(raw)
    return cls(
        mode=str(raw.get("mode", "unknown")),
        dry_run=bool(raw.get("dry_run", False)),
        fps=float(raw.get("fps", 10.0)),
        resolution=[int(v) for v in raw.get("resolution", [0, 0])],
        frame_count=int(raw.get("frame_count", 0)),
        state_sample_count=int(raw.get("state_sample_count", 0)),
        duration_s=float(raw.get("duration_s", 0.0)),
        phases=[str(p) for p in raw.get("phases", [])],
        z_table_m=float(raw.get("z_table_m", 0.0)),
        tcp_offset_z_m=float(raw.get("tcp_offset_z_m", 0.145)),
        mp4_path=raw.get("mp4_path"),
        jsonl_path=str(raw.get("jsonl_path", "")),
        first_frame_path=raw.get("first_frame_path"),
        last_frame_path=raw.get("last_frame_path"),
        error=str(raw["error"]) if raw.get("error") is not None else None,
    )


@dataclass
class PhaseSlice(_DictCompatMixin):
  """Frame index range and list for one named episode phase."""

  count: int
  start_idx: int
  end_idx: int
  indices: list[int] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    """Converts the phase slice to a plain dictionary."""
    return {
        "count": int(self.count),
        "start_idx": int(self.start_idx),
        "end_idx": int(self.end_idx),
        "indices": list(self.indices),
    }


@dataclass
class RecordedEpisode(_DictCompatMixin):
  """Loaded episode telemetry, phase slices, and aggregate safety metrics."""

  summary: EpisodeSummary | dict[str, Any]
  frames: list[RecordedFrame]
  state_16d: np.ndarray
  elapsed_s: np.ndarray
  phases: list[str]
  phase_slices: dict[str, PhaseSlice]
  peak_f_ext_norm_n: dict[str, float]
  min_z_tcp_m: dict[str, float | None]
  frame_count: int
  state_sample_count: int
  duration_s: float

  def to_dict(self) -> dict[str, Any]:
    """Converts the loaded episode to a plain dictionary."""
    summary_val = (
        self.summary.to_dict()
        if isinstance(self.summary, EpisodeSummary)
        else dict(self.summary)
    )
    return {
        "summary": summary_val,
        "frames": [f.to_dict() for f in self.frames],
        "state_16d": self.state_16d.tolist(),
        "elapsed_s": self.elapsed_s.tolist(),
        "phases": list(self.phases),
        "phase_slices": {k: v.to_dict() for k, v in self.phase_slices.items()},
        "peak_f_ext_norm_n": dict(self.peak_f_ext_norm_n),
        "min_z_tcp_m": dict(self.min_z_tcp_m),
        "frame_count": int(self.frame_count),
        "state_sample_count": int(self.state_sample_count),
        "duration_s": float(self.duration_s),
    }


@dataclass
class ReplayResult(_DictCompatMixin):
  """Result returned by `replay_episode_in_sim()`."""

  frames_replayed: int
  mp4_path: str | None
  ee_trajectories: dict[str, list[list[float]]]
  min_z_tcp_m: float | None
  min_z_tcp_by_arm_m: dict[str, float | None]

  def to_dict(self) -> dict[str, Any]:
    """Converts the replay result to a JSON-serializable dictionary."""
    return {
        "frames_replayed": int(self.frames_replayed),
        "mp4_path": self.mp4_path,
        "ee_trajectories": {
            k: [list(pt) for pt in pts]
            for k, pts in self.ee_trajectories.items()
        },
        "min_z_tcp_m": self.min_z_tcp_m,
        "min_z_tcp_by_arm_m": dict(self.min_z_tcp_by_arm_m),
    }


def _validate_positive_fps(fps: float) -> float:
  fps_val = float(fps)
  if fps_val <= 0.0:
    raise ValueError(f"fps must be > 0, got {fps!r}")
  return fps_val


class MosaicVideoWriter:
  """Lazily initialized 2x2 contact-sheet MP4 and boundary PNG writer."""

  def __init__(
      self,
      mp4_path: str | Path,
      *,
      fps: float = 10.0,
      first_frame_path: str | Path | None = None,
      last_frame_path: str | Path | None = None,
  ) -> None:
    self.fps = _validate_positive_fps(fps)
    self.mp4_path = Path(mp4_path)
    self.first_frame_path = (
        Path(first_frame_path) if first_frame_path is not None else None
    )
    self.last_frame_path = (
        Path(last_frame_path) if last_frame_path is not None else None
    )
    self._writer: Any = None
    self.frame_count: int = 0
    self.resolution: list[int] = [0, 0]
    self.first_frame_written: bool = False
    self.last_frame_written: bool = False
    self._last_rgb: np.ndarray | None = None

  def write_batch(
      self, frames: Mapping[str, CameraFrame | np.ndarray]
  ) -> tuple[int, np.ndarray]:
    """Renders a 2x2 RGB mosaic from `frames` and appends it to the MP4."""
    mosaic_rgb = render_contact_sheet(frames)
    h, w = int(mosaic_rgb.shape[0]), int(mosaic_rgb.shape[1])
    if self._writer is None:
      self._writer = open_mp4_writer(
          self.mp4_path, width=w, height=h, fps=self.fps
      )
      self.resolution = [w, h]
    if (
        not self.first_frame_written
        and self.first_frame_path is not None
    ):
      save_rgb_png(mosaic_rgb, self.first_frame_path)
      self.first_frame_written = True

    bgr = np.ascontiguousarray(mosaic_rgb[:, :, ::-1])
    self._writer.write(bgr)
    frame_idx = self.frame_count
    self.frame_count += 1
    self._last_rgb = mosaic_rgb
    return frame_idx, mosaic_rgb

  def close(self) -> MosaicVideoSummary:
    """Writes `last_frame_path` if captured and releases the MP4 writer."""
    try:
      if self._last_rgb is not None and self.last_frame_path is not None:
        save_rgb_png(self._last_rgb, self.last_frame_path)
        self.last_frame_written = True
    finally:
      if self._writer is not None:
        try:
          self._writer.release()
        finally:
          self._writer = None

    return MosaicVideoSummary(
        mp4_path=str(self.mp4_path) if self.frame_count > 0 else None,
        frame_count=int(self.frame_count),
        resolution=list(self.resolution),
        first_frame_path=(
            str(self.first_frame_path) if self.first_frame_written else None
        ),
        last_frame_path=(
            str(self.last_frame_path) if self.last_frame_written else None
        ),
    )


def record_mosaic_clip(
    capture_fn: Callable[[], Mapping[str, CameraFrame | np.ndarray]],
    mp4_path: str | Path,
    jsonl_path: str | Path | None = None,
    *,
    duration_s: float = 2.0,
    fps: float = 10.0,
    first_frame_path: str | Path | None = None,
    last_frame_path: str | Path | None = None,
    event_builder: (
        Callable[
            [int, Mapping[str, CameraFrame | np.ndarray], float],
            Mapping[str, Any],
        ]
        | None
    ) = None,
    stop_event: threading.Event | None = None,
) -> MosaicVideoSummary:
  """Synchronously records a timed 2x2 camera mosaic MP4 and optional JSONL."""
  fps_val = _validate_positive_fps(fps)
  total_frames = max(2, int(round(max(0.1, float(duration_s)) * fps_val)))
  dt_s = 1.0 / max(1e-3, fps_val)
  writer = MosaicVideoWriter(
      mp4_path,
      fps=fps_val,
      first_frame_path=first_frame_path,
      last_frame_path=last_frame_path,
  )
  log_f = None
  try:
    if jsonl_path is not None:
      jpath = Path(jsonl_path)
      jpath.parent.mkdir(parents=True, exist_ok=True)
      log_f = jpath.open("w", encoding="utf-8")

    for idx in range(total_frames):
      if stop_event is not None and stop_event.is_set():
        break
      t0 = time.monotonic()
      batch = capture_fn()
      frame_idx, _ = writer.write_batch(batch)
      if log_f is not None:
        if event_builder is not None:
          row = dict(event_builder(frame_idx, batch, t0))
        else:
          row = {
              "frame_index": frame_idx,
              "captured_monotonic_s": round(t0, 6),
              "streams": {
                  name: {
                      "timestamp_s": round(float(frm.timestamp_s), 6),
                      "brightness_mean": round(float(frm.brightness_mean), 2),
                  }
                  for name, frm in batch.items()
                  if isinstance(frm, CameraFrame)
              },
          }
        log_f.write(json.dumps(row) + "\n")
      elapsed = time.monotonic() - t0
      remaining_s = max(0.0, dt_s - elapsed)
      if idx + 1 < total_frames and remaining_s > 0.0:
        if stop_event is not None:
          if stop_event.wait(remaining_s):
            break
        else:
          time.sleep(remaining_s)
  finally:
    if log_f is not None:
      log_f.close()
    summary = writer.close()

  summary.jsonl_path = str(jsonl_path) if jsonl_path is not None else None
  return summary


def _extract_arms_and_state16d(
    snapshot: Any,
) -> tuple[list[float], dict[str, RecordedArmTelemetry]]:
  """Formats `state_16d` and per-arm telemetry from a robot state snapshot."""
  raw_s16 = snapshot["state_16d"] if "state_16d" in snapshot else None
  has_s16 = raw_s16 is not None and len(raw_s16) == 16
  arms_out: dict[str, RecordedArmTelemetry] = {}
  state_16d: list[float] = [float(x) for x in raw_s16] if has_s16 else []
  for side in ("left", "right"):
    arm_data = snapshot[side]
    q_rad = [float(x) for x in arm_data["q_rad"]]
    tcp_pos = [float(x) for x in arm_data["tcp_pos_base"]]
    g_ratio = float(arm_data["gripper_ratio"])
    wrench = np.asarray(
        arm_data.get("tau_ext_base_6d", [0.0] * 6), dtype=np.float64
    )
    f_ext_norm_n = float(np.linalg.norm(wrench[:3]))
    arms_out[side] = RecordedArmTelemetry(
        q_rad=q_rad,
        tcp_pos_base_m=tcp_pos,
        gripper_ratio=g_ratio,
        f_ext_norm_n=f_ext_norm_n,
    )
    if not has_s16:
      state_16d.extend(q_rad)
      state_16d.append(g_ratio)
  return state_16d, arms_out


class EpisodeRecorder:
  """Thread-safe background/event-driven recorder for FR3 Duo episodes.

  Captures periodic `2x2` camera mosaic frames (`episode_mosaic.mp4`),
  boundary PNGs (`first_frame.png`, `last_frame.png`), synchronized per-tick and
  per-event telemetry (`episode_frames.jsonl`), and an `episode_summary.json`
  manifest.

  Hardware thread-safety: background ticks NEVER call `ArmWorkerProcess` or
  Robotiq serial status methods directly; instead they read
  `robot.get_latest_state_snapshot()` and guard camera capture via the robot's
  dedicated camera lock.
  """

  def __init__(
      self,
      robot: Any,
      output_dir: str | Path,
      *,
      fps: float = 10.0,
      width: int = 1280,
      height: int = 720,
      phase: str = "episode",
      record_cameras: bool = True,
  ) -> None:
    self._fps = _validate_positive_fps(fps)
    self._robot = robot
    self._output_dir = Path(output_dir)
    self._width = int(width)
    self._height = int(height)
    self._phase = str(phase)
    self._record_cameras = bool(record_cameras)

    self._mp4_path = self._output_dir / "episode_mosaic.mp4"
    self._jsonl_path = self._output_dir / "episode_frames.jsonl"
    self._summary_path = self._output_dir / "episode_summary.json"
    self._first_frame_path = self._output_dir / "first_frame.png"
    self._last_frame_path = self._output_dir / "last_frame.png"

    self._lock = threading.Lock()
    self._stop_event = threading.Event()
    self._thread: threading.Thread | None = None
    self._mosaic_writer: MosaicVideoWriter | None = None
    self._jsonl_file: Any = None

    self._started = False
    self._starting = False
    self._stopped = False
    self._start_mono_s = 0.0
    self._state_sample_count = 0
    self._state_seq = 0
    self._last_video_state_seq = -1
    self._phases: list[str] = []
    self._error: str | None = None
    self._summary: EpisodeSummary | None = None

  @property
  def output_dir(self) -> Path:
    """Returns the episode output directory."""
    return self._output_dir

  @property
  def is_recording(self) -> bool:
    """Returns True while the recorder is active and not yet stopped."""
    with self._lock:
      return self._started and not self._stopped

  @property
  def summary(self) -> EpisodeSummary | None:
    """Returns the finalized `EpisodeSummary` after `stop()`."""
    with self._lock:
      return (
          copy.deepcopy(self._summary) if self._summary is not None else None
      )

  def _record_phase_unlocked(self, phase_name: str) -> None:
    if phase_name not in self._phases:
      self._phases.append(phase_name)

  def _is_active_or_starting(self) -> bool:
    with self._lock:
      return (self._started or self._starting) and not self._stopped

  def _cleanup_failed_start(self) -> None:
    self._stop_event.set()
    with self._lock:
      thread = self._thread
    if (
        thread is not None
        and thread.is_alive()
        and thread is not threading.current_thread()
    ):
      thread.join(timeout=5.0)
    unregister_fn = getattr(self._robot, "_unregister_active_recorder", None)
    if callable(unregister_fn):
      try:
        unregister_fn(self)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    with self._lock:
      self._thread = None
      self._started = False
      self._starting = False
      if self._mosaic_writer is not None:
        try:
          self._mosaic_writer.close()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        finally:
          self._mosaic_writer = None
      if self._jsonl_file is not None:
        try:
          self._jsonl_file.close()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        finally:
          self._jsonl_file = None

  def start(self) -> EpisodeRecorder:
    """Starts the background recording loop and opens episode outputs."""
    with self._lock:
      if self._stopped:
        raise RuntimeError("Cannot restart a stopped EpisodeRecorder.")
      if self._started:
        return self
      if self._starting:
        raise RuntimeError("EpisodeRecorder start is already in progress.")
      self._starting = True

    try:
      register_fn = getattr(self._robot, "_register_active_recorder", None)
      if callable(register_fn):
        register_fn(self)

      # Seed initial state on the caller thread before spawning ticks.
      if hasattr(self._robot, "read_state") and (
          getattr(self._robot, "_connected", False)
          or getattr(self._robot, "mode", "sim") == "sim"
          or getattr(self._robot, "dry_run", False)
      ):
        self._robot.read_state()

      with self._lock:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_file = self._jsonl_path.open("w", encoding="utf-8")
        if self._record_cameras:
          self._mosaic_writer = MosaicVideoWriter(
              self._mp4_path,
              fps=self._fps,
              first_frame_path=self._first_frame_path,
              last_frame_path=self._last_frame_path,
          )
        self._start_mono_s = time.monotonic()
        self._stop_event.clear()
        self._started = True
        self._starting = False
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="fr3_episode_recorder",
            daemon=True,
        )
        self._thread.start()
    except Exception:
      self._cleanup_failed_start()
      raise
    return self

  def __enter__(self) -> EpisodeRecorder:
    return self.start()

  def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    self.stop()

  def set_phase(self, phase: str) -> None:
    """Updates the active episode phase label for subsequent samples."""
    with self._lock:
      self._phase = str(phase)

  def _write_jsonl_row_unlocked(
      self,
      *,
      frame_index: int | None,
      event: str,
      phase_name: str,
      snapshot: Any,
      camera_timestamps: Mapping[str, float],
      now_s: float,
  ) -> RecordedFrame:
    state_16d, arms = _extract_arms_and_state16d(snapshot)
    self._record_phase_unlocked(phase_name)
    row = RecordedFrame(
        frame_index=frame_index,
        elapsed_s=round(max(0.0, now_s - self._start_mono_s), 6),
        timestamp_s=float(now_s),
        phase=phase_name,
        event=str(event),
        state_16d=state_16d,
        arms=arms,
        camera_timestamps={
            str(k): float(v) for k, v in camera_timestamps.items()
        },
    )
    self._jsonl_file.write(json.dumps(row.to_dict()) + "\n")
    self._jsonl_file.flush()
    self._state_sample_count += 1
    self._state_seq += 1
    return row

  def log_state(
      self,
      event: str = "state",
      phase: str | None = None,
      *,
      snapshot: Any = None,
  ) -> RecordedFrame | None:
    """Logs a synchronized state event using cached/live state safely."""
    snap = (
        snapshot
        if snapshot is not None
        else self._robot.get_latest_state_snapshot()
    )
    now_s = time.monotonic()
    with self._lock:
      if not self._started or self._stopped or self._jsonl_file is None:
        return None
      active_phase = self._phase if phase is None else str(phase)
      return self._write_jsonl_row_unlocked(
          frame_index=None,
          event=event,
          phase_name=active_phase,
          snapshot=snap,
          camera_timestamps={},
          now_s=now_s,
      )

  def _capture_and_log_tick(self) -> None:
    if self._record_cameras:
      contact_frames, cam_timestamps = (
          self._robot._capture_recording_frames(
              width=self._width, height=self._height
          )
      )
    else:
      contact_frames = {}
      cam_timestamps = {}

    snap = self._robot.get_latest_state_snapshot()
    now_s = time.monotonic()
    with self._lock:
      if not self._started or self._stopped or self._jsonl_file is None:
        return
      frame_idx: int | None = None
      if self._record_cameras and self._mosaic_writer is not None:
        frame_idx, _ = self._mosaic_writer.write_batch(contact_frames)
      self._write_jsonl_row_unlocked(
          frame_index=frame_idx,
          event="tick",
          phase_name=self._phase,
          snapshot=snap,
          camera_timestamps=cam_timestamps,
          now_s=now_s,
      )
      if frame_idx is not None:
        self._last_video_state_seq = self._state_seq

  def _worker_loop(self) -> None:
    period_s = 1.0 / self._fps
    while not self._stop_event.is_set():
      t0 = time.monotonic()
      try:
        self._capture_and_log_tick()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        with self._lock:
          if self._error is None:
            self._error = f"{type(exc).__name__}: {exc}"
        break
      wait_s = max(0.0, period_s - (time.monotonic() - t0))
      if self._stop_event.wait(wait_s):
        break

  def stop(self) -> EpisodeSummary:
    """Stops recording, finalizes MP4/PNG/JSONL/summary, and returns summary."""
    with self._lock:
      if self._stopped and self._summary is not None:
        return copy.deepcopy(self._summary)
      if not self._started:
        raise RuntimeError("EpisodeRecorder was never started.")
      self._stop_event.set()
      thread = self._thread

    if (
        thread is not None
        and thread.is_alive()
        and thread is not threading.current_thread()
    ):
      thread.join(timeout=5.0)

    unregister_fn = getattr(self._robot, "_unregister_active_recorder", None)
    try:
      # Capture a final frame if no frames were recorded or state advanced
      # since the last video tick, so short episodes keep distinct endpoints.
      need_final_tick = False
      with self._lock:
        if (
            self._record_cameras
            and self._error is None
            and self._mosaic_writer is not None
            and (
                self._mosaic_writer.frame_count == 0
                or self._last_video_state_seq != self._state_seq
            )
        ):
          need_final_tick = True

      if need_final_tick:
        try:
          self._capture_and_log_tick()
        except Exception as exc:  # pylint: disable=broad-exception-caught
          with self._lock:
            if self._error is None:
              self._error = f"{type(exc).__name__}: {exc}"

      with self._lock:
        self._stopped = True
        video_info = MosaicVideoSummary(
            mp4_path=None,
            frame_count=0,
            resolution=[0, 0],
            first_frame_path=None,
            last_frame_path=None,
        )
        try:
          if self._mosaic_writer is not None:
            try:
              video_info = self._mosaic_writer.close()
            except Exception as exc:  # pylint: disable=broad-exception-caught
              if self._error is None:
                self._error = f"{type(exc).__name__}: {exc}"
            finally:
              self._mosaic_writer = None
        finally:
          if self._jsonl_file is not None:
            try:
              self._jsonl_file.close()
            finally:
              self._jsonl_file = None

        duration_s = max(0.0, time.monotonic() - self._start_mono_s)
        cfg = getattr(self._robot, "config", None)
        z_table_m = float(getattr(cfg, "z_table_m", 0.0))
        tcp_offset_z_m = float(getattr(cfg, "tcp_offset_z_m", 0.145))
        phases = list(self._phases) if self._phases else [self._phase]

        summary = EpisodeSummary(
            mode=str(getattr(self._robot, "mode", "unknown")),
            dry_run=bool(getattr(self._robot, "dry_run", False)),
            fps=float(self._fps),
            resolution=list(video_info.resolution),
            frame_count=int(video_info.frame_count),
            state_sample_count=int(self._state_sample_count),
            duration_s=round(duration_s, 6),
            phases=phases,
            z_table_m=z_table_m,
            tcp_offset_z_m=tcp_offset_z_m,
            mp4_path=video_info.mp4_path,
            jsonl_path=str(self._jsonl_path),
            first_frame_path=video_info.first_frame_path,
            last_frame_path=video_info.last_frame_path,
            error=self._error,
        )

        self._summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        self._summary = summary
        return copy.deepcopy(summary)
    finally:
      if callable(unregister_fn):
        try:
          unregister_fn(self)
        except Exception:  # pylint: disable=broad-exception-caught
          pass


def _resolve_episode_paths(path: str | Path) -> tuple[Path | None, Path | None]:
  target = Path(path)
  if not target.exists():
    raise FileNotFoundError(f"Episode path does not exist: {target}")
  if target.is_dir():
    s_path = target / "episode_summary.json"
    j_path = target / "episode_frames.jsonl"
    return (
        s_path if s_path.is_file() else None,
        j_path if j_path.is_file() else None,
    )
  if target.suffix == ".json" or target.name == "episode_summary.json":
    s_path = target
    j_path = target.parent / "episode_frames.jsonl"
    try:
      summary_raw = json.loads(s_path.read_text(encoding="utf-8"))
      cand = summary_raw.get("jsonl_path")
      if cand and Path(cand).is_file():
        j_path = Path(cand)
    except (OSError, ValueError):
      pass
    return s_path, (j_path if j_path.is_file() else None)
  j_path = target
  s_path = target.parent / "episode_summary.json"
  return (s_path if s_path.is_file() else None), j_path


def _build_recorded_episode(
    frames: list[RecordedFrame],
    summary: EpisodeSummary | dict[str, Any],
) -> RecordedEpisode:
  """Aggregates arrays, phase slices, peak forces, and min Z from `frames`."""
  if frames:
    state_rows = [row.state_16d for row in frames if len(row.state_16d) == 16]
    state_16d = (
        np.asarray(state_rows, dtype=np.float64)
        if state_rows
        else np.zeros((0, 16), dtype=np.float64)
    )
    elapsed_s = np.asarray(
        [float(row.elapsed_s) for row in frames],
        dtype=np.float64,
    )
  else:
    state_16d = np.zeros((0, 16), dtype=np.float64)
    elapsed_s = np.zeros((0,), dtype=np.float64)

  phases: list[str] = []
  phase_slices: dict[str, PhaseSlice] = {}
  peak_force: dict[str, float] = {"left": 0.0, "right": 0.0}
  z_vals: dict[str, list[float]] = {"left": [], "right": []}

  for idx, row in enumerate(frames):
    if row.phase:
      ph_str = str(row.phase)
      if ph_str not in phases:
        phases.append(ph_str)
      if ph_str not in phase_slices:
        phase_slices[ph_str] = PhaseSlice(
            count=0, start_idx=idx, end_idx=idx, indices=[]
        )
      bucket = phase_slices[ph_str]
      bucket.count += 1
      bucket.end_idx = idx
      bucket.indices.append(idx)

    for side in ("left", "right"):
      arm_entry = row.arms.get(side)
      if arm_entry is None:
        continue
      peak_force[side] = max(peak_force[side], float(arm_entry.f_ext_norm_n))
      if len(arm_entry.tcp_pos_base_m) >= 3:
        z_vals[side].append(float(arm_entry.tcp_pos_base_m[2]))

  summary_phases = summary.get("phases") if summary else None
  if not phases and isinstance(summary_phases, list):
    phases = [str(p) for p in summary_phases]

  all_z = z_vals["left"] + z_vals["right"]
  default_dur = (
      float(elapsed_s[-1] - elapsed_s[0]) if len(elapsed_s) >= 2 else 0.0
  )
  default_frame_count = sum(
      1 for row in frames if row.frame_index is not None
  )
  return RecordedEpisode(
      summary=summary,
      frames=frames,
      state_16d=state_16d,
      elapsed_s=elapsed_s,
      phases=phases,
      phase_slices=phase_slices,
      peak_f_ext_norm_n={
          "left": peak_force["left"],
          "right": peak_force["right"],
          "overall": max(peak_force["left"], peak_force["right"]),
      },
      min_z_tcp_m={
          "left": min(z_vals["left"]) if z_vals["left"] else None,
          "right": min(z_vals["right"]) if z_vals["right"] else None,
          "overall": min(all_z) if all_z else None,
      },
      frame_count=int(summary.get("frame_count", default_frame_count)),
      state_sample_count=len(frames),
      duration_s=float(summary.get("duration_s", default_dur)),
  )


def load_recorded_episode(path: str | Path) -> RecordedEpisode:
  """Loads a recorded episode directory, summary JSON, or frames JSONL."""
  summary_path, jsonl_path = _resolve_episode_paths(path)
  if summary_path is None and jsonl_path is None:
    raise FileNotFoundError(
        f"No episode_summary.json or episode_frames.jsonl found at {path}"
    )

  summary: EpisodeSummary | dict[str, Any] = {}
  if summary_path is not None:
    summary_raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(summary_raw, Mapping) and summary_raw:
      summary = EpisodeSummary.from_mapping(summary_raw)

  frames: list[RecordedFrame] = []
  if jsonl_path is not None:
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
      stripped = line.strip()
      if stripped:
        frames.append(RecordedFrame.from_mapping(json.loads(stripped)))

  return _build_recorded_episode(frames, summary)


def _extract_frame_joints_and_grippers(
    frame: RecordedFrame | Mapping[str, Any],
) -> tuple[list[float], list[float], float, float]:
  arms = frame.arms if isinstance(frame, RecordedFrame) else frame.get("arms")
  s16 = (
      frame.state_16d
      if isinstance(frame, RecordedFrame)
      else frame.get("state_16d")
  )
  if (
      isinstance(arms, Mapping)
      and "left" in arms
      and "right" in arms
  ):
    left = arms["left"]
    right = arms["right"]
    q_left = [float(x) for x in left["q_rad"]]
    q_right = [float(x) for x in right["q_rad"]]
    g_left = float(left.get("gripper_ratio", 0.0))
    g_right = float(right.get("gripper_ratio", 0.0))
    return q_left, q_right, g_left, g_right
  if isinstance(s16, Sequence) and len(s16) == 16:
    return (
        [float(x) for x in s16[0:7]],
        [float(x) for x in s16[8:15]],
        float(s16[7]),
        float(s16[15]),
    )
  raise ValueError(
      "Recorded frame must contain either `arms.{left,right}` or 16-D "
      "`state_16d`."
  )


def _subsample_frames(
    frames: Sequence[RecordedFrame | Mapping[str, Any]],
    max_frames: int | None,
) -> list[RecordedFrame | Mapping[str, Any]]:
  selected = list(frames)
  if max_frames is None:
    return selected
  cap = int(max_frames)
  if cap <= 0:
    return []
  if len(selected) <= cap:
    return selected
  if cap == 1:
    return [selected[0]]
  indices = np.linspace(0, len(selected) - 1, cap, dtype=np.intp)
  return [selected[int(i)] for i in indices]


def _step_replay_frame(
    sim_robot: Any,
    frame: RecordedFrame | Mapping[str, Any],
    prev_grip: dict[str, float | None],
    ee_trajectories: dict[str, list[list[float]]],
    min_z_by_arm: dict[str, float | None],
    *,
    writer: MosaicVideoWriter | None,
    width: int,
    height: int,
    frame_idx: int,
) -> None:
  q_left, q_right, g_left, g_right = _extract_frame_joints_and_grippers(frame)
  arm_states = (("left", q_left, g_left), ("right", q_right, g_right))
  with sim_robot._sim_lock:
    for side, q_vec, _ in arm_states:
      sim_robot._sim_backend.sync_qpos_no_actuators(side, q_vec)
    moved_gripper = False
    for side, _, g_ratio in arm_states:
      if (
          prev_grip[side] is None
          or abs(g_ratio - float(prev_grip[side])) > 1e-3
      ):
        sim_robot._sim_backend.sync_ctrl_to_qpos()
        sim_robot._sim_backend.move_gripper(side, g_ratio)
        prev_grip[side] = g_ratio
        moved_gripper = True
    if moved_gripper:
      for side, q_vec, _ in arm_states:
        sim_robot._sim_backend.sync_qpos_no_actuators(side, q_vec)

  for side, q_vec, _ in arm_states:
    tcp_pos, _ = fr3_kinematics.compute_arm_fk(sim_robot.mj_model, side, q_vec)
    tcp_list = [float(x) for x in tcp_pos]
    ee_trajectories[side].append(tcp_list)
    z_val = float(tcp_list[2])
    cur_min = min_z_by_arm[side]
    min_z_by_arm[side] = z_val if cur_min is None else min(cur_min, z_val)

  if writer is not None:
    sim_frames = sim_robot.render_sim_cameras(
        width=int(width), height=int(height)
    )
    ts_val = float(frame.get("elapsed_s", frame_idx))
    writer.write_batch(build_sim_camera_frames(sim_frames, ts_val))


def replay_episode_in_sim(
    episode_or_dir: str | Path | RecordedEpisode | Mapping[str, Any],
    *,
    cfg: RobotConfig | Mapping[str, Any] | None = None,
    output_mp4: str | Path | None = None,
    fps: float = 10.0,
    width: int = 640,
    height: int = 360,
    max_frames: int | None = None,
) -> ReplayResult:
  """Replays a recorded episode kinematically in MuJoCo simulation."""
  if isinstance(episode_or_dir, (str, Path)):
    episode: RecordedEpisode | Mapping[str, Any] = load_recorded_episode(
        episode_or_dir
    )
  elif isinstance(episode_or_dir, (RecordedEpisode, Mapping)):
    if "frames" not in episode_or_dir:
      raise ValueError(
          "Episode mapping passed to `replay_episode_in_sim` must contain "
          "'frames'."
      )
    episode = episode_or_dir
  else:
    raise TypeError(
        f"Expected path or episode mapping, got {type(episode_or_dir)!r}"
    )

  frames = _subsample_frames(episode.get("frames") or [], max_frames)
  from fr3_duo_robot import Fr3DuoRobot  # pylint: disable=g-import-not-at-top

  robot_cfg = (
      normalize_robot_config(cfg) if cfg is not None else default_robot_config()
  )
  ee_trajectories: dict[str, list[list[float]]] = {"left": [], "right": []}
  min_z_by_arm: dict[str, float | None] = {"left": None, "right": None}
  prev_grip: dict[str, float | None] = {"left": None, "right": None}
  writer: MosaicVideoWriter | None = (
      MosaicVideoWriter(output_mp4, fps=fps) if output_mp4 is not None else None
  )
  frames_replayed = 0
  written_mp4_path: str | None = None

  try:
    with Fr3DuoRobot(robot_cfg, mode="sim") as sim_robot:
      for frame in frames:
        _step_replay_frame(
            sim_robot,
            frame,
            prev_grip,
            ee_trajectories,
            min_z_by_arm,
            writer=writer,
            width=width,
            height=height,
            frame_idx=frames_replayed,
        )
        frames_replayed += 1
  finally:
    if writer is not None:
      writer_summary = writer.close()
      written_mp4_path = writer_summary.mp4_path

  z_candidates = [
      val
      for val in (min_z_by_arm["left"], min_z_by_arm["right"])
      if val is not None
  ]
  min_z_tcp_m = min(z_candidates) if z_candidates else None

  return ReplayResult(
      frames_replayed=frames_replayed,
      mp4_path=written_mp4_path,
      ee_trajectories=ee_trajectories,
      min_z_tcp_m=min_z_tcp_m,
      min_z_tcp_by_arm_m=min_z_by_arm,
  )
