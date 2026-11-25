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

"""Visual VQA and Gemini Decisions API helpers for FR3 Duo control.

Provides two scene-understanding primitives over `SceneObservation` objects
from `Fr3DuoRobot.observe_scene()` (or `load_saved_observation()`):

1. `ask_scene_vqa`: Open-ended multimodal VQA via `google-genai`
   `client.models.generate_content(...)`.
2. `decide_scene`: Closed-set calibrated yes/no (`noul`), `choice`, and `score`
   gates via the Gemini Decisions API (`client.decisions.create` with
   `gemini-decision-1`, zero output tokens).

When `GEMINI_API_KEY` is unset (or `google-genai` is unavailable), both helpers
save the camera PNGs, `contact_sheet_2x2.png`, `observation.json`, and the
pending question/decision payload to a directory so the host VLM agent can
inspect the images directly with `read_file`.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import (  # pylint: disable=g-importing-member
    Mapping,
    Sequence,
)
import datetime
import json
import os
from pathlib import Path  # pylint: disable=g-importing-member
import sys
from typing import Any

# pylint: disable=g-import-not-at-top
try:
  from google import genai  # type: ignore[import-not-found]
except ImportError:
  genai = None  # type: ignore[assignment]

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS_DIR))

from fr3_camera_controller import CANONICAL_CAMERAS  # noqa: E402
from fr3_camera_controller import render_contact_sheet  # noqa: E402
from fr3_camera_controller import require_cv2  # noqa: E402
from fr3_config import default_robot_config  # noqa: E402
from fr3_config import load_robot_config  # noqa: E402
from fr3_config import resolve_stage_output_dir  # noqa: E402
from fr3_config import save_rgb_png  # noqa: E402
import fr3_duo_robot  # noqa: E402
import numpy as np
# pylint: enable=g-import-not-at-top

DEFAULT_VQA_MODEL = "gemini-4-hydrogen"
DEFAULT_DECISION_MODEL = "gemini-decision-1"


def _to_json_safe(value: Any) -> Any:
  """Recursively converts numpy/Path/SDK values to JSON-safe types."""
  if hasattr(value, "to_json_dict") and callable(value.to_json_dict):
    return _to_json_safe(value.to_json_dict())
  if hasattr(value, "model_dump") and callable(value.model_dump):
    return _to_json_safe(value.model_dump(mode="json"))
  if hasattr(value, "to_dict") and callable(value.to_dict):
    return _to_json_safe(value.to_dict())
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, Mapping):
    return {str(k): _to_json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_to_json_safe(v) for v in value]
  return value


def _validate_cameras(cameras: Sequence[str]) -> tuple[str, ...]:
  if not cameras:
    raise ValueError("cameras must contain at least one camera name.")
  out: list[str] = []
  for cam in cameras:
    if cam not in CANONICAL_CAMERAS:
      raise ValueError(
          f"Unknown camera {cam!r}; expected one of {CANONICAL_CAMERAS}."
      )
    if cam not in out:
      out.append(cam)
  return tuple(out)


def format_robot_state_summary(obs: fr3_duo_robot.SceneObservation) -> str:
  """Formats a compact text summary of a `SceneObservation` for VLM prompts.

  Args:
    obs: Synchronized `SceneObservation` with `mode`, `dry_run`, `timestamp_s`,
      `z_table_m`, and `state` (`RobotState` containing `left`/`right`
      `ArmPoseState` with `q_rad`, `gripper_ratio`, and `tcp_pos_base`).

  Returns:
    Multi-line human-readable state summary string.
  """
  lines = [
      f"FR3 Duo scene observation (mode={obs.mode}, dry_run={obs.dry_run}, "
      f"t={float(obs.timestamp_s):.3f}s, z_table_m={float(obs.z_table_m):.4f})",
  ]
  for side, arm_state in (
      ("left", obs.state.left),
      ("right", obs.state.right),
  ):
    q_fmt = ", ".join(f"{float(v):.3f}" for v in arm_state.q_rad)
    tcp_fmt = ", ".join(f"{float(v):.3f}" for v in arm_state.tcp_pos_base)
    tcp_str = f" tcp_pos_base=[{tcp_fmt}]" if tcp_fmt else ""
    lines.append(
        f"  {side}: q_rad=[{q_fmt}] "
        f"gripper={float(arm_state.gripper_ratio):.3f}{tcp_str}"
    )
  return "\n".join(lines)


def encode_frame_jpeg_bytes(
    frame: np.ndarray, *, quality: int = 85
) -> bytes:
  """Encodes an `(H, W, 3)` uint8 RGB array as JPEG bytes via OpenCV.

  Args:
    frame: RGB image array with shape `(H, W, 3)`.
    quality: JPEG compression quality in `[0, 100]`.

  Returns:
    Encoded JPEG byte string.

  Raises:
    ValueError: If `frame` does not have shape `(H, W, 3)`.
    RuntimeError: If OpenCV is unavailable or `cv2.imencode` fails.
  """
  rgb = np.ascontiguousarray(frame, dtype=np.uint8)
  if rgb.ndim != 3 or rgb.shape[2] != 3:
    raise ValueError(f"Expected (H, W, 3) RGB frame, got {rgb.shape}.")
  cv2 = require_cv2()
  bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
  ok, encoded = cv2.imencode(
      ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
  )
  if not ok:
    raise RuntimeError("cv2.imencode('.jpg', ...) failed.")
  return encoded.tobytes()


def _maybe_contact_sheet(
    frames: Mapping[str, np.ndarray],
) -> np.ndarray | None:
  if all(cam in frames for cam in CANONICAL_CAMERAS):
    return render_contact_sheet(frames)
  return None


def _resolve_bundle_dir(
    output_dir: str | Path | None, *, prefix: str
) -> Path:
  if output_dir is not None:
    return Path(output_dir)
  base = resolve_stage_output_dir("vqa", None, dry_run=False)
  assert base is not None
  stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y%m%dT%H%M%SZ"
  )
  return base / f"{prefix}_{stamp}"


def save_vqa_bundle(
    obs: fr3_duo_robot.SceneObservation,
    *,
    output_dir: str | Path | None = None,
    request_payload: Mapping[str, Any] | None = None,
    request_filename: str = "vqa_request.json",
    prefix: str = "vqa",
) -> dict[str, str]:
  """Writes camera PNGs, contact sheet, observation.json, and request JSON.

  Args:
    obs: `SceneObservation` whose `frames` and metadata are written to disk.
    output_dir: Target directory. When `None`, defaults to a timestamped
      subdirectory under `resolve_stage_output_dir("vqa", None, dry_run=False)`.
    request_payload: Optional JSON-serializable request/result payload to write.
    request_filename: Filename for `request_payload` inside `output_dir`.
    prefix: Directory name prefix used when `output_dir` is `None`.

  Returns:
    Dictionary mapping `"output_dir"`, present canonical camera names,
    `"contact_sheet_2x2"` (when all 4 canonical cameras are present),
    `"observation_json"`, and `"request_json"` (when `request_payload` is
    provided) to their saved file paths.
  """
  out_dir = _resolve_bundle_dir(output_dir, prefix=prefix)
  out_dir.mkdir(parents=True, exist_ok=True)
  saved: dict[str, str] = {"output_dir": str(out_dir)}
  for cam_name in CANONICAL_CAMERAS:
    if cam_name not in obs.frames:
      continue
    cam_path = out_dir / f"{cam_name}.png"
    save_rgb_png(obs.frames[cam_name], cam_path)
    saved[cam_name] = str(cam_path)

  sheet = _maybe_contact_sheet(obs.frames)
  if sheet is not None:
    sheet_path = out_dir / "contact_sheet_2x2.png"
    save_rgb_png(sheet, sheet_path)
    saved["contact_sheet_2x2"] = str(sheet_path)

  summary = {
      k: v
      for k, v in obs.to_dict().items()
      if k not in ("frames", "saved_paths")
  }
  summary["saved_paths"] = dict(saved)
  obs_json_path = out_dir / "observation.json"
  obs_json_path.write_text(
      json.dumps(_to_json_safe(summary), indent=2) + "\n",
      encoding="utf-8",
  )
  saved["observation_json"] = str(obs_json_path)

  if request_payload is not None:
    req_path = out_dir / request_filename
    req_path.write_text(
        json.dumps(_to_json_safe(request_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    saved["request_json"] = str(req_path)

  return saved


def _collect_selected_frames(
    obs: fr3_duo_robot.SceneObservation,
    cameras: Sequence[str],
    *,
    include_contact_sheet: bool,
) -> list[tuple[str, np.ndarray]]:
  """Extracts validated camera frames and optional 2x2 contact sheet."""
  selected: list[tuple[str, np.ndarray]] = []
  for cam_name in _validate_cameras(cameras):
    if cam_name not in obs.frames:
      raise KeyError(
          f"Camera {cam_name!r} missing from observation frames "
          f"(available: {sorted(obs.frames.keys())})."
      )
    selected.append((cam_name, obs.frames[cam_name]))
  if include_contact_sheet:
    sheet = _maybe_contact_sheet(obs.frames)
    if sheet is not None:
      selected.append(("contact_sheet_2x2", sheet))
  return selected


def build_vqa_contents(
    obs: fr3_duo_robot.SceneObservation,
    question: str,
    *,
    cameras: Sequence[str] = ("head_camera_left",),
    include_contact_sheet: bool = False,
    jpeg_quality: int = 85,
) -> list[Any]:
  """Builds multimodal `contents` for `client.models.generate_content`.

  Args:
    obs: `SceneObservation` containing robot state and camera frames.
    question: Non-empty natural-language question about the scene.
    cameras: Canonical camera names to attach.
    include_contact_sheet: Whether to append `contact_sheet_2x2` when all 4
      canonical cameras are present.
    jpeg_quality: JPEG compression quality for attached images.

  Returns:
    List of text prompt strings, camera label strings, and inline-data image
    part dicts accepted by `client.models.generate_content`.

  Raises:
    ValueError: If `question` is empty or `cameras` is empty/unknown.
    KeyError: If any requested camera is missing from `obs.frames`.
  """
  if not question.strip():
    raise ValueError("question must be a non-empty string.")
  selected = _collect_selected_frames(
      obs, cameras, include_contact_sheet=include_contact_sheet
  )
  prompt_text = (
      f"{format_robot_state_summary(obs)}\n"
      f"Camera views attached: {[name for name, _ in selected]}\n"
      f"Question: {question.strip()}"
  )
  contents: list[Any] = [prompt_text]
  for cam_name, frame in selected:
    jpeg_bytes = encode_frame_jpeg_bytes(frame, quality=jpeg_quality)
    contents.append(f"[Camera: {cam_name}]")
    if genai is not None:
      contents.append(
          genai.types.Part.from_bytes(
              data=jpeg_bytes, mime_type="image/jpeg"
          )
      )
    else:
      contents.append({
          "inline_data": {
              "mime_type": "image/jpeg",
              "data": base64.urlsafe_b64encode(jpeg_bytes).decode("ascii"),
          }
      })
  return contents


def build_decision_input(
    obs: fr3_duo_robot.SceneObservation,
    *,
    cameras: Sequence[str] = ("head_camera_left",),
    resolution: str = "medium",
    include_contact_sheet: bool = False,
    jpeg_quality: int = 85,
) -> list[dict[str, Any]]:
  """Builds shared multimodal `input` blocks for `client.decisions.create`.

  Args:
    obs: `SceneObservation` containing robot state and camera frames.
    cameras: Canonical camera names to attach.
    resolution: Decisions API resolution tier (`"low"`, `"medium"`, `"high"`,
      or `"ultra_high"`).
    include_contact_sheet: Whether to append `contact_sheet_2x2` when all 4
      canonical cameras are present.
    jpeg_quality: JPEG compression quality for attached images.

  Returns:
    List of block dicts starting with a `"text"` state summary block followed
    by base64-encoded `"image"` blocks.

  Raises:
    ValueError: If `cameras` is empty/unknown.
    KeyError: If any requested camera is missing from `obs.frames`.
  """
  selected = _collect_selected_frames(
      obs, cameras, include_contact_sheet=include_contact_sheet
  )
  blocks: list[dict[str, Any]] = [
      {
          "type": "text",
          "text": (
              f"{format_robot_state_summary(obs)}\n"
              f"Attached camera views: {[name for name, _ in selected]}"
          ),
      }
  ]
  for _, frame in selected:
    jpeg_bytes = encode_frame_jpeg_bytes(frame, quality=jpeg_quality)
    blocks.append({
        "type": "image",
        "data": base64.b64encode(jpeg_bytes).decode("utf-8"),
        "mime_type": "image/jpeg",
        "resolution": resolution,
    })
  return blocks


def _load_dotenv_api_key() -> str:
  """Reads `GEMINI_API_KEY` from `.env` in CWD or skill root if present."""
  for env_path in (Path.cwd() / ".env", _SCRIPTS_DIR.parent / ".env"):
    if not env_path.is_file():
      continue
    try:
      lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
      continue
    for raw_line in lines:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      if line.startswith("export "):
        line = line[len("export ") :].strip()
      if "=" not in line:
        continue
      key, val = line.split("=", 1)
      if key.strip() != "GEMINI_API_KEY":
        continue
      cleaned = val.strip().strip("'\"")
      if cleaned:
        return cleaned
  return ""


def _get_genai_client(
    *,
    client: Any = None,
    api_key: str | None = None,
    api_version: str | None = None,
) -> tuple[Any | None, str | None]:
  """Returns an initialized `genai.Client` or `(None, fallback_reason)`."""
  if client is not None:
    return client, None
  if genai is None:
    return None, "google-genai is not installed"
  resolved_key = (
      (api_key or "").strip()
      or os.environ.get("GEMINI_API_KEY", "").strip()
      or _load_dotenv_api_key()
  )
  if not resolved_key:
    return None, "GEMINI_API_KEY is not set"
  kwargs: dict[str, Any] = {"api_key": resolved_key}
  if api_version is not None:
    kwargs["http_options"] = {"api_version": api_version}
  return genai.Client(**kwargs), None


def _serialize_decision_answer(answer: Any) -> dict[str, Any]:
  """Converts a Gemini Decisions answer object into a plain dict."""
  converted = _to_json_safe(answer)
  if isinstance(converted, Mapping):
    return dict(converted)
  if hasattr(answer, "__dict__"):
    return {
        k: _to_json_safe(v)
        for k, v in vars(answer).items()
        if v is not None
    }
  return {"value": converted}


def _save_and_attach_artifacts(
    result: dict[str, Any],
    obs: fr3_duo_robot.SceneObservation,
    *,
    output_dir: str | Path | None,
    request_payload: Mapping[str, Any],
    request_filename: str,
    prefix: str,
) -> None:
  """Saves VQA/decision bundle to disk and attaches paths to `result`."""
  saved = save_vqa_bundle(
      obs,
      output_dir=output_dir,
      request_payload=request_payload,
      request_filename=request_filename,
      prefix=prefix,
  )
  result["output_dir"] = saved["output_dir"]
  result["saved_paths"] = saved


def _fallback_response(
    obs: fr3_duo_robot.SceneObservation,
    *,
    reason: str | None,
    model: str,
    cameras: Sequence[str],
    instructions: str,
    extra: Mapping[str, Any],
    output_dir: str | Path | None,
    request_payload: Mapping[str, Any],
    request_filename: str,
    prefix: str,
) -> dict[str, Any]:
  """Builds and saves the local-VLM fallback result payload."""
  fallback: dict[str, Any] = {
      "status": "fallback_saved",
      "backend": "local_vlm_fallback",
      "reason": reason,
      "model": model,
      "cameras": list(cameras),
      **extra,
      "instructions": instructions,
  }
  _save_and_attach_artifacts(
      fallback,
      obs,
      output_dir=output_dir,
      request_payload=request_payload,
      request_filename=request_filename,
      prefix=prefix,
  )
  return fallback


def ask_scene_vqa(
    obs: fr3_duo_robot.SceneObservation,
    question: str,
    *,
    cameras: Sequence[str] = ("head_camera_left",),
    include_contact_sheet: bool = False,
    model: str = DEFAULT_VQA_MODEL,
    config: Any = None,
    client: Any = None,
    api_key: str | None = None,
    output_dir: str | Path | None = None,
    save_artifacts: bool = False,
    jpeg_quality: int = 85,
) -> dict[str, Any]:
  """Answers an open-ended visual question, or saves artifacts for VLM fallback.

  Args:
    obs: `SceneObservation` containing robot state and camera frames.
    question: Non-empty natural-language question about the scene.
    cameras: Canonical camera names to attach.
    include_contact_sheet: Whether to include `contact_sheet_2x2`.
    model: Gemini VQA model name.
    config: Optional `google-genai` config (`GenerateContentConfig` or dict)
      forwarded directly to `client.models.generate_content(...)`.
    client: Optional pre-configured `google-genai` client (useful for tests).
    api_key: Optional API key override (defaults to `GEMINI_API_KEY`).
    output_dir: Directory for saved PNGs/JSON; implies `save_artifacts=True`.
    save_artifacts: Always write observation PNGs and `vqa_result.json`.
    jpeg_quality: JPEG compression quality for API image payloads.

  Returns:
    Dictionary with either:
    - `status="ok"`, `backend="gemini_vqa"`, `model`, `question`, `cameras`,
      `answer` (plus `output_dir`/`saved_paths` if artifacts were saved).
    - `status="fallback_saved"`, `backend="local_vlm_fallback"`, `reason`,
      `model`, `question`, `cameras`, `output_dir`, `saved_paths`,
      `instructions` when `GEMINI_API_KEY` is unset or `google-genai` is
      unavailable.

  Raises:
    ValueError: If `question` is empty or `cameras` is empty/unknown.
    KeyError: If any requested camera is missing from `obs.frames`.
  """
  validated_cams = _validate_cameras(cameras)
  clean_question = question.strip()
  if not clean_question:
    raise ValueError("question must be a non-empty string.")
  active_client, fallback_reason = _get_genai_client(
      client=client, api_key=api_key
  )
  request_payload = {
      "mode": "vqa",
      "model": model,
      "cameras": list(validated_cams),
      "include_contact_sheet": bool(include_contact_sheet),
      "question": clean_question,
      "state_summary": format_robot_state_summary(obs),
  }
  if active_client is None:
    return _fallback_response(
        obs,
        reason=fallback_reason,
        model=model,
        cameras=validated_cams,
        extra={"question": clean_question},
        instructions=(
            "Inspect contact_sheet_2x2.png and camera PNGs via read_file to "
            "answer the question directly."
        ),
        output_dir=output_dir,
        request_payload=request_payload,
        request_filename="vqa_request.json",
        prefix="vqa",
    )

  contents = build_vqa_contents(
      obs,
      clean_question,
      cameras=validated_cams,
      include_contact_sheet=include_contact_sheet,
      jpeg_quality=jpeg_quality,
  )
  gen_kwargs: dict[str, Any] = {
      "model": model,
      "contents": contents,
  }
  if config is not None:
    gen_kwargs["config"] = config
  response = active_client.models.generate_content(**gen_kwargs)
  answer_text = str(getattr(response, "text", "") or "").strip()
  result: dict[str, Any] = {
      "status": "ok",
      "backend": "gemini_vqa",
      "model": model,
      "question": clean_question,
      "cameras": list(validated_cams),
      "answer": answer_text,
  }
  if save_artifacts or output_dir is not None:
    _save_and_attach_artifacts(
        result,
        obs,
        output_dir=output_dir,
        request_payload={**request_payload, "answer": answer_text},
        request_filename="vqa_result.json",
        prefix="vqa",
    )
  return result


def decide_scene(
    obs: fr3_duo_robot.SceneObservation,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    cameras: Sequence[str] = ("head_camera_left",),
    resolution: str = "medium",
    include_contact_sheet: bool = False,
    model: str = DEFAULT_DECISION_MODEL,
    client: Any = None,
    api_key: str | None = None,
    output_dir: str | Path | None = None,
    save_artifacts: bool = False,
    jpeg_quality: int = 85,
) -> dict[str, Any]:
  """Evaluates typed Decisions API questions, or saves artifacts for fallback.

  Args:
    obs: `SceneObservation` containing robot state and camera frames.
    questions: Non-empty `questions` mapping forwarded directly to
      `client.decisions.create(...)` (e.g.
      `{"ok": {"type": "noul", "instructions": "Is the target centered?"}}`).
      See https://aistudio.google.com/docs/decisions-api for the full schema.
    cameras: Canonical camera names to attach.
    resolution: Decisions API resolution tier (`"low"`, `"medium"`, `"high"`,
      or `"ultra_high"`).
    include_contact_sheet: Whether to include `contact_sheet_2x2`.
    model: Decisions model name (default `"gemini-decision-1"`).
    client: Optional pre-configured `google-genai` client (useful for tests).
    api_key: Optional API key override (defaults to `GEMINI_API_KEY`).
    output_dir: Directory for saved PNGs/JSON; implies `save_artifacts=True`.
    save_artifacts: Always write observation PNGs and `decision_result.json`.
    jpeg_quality: JPEG compression quality for API image payloads.

  Returns:
    Dictionary with either:
    - `status="ok"`, `backend="gemini_decisions"`, `model`, `cameras`,
      `resolution`, `answers` (per-question serialized fields such as `noul`,
      `choice`, `score`, `confidence`, `probabilities`, `legend`), plus
      `output_dir`/`saved_paths` if artifacts were saved.
    - `status="fallback_saved"`, `backend="local_vlm_fallback"`, `reason`,
      `model`, `cameras`, `resolution`, `questions`, `output_dir`,
      `saved_paths`, `instructions` when `GEMINI_API_KEY` is unset or
      `google-genai` is unavailable.

  Raises:
    ValueError: If `questions` is empty/non-mapping or `cameras` is
      empty/unknown.
    KeyError: If any requested camera is missing from `obs.frames`.
  """
  validated_cams = _validate_cameras(cameras)
  if not isinstance(questions, Mapping) or not questions:
    raise ValueError("questions must be a non-empty mapping.")

  active_client, fallback_reason = _get_genai_client(
      client=client, api_key=api_key, api_version="v1beta"
  )
  request_payload = {
      "mode": "decide",
      "model": model,
      "cameras": list(validated_cams),
      "resolution": resolution,
      "include_contact_sheet": bool(include_contact_sheet),
      "questions": _to_json_safe(questions),
      "state_summary": format_robot_state_summary(obs),
  }
  if active_client is None:
    return _fallback_response(
        obs,
        reason=fallback_reason,
        model=model,
        cameras=validated_cams,
        extra={
            "resolution": resolution,
            "questions": _to_json_safe(questions),
        },
        instructions=(
            "Inspect contact_sheet_2x2.png and camera PNGs via read_file to "
            "evaluate the typed questions in decision_request.json."
        ),
        output_dir=output_dir,
        request_payload=request_payload,
        request_filename="decision_request.json",
        prefix="decide",
    )

  input_blocks = build_decision_input(
      obs,
      cameras=validated_cams,
      resolution=resolution,
      include_contact_sheet=include_contact_sheet,
      jpeg_quality=jpeg_quality,
  )
  response = active_client.decisions.create(
      model=model,
      input=input_blocks,
      questions=dict(questions),
  )
  resp_json = _to_json_safe(response)
  if isinstance(resp_json, Mapping):
    raw_answers = resp_json.get("answers")
    prompt_feedback = resp_json.get("prompt_feedback")
  else:
    raw_answers = getattr(response, "answers", None)
    raw_pf = getattr(response, "prompt_feedback", None)
    prompt_feedback = _to_json_safe(raw_pf) if raw_pf is not None else None
    if isinstance(raw_answers, Mapping):
      raw_answers = {
          str(qid): _serialize_decision_answer(ans)
          for qid, ans in raw_answers.items()
      }
  answers = dict(raw_answers) if isinstance(raw_answers, Mapping) else {}
  status = "blocked" if (not answers and prompt_feedback) else "ok"
  result: dict[str, Any] = {
      "status": status,
      "backend": "gemini_decisions",
      "model": model,
      "cameras": list(validated_cams),
      "resolution": resolution,
      "answers": answers,
  }
  if prompt_feedback:
    result["prompt_feedback"] = prompt_feedback
  if save_artifacts or output_dir is not None:
    _save_and_attach_artifacts(
        result,
        obs,
        output_dir=output_dir,
        request_payload={**request_payload, "answers": answers},
        request_filename="decision_result.json",
        prefix="decide",
    )
  return result


def _arm_pose_from_dict(raw: Mapping[str, Any]) -> fr3_duo_robot.ArmPoseState:
  return fr3_duo_robot.ArmPoseState(
      q_rad=[float(v) for v in raw.get("q_rad", ())],
      dq_rad=[float(v) for v in raw.get("dq_rad", ())],
      tau_j=[float(v) for v in raw.get("tau_j", ())],
      tau_ext_base_6d=[float(v) for v in raw.get("tau_ext_base_6d", ())],
      tcp_pos_base=[float(v) for v in raw.get("tcp_pos_base", ())],
      tcp_rot_base=[list(r) for r in raw.get("tcp_rot_base", ())],
      o_t_ee_4x4=[list(r) for r in raw.get("o_t_ee_4x4", ())],
      fk_pos_error_m=float(raw.get("fk_pos_error_m", 0.0)),
      gripper_ratio=float(raw.get("gripper_ratio", 0.0)),
      gripper_status=dict(raw.get("gripper_status", {})),
      current_errors=[str(e) for e in raw.get("current_errors", ())],
      desk_audit=dict(raw.get("desk_audit", {})),
  )


def load_saved_observation(
    observation_dir: str | Path,
) -> fr3_duo_robot.SceneObservation:
  """Loads an `observation.json` + saved camera PNGs as a `SceneObservation`.

  Args:
    observation_dir: Directory containing `observation.json` and camera PNGs.

  Returns:
    Reconstructed `SceneObservation` with `frames: dict[str, np.ndarray]` as
    uint8 RGB arrays.

  Raises:
    FileNotFoundError: If `observation.json` is missing in `observation_dir`.
    RuntimeError: If OpenCV is unavailable when reading camera PNGs.
  """
  obs_dir = Path(observation_dir)
  obs_json = obs_dir / "observation.json"
  if not obs_json.is_file():
    raise FileNotFoundError(f"Missing observation.json in {obs_dir}")
  data = json.loads(obs_json.read_text(encoding="utf-8"))
  cv2 = require_cv2()
  frames: dict[str, np.ndarray] = {}
  for cam_name in CANONICAL_CAMERAS:
    cam_path = obs_dir / f"{cam_name}.png"
    if not cam_path.is_file():
      continue
    bgr = cv2.imread(str(cam_path), cv2.IMREAD_COLOR)
    if bgr is not None:
      frames[cam_name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

  raw_state = data.get("state", {})
  timestamp_s = float(data.get("timestamp_s", 0.0))
  state = fr3_duo_robot.RobotState(
      timestamp_s=float(raw_state.get("timestamp_s", timestamp_s)),
      left=_arm_pose_from_dict(raw_state.get("left", {})),
      right=_arm_pose_from_dict(raw_state.get("right", {})),
      state_16d=[float(v) for v in raw_state.get("state_16d", ())],
  )
  camera_meta = {
      cam: fr3_duo_robot.CameraMeta(
          backend=str(meta.get("backend", "saved")),
          width=int(meta.get("width", 0)),
          height=int(meta.get("height", 0)),
          timestamp_s=meta.get("timestamp_s"),
          age_s=meta.get("age_s"),
          rectified=bool(meta.get("rectified", False)),
      )
      for cam, meta in data.get("camera_meta", {}).items()
  }
  raw_bounds = data.get("table_box_bounds", {})
  bounds = fr3_duo_robot.TableBoxBounds(
      center=[float(v) for v in raw_bounds.get("center", ())],
      half_extents=[float(v) for v in raw_bounds.get("half_extents", ())],
  )
  return fr3_duo_robot.SceneObservation(
      timestamp_s=timestamp_s,
      mode=str(data.get("mode", "unknown")),
      dry_run=bool(data.get("dry_run", False)),
      state=state,
      state_16d=[float(v) for v in data.get("state_16d", ())],
      frames=frames,
      camera_meta=camera_meta,
      calibration=dict(data.get("calibration", {})),
      z_table_m=float(data.get("z_table_m", 0.0)),
      table_box_bounds=bounds,
      saved_paths=dict(data.get("saved_paths", {})),
  )


def _build_arg_parser() -> argparse.ArgumentParser:
  """Builds the CLI argument parser for `fr3_vqa.py`."""
  parser = argparse.ArgumentParser(
      description="Run FR3 Duo VQA or Gemini Decisions API over scene cameras."
  )
  parser.add_argument(
      "--mode",
      choices=("vqa", "decide"),
      required=True,
      help="Whether to run open-ended VQA or closed-set Gemini Decisions.",
  )
  parser.add_argument(
      "--question",
      default="",
      help="Question text when --mode=vqa.",
  )
  parser.add_argument(
      "--questions-json",
      default="",
      help="JSON object mapping question IDs to specs when --mode=decide.",
  )
  parser.add_argument(
      "--cameras",
      nargs="+",
      default=["head_camera_left"],
      choices=CANONICAL_CAMERAS,
      help="Canonical camera stream(s) to include.",
  )
  parser.add_argument(
      "--resolution",
      default="medium",
      help="Decisions API image resolution tier.",
  )
  parser.add_argument(
      "--include-contact-sheet",
      action="store_true",
      help="Also attach the 2x2 contact sheet.",
  )
  parser.add_argument(
      "--model",
      default="",
      help="Optional model override.",
  )
  parser.add_argument(
      "--robot-config",
      default="",
      help="Path to config/robot_config.json (defaults to sim config).",
  )
  parser.add_argument(
      "--hardware",
      action="store_true",
      help="Observe live hardware cameras instead of sim.",
  )
  parser.add_argument(
      "--observation-dir",
      default="",
      help="Reuse an existing observation directory instead of capturing live.",
  )
  parser.add_argument(
      "--output-dir",
      default="",
      help="Directory where observation PNGs and request/result JSON go.",
  )
  parser.add_argument(
      "--save-artifacts",
      action="store_true",
      help="Always write observation PNGs and JSON even when API succeeds.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point for FR3 Duo VQA and Decisions API queries."""
  parser = _build_arg_parser()
  args = parser.parse_args(list(argv) if argv is not None else None)
  if args.mode == "vqa" and not args.question.strip():
    parser.error("--question is required when --mode=vqa.")
  questions_dict: dict[str, Any] = {}
  if args.mode == "decide":
    if not args.questions_json.strip():
      parser.error("--questions-json is required when --mode=decide.")
    try:
      parsed_q = json.loads(args.questions_json)
    except json.JSONDecodeError as exc:
      parser.error(f"Invalid --questions-json: {exc}")
    if not isinstance(parsed_q, dict) or not parsed_q:
      parser.error("--questions-json must decode to a non-empty JSON object.")
    questions_dict = parsed_q

  if args.observation_dir:
    obs = load_saved_observation(args.observation_dir)
  else:
    cfg = (
        load_robot_config(args.robot_config)
        if args.robot_config
        else default_robot_config()
    )
    robot_mode = "hardware" if args.hardware else "sim"
    with fr3_duo_robot.Fr3DuoRobot(
        cfg, mode=robot_mode, dry_run=False, read_only=args.hardware
    ) as robot:
      obs = robot.observe_scene()

  out_dir = Path(args.output_dir) if args.output_dir else None
  if args.mode == "vqa":
    result = ask_scene_vqa(
        obs,
        args.question,
        cameras=args.cameras,
        include_contact_sheet=args.include_contact_sheet,
        model=args.model or DEFAULT_VQA_MODEL,
        output_dir=out_dir,
        save_artifacts=args.save_artifacts,
    )
  else:
    result = decide_scene(
        obs,
        questions_dict,
        cameras=args.cameras,
        resolution=args.resolution,
        include_contact_sheet=args.include_contact_sheet,
        model=args.model or DEFAULT_DECISION_MODEL,
        output_dir=out_dir,
        save_artifacts=args.save_artifacts,
    )
  print(json.dumps(_to_json_safe(result), indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
