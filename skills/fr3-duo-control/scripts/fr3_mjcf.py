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

"""Franka Research 3 Duo MJCF builder and canonical model constants.

Composes a dual-arm FR3 + Robotiq 2F-85 scene in the canonical world/base frame
(+X forward across the tabletop, +Y robot-left, +Z up) from
`mujoco_menagerie` (`franka_fr3_duo/fr3_duo_with_spine.xml` +
`robotiq_2f85/2f85.xml`), normalized to `{left,right}/fr3_joint1..7`,
`{left,right}/robotiq_gripper_{left,right}/{pinch_site,fingers_actuator}`,
and `head_camera_{left,right}` / `{side}/wrist_camera/wrist_camera_{side}`
names.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping
import copy
import dataclasses
import fcntl
import functools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

try:
  import mujoco  # pytype: disable=import-error
except ImportError:
  mujoco: Any = None

FR3_JOINT_LIMITS_RAD = np.array(
    [
        [-2.7437, 2.7437],
        [-1.7837, 1.7837],
        [-2.9007, 2.9007],
        [-3.0421, -0.1518],
        [-2.8065, 2.8065],
        [0.5445, 4.5169],
        [-3.0159, 3.0159],
    ],
    dtype=np.float64,
)

FR3_IK_MARGINS_RAD = (0.025, 0.025, 0.025, 0.100, 0.025, 0.025, 0.025)

WRIST_CAMERA_POS_LINK7 = (0.0639045, 0.0, 0.173407)
WRIST_CAMERA_QUAT_LINK7_WXYZ = (0.122788, 0.696364, 0.696364, 0.122788)

SPINE_TABLETOP_POS_XYZ = (-0.244844, 0.0, -0.75)
BRACKET_UNDERSIDE_FRONT_X_M = 0.08437
HEAD_CAMERA_POS_LEFT_HEAD_LINK = (0.069440, 0.0, 0.230128)
HEAD_CAMERA_POS_RIGHT_HEAD_LINK = (0.069440, -0.063, 0.230128)
HEAD_CAMERA_QUAT_HEAD_LINK_WXYZ = (0.643441, 0.293230, -0.293230, -0.643441)

DEFAULT_REST_POSE_RAD: dict[str, list[float]] = {
    "left": [0.90, -0.30, -0.4330, -2.1180, 0.8727, 1.9509, -0.7647],
    "right": [-0.90, -0.30, 0.4330, -2.1180, -0.8727, 1.9509, 0.7647],
}

CANONICAL_TO_MJCF_CAMERA_NAME: dict[str, str] = {
    "head_camera_left": "head_camera_left",
    "head_camera_right": "head_camera_right",
    "wrist_camera_left": "left/wrist_camera/wrist_camera_left",
    "wrist_camera_right": "right/wrist_camera/wrist_camera_right",
}

_FLANGE_QUAT = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
_CAMERA_RESOLUTION_WH = (1280, 720)
_HEAD_CAMERA_SENSOR_SIZE = (0.00512, 0.00288)
_HEAD_CAMERA_FOCAL_PIXEL = (710.0, 710.0)
_HEAD_CAMERA_PRINCIPAL_PIXEL = (0.0, 0.0)
_WRIST_CAMERA_SENSOR_SIZE = (0.00193, 0.00193)
_WRIST_CAMERA_FOCAL_PIXEL = (635.0, 635.0)
_WRIST_CAMERA_PRINCIPAL_PIXEL = (0.0, 0.0)

_MENAGERIE_REPO_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
_MENAGERIE_SPARSE_PATHS = ("franka_fr3_duo", "robotiq_2f85")


def gripper_prefix(arm: str) -> str:
  """Returns canonical Robotiq 2F-85 prefix for `arm` (`'left'`/`'right'`)."""
  return f"{arm}/robotiq_gripper_{arm}/"


def gripper_pinch_site_name(arm: str) -> str:
  """Returns canonical gripper TCP `pinch_site` name for `arm`."""
  return f"{gripper_prefix(arm)}pinch_site"


def gripper_actuator_name(arm: str) -> str:
  """Returns canonical gripper actuator name for `arm`."""
  return f"{gripper_prefix(arm)}fingers_actuator"


def canonical_camera_name(mjcf_cam_name: str) -> str:
  """Maps an MJCF camera name (e.g. wrist camera path) to stream key."""
  for canonical, mjcf_name in CANONICAL_TO_MJCF_CAMERA_NAME.items():
    if mjcf_cam_name == mjcf_name:
      return canonical
  return mjcf_cam_name


def resolve_mjcf_camera_name(mj_model: Any, canonical_name: str) -> str:
  """Resolves a canonical camera stream key to the MJCF camera name."""
  mj = _require_mujoco()
  candidates = [canonical_name]
  mapped = CANONICAL_TO_MJCF_CAMERA_NAME.get(canonical_name)
  if mapped and mapped not in candidates:
    candidates.insert(0, mapped)
  for cand in candidates:
    if mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_CAMERA, cand) >= 0:
      return cand
  return canonical_name


def _lexical_abspath(path: str | Path) -> Path:
  """Normalizes `path` to an absolute lexical `Path` without symlink resolve."""
  return Path(os.path.abspath(os.path.expanduser(str(path))))


def _is_valid_menagerie_dir(candidate: Path) -> bool:
  return (
      candidate.is_dir()
      and (candidate / "franka_fr3_duo" / "fr3_duo_with_spine.xml").is_file()
      and (candidate / "robotiq_2f85" / "2f85.xml").is_file()
  )


def _find_runfiles_menagerie_dir() -> Path | None:
  """Checks `RUNFILES_DIR` / `TEST_SRCDIR` for `mujoco_menagerie`."""
  candidates: list[Path] = []
  for env_var in ("RUNFILES_DIR", "TEST_SRCDIR"):
    root_str = os.environ.get(env_var, "").strip()
    if not root_str:
      continue
    root = _lexical_abspath(root_str)
    if not root.is_dir():
      continue
    candidates.append(root / "third_party" / "mujoco_menagerie")
    candidates.append(root / "mujoco_menagerie")
    try:
      for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.is_dir():
          candidates.append(child / "third_party" / "mujoco_menagerie")
          candidates.append(child / "mujoco_menagerie")
    except OSError:
      # Unreadable runfiles child directory; continue probing remaining roots.
      pass

  seen: set[str] = set()
  for cand in candidates:
    norm_cand = _lexical_abspath(cand)
    key = str(norm_cand)
    if key in seen:
      continue
    seen.add(key)
    if _is_valid_menagerie_dir(norm_cand):
      return norm_cand
  return None


def _menagerie_cache_dir() -> Path:
  """Returns `$XDG_CACHE_HOME/mujoco_menagerie` or `~/.cache/...`."""
  xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
  base = (
      _lexical_abspath(xdg_cache)
      if xdg_cache
      else _lexical_abspath(Path.home() / ".cache")
  )
  return base / "mujoco_menagerie"


def _sparse_checkout_menagerie(target_dir: Path) -> Path:
  """Populates `target_dir` via sparse git checkout with an `fcntl` lock."""
  target_dir.parent.mkdir(parents=True, exist_ok=True)
  lock_path = target_dir.parent / f"{target_dir.name}.lock"
  with open(lock_path, "w", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    if _is_valid_menagerie_dir(target_dir):
      return target_dir
    target = str(target_dir)
    sparse_cfg = target_dir / ".git" / "info" / "sparse-checkout"
    try:
      fresh_clone = not sparse_cfg.exists()
      if fresh_clone:
        if target_dir.exists():
          shutil.rmtree(target_dir)
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "--filter=blob:none",
                _MENAGERIE_REPO_URL,
                target,
            ],
            check=True,
        )
      subprocess.run(
          [
              "git",
              "-C",
              target,
              "sparse-checkout",
              "set",
              *_MENAGERIE_SPARSE_PATHS,
          ],
          check=True,
      )
      checkout_cmd = (
          ["git", "-C", target, "checkout"]
          if fresh_clone
          else ["git", "-C", target, "checkout", "--", *_MENAGERIE_SPARSE_PATHS]
      )
      subprocess.run(checkout_cmd, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
      raise FileNotFoundError(
          f"Failed to sparse-clone mujoco_menagerie into {target_dir}: {exc}"
      ) from exc

    if not _is_valid_menagerie_dir(target_dir):
      raise FileNotFoundError(
          "Sparse-cloned mujoco_menagerie missing required assets in "
          f"{target_dir}"
      )
    return target_dir


def _resolve_menagerie_dir(menagerie_dir: str | Path | None = None) -> Path:
  """Resolves `mujoco_menagerie` or raises `FileNotFoundError`."""
  if menagerie_dir is not None:
    raw = str(menagerie_dir).strip()
    cand = _lexical_abspath(raw) if raw else None
    if cand is None or not _is_valid_menagerie_dir(cand):
      raise FileNotFoundError(
          "mujoco_menagerie `franka_fr3_duo` assets not found "
          f"(menagerie_dir={menagerie_dir!r})."
      )
    return cand

  runfiles_dir = _find_runfiles_menagerie_dir()
  if runfiles_dir is not None:
    return runfiles_dir

  cache_dir = _menagerie_cache_dir()
  if _is_valid_menagerie_dir(cache_dir):
    return cache_dir
  return _sparse_checkout_menagerie(cache_dir)


def find_menagerie_dir(
    explicit_path: str | Path | None = None,
) -> Path | None:
  """Resolves `mujoco_menagerie` with `franka_fr3_duo` & `robotiq_2f85`."""
  try:
    return _resolve_menagerie_dir(explicit_path)
  except (OSError, subprocess.SubprocessError):
    return None


def _coerce_config_mapping(config: Any) -> dict[str, Any]:
  if config is None:
    return {}
  if dataclasses.is_dataclass(config) and not isinstance(config, type):
    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
  if isinstance(config, Mapping):
    return dict(config)
  raise TypeError(f"Unsupported config type: {type(config)!r}")


def _absolutize_spec_assets(spec: Any, mjcf_file: Path) -> None:
  """Resolves `spec.meshes` / `spec.textures` relative paths to absolute."""
  base_dir = _lexical_abspath(mjcf_file).parent
  meshdir = str(getattr(spec.compiler, "meshdir", "") or "")
  texturedir = str(getattr(spec.compiler, "texturedir", "") or "")
  mesh_base = _lexical_abspath(base_dir / meshdir) if meshdir else base_dir
  tex_base = _lexical_abspath(base_dir / texturedir) if texturedir else base_dir
  for mesh in spec.meshes:
    if mesh.file and not Path(mesh.file).is_absolute():
      mesh.file = str(_lexical_abspath(mesh_base / mesh.file))
  for tex in spec.textures:
    if tex.file and not Path(tex.file).is_absolute():
      tex.file = str(_lexical_abspath(tex_base / tex.file))
  spec.compiler.meshdir = ""
  spec.compiler.texturedir = ""


def _name_unnamed_geoms(spec: Any) -> Any:
  """Ensures all geoms have names so they receive prefixes when attached."""
  for body in spec.bodies:
    for i, geom in enumerate(body.geoms):
      if not geom.name:
        geom.name = f"{body.name}_geom_{i}"
  return spec


def _strip_keys(spec: Any) -> Any:
  """Removes all keyframes from `spec` before composition/attachment."""
  while spec.keys:
    spec.delete(spec.keys[0])
  return spec


def _rename_arm_prefix(name: str) -> str:
  """Converts `{left,right}_fr3v2_*` names to `{left,right}/fr3_*`."""
  for side in ("left", "right"):
    prefix = f"{side}_fr3v2_"
    if name.startswith(prefix):
      return f"{side}/fr3_{name[len(prefix):]}"
  return name


def _rename_arm_elements(spec: Any) -> None:
  """Applies `{left,right}/fr3_*` naming across bodies, joints, exclusions."""
  for items in (spec.bodies, spec.joints, spec.geoms):
    for item in items:
      item.name = _rename_arm_prefix(item.name)
  for act in spec.actuators:
    act.name = _rename_arm_prefix(act.name)
    act.target = _rename_arm_prefix(act.target)
  for exc in spec.excludes:
    exc.bodyname1 = _rename_arm_prefix(exc.bodyname1)
    exc.bodyname2 = _rename_arm_prefix(exc.bodyname2)


def _add_cameras(spec: Any) -> None:
  """Adds calibrated ZED Mini stereo head and D405 wrist cameras to `spec`."""
  head_link = spec.body("head_link")
  if head_link is None:
    raise ValueError("Body not found: head_link")
  for cam_name, cam_pos in (
      ("head_camera_left", HEAD_CAMERA_POS_LEFT_HEAD_LINK),
      ("head_camera_right", HEAD_CAMERA_POS_RIGHT_HEAD_LINK),
  ):
    if spec.camera(cam_name) is None:
      head_link.add_camera(
          name=CANONICAL_TO_MJCF_CAMERA_NAME[cam_name],
          pos=list(cam_pos),
          quat=list(HEAD_CAMERA_QUAT_HEAD_LINK_WXYZ),
          sensor_size=list(_HEAD_CAMERA_SENSOR_SIZE),
          focal_pixel=list(_HEAD_CAMERA_FOCAL_PIXEL),
          principal_pixel=list(_HEAD_CAMERA_PRINCIPAL_PIXEL),
          resolution=list(_CAMERA_RESOLUTION_WH),
      )

  for side in ("left", "right"):
    cam_name = CANONICAL_TO_MJCF_CAMERA_NAME[f"wrist_camera_{side}"]
    if spec.camera(cam_name) is None:
      link7_name = f"{side}/fr3_link7"
      link7 = spec.body(link7_name)
      if link7 is None:
        raise ValueError(f"Body not found: {link7_name}")
      link7.add_camera(
          name=cam_name,
          pos=list(WRIST_CAMERA_POS_LINK7),
          quat=list(WRIST_CAMERA_QUAT_LINK7_WXYZ),
          sensor_size=list(_WRIST_CAMERA_SENSOR_SIZE),
          focal_pixel=list(_WRIST_CAMERA_FOCAL_PIXEL),
          principal_pixel=list(_WRIST_CAMERA_PRINCIPAL_PIXEL),
          resolution=list(_CAMERA_RESOLUTION_WH),
      )


def _cfg_float(
    config: Mapping[str, Any], key: str, default: float
) -> float:
  raw = config.get(key)
  return float(raw) if raw is not None else default


def _resolve_table_geometry(
    config: Mapping[str, Any],
) -> tuple[float, float, float, float]:
  """Extracts `(z_table_m, table_front_offset_m, width_m, depth_m)`."""
  if (
      config.get("base_height_m") is not None
      or config.get("table_height_m") is not None
  ):
    base_h = _cfg_float(config, "base_height_m", 0.913887)
    table_h = _cfg_float(config, "table_height_m", 0.75)
    z_table_m = round(0.163887 - (base_h - table_h), 6)
  else:
    z_table_m = _cfg_float(config, "z_table_m", 0.0)
  return (
      z_table_m,
      _cfg_float(config, "table_front_offset_m", 0.25),
      _cfg_float(config, "table_width_m", 1.22),
      _cfg_float(config, "table_depth_m", 0.74),
  )


def _add_arena_props(spec: Any, config: Mapping[str, Any]) -> None:
  """Adds canonical overhead light, tabletop collider, and workspace marker."""
  mj = _require_mujoco()
  z_table_m, front_offset, width, depth = _resolve_table_geometry(config)
  table_thickness = 0.04
  x_front = BRACKET_UNDERSIDE_FRONT_X_M + front_offset
  table_center_x = round(x_front + 0.5 * depth, 6)
  table_center_z = round(z_table_m - 0.5 * table_thickness, 6)
  table_half_extents = [
      round(0.5 * depth, 6),
      round(0.5 * width, 6),
      round(0.5 * table_thickness, 6),
  ]
  if spec.light("overhead_light") is None:
    spec.worldbody.add_light(
        name="overhead_light",
        pos=[0.45, 0.0, 1.6],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.85, 0.85, 0.88],
    )
  if spec.geom("table") is None:
    spec.worldbody.add_geom(
        name="table",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=table_half_extents,
        pos=[table_center_x, 0.0, table_center_z],
        rgba=[0.82, 0.79, 0.73, 1.0],
        contype=1,
        conaffinity=1,
    )
  if spec.geom("workspace_center_marker") is None:
    spec.worldbody.add_geom(
        name="workspace_center_marker",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        size=[0.035, 0.002, 0.0],
        pos=[table_center_x, 0.0, float(z_table_m) + 0.002],
        rgba=[0.16, 0.48, 0.78, 1.0],
        group=2,
        contype=0,
        conaffinity=0,
    )


def _require_mujoco() -> Any:
  if mujoco is None:
    raise RuntimeError(
        "`mujoco` is required for FR3 Duo kinematics and simulation. "
        "Install it with `pip install mujoco`."
    )
  return mujoco


_MJCF_BUILD_CACHE_LOCK = threading.Lock()
_MJCF_BUILD_CACHE: collections.OrderedDict[tuple[Any, ...], tuple[Any, str]] = (
    collections.OrderedDict()
)
_MJCF_BUILD_CACHE_MAX = 16


@functools.lru_cache(maxsize=None)
def _probe_sphere_contact(pos_x: float, margin: float, gap: float) -> bool:
  mj = _require_mujoco()
  geom = (
      f'<geom type="sphere" size="0.01" margin="{margin}" gap="{gap}" '
      'contype="1" conaffinity="1"/>'
  )
  probe_xml = (
      '<mujoco model="margin_probe"><worldbody>'
      f'<body name="b1" pos="0 0 0">{geom}</body>'
      f'<body name="b2" pos="{pos_x} 0 0"><freejoint/>{geom}</body>'
      "</worldbody></mujoco>"
  )
  m = mj.MjModel.from_xml_string(probe_xml)
  d = mj.MjData(m)
  mj.mj_kinematics(m, d)
  mj.mj_collision(m, d)
  return int(d.ncon) > 0


def uses_additive_margin_gap() -> bool:
  """Returns True on MuJoCo >= 3.9 where detection uses `margin + gap`."""
  return _probe_sphere_contact(0.03, 0.0, 0.02)


def resolve_min_inter_arm_clearance(config: Any) -> float:
  """Extracts non-negative `min_inter_arm_clearance_m` from `config`."""
  cfg = _coerce_config_mapping(config)
  return max(0.0, _cfg_float(cfg, "min_inter_arm_clearance_m", 0.030))


def uses_pairwise_additive_margin() -> bool:
  """Returns True on MuJoCo >= 3.5 where pairwise margin/gap is summed."""
  return _probe_sphere_contact(0.045, 0.015, 0.0)


def _apply_inter_arm_clearance_margins(
    spec: Any, min_clearance_m: float
) -> None:
  """Applies clearance `gap` (and legacy `margin`) to arm geoms."""
  scale = 0.5 if uses_pairwise_additive_margin() else 1.0
  clearance = max(0.0, scale * float(min_clearance_m))
  if clearance <= 0.0:
    return
  additive_mode = uses_additive_margin_gap()
  for geom in spec.geoms:
    if int(geom.contype) == 0 and int(geom.conaffinity) == 0:
      continue
    parent = geom.parent
    bname = str(parent.name) if parent is not None else ""
    if not bname.startswith(("left/", "right/")):
      continue
    geom.gap = max(float(geom.gap), clearance)
    if not additive_mode:
      geom.margin = max(float(geom.margin), clearance)


def _mjcf_cache_payload(config: Mapping[str, Any]) -> dict[str, Any]:
  z_table_m, front_offset, width, depth = _resolve_table_geometry(config)
  return {
      "z_table_m": z_table_m,
      "table_front_offset_m": front_offset,
      "table_width_m": width,
      "table_depth_m": depth,
      "min_inter_arm_clearance_m": resolve_min_inter_arm_clearance(config),
      "tcp_offset_z_m": _cfg_float(config, "tcp_offset_z_m", 0.145),
  }


def _strip_unnamed_default_wrappers(xml_str: str) -> str:
  """Removes or flattens unnamed `<default>` wrappers from serialized XML."""
  root = ET.fromstring(xml_str)
  any_changed = False

  def _clean(parent: ET.Element, in_default: bool) -> None:
    nonlocal any_changed
    local_changed = False
    new_children: list[ET.Element] = []
    for child in list(parent):
      if child.tag == "default":
        _clean(child, in_default=True)
        cls_name = child.attrib.get("class", "").strip()
        if not cls_name:
          children_of_child = list(child)
          if not children_of_child:
            local_changed = True
            continue
          if in_default:
            new_children.extend(children_of_child)
            local_changed = True
            continue
      new_children.append(child)
    if local_changed:
      parent[:] = new_children
      any_changed = True

  _clean(root, in_default=False)
  return ET.tostring(root, encoding="unicode") if any_changed else xml_str


def _assemble_duo_spec(
    cfg: Mapping[str, Any],
    menagerie_root: Path,
    *,
    include_arena: bool,
    add_gravcomp: bool,
) -> Any:
  """Assembles the single-level Menagerie FR3 Duo + Robotiq 2F-85 `MjSpec`."""
  mj = _require_mujoco()
  duo_xml = menagerie_root / "franka_fr3_duo" / "fr3_duo_with_spine.xml"
  rq_xml = menagerie_root / "robotiq_2f85" / "2f85.xml"
  if not duo_xml.is_file():
    raise FileNotFoundError(
        f"Menagerie FR3 Duo spine XML not found: {duo_xml}"
    )
  if not rq_xml.is_file():
    raise FileNotFoundError(f"Menagerie Robotiq 2F-85 XML not found: {rq_xml}")

  spec = mj.MjSpec.from_file(str(duo_xml))
  _absolutize_spec_assets(spec, duo_xml)
  spec = _strip_keys(_name_unnamed_geoms(spec))
  _rename_arm_elements(spec)
  spec.option.integrator = mj.mjtIntegrator.mjINT_RK4
  spec.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10

  gripper_spec = mj.MjSpec.from_file(str(rq_xml))
  _absolutize_spec_assets(gripper_spec, rq_xml)
  gripper_spec = _strip_keys(_name_unnamed_geoms(gripper_spec))
  pinch = gripper_spec.site("pinch")
  if pinch is None:
    raise ValueError("Gripper site not found: pinch")
  pinch.name = "pinch_site"
  pinch.pos = [0.0, 0.0, _cfg_float(cfg, "tcp_offset_z_m", 0.145)]

  for side in ("left", "right"):
    prefix = gripper_prefix(side)
    site = spec.site(f"{side}_attachment_site")
    if site is None:
      raise ValueError(f"Attachment site not found: {side}_attachment_site")
    site.quat = list(_FLANGE_QUAT)
    spec.attach(gripper_spec.copy(), prefix=prefix, site=site)

    link6 = f"{side}/fr3_link6"
    link7 = f"{side}/fr3_link7"
    spec.add_exclude(bodyname1=link7, bodyname2=f"{prefix}base_mount")
    spec.add_exclude(bodyname1=link7, bodyname2=f"{prefix}base")
    spec.add_exclude(bodyname1=link6, bodyname2=f"{prefix}base_mount")

    for j_idx in range(7):
      jnt = spec.joint(f"{side}/fr3_joint{j_idx + 1}")
      if jnt is not None:
        lo, hi = FR3_JOINT_LIMITS_RAD[j_idx]
        jnt.range = [
            max(float(jnt.range[0]), float(lo)),
            min(float(jnt.range[1]), float(hi)),
        ]

  spec.add_exclude(bodyname1="left/fr3_link0", bodyname2="right/fr3_link0")
  _add_cameras(spec)
  spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), 1280)
  spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), 720)

  spine_body = spec.body("franka_spine")
  if spine_body is not None:
    spine_body.pos = list(SPINE_TABLETOP_POS_XYZ)

  if include_arena:
    _add_arena_props(spec, cfg)

  if add_gravcomp:
    for body in spec.bodies:
      if str(body.name).startswith(("left/", "right/")):
        body.gravcomp = 1.0

  _apply_inter_arm_clearance_margins(
      spec, resolve_min_inter_arm_clearance(cfg)
  )
  return spec


def build_fr3_duo_spec(
    config: Any = None,
    *,
    menagerie_dir: str | Path | None = None,
    include_arena: bool = True,
    add_gravcomp: bool = False,
) -> Any:
  """Builds the canonical dual-arm FR3 + Robotiq 2F-85 `MjSpec`."""
  _require_mujoco()
  cfg = _coerce_config_mapping(config)
  resolved_menagerie = _resolve_menagerie_dir(menagerie_dir)
  return _assemble_duo_spec(
      cfg,
      resolved_menagerie,
      include_arena=bool(include_arena),
      add_gravcomp=bool(add_gravcomp),
  )


def build_fr3_duo_mjcf(
    config: Any = None,
    *,
    menagerie_dir: str | Path | None = None,
    include_arena: bool = True,
    add_gravcomp: bool = False,
) -> tuple[Any, str]:
  """Builds and compiles the dual-arm FR3 + Robotiq 2F-85 `(MjModel, xml)`."""
  _require_mujoco()
  cfg = _coerce_config_mapping(config)
  resolved_menagerie = _resolve_menagerie_dir(menagerie_dir)
  arena_flag = bool(include_arena)
  gravcomp_flag = bool(add_gravcomp)
  payload_json = json.dumps(_mjcf_cache_payload(cfg), sort_keys=True)
  add_gap = uses_additive_margin_gap()
  cache_key = (
      str(resolved_menagerie),
      add_gap,
      arena_flag,
      gravcomp_flag,
      payload_json,
  )

  with _MJCF_BUILD_CACHE_LOCK:
    cached = _MJCF_BUILD_CACHE.get(cache_key)
    if cached is not None:
      _MJCF_BUILD_CACHE.move_to_end(cache_key)
      cached_model, cached_xml = cached
      return copy.copy(cached_model), cached_xml

  spec = _assemble_duo_spec(
      cfg,
      resolved_menagerie,
      include_arena=arena_flag,
      add_gravcomp=gravcomp_flag,
  )
  xml_str = _strip_unnamed_default_wrappers(spec.to_xml())
  mj_model = spec.compile()
  cached_copy = copy.copy(mj_model)
  with _MJCF_BUILD_CACHE_LOCK:
    _MJCF_BUILD_CACHE[cache_key] = (cached_copy, xml_str)
    _MJCF_BUILD_CACHE.move_to_end(cache_key)
    while len(_MJCF_BUILD_CACHE) > _MJCF_BUILD_CACHE_MAX:
      _MJCF_BUILD_CACHE.popitem(last=False)
  return mj_model, xml_str


def resolve_ee_site_id(mj_model: Any, arm: str) -> int:
  """Resolves the canonical gripper `pinch_site` ID for `arm` in `mj_model`."""
  mj = _require_mujoco()
  sname = gripper_pinch_site_name(arm)
  sid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SITE, sname)
  if sid >= 0:
    return int(sid)
  raise KeyError(f"End-effector site {sname!r} not found in MjModel.")


def arm_joint_qpos_and_dof_addrs(
    mj_model: Any, arm: str
) -> tuple[list[int], list[int]]:
  """Returns `(qpos_addrs, dof_addrs)` for `{arm}/fr3_joint1..7`."""
  mj = _require_mujoco()
  if arm not in ("left", "right"):
    raise ValueError(f"Invalid arm={arm!r}; expected 'left' or 'right'")
  qpos_addrs: list[int] = []
  dof_addrs: list[int] = []
  for idx in range(7):
    jname = f"{arm}/fr3_joint{idx + 1}"
    jid = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
      raise KeyError(f"Joint {jname!r} not found in MjModel.")
    qpos_addrs.append(int(mj_model.jnt_qposadr[jid]))
    dof_addrs.append(int(mj_model.jnt_dofadr[jid]))
  return qpos_addrs, dof_addrs
