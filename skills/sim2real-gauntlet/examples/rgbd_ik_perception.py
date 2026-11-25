# Copyright 2026 DeepMind Technologies Limited
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
"""Unprivileged RGB-D and RGB-only perception + damped-least-squares IK utilities.

Operates strictly on camera arrays (`uint8 [H, W, 3]` RGB and optional
`float32 [H, W]` metric depth), joint positions (`q_robot`), and a robot-only
nominal `MjModel`/`MjData`. Contains zero references to any external evaluation
harness.
"""

from collections.abc import Sequence
import dataclasses
import math
import os
import pathlib
import re
import zipfile

import mujoco
from mujoco_playground import asset as mp_asset
from mujoco_playground.robots.so101.sim import scene as so101_scene
from mujoco_playground.robots.yam.sim import scene as yam_scene
import numpy as np

OVERHEAD_CAM = "overhead_cam"
WRIST_CAM = "wrist_camera"
CONTROL_DT_S = 0.02
_SO101_XML_URI = (
    "github:google-deepmind/mujoco_menagerie/robotstudio_so101/so101.xml"
)


def _populate_assets(spec: mujoco.MjSpec, xml_uri: str) -> None:
  """Loads robot meshes into `spec.assets` using only public `asset.get`."""
  for name in ("table/tabletop", "table/tablelegs"):
    for item in (spec.geom(name), spec.mesh(name)):
      if item is not None:
        spec.delete(item)
  for item in (
      spec.material("table/table"),
      spec.texture("table/small_meta_table_diffuse"),
  ):
    if item is not None:
      spec.delete(item)
  table = spec.geom("table/table_collision")
  table.group, table.rgba = 2, [0.50, 0.42, 0.34, 1.0]
  raw_path = str(mp_asset.get(xml_uri).path)
  rel = raw_path.removeprefix("runfile:")
  ws = os.environ.get("TEST_WORKSPACE", "")
  stripped = rel[rel.find("/") + 1:] if "/" in rel else rel
  candidates = [pathlib.Path(rel), pathlib.Path(stripped)]
  for env_key in ("RUNFILES_DIR", "TEST_SRCDIR"):
    root = os.environ.get(env_key, "")
    if root:
      candidates.append(pathlib.Path(root) / rel)
      if ws:
        candidates.append(pathlib.Path(root) / ws / stripped)
  base = next(
      (p.parent for p in candidates if p.exists()), candidates[1].parent
  )
  assets = dict(spec.assets)
  missing: list[str] = []
  for mesh in spec.meshes:
    if not mesh.file or mesh.file in assets:
      continue
    found = False
    for cand in (base / "assets" / mesh.file, base / mesh.file):
      if cand.is_file():
        assets[mesh.file] = cand.read_bytes()
        found = True
        break
    if not found:
      missing.append(mesh.file)
  if missing:
    for part in pathlib.Path(__file__).absolute().parents:
      if part.is_file() and zipfile.is_zipfile(part):
        rel_parent = pathlib.Path(rel).parent
        stripped_parent = pathlib.Path(stripped).parent
        with zipfile.ZipFile(part) as zf:
          names = set(zf.namelist())
          for mfile in missing:
            for zcand in (
                f"{rel_parent}/assets/{mfile}",
                f"{rel_parent}/{mfile}",
                f"{stripped_parent}/assets/{mfile}",
                f"{stripped_parent}/{mfile}",
            ):
              if zcand in names:
                assets[mfile] = zf.read(zcand)
                break
        break
  spec.assets = assets


def nominal_table_z(robot: str) -> float:
  """Returns the calibrated ALOHA tabletop height for `robot`."""
  return float(
      yam_scene.TABLE_HEIGHT if robot == "yam" else so101_scene.TABLE_HEIGHT
  )


def nominal_robot_spec(robot: str) -> mujoco.MjSpec:
  """The calibrated rig a controller may know: robot, table, and cameras."""
  tz = nominal_table_z(robot)
  if robot == "yam":
    spec = yam_scene.yam_aloha_table()
    _populate_assets(spec, yam_scene.YAM_XML_URI)
    cam_pos = [0.0, 0.04, tz + 0.55]
  else:
    spec = so101_scene.so101_aloha_table()
    _populate_assets(spec, _SO101_XML_URI)
    cam_pos = [0.0, -0.06, tz + 0.50]
  spec.option.timestep = 0.002
  spec.worldbody.add_camera(
      name=OVERHEAD_CAM, pos=cam_pos, xyaxes=[1, 0, 0, 0, 1, 0], fovy=55.0
  )
  return spec


DOWNWARD_ROT = np.eye(3, dtype=np.float64)
TCP_SITE_CANDIDATES = (
    "robot/grasp_site",
    "grasp_site",
    "robot/pinch_site",
    "pinch_site",
    "robot/gripperframe",
    "gripperframe",
    "tcp",
    "ee_site",
)
GRIPPER_TOKENS = ("finger", "gripper", "jaw")

# Declared object-size priors (meters): `(prior_obj_h, grasp_below_top_m,
# prior_support_dz)`. Nominal XY coordinates are never stored here.
TASK_PRIORS: dict[str, tuple[float, float, float]] = {
    "cube_in_bowl": (0.040, 0.016, 0.008),
    "peg_in_plate": (0.080, 0.018, 0.008),
    "two_block_stack": (0.040, 0.016, 0.040),
}

TABLE_ROI_BOX: tuple[float, float, float, float, float, float] = (
    -0.36,
    0.62,
    -0.34,
    0.34,
    -0.05,
    0.90,
)
WORKSPACE_BOX: tuple[float, float, float, float, float, float] = (
    -0.34,
    0.58,
    -0.26,
    0.32,
    0.003,
    0.120,
)
PEG_MIN_DZ_M: float = 0.055
BLOCK_MIN_DZ_M: float = 0.014
SOCKET_DZ_BAND_M: tuple[float, float] = (0.014, 0.042)
WRIST_DZ_BANDS_M: dict[str, tuple[float, float]] = {
    "peg": (0.058, 0.115),
    "target": (0.020, 0.115),
    "bowl": (0.002, 0.058),
    "block_base": (0.018, 0.058),
    "socket": (0.014, 0.042),
}

_SUPPORTED_COLORS: dict[str, str] = {
    "red": "red",
    "orange": "orange",
    "green": "green",
    "blue": "blue",
    "gray": "gray",
    "grey": "gray",
    "slate": "gray",
}
_NEUTRAL_MODIFIERS: frozenset[str] = frozenset({
    "small",
    "large",
    "tall",
    "short",
    "wooden",
    "plastic",
    "target",
    "top",
    "bottom",
    "base",
    "cylindrical",
    "square",
    "round",
    "flat",
    "fragile",
    "distractor",
})
_STOP_WORDS: frozenset[str] = frozenset({
    "the",
    "a",
    "an",
    "this",
    "that",
    "into",
    "in",
    "inside",
    "on",
    "onto",
    "of",
    "to",
    "and",
    "or",
    "with",
    "from",
    "place",
    "put",
    "insert",
    "stack",
    "move",
    "pick",
    "drop",
    "sort",
    "gently",
    "cleanly",
    "upright",
    "carefully",
})
_TARGET_NOUNS: frozenset[str] = frozenset({"cube", "peg", "block", "cup"})
_DEST_NOUNS: frozenset[str] = frozenset({
    "bowl",
    "socket",
    "base",
    "plate",
    "coaster",
})
_ALL_NOUNS: frozenset[str] = _TARGET_NOUNS | _DEST_NOUNS
_PROHIBITED_CLAUSE_WORDS: frozenset[str] = frozenset({
    "untouched",
    "ignore",
    "avoid",
    "without",
    "not",
})


@dataclasses.dataclass(frozen=True)
class TaskPlan:
  """Prompt-parsed color words paired with declared `TASK_PRIORS` sizes."""

  family: str
  target_color: str
  dest_kind: str
  dest_color: str
  prior_obj_h: float
  grasp_below_top_m: float
  prior_support_dz: float


def _extract_noun_color(
    tokens: Sequence[str], idx: int, *, allow_uncolored_gray: bool = False
) -> str:
  """Extracts a validated noun-adjacent color before `tokens[idx]`."""
  adjs: list[str] = []
  j = idx - 1
  while j >= max(0, idx - 3):
    tok = tokens[j]
    if tok in _STOP_WORDS or (
        tok in _ALL_NOUNS and tok not in _NEUTRAL_MODIFIERS
    ):
      break
    adjs.append(tok)
    j -= 1
  adjs.reverse()
  cols: list[str] = []
  for adj in adjs:
    if adj in _SUPPORTED_COLORS:
      norm = _SUPPORTED_COLORS[adj]
      if norm not in cols:
        cols.append(norm)
    elif adj not in _NEUTRAL_MODIFIERS:
      return "unsupported"
  if not cols:
    return "gray" if allow_uncolored_gray else "unsupported"
  return cols[0] if len(cols) == 1 else "unsupported"


def _is_dest_block_phrase(tokens: Sequence[str], idx: int) -> bool:
  """Returns True when `block` at `idx` refers to a support/base block."""
  window = tokens[max(0, idx - 4) : idx]
  if "bottom" in window or "base" in window:
    return True
  return any(prep in window for prep in ("on", "onto", "of"))


def parse_task_prompt(prompt: str) -> TaskPlan:
  """Parses noun-adjacent `<color> <noun>` phrases and fails safe on unknown."""
  clauses = [
      re.findall(r"[a-z0-9]+", cl)
      for cl in re.split(r"[;.]", prompt.lower())
  ]
  active = [
      cl for cl in clauses if cl and not (_PROHIBITED_CLAUSE_WORDS & set(cl))
  ]
  tokens: list[str] = []
  for cl in active or clauses:
    tokens.extend(cl)
  t_nouns: list[tuple[str, str]] = []
  d_nouns: list[tuple[str, str]] = []
  for idx, tok in enumerate(tokens):
    if (
        tok == "base"
        and idx + 1 < len(tokens)
        and tokens[idx + 1] in _TARGET_NOUNS
    ):
      continue
    if tok == "block" and _is_dest_block_phrase(tokens, idx):
      d_nouns.append(("base", _extract_noun_color(tokens, idx)))
    elif tok in _TARGET_NOUNS:
      t_nouns.append((tok, _extract_noun_color(tokens, idx)))
    elif tok in _DEST_NOUNS:
      allow_gray = tok in ("socket", "plate")
      d_nouns.append((
          tok,
          _extract_noun_color(tokens, idx, allow_uncolored_gray=allow_gray),
      ))

  t_cols = {c for _, c in t_nouns}
  t_col = next(iter(t_cols)) if len(t_cols) == 1 else "unsupported"
  d_cols = {c for _, c in d_nouns}
  d_col = next(iter(d_cols)) if len(d_cols) == 1 else "unsupported"
  if t_col == "unsupported" or d_col == "unsupported":
    p = TASK_PRIORS["cube_in_bowl"]
    return TaskPlan("unsupported", t_col, "unsupported", d_col, *p)

  t_kinds = {n for n, _ in t_nouns}
  d_kinds = {n for n, _ in d_nouns}
  if "peg" in t_kinds or "socket" in d_kinds or "plate" in d_kinds:
    p = TASK_PRIORS["peg_in_plate"]
    return TaskPlan("peg_in_plate", t_col, "socket", d_col, *p)
  if "stack" in tokens or "base" in d_kinds:
    p = TASK_PRIORS["two_block_stack"]
    return TaskPlan("two_block_stack", t_col, "block_base", d_col, *p)
  p = TASK_PRIORS["cube_in_bowl"]
  return TaskPlan("cube_in_bowl", t_col, "bowl", d_col, *p)


def _find_id(model: mujoco.MjModel, obj_type: int, names: Sequence[str]) -> int:
  for name in names:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx >= 0:
      return int(idx)
  return -1


def _resolve_pinch_ids(model: mujoco.MjModel) -> tuple[int, int, int, str]:
  """Resolves `(left_geom_id, right_geom_id, body_id, kind)` for YAM/SO101."""
  m_geom = int(mujoco.mjtObj.mjOBJ_GEOM)
  m_body = int(mujoco.mjtObj.mjOBJ_BODY)
  lid = _find_id(model, m_geom, ("robot/left_pad", "left_pad"))
  rid = _find_id(model, m_geom, ("robot/right_pad", "right_pad"))
  if lid >= 0 and rid >= 0:
    return lid, rid, -1, "yam"
  lid = _find_id(model, m_geom, ("robot/fixed_jaw_box5", "fixed_jaw_box5"))
  rid = _find_id(model, m_geom, ("robot/moving_jaw_box3", "moving_jaw_box3"))
  bid = _find_id(model, m_body, ("robot/gripper", "gripper"))
  if lid >= 0 and rid >= 0 and bid >= 0:
    return lid, rid, bid, "so101"
  return -1, -1, -1, "site"


def _set_nominal_pinch_fingers(
    model: mujoco.MjModel, data: mujoco.MjData, kind: str
) -> None:
  m_jnt = int(mujoco.mjtObj.mjOBJ_JOINT)
  if kind == "yam":
    for jn in (
        "robot/left_finger",
        "left_finger",
        "robot/right_finger",
        "right_finger",
    ):
      jid = mujoco.mj_name2id(model, m_jnt, jn)
      if jid >= 0:
        data.qpos[int(model.jnt_qposadr[jid])] = 0.020
  elif kind == "so101":
    for jn in ("robot/gripper", "gripper"):
      jid = mujoco.mj_name2id(model, m_jnt, jn)
      if jid >= 0:
        data.qpos[int(model.jnt_qposadr[jid])] = 0.42


@dataclasses.dataclass(frozen=True)
class SpecKinematics:
  """Nominal robot model arm/gripper index split and wrist-camera XY offset."""

  site_id: int
  arm_act_ids: tuple[int, ...]
  arm_jnt_ids: tuple[int, ...]
  arm_qpos_ids: tuple[int, ...]
  arm_dof_ids: tuple[int, ...]
  grip_act_ids: tuple[int, ...]
  all_qpos_adrs: tuple[int, ...]
  wrist_xy_off: np.ndarray


def parse_spec_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_robot: Sequence[float] | np.ndarray,
) -> SpecKinematics:
  """Extracts arm/gripper indices and wrist-camera offset from nominal model."""
  site_id = _find_id(model, int(mujoco.mjtObj.mjOBJ_SITE), TCP_SITE_CANDIDATES)
  if site_id < 0:
    raise ValueError(f"No TCP site in model matching {TCP_SITE_CANDIDATES}.")
  arm_act, arm_jnt, arm_qpos, arm_dof, grip_act = [], [], [], [], []
  for aid in range(int(model.nu)):
    aname = (model.actuator(aid).name or "").lower()
    jid = int(model.actuator_trnid[aid, 0])
    jn = (model.joint(jid).name or "").lower() if 0 <= jid < model.njnt else ""
    if any(tok in aname or tok in jn for tok in GRIPPER_TOKENS):
      grip_act.append(aid)
    elif 0 <= jid < int(model.njnt):
      arm_act.append(aid)
      arm_jnt.append(jid)
      arm_qpos.append(int(model.jnt_qposadr[jid]))
      arm_dof.append(int(model.jnt_dofadr[jid]))
  if not arm_act or not grip_act:
    raise ValueError("Could not split arm vs gripper actuators from model.")
  all_qpos = tuple(
      int(model.jnt_qposadr[j])
      for j in range(int(model.njnt))
      if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)
  )
  wcid = _find_id(
      model, int(mujoco.mjtObj.mjOBJ_CAMERA), ("wrist_cam", "robot/wrist_cam")
  )
  if wcid < 0:
    wrist_off = np.zeros(2, dtype=np.float64)
  else:
    q_seed = np.asarray(q_robot[: len(arm_act)], dtype=np.float64)
    base_bid = int(model.jnt_bodyid[arm_jnt[0]])
    data.qpos[arm_qpos] = q_seed
    mujoco.mj_kinematics(model, data)
    base_pos = np.asarray(data.xpos[base_bid], dtype=np.float64)
    ref_xyz = (
        (0.0, 0.0, float(base_pos[2]) + 0.16)
        if float(base_pos[1]) < -0.10
        else (0.28, 0.0, float(base_pos[2]) + 0.16)
    )
    q_ref = solve_downward_ik(
        model, data, site_id, arm_qpos, arm_dof, arm_jnt, ref_xyz, q_seed
    )
    tcp_ref = nominal_tcp_xpos(model, data, site_id, arm_qpos, q_ref)
    mujoco.mj_camlight(model, data)
    wrist_off = np.asarray(data.cam_xpos[wcid][:2] - tcp_ref[:2], float)
  return SpecKinematics(
      site_id,
      tuple(arm_act),
      tuple(arm_jnt),
      tuple(arm_qpos),
      tuple(arm_dof),
      tuple(grip_act),
      all_qpos,
      wrist_off,
  )


def solve_downward_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    arm_qpos_ids: Sequence[int],
    arm_dof_ids: Sequence[int],
    arm_jnt_ids: Sequence[int],
    target_xyz: Sequence[float] | np.ndarray,
    init_q: Sequence[float] | np.ndarray,
    *,
    max_iters: int = 110,
    damping: float = 0.035,
    lock_world_yaw: bool = False,
) -> np.ndarray:
  """Solves damped least-squares IK on nominal model with downward TCP."""
  n_arm = len(arm_qpos_ids)
  q = np.asarray(init_q[:n_arm], dtype=np.float64).copy()
  lo = np.asarray([model.jnt_range[j, 0] for j in arm_jnt_ids]) + 0.055
  hi = np.asarray([model.jnt_range[j, 1] for j in arm_jnt_ids]) - 0.055
  goal = np.asarray(target_xyz, dtype=np.float64)
  lid, rid, bid, kind = _resolve_pinch_ids(model)
  _set_nominal_pinch_fingers(model, data, kind)
  jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
  jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
  jacp_r = np.zeros((3, int(model.nv)), dtype=np.float64)
  reg = (damping * damping) * np.eye(6)
  base_bid = int(model.jnt_bodyid[arm_jnt_ids[0]])
  data.qpos[list(arm_qpos_ids)] = np.clip(q, lo, hi)
  mujoco.mj_kinematics(model, data)
  base_xy = np.asarray(data.xpos[base_bid][:2], dtype=np.float64)
  rad_xy = goal[:2] - base_xy
  rad_norm = float(np.linalg.norm(rad_xy))
  u_rad = (
      np.array([rad_xy[0] / rad_norm, rad_xy[1] / rad_norm, 0.0])
      if rad_norm > 1e-4
      else np.array([0.0, 1.0, 0.0])
  )
  u_perp = np.array([-u_rad[1], u_rad[0], 0.0], dtype=np.float64)
  v_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  if kind == "so101" and n_arm == 5 and lock_world_yaw:
    pan_g = math.atan2(-float(rad_xy[0]), max(1e-4, float(rad_xy[1])))
    if abs(float(q[0]) - pan_g) > 0.45:
      q = np.array([pan_g, -0.26, 0.45, 1.05, -pan_g], dtype=np.float64)

  for _ in range(max_iters):
    data.qpos[list(arm_qpos_ids)] = np.clip(q, lo, hi)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    if kind in ("yam", "so101"):
      p_cur = 0.5 * (
          np.asarray(data.geom_xpos[lid], float)
          + np.asarray(data.geom_xpos[rid], float)
      )
      pos_err = goal - p_cur
      if kind == "yam":
        s_mat = np.asarray(data.site_xmat[site_id], float).reshape(3, 3)
        v_app, v_jaw = s_mat[:, 2], s_mat[:, 1]
        t_jaw = u_perp if float(np.dot(v_jaw, u_perp)) >= 0.0 else -u_perp
        r_err = 0.65 * np.cross(v_app, v_down) + 0.45 * np.cross(v_jaw, t_jaw)
        mujoco.mj_jacSite(model, data, jacp_r, jacr, site_id)
      else:
        b_mat = np.asarray(data.xmat[bid], float).reshape(3, 3)
        v_app, v_jaw = -b_mat[:, 2], b_mat[:, 0]
        if lock_world_yaw:
          t_jaw = np.array([1.0, 0.0, 0.0], dtype=np.float64)
          r_err = 0.75 * np.cross(v_app, v_down) + 0.35 * np.cross(v_jaw, t_jaw)
        else:
          t_jaw = u_perp if float(np.dot(v_jaw, u_perp)) >= 0.0 else -u_perp
          r_err = 0.65 * np.cross(v_app, v_down) + 0.45 * np.cross(v_jaw, t_jaw)
        mujoco.mj_jacBody(model, data, jacp_r, jacr, bid)
      mujoco.mj_jacGeom(model, data, jacp, None, lid)
      mujoco.mj_jacGeom(model, data, jacp_r, None, rid)
      jacp[:] = 0.5 * (jacp + jacp_r)
    else:
      pos_err = goal - np.asarray(data.site_xpos[site_id], dtype=np.float64)
      rot_cur = np.asarray(data.site_xmat[site_id], float).reshape(3, 3)
      r_err = 0.5 * sum(
          np.cross(rot_cur[:, k], DOWNWARD_ROT[:, k]) for k in range(3)
      )
      mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    if float(np.linalg.norm(pos_err)) < 3e-4 and (
        n_arm < 5 or float(np.linalg.norm(r_err)) < 8e-3
    ):
      break
    err6 = np.concatenate((pos_err, r_err))
    jac_arm = np.vstack((jacp[:, arm_dof_ids], jacr[:, arm_dof_ids]))
    dq = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + reg, err6)
    blocked = ((q >= hi - 1e-6) & (dq > 0.0)) | ((q <= lo + 1e-6) & (dq < 0.0))
    if np.any(blocked):
      jac_free = jac_arm.copy()
      jac_free[:, blocked] = 0.0
      dq = jac_free.T @ np.linalg.solve(jac_free @ jac_free.T + reg, err6)
    q = np.clip(q + np.clip(dq, -0.20, 0.20), lo, hi)
  return np.clip(q, lo, hi)


def nominal_tcp_xpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    arm_qpos_ids: Sequence[int],
    q_arm: Sequence[float] | np.ndarray,
) -> np.ndarray:
  """Evaluates nominal forward kinematics TCP position `(x, y, z)`."""
  lid, rid, _, kind = _resolve_pinch_ids(model)
  _set_nominal_pinch_fingers(model, data, kind)
  data.qpos[list(arm_qpos_ids)] = np.asarray(q_arm[: len(arm_qpos_ids)], float)
  mujoco.mj_kinematics(model, data)
  if lid >= 0 and rid >= 0:
    return 0.5 * (
        np.asarray(data.geom_xpos[lid], dtype=np.float64)
        + np.asarray(data.geom_xpos[rid], dtype=np.float64)
    )
  return np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()


def tcp_near_zones(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    arm_qpos_ids: Sequence[int],
    q_arm: Sequence[float] | np.ndarray,
    zones: Sequence[tuple[float, float, float]],
) -> bool:
  """Returns True when nominal-FK TCP is within 11 cm XY & +7.5 cm Z of zone."""
  if not zones:
    return False
  tcp = nominal_tcp_xpos(model, data, site_id, arm_qpos_ids, q_arm)
  return any(
      math.hypot(float(tcp[0]) - gx, float(tcp[1]) - gy) < 0.11
      and float(tcp[2]) < gz + 0.075
      for gx, gy, gz in zones
  )


def build_guard_zones(
    t_xy: np.ndarray,
    d_xy: np.ndarray,
    obj_h: float,
    sup_dz: float,
    plane: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
  """Builds `(x, y, z_top)` proximity guard zones for target & destination."""
  zt = plane_z(plane, float(t_xy[0]), float(t_xy[1])) + obj_h
  zd = plane_z(plane, float(d_xy[0]), float(d_xy[1])) + sup_dz
  return [
      (float(t_xy[0]), float(t_xy[1]), zt),
      (float(d_xy[0]), float(d_xy[1]), zd),
  ]


def color_mask(colors: np.ndarray, kind: str) -> np.ndarray:
  """Chromaticity mask tolerant to HSV/exposure shifts while rejecting yellow."""
  norm = _SUPPORTED_COLORS.get(kind, kind)
  r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
  if norm == "red":
    return (r - b > 32.0) & (r - g > 26.0) & (g < 0.70 * r)
  if norm == "orange":
    return (
        (r - b > 32.0)
        & (r - g > 26.0)
        & (b < 0.58 * r)
        & ((r - g) > 0.42 * np.maximum(0.0, g - b))
    )
  if norm == "green":
    return (g - np.maximum(r, b)) > 14.0
  if norm == "blue":
    return (b - r > 16.0) & (b > g - 10.0)
  if norm == "gray":
    span = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return (
        (r > 48.0)
        & (span < 58.0)
        & (b >= r - 12.0)
        & (g <= np.maximum(r, b) + 12.0)
    )
  return np.zeros(len(colors), dtype=bool)


def backproject_rgbd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_adrs: Sequence[int],
    q_robot: Sequence[float] | np.ndarray,
    cam_name: str,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    *,
    nominal_table_z_m: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
  """Back-projects `(rgb, depth)` (or RGB-only ray-plane) into `(pts [N,3], colors [N,3])`."""
  n_cp = min(len(qpos_adrs), len(q_robot))
  data.qpos[list(qpos_adrs[:n_cp])] = np.asarray(q_robot[:n_cp], float)
  mujoco.mj_kinematics(model, data)
  mujoco.mj_camlight(model, data)
  cid = _find_id(
      model,
      int(mujoco.mjtObj.mjOBJ_CAMERA),
      (
          cam_name,
          f"robot/{cam_name}",
          cam_name.rsplit("/", 1)[-1],
          "robot/wrist_cam" if "wrist" in cam_name.lower() else cam_name,
      ),
  )
  if cid < 0:
    raise KeyError(f"Camera {cam_name!r} absent from nominal model.")

  if depth is not None:
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
      ry = np.minimum((np.arange(h) * rgb.shape[0]) // h, rgb.shape[0] - 1)
      rx = np.minimum((np.arange(w) * rgb.shape[1]) // w, rgb.shape[1] - 1)
      rgb = rgb[ry[:, None], rx[None, :]]
  else:
    h, w = rgb.shape[:2]

  aspect = float(w) / float(max(1, h))
  vcam = mujoco.MjvCamera()
  vcam.type = int(mujoco.mjtCamera.mjCAMERA_FIXED)
  vcam.fixedcamid = cid
  gl_cam = mujoco.mjv_camera2GLCamera(model, data, vcam)
  cam_pos = np.asarray(gl_cam.pos, dtype=np.float64)
  cam_fwd = np.asarray(gl_cam.forward, dtype=np.float64)
  cam_fwd = cam_fwd / max(float(np.linalg.norm(cam_fwd)), 1e-9)
  cam_up = np.asarray(gl_cam.up, dtype=np.float64)
  cam_up = cam_up / max(float(np.linalg.norm(cam_up)), 1e-9)
  cam_right = np.cross(cam_fwd, cam_up)
  cam_right = cam_right / max(float(np.linalg.norm(cam_right)), 1e-9)
  f_span = float(gl_cam.frustum_top - gl_cam.frustum_bottom)
  half_w = float(gl_cam.frustum_width) or 0.5 * aspect * f_span
  f_left = float(gl_cam.frustum_center) - half_w
  f_right = float(gl_cam.frustum_center) + half_w
  f_bot = float(gl_cam.frustum_bottom)
  f_top = float(gl_cam.frustum_top)
  f_near = max(float(gl_cam.frustum_near), 1e-6)

  if depth is not None:
    rows, cols = np.nonzero(
        (depth > 0.05) & (depth < 1.35) & np.isfinite(depth)
    )
    z = depth[rows, cols].astype(np.float64)
  else:
    grid_r, grid_c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rows = grid_r.reshape(-1)
    cols = grid_c.reshape(-1)
    flat_c = rgb.reshape(-1, 3).astype(np.float64)
    z_plane = np.full(len(rows), float(nominal_table_z_m), dtype=np.float64)
    for col_name, dz_est in (
        ("red", 0.032),
        ("orange", 0.032),
        ("blue", 0.026),
        ("gray", 0.024),
        ("green", 0.064),
    ):
      z_plane[color_mask(flat_c, col_name)] = float(nominal_table_z_m + dz_est)
    relx_all = (cols.astype(np.float64) + 0.5) / float(w)
    rely_all = 1.0 - (rows.astype(np.float64) + 0.5) / float(h)
    u_x = (f_left + relx_all * (f_right - f_left)) / f_near
    u_y = (f_bot + rely_all * (f_top - f_bot)) / f_near
    ray_z = cam_fwd[2] + u_x * cam_right[2] + u_y * cam_up[2]
    valid_ray = ray_z < -0.10
    rows = rows[valid_ray]
    cols = cols[valid_ray]
    z = (z_plane[valid_ray] - float(cam_pos[2])) / ray_z[valid_ray]
    keep_z = (z > 0.05) & (z < 1.35)
    rows = rows[keep_z]
    cols = cols[keep_z]
    z = z[keep_z]

  relx = (cols.astype(np.float64) + 0.5) / float(w)
  rely = 1.0 - (rows.astype(np.float64) + 0.5) / float(h)
  xc = (f_left + relx * (f_right - f_left)) / f_near * z
  yc = (f_bot + rely * (f_top - f_bot)) / f_near * z
  pts = (
      cam_pos
      + z[:, None] * cam_fwd
      + xc[:, None] * cam_right
      + yc[:, None] * cam_up
  )
  cols_out = rgb[rows, cols].astype(np.float64)
  if "wrist" not in cam_name.lower() and len(pts) > 0:
    r_bids = {
        b
        for b in range(1, int(model.nbody))
        if "table" not in (model.body(b).name or "").lower()
    }
    keep = np.ones(len(pts), dtype=bool)
    for g in range(int(model.ngeom)):
      if int(model.geom_bodyid[g]) in r_bids:
        g_p = np.asarray(data.geom_xpos[g], dtype=np.float64)
        g_r = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        ab = np.asarray(model.geom_aabb[g], dtype=np.float64)
        loc = np.abs((pts - g_p) @ g_r - ab[:3])
        keep &= ~np.all(loc <= ab[3:] + 0.022, axis=1)
    pts, cols_out = pts[keep], cols_out[keep]
  return pts, cols_out


def fit_table_plane(
    pts: np.ndarray,
    default_plane: tuple[float, float, float] = (0.0, 0.0, 0.75),
) -> tuple[float, float, float]:
  """Fits local tabletop plane `Z = a*X + b*Y + c` via trimmed least squares."""
  x0, x1, y0, y1, z0, z1 = TABLE_ROI_BOX
  in_box = (
      (pts[:, 0] >= x0)
      & (pts[:, 0] <= x1)
      & (pts[:, 1] >= y0)
      & (pts[:, 1] <= y1)
      & (pts[:, 2] >= z0)
      & (pts[:, 2] <= z1)
  )
  sub = pts[in_box]
  if len(sub) < 40:
    return default_plane
  z_mode = float(np.percentile(sub[:, 2], 35.0))
  sub = sub[np.abs(sub[:, 2] - z_mode) <= 0.08]
  if len(sub) < 40:
    return default_plane
  inliers = np.abs(sub[:, 2] - float(np.percentile(sub[:, 2], 35.0))) <= 0.012
  plane = default_plane
  for _ in range(3):
    if int(np.count_nonzero(inliers)) < 25:
      break
    inp = sub[inliers]
    mat = np.column_stack((inp[:, 0], inp[:, 1], np.ones(len(inp))))
    coef, _, _, _ = np.linalg.lstsq(mat, inp[:, 2], rcond=None)
    plane = (float(coef[0]), float(coef[1]), float(coef[2]))
    z_hat = plane[0] * sub[:, 0] + plane[1] * sub[:, 1] + plane[2]
    inliers = np.abs(sub[:, 2] - z_hat) <= 0.0055
  return plane


def plane_z(plane: tuple[float, float, float], x: float, y: float) -> float:
  """Evaluates fitted tabletop height `Z = a*x + b*y + c` at `(x, y)`."""
  return plane[0] * x + plane[1] * y + plane[2]


def robust_xy(pts: np.ndarray, radius_m: float = 0.065) -> np.ndarray | None:
  """Returns 5-95% bounding-box midpoint around median XY, or `None`."""
  if len(pts) < 4:
    return None
  med = np.median(pts[:, :2], axis=0)
  keep = pts[np.linalg.norm(pts[:, :2] - med[None, :], axis=1) <= radius_m]
  if len(keep) < 4:
    sub = pts[:: max(1, len(pts) // 64), :2]
    dmat = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    med = sub[int(np.argmax(np.count_nonzero(dmat <= radius_m, axis=1)))]
    keep = pts[np.linalg.norm(pts[:, :2] - med[None, :], axis=1) <= radius_m]
    if len(keep) < 4:
      return None
  p5, p95 = np.percentile(keep[:, :2], [5.0, 95.0], axis=0)
  return 0.5 * (p5 + p95)


def cluster_height(
    pts: np.ndarray,
    dz: np.ndarray,
    xy: np.ndarray,
    default_h: float,
    bounds: tuple[float, float],
) -> float:
  """Estimates object top height (95th-pct `dz` within 4.5 cm of `xy`)."""
  local = dz[np.linalg.norm(pts[:, :2] - xy[None, :], axis=1) <= 0.045]
  if len(local) < 6:
    return default_h
  return float(np.clip(np.percentile(local, 95.0), bounds[0], bounds[1]))


def localize_overhead_scene(
    pts: np.ndarray,
    cols: np.ndarray,
    plan: TaskPlan,
    *,
    nominal_table_z_m: float = 0.75,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    float,
    float,
    tuple[float, float, float],
]:
  """Extracts `(t_xy, d_xy, obj_h, sup_dz, plane)` from overhead cloud."""
  plane = fit_table_plane(pts, default_plane=(0.0, 0.0, nominal_table_z_m))
  dz = pts[:, 2] - (plane[0] * pts[:, 0] + plane[1] * pts[:, 1] + plane[2])
  wx0, wx1, wy0, wy1, wz0, wz1 = WORKSPACE_BOX
  in_xy = (
      (pts[:, 0] >= wx0)
      & (pts[:, 0] <= wx1)
      & (pts[:, 1] >= wy0)
      & (pts[:, 1] <= wy1)
  )
  in_ws = in_xy & (dz >= wz0) & (dz <= wz1)
  fg_p, fg_c, fg_dz = pts[in_ws], cols[in_ws], dz[in_ws]
  t_min = PEG_MIN_DZ_M if plan.family == "peg_in_plate" else BLOCK_MIN_DZ_M
  t_xy = robust_xy(fg_p[color_mask(fg_c, plan.target_color) & (fg_dz >= t_min)])
  if plan.dest_kind == "bowl":
    d_xy = robust_xy(fg_p[color_mask(fg_c, plan.dest_color)], 0.115)
  elif plan.dest_kind == "block_base":
    d_xy = robust_xy(
        fg_p[color_mask(fg_c, plan.dest_color) & (fg_dz >= BLOCK_MIN_DZ_M)]
    )
  elif t_xy is not None:
    far = np.linalg.norm(fg_p[:, :2] - t_xy[None, :], axis=1) >= 0.070
    sz_lo, sz_hi = SOCKET_DZ_BAND_M
    s_z = (fg_dz >= sz_lo) & (fg_dz <= sz_hi)
    d_xy = robust_xy(fg_p[color_mask(fg_c, plan.dest_color) & far & s_z])
  else:
    d_xy = None
  if t_xy is not None:
    obj_h = cluster_height(fg_p, fg_dz, t_xy, plan.prior_obj_h, (0.024, 0.105))
  else:
    obj_h = plan.prior_obj_h
  if d_xy is not None and plan.dest_kind == "block_base":
    sup_dz = cluster_height(
        fg_p, fg_dz, d_xy, plan.prior_support_dz, (0.020, 0.070)
    )
  else:
    sup_dz = plan.prior_support_dz
  return t_xy, d_xy, obj_h, sup_dz, plane


def refine_wrist_cluster(
    pts: np.ndarray,
    cols: np.ndarray,
    approx_xy: np.ndarray,
    default_plane: tuple[float, float, float],
    kind: str,
    color: str = "",
) -> tuple[np.ndarray, tuple[float, float, float]]:
  """Refines cluster XY and local tabletop plane from a wrist cloud."""
  p_ref = fit_table_plane(pts, default_plane=default_plane)
  if (
      abs(p_ref[0] - default_plane[0]) > 0.035
      or abs(p_ref[1] - default_plane[1]) > 0.035
  ):
    p_ref = default_plane
  dz = pts[:, 2] - (p_ref[0] * pts[:, 0] + p_ref[1] * pts[:, 1] + p_ref[2])
  r_s = 0.115 if kind == "bowl" else 0.095
  near = np.linalg.norm(pts[:, :2] - approx_xy[None, :], axis=1) <= r_s
  if kind in ("bowl", "block_base", "socket"):
    dz_lo, dz_hi = WRIST_DZ_BANDS_M[kind]
    eff_col = color or ("gray" if kind == "socket" else "blue")
    mask = near & (dz >= dz_lo) & (dz <= dz_hi) & color_mask(cols, eff_col)
  else:
    is_peg = kind == "peg" or (not color and kind == "green")
    dz_lo, dz_hi = WRIST_DZ_BANDS_M["peg" if is_peg else "target"]
    eff_col = color or kind
    mask = near & (dz >= dz_lo) & (dz <= dz_hi) & color_mask(cols, eff_col)
  ref = robust_xy(pts[mask], r_s)
  return (ref if ref is not None else approx_xy), p_ref


def compute_scan_pose(
    q_home: np.ndarray, jnt0_range: Sequence[float] | np.ndarray
) -> np.ndarray:
  """Returns retracted overhead-camera scan joint configuration."""
  q_scan = np.asarray(q_home, dtype=np.float64).copy()
  q_scan[0] = float(
      np.clip(
          q_home[0] - 1.25,
          float(jnt0_range[0]) + 0.08,
          float(jnt0_range[1]) - 0.08,
      )
  )
  return q_scan
