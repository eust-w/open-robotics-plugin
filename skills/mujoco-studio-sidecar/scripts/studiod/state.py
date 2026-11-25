# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thread-safe daemon status, context snapshots, and hot-reload health."""

import collections
from collections.abc import Callable
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import sys
import tempfile
import threading
import time
import traceback
from typing import Any

import mujoco
from mujoco.studio import hot_reload
from mujoco.studio import messages
import numpy as np
from studiod import discovery

logger = logging.getLogger(__name__)


StatusListener = Callable[[str, str | None], None]

# Listener key for model file load results. Not a valid module name, so a user
# helper module called `scene.py` is never mistaken for the model.
MODEL_STATUS_KEY = '<model>'


@dataclasses.dataclass(frozen=True)
class RequestContextSnapshotEvent(messages.Event):
  """Triggers an immediate agent context snapshot from the simulation thread."""

  note: str = ''
  selected_body_id: int = -1
  history_index: int = 0
  history_size: int = 0


def default_runtime_dir(
    workspace_dir: pathlib.Path, explicit_dir: str = ''
) -> pathlib.Path:
  """Returns an out-of-workspace directory for runtime status and snapshots."""
  if explicit_dir:
    return pathlib.Path(explicit_dir).resolve()
  env_dir = os.environ.get('MUJOCO_STUDIO_RUNTIME_DIR', '')
  if env_dir:
    return pathlib.Path(env_dir).resolve()
  ws_resolved = workspace_dir.resolve()
  digest = hashlib.sha256(str(ws_resolved).encode('utf-8')).hexdigest()[:12]
  slug = f'{ws_resolved.name or "workspace"}_{digest}'
  for env_var in ('PLUGIN_DATA', 'ANTIGRAVITY_EXECUTABLE_DATA_DIR'):
    base = os.environ.get(env_var, '')
    if base:
      return pathlib.Path(base).resolve() / 'workspaces' / slug
  return (
      pathlib.Path(tempfile.gettempdir())
      / f'mujoco_studio_{discovery.uid_tag()}'
      / slug
  )


def make_hot_reload_status_listener(
    daemon_state: 'DaemonState',
) -> StatusListener:
  """Builds a structured hot-reload status listener updating DaemonState."""
  failed_helpers: dict[str, str] = {}

  def _on_status(module_name: str, error: str | None) -> None:
    if module_name == MODEL_STATUS_KEY:
      daemon_state.record_status('scene', ok=error is None, error=error)
      return
    leaf = module_name.rsplit('.', 1)[-1]
    if leaf in ('messages', 'sim_plugin', 'viewer_plugin') and (
        module_name == leaf or module_name.startswith('studio_plugins.')
    ):
      # Helper failures are reported under sim_plugin, so a successful reload
      # of sim_plugin.py must not hide a helper that is still broken.
      if error is None and leaf == 'sim_plugin' and failed_helpers:
        error = next(iter(failed_helpers.values()))
      daemon_state.record_status(leaf, ok=error is None, error=error)
      return
    if error is not None:
      failed_helpers[module_name] = error
      daemon_state.record_status('sim_plugin', ok=False, error=error)
    elif module_name in failed_helpers:
      failed_helpers.pop(module_name, None)
      remaining = next(iter(failed_helpers.values()), None)
      daemon_state.record_status(
          'sim_plugin', ok=remaining is None, error=remaining
      )

  return _on_status


def _classify_plugin_exception(cls_prefix: str, err_text: str) -> str:
  """Maps a plugin class name and traceback to `sim_plugin` or `viewer_plugin`."""
  for comp in ('sim_plugin', 'viewer_plugin'):
    mod = sys.modules.get(f'studio_plugins.{comp}')
    if mod is not None and isinstance(getattr(mod, cls_prefix, None), type):
      return comp
  if '/viewer_plugin.py"' in err_text or '\\viewer_plugin.py"' in err_text:
    return 'viewer_plugin'
  if '/sim_plugin.py"' in err_text or '\\sim_plugin.py"' in err_text:
    return 'sim_plugin'
  if cls_prefix == 'ViewerPlugin':
    return 'viewer_plugin'
  return 'sim_plugin'


def subscribe_hot_reload_status(listener: StatusListener) -> Callable[[], None]:
  """Subscribes `listener(module_name, error_or_none)` to Studio hot-reload."""
  subscribe_fn = getattr(hot_reload, 'subscribe_status', None)
  if callable(subscribe_fn):
    return subscribe_fn(listener)

  prefix = '[Studio hot reload] '
  failed_marker = ' failed; keeping the previous code.\n'
  raised_marker = ' raised (repeats hidden until the next reload):\n'
  model_failed_suffix = ' failed; keeping the previous model.'

  class _HotReloadLogHandler(logging.Handler):
    """Parses structured reload messages emitted by `mujoco.studio.hot_reload`."""

    def emit(self, record: logging.LogRecord) -> None:
      try:
        msg = record.getMessage()
        if not msg.startswith(prefix):
          return
        body = msg[len(prefix) :]
        if body.startswith('Reloading ') and failed_marker in body:
          mod_name, err_text = body[len('Reloading ') :].split(failed_marker, 1)
          listener(mod_name.strip(), err_text)
        elif body.startswith('Reloaded '):
          for mod_name in body[len('Reloaded ') :].split(', '):
            if mod_name.strip():
              listener(mod_name.strip(), None)
        elif raised_marker in body:
          label, err_text = body.split(raised_marker, 1)
          cls_prefix = label.split('.', 1)[0]
          listener(_classify_plugin_exception(cls_prefix, err_text), err_text)
      except Exception:  # pylint: disable=broad-exception-caught
        logger.debug('Hot-reload log handler failed', exc_info=True)

  class _ModelLoadLogHandler(logging.Handler):
    """Captures model file load failures from `mujoco.studio.viewer_handle`."""

    def emit(self, record: logging.LogRecord) -> None:
      try:
        msg = record.getMessage()
        if msg.startswith('Loading ') and msg.endswith(model_failed_suffix):
          err = msg
          if record.exc_info and record.exc_info[0] is not None:
            exc_lines = traceback.format_exception(*record.exc_info)
            err = f'{msg}\n' + ''.join(exc_lines).rstrip('\n')
          listener(MODEL_STATUS_KEY, err)
      except Exception:  # pylint: disable=broad-exception-caught
        logger.debug('Model-load log handler failed', exc_info=True)

  target_logger = getattr(
      hot_reload, 'logger', logging.getLogger('mujoco.studio.hot_reload')
  )
  vh_logger = logging.getLogger('mujoco.studio.viewer_handle')
  handler = _HotReloadLogHandler()
  model_handler = _ModelLoadLogHandler()
  target_logger.addHandler(handler)
  vh_logger.addHandler(model_handler)

  def _unsubscribe() -> None:
    target_logger.removeHandler(handler)
    vh_logger.removeHandler(model_handler)

  return _unsubscribe


class DaemonState:
  """Thread-safe status, context snapshot, and port registry for studiod."""

  def __init__(
      self,
      workspace_dir: pathlib.Path,
      scene_file: str = '',
      runtime_dir: pathlib.Path | None = None,
      on_context_captured: Callable[[dict[str, Any]], None] | None = None,
      on_status_changed: Callable[[dict[str, Any]], None] | None = None,
  ) -> None:
    self._lock = threading.RLock()
    self._workspace_dir = workspace_dir.resolve()
    self._runtime_dir = (
        runtime_dir or default_runtime_dir(self._workspace_dir)
    ).resolve()
    self._scene_file = scene_file
    self._on_context_captured = on_context_captured
    self._on_status_changed = on_status_changed
    self._viewer_http_port = 0
    self._viewer_port_ready = threading.Event()
    self._selected_body_id = -1
    self._history_index = 0
    self._history_size = 0
    self._sim_head_time = 0.0
    self._recent_trajectory: collections.deque[dict[str, Any]] = (
        collections.deque(maxlen=50)
    )
    self._latest_snapshot: dict[str, Any] = {}
    self._status: dict[str, Any] = {
        'ok': True,
        'timestamp': time.time(),
        'workspace_dir': str(self._workspace_dir),
        'runtime_dir': str(self._runtime_dir),
        'status_file': str(self._runtime_dir / 'status.json'),
        'snapshot_file': str(self._runtime_dir / 'latest_snapshot.json'),
        'scene': {
            'ok': True,
            'source': self.scene_source_name(),
            'error': None,
        },
        'messages': {'ok': True, 'error': None},
        'sim_plugin': {'ok': True, 'error': None},
        'viewer_plugin': {'ok': True, 'error': None},
    }
    self._write_status_file()

  @property
  def workspace_dir(self) -> pathlib.Path:
    with self._lock:
      return self._workspace_dir

  @property
  def runtime_dir(self) -> pathlib.Path:
    with self._lock:
      return self._runtime_dir

  @property
  def status_file(self) -> pathlib.Path:
    with self._lock:
      return self._runtime_dir / 'status.json'

  @property
  def snapshot_file(self) -> pathlib.Path:
    with self._lock:
      return self._runtime_dir / 'latest_snapshot.json'

  def resolve_scene_path(self) -> pathlib.Path | None:
    with self._lock:
      ws = self._workspace_dir
      if self._scene_file:
        candidate = ws / self._scene_file
        return candidate if candidate.is_file() else None
      candidate = ws / 'scene.xml'
      if candidate.is_file():
        return candidate
      xml_files = sorted(ws.glob('*.xml'))
      return xml_files[0] if xml_files else None

  def scene_source_name(self) -> str:
    path = self.resolve_scene_path()
    if path is not None:
      return path.name
    with self._lock:
      if self._scene_file:
        return self._scene_file
      if (self._workspace_dir / 'sim_plugin.py').is_file():
        return 'sim_plugin.py'
      return 'scene.xml'

  def set_viewer_http_port(self, port: int) -> None:
    with self._lock:
      self._viewer_http_port = int(port)
      if port > 0:
        self._viewer_port_ready.set()

  def wait_for_viewer_http_port(
      self,
      timeout: float = 10.0,
      is_alive_fn: Callable[[], bool] | None = None,
  ) -> int:
    """Waits for the WebViewer HTTP port or returns early if the viewer exits."""
    if is_alive_fn is None:
      self._viewer_port_ready.wait(timeout=timeout)
    else:
      deadline = time.monotonic() + max(0.0, timeout)
      while not self._viewer_port_ready.is_set():
        try:
          if not is_alive_fn():
            break
        except Exception:  # pylint: disable=broad-exception-caught
          break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          break
        self._viewer_port_ready.wait(timeout=min(0.05, remaining))
    with self._lock:
      return self._viewer_http_port

  def update_ui_metadata(
      self,
      selected_body_id: int,
      history_index: int,
      history_size: int,
      sim_head_time: float,
  ) -> None:
    with self._lock:
      self._selected_body_id = selected_body_id
      self._history_index = history_index
      self._history_size = history_size
      self._sim_head_time = sim_head_time

  def get_ui_metadata(self) -> tuple[int, int, int, float]:
    with self._lock:
      return (
          self._selected_body_id,
          self._history_index,
          self._history_size,
          self._sim_head_time,
      )

  def record_status(
      self,
      component: str,
      *,
      ok: bool,
      error: str | None = None,
      source: str | None = None,
  ) -> None:
    """Updates health status for `component` and persists `status.json`."""
    with self._lock:
      entry: dict[str, Any] = {
          'ok': ok,
          'error': error,
          'updated_at': time.time(),
      }
      if source is not None:
        entry['source'] = source
      elif component == 'scene':
        entry['source'] = self.scene_source_name()
      self._status[component] = entry
      self._status['timestamp'] = time.time()
      self._status['workspace_dir'] = str(self._workspace_dir)
      self._status['runtime_dir'] = str(self._runtime_dir)
      self._status['status_file'] = str(self._runtime_dir / 'status.json')
      self._status['snapshot_file'] = str(
          self._runtime_dir / 'latest_snapshot.json'
      )
      self._status['ok'] = all(
          self._status.get(k, {}).get('ok', True)
          for k in ('scene', 'messages', 'sim_plugin', 'viewer_plugin')
      )
      snapshot = json.loads(json.dumps(self._status))
      self._write_status_file()
    if self._on_status_changed is not None:
      try:
        self._on_status_changed(snapshot)
      except Exception:  # pylint: disable=broad-exception-caught
        logger.debug('on_status_changed callback failed', exc_info=True)

  def _write_status_file(self) -> None:
    try:
      status_path = self._runtime_dir / 'status.json'
      discovery.atomic_write_text(
          status_path, json.dumps(self._status, indent=2)
      )
    except OSError:
      logger.debug('Could not write status.json', exc_info=True)

  def get_status(self) -> dict[str, Any]:
    with self._lock:
      return json.loads(json.dumps(self._status))

  def record_trajectory_sample(
      self, model: mujoco.MjModel, data: mujoco.MjData
  ) -> None:
    """Appends a decimated trajectory sample (~50 Hz) to the ring buffer."""
    with self._lock:
      if (
          self._recent_trajectory
          and abs(float(data.time) - self._recent_trajectory[-1]['time']) < 0.02
      ):
        return
      sample: dict[str, Any] = {
          'time': round(float(data.time), 4),
          'qpos_head': [
              round(float(x), 5) for x in data.qpos[: min(16, model.nq)]
          ],
          'qvel_head': [
              round(float(x), 5) for x in data.qvel[: min(16, model.nv)]
          ],
          'ncon': int(data.ncon),
      }
      self._recent_trajectory.append(sample)

  def capture_context_snapshot(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      event: RequestContextSnapshotEvent | None = None,
  ) -> dict[str, Any]:
    """Serializes the current MjModel/MjData state to `runtime_dir`."""
    sel_body, hist_idx, hist_size, head_time = self.get_ui_metadata()
    if event is not None:
      if event.selected_body_id >= 0:
        sel_body = event.selected_body_id
      if event.history_size > 0:
        hist_idx = event.history_index
        hist_size = event.history_size

    selected_body_name = ''
    if 0 <= sel_body < model.nbody:
      selected_body_name = (
          mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_BODY), sel_body)
          or f'body_{sel_body}'
      )

    named_qpos: dict[str, list[float]] = {}
    named_qvel: dict[str, list[float]] = {}
    for j in range(model.njnt):
      jname = (
          mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_JOINT), j)
          or f'joint_{j}'
      )
      jtype = int(model.jnt_type[j])
      qpos_adr = int(model.jnt_qposadr[j])
      qvel_adr = int(model.jnt_dofadr[j])
      nq = {
          int(mujoco.mjtJoint.mjJNT_FREE): 7,
          int(mujoco.mjtJoint.mjJNT_BALL): 4,
          int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
          int(mujoco.mjtJoint.mjJNT_HINGE): 1,
      }.get(jtype, 0)
      nv = {
          int(mujoco.mjtJoint.mjJNT_FREE): 6,
          int(mujoco.mjtJoint.mjJNT_BALL): 3,
          int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
          int(mujoco.mjtJoint.mjJNT_HINGE): 1,
      }.get(jtype, 0)
      if nq > 0:
        named_qpos[jname] = [
            round(float(x), 6) for x in data.qpos[qpos_adr : qpos_adr + nq]
        ]
      if nv > 0:
        named_qvel[jname] = [
            round(float(x), 6) for x in data.qvel[qvel_adr : qvel_adr + nv]
        ]

    named_ctrl: dict[str, float] = {}
    for a in range(model.nu):
      aname = (
          mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_ACTUATOR), a)
          or f'actuator_{a}'
      )
      named_ctrl[aname] = round(float(data.ctrl[a]), 6)

    applied_wrenches: dict[str, list[float]] = {}
    for b in range(model.nbody):
      wrench = data.xfrc_applied[b]
      if np.any(np.abs(wrench) > 1e-9):
        bname = (
            mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_BODY), b)
            or f'body_{b}'
        )
        applied_wrenches[bname] = [round(float(x), 5) for x in wrench]

    contacts: list[dict[str, Any]] = []
    for c_idx in range(min(int(data.ncon), 32)):
      con = data.contact[c_idx]
      g1 = int(con.geom1)
      g2 = int(con.geom2)
      g1_name = (
          mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_GEOM), g1)
          or f'geom_{g1}'
      )
      g2_name = (
          mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_GEOM), g2)
          or f'geom_{g2}'
      )
      contacts.append({
          'geom1': g1_name,
          'geom2': g2_name,
          'dist': round(float(con.dist), 6),
          'pos': [round(float(x), 5) for x in con.pos],
      })

    qpos_str = ' '.join(f'{float(x):.6g}' for x in data.qpos)
    qvel_str = ' '.join(f'{float(x):.6g}' for x in data.qvel)
    ctrl_str = ' '.join(f'{float(x):.6g}' for x in data.ctrl)
    keyframe_xml = (
        f'<key name="captured_{int(time.time())}" time="{float(data.time):.4f}"'
        f' qpos="{qpos_str}" qvel="{qvel_str}" ctrl="{ctrl_str}"/>'
    )

    with self._lock:
      ws = self._workspace_dir
      latest_path = self._runtime_dir / 'latest_snapshot.json'
      snapshot: dict[str, Any] = {
          'timestamp': time.time(),
          'workspace_dir': str(ws),
          'snapshot_file': str(latest_path),
          'scene_source': self.scene_source_name(),
          'note': event.note if event is not None else '',
          'sim_time': round(float(data.time), 5),
          'sim_head_time': round(float(head_time), 5),
          'history_index': int(hist_idx),
          'history_size': int(hist_size),
          'selected_body_id': int(sel_body),
          'selected_body_name': selected_body_name,
          'applied_wrenches': applied_wrenches,
          'qpos': named_qpos,
          'qvel': named_qvel,
          'ctrl': named_ctrl,
          'ncon': int(data.ncon),
          'contacts': contacts,
          'keyframe_xml': keyframe_xml,
          'recent_trajectory': list(self._recent_trajectory),
      }
      self._latest_snapshot = snapshot
      try:
        discovery.atomic_write_text(latest_path, json.dumps(snapshot, indent=2))
      except OSError:
        logger.debug('Could not write latest_snapshot.json', exc_info=True)

    if self._on_context_captured is not None:
      try:
        self._on_context_captured(snapshot)
      except Exception:  # pylint: disable=broad-exception-caught
        logger.debug('on_context_captured callback failed', exc_info=True)
    return snapshot

  def get_latest_snapshot(self) -> dict[str, Any]:
    with self._lock:
      if self._latest_snapshot:
        return json.loads(json.dumps(self._latest_snapshot))
      latest_path = self._runtime_dir / 'latest_snapshot.json'
    if latest_path.is_file():
      try:
        return json.loads(latest_path.read_text(encoding='utf-8'))
      except (OSError, json.JSONDecodeError):
        pass
    return {}
