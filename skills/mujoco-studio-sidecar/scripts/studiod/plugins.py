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
"""Built-in Studio plugins registered by `studiod.py`."""

import logging
import os
import traceback

from mujoco.studio import app as studio_app
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio.app import plugins as app_plugins
from studiod import sim_state
from studiod import state

imgui = app_plugins.imgui
logger = logging.getLogger(__name__)


class DaemonSimPlugin(plugin.SimPlugin):
  """Sim-side helper plugin for scene loading, state preservation, and context."""

  def __init__(self, daemon_state: state.DaemonState) -> None:
    self._daemon_state = daemon_state
    self._preserved_state: sim_state.PreservedSimState | None = None
    self._capture_on_load: bool = True

  # Below Priority.INTERNAL so StepControl and SimHistory finish their own init
  # before the first model loads; otherwise they reset after it and the first
  # reload loses pause and speed.
  @messages.handler(priority=messages.Priority.INTERNAL - 1)
  def on_sim_init(self, event: plugin.SimInitEvent) -> None:
    del event
    self.reload_scene(preserve_state=False)

  def reload_scene(self, preserve_state: bool = True) -> None:
    """Dispatches LoadModelEvent for the workspace scene."""
    scene_path = self._daemon_state.resolve_scene_path()
    path = str(scene_path) if scene_path is not None else ''
    prev_capture = self._capture_on_load
    self._capture_on_load = preserve_state
    if not preserve_state:
      self._preserved_state = None
    try:
      self.handle.dispatch(messages.LoadModelEvent(path=path))
      if scene_path is None and self.handle.model is None:
        self._daemon_state.record_status(
            'scene',
            ok=False,
            error=(
                f'No scene XML in {self._daemon_state.workspace_dir} and no'
                ' sim plugin handled LoadModelEvent. Add scene.xml, or an'
                ' on_load_model handler that calls self.handle.set_model().'
            ),
        )
    except Exception:  # pylint: disable=broad-exception-caught
      tb = traceback.format_exc()
      logger.error('Scene load error:\n%s', tb)
      component = 'scene' if scene_path is not None else 'sim_plugin'
      self._daemon_state.record_status(component, ok=False, error=tb)
    finally:
      self._capture_on_load = prev_capture

  @messages.handler(priority=messages.Priority.CRITICAL)
  def on_load_model(self, event: messages.LoadModelEvent) -> None:
    del event
    if (
        self._capture_on_load
        and self.handle.model is not None
        and self.handle.data is not None
    ):
      self._preserved_state = sim_state.capture_preserved_state(
          self.handle.model, self.handle.data
      )

  @messages.handler(priority=messages.Priority.CRITICAL)
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    if self._preserved_state is not None:
      sim_state.apply_preserved_state(
          event.model, event.data, self._preserved_state
      )
      self._preserved_state = None
    self._daemon_state.record_status(
        'scene',
        ok=True,
        error=None,
        source=(
            os.path.basename(event.path)
            if event.path
            else self._daemon_state.scene_source_name()
        ),
    )

  @messages.handler(priority=messages.Priority.CRITICAL)
  def on_reset(self, event: messages.ResetEvent) -> None:
    del event
    self._preserved_state = None

  @messages.handler(priority=messages.Priority.LIBRARY)
  def on_post_step(self, event: messages.PostStepEvent) -> None:
    self._daemon_state.record_trajectory_sample(event.model, event.data)

  @messages.handler
  def on_request_context_snapshot(
      self, event: state.RequestContextSnapshotEvent
  ) -> None:
    if self.handle.model is not None and self.handle.data is not None:
      self._daemon_state.capture_context_snapshot(
          self.handle.model, self.handle.data, event=event
      )


class DaemonViewerPlugin(plugin.ViewerPlugin):
  """Viewer-side helper plugin for port discovery and agent context menu."""

  def __init__(
      self,
      daemon_state: state.DaemonState,
      *,
      compact_ui: bool = True,
  ) -> None:
    self._daemon_state = daemon_state
    self._compact_ui = compact_ui
    self._app: studio_app.App | None = None
    self._last_ui_key: tuple[int, int, int, float] | None = None

  @messages.handler(priority=messages.Priority.CRITICAL)
  def on_viewer_init(self, event: plugin.ViewerInitEvent) -> None:
    port = getattr(event.viewer, 'http_port', 0) or getattr(
        event.viewer, '_http_port', 0
    )
    if port:
      self._daemon_state.set_viewer_http_port(int(port))

  @messages.handler(priority=messages.Priority.CRITICAL)
  def on_app_init(self, event: studio_app.InitEvent) -> None:
    self._app = event.app
    if self._compact_ui and hasattr(event.app, 'config'):
      event.app.config.show_options = False
      event.app.config.show_inspector = False

  @messages.handler(priority=messages.Priority.LIBRARY)
  def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
    del event
    selected_body = -1
    history_idx = 0
    history_size = 0
    head_time = 0.0
    if self._app is not None:
      ux_state = getattr(self._app, 'ux_state', None)
      if ux_state is not None:
        selected_body = int(getattr(ux_state, 'selected_body_id', -1))
        history_idx = int(getattr(ux_state, 'history_index', 0))
        history_size = int(getattr(ux_state, 'history_size', 0))
        head_time = float(getattr(ux_state, 'sim_head_time', 0.0))
      self._daemon_update_ui(
          selected_body, history_idx, history_size, head_time
      )

    if imgui.BeginMainMenuBar():
      try:
        if imgui.BeginMenu('Simulation'):
          try:
            imgui.Separator()
            if imgui.MenuItem(
                'Capture Frame for Agent Context', 'Ctrl+Shift+C'
            ):
              self.viewer.send_to_sim(
                  state.RequestContextSnapshotEvent(
                      note='Captured from MuJoCo Studio Simulation menu',
                      selected_body_id=selected_body,
                      history_index=history_idx,
                      history_size=history_size,
                  )
              )
              if self._app is not None:
                self._app.status = (
                    'Captured frame for Agent Context (/_studio/context)'
                )
          finally:
            imgui.EndMenu()
      finally:
        imgui.EndMainMenuBar()

  def _daemon_update_ui(
      self,
      selected_body: int,
      history_idx: int,
      history_size: int,
      head_time: float,
  ) -> None:
    key = (selected_body, history_idx, history_size, round(head_time, 3))
    if key == self._last_ui_key:
      return
    self._last_ui_key = key
    self._daemon_state.update_ui_metadata(
        selected_body, history_idx, history_size, head_time
    )
