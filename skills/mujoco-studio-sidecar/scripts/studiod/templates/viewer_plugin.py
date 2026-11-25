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
"""Viewer-side Dear ImGui Studio plugin (hot-reloaded on save)."""

import os
import pathlib

import messages as user_messages
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio.app import plugins as app_plugins

imgui = app_plugins.imgui

# This file lives in the workspace, so its folder is where the editable files
# are. Shown in the panel so you always know what hot reload is watching.
WORKSPACE_DIR = pathlib.Path(os.path.abspath(__file__)).parent
EDITABLE_FILES = (
    'messages.py',
    'sim_plugin.py',
    'viewer_plugin.py',
)


def _short_path(path: pathlib.Path) -> str:
  home = pathlib.Path.home()
  return (
      '~/' + str(path.relative_to(home))
      if path.is_relative_to(home)
      else str(path)
  )


class ViewerPlugin(plugin.ViewerPlugin):
  """Simulation Playground: tweak the toy-drop scene and drop again."""

  seed: int = 0
  drop_height: float = 0.5
  toy_scale: float = 1.0
  telemetry: user_messages.DropTelemetry = user_messages.DropTelemetry()
  window_open: bool = True

  @messages.handler
  def on_telemetry(self, snapshot: user_messages.DropTelemetry) -> None:
    self.telemetry = snapshot

  def _send_config(self) -> None:
    self.viewer.send_to_sim(
        user_messages.SceneConfig(
            seed=self.seed,
            drop_height=self.drop_height,
            toy_scale=self.toy_scale,
        )
    )

  @messages.handler
  def on_build_gui(self, event: messages.BuildGuiEvent) -> None:
    del event
    if not self.window_open:
      return
    expanded, self.window_open = imgui.Begin(
        'Simulation Playground###SimulationPlaygroundWindow', self.window_open
    )
    try:
      if not expanded:
        return
      imgui.TextUnformatted(f'Workspace: {_short_path(WORKSPACE_DIR)}')
      imgui.SetItemTooltip(
          'Edit and save any of these to hot-reload the running simulation:\n'
          + '\n'.join(f'  {name}' for name in EDITABLE_FILES)
      )
      imgui.Separator()

      # Each control changes the scene built in sim_plugin.py, which is then
      # rebuilt and the toys dropped again.
      changed = False
      seed_changed, seed = imgui.InputInt('Seed', self.seed)
      if seed_changed:
        self.seed = max(0, seed)
        changed = True
      height_changed, self.drop_height = imgui.SliderFloat(
          'Drop height (m)', self.drop_height, 0.2, 2.0
      )
      scale_changed, self.toy_scale = imgui.SliderFloat(
          'Toy size', self.toy_scale, 0.5, 2.0
      )
      if changed or height_changed or scale_changed:
        self._send_config()
      if imgui.Button('Drop again'):
        self._send_config()

      t = self.telemetry
      if t.settled_at is None:
        imgui.TextUnformatted(f'Moving...  t = {t.sim_time:.2f} s')
      else:
        imgui.TextUnformatted(f'Settled at {t.settled_at:.2f} s')

      imgui.Separator()
      if imgui.Button('Bake to scene.xml'):
        self.viewer.send_to_sim(user_messages.BakeEvent())
      imgui.SetItemTooltip(
          'Write the current scene to scene.xml in the workspace and show that'
          ' file from now on. The controls above stop changing the scene.'
      )
    finally:
      imgui.End()
