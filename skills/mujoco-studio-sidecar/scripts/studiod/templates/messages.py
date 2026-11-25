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
"""Shared message definitions between sim_plugin.py and viewer_plugin.py."""

import dataclasses
from mujoco.studio import messages


@dataclasses.dataclass(frozen=True)
class SceneConfig(messages.Snapshot):
  """UI -> sim: scene parameters; any change rebuilds the scene and drops."""

  seed: int = 0
  drop_height: float = 0.5  # Height of the lowest toy above the floor.
  toy_scale: float = 1.0  # Overall toy size.


@dataclasses.dataclass(frozen=True)
class BakeEvent(messages.Event):
  """UI -> sim: write the current scene to scene.xml and show that file."""


@dataclasses.dataclass(frozen=True)
class DropTelemetry(messages.Snapshot):
  """sim -> UI: latest-wins summary of the current drop."""

  seed: int = 0
  sim_time: float = 0.0
  # Sim time at which every toy came to rest, or None while still moving.
  settled_at: float | None = None
