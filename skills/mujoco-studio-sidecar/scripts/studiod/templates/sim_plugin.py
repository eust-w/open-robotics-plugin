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
"""Simulation-side Studio plugin (hot-reloaded on save).

Builds the whole scene in Python: a floor and four toys whose sizes and drop
poses come from a seed. Change a parameter in the panel, or edit this file and
save, and the scene is rebuilt in place. Ask the agent to bake the scene into
scene.xml once you are happy with it.
"""

import os

import messages as user_messages
import mujoco
from mujoco.studio import messages
from mujoco.studio import plugin
import numpy as np

# Each toy's geom type, base size, and color (Google blue, red, yellow, green).
TOYS = {
    'box': (
        mujoco.mjtGeom.mjGEOM_BOX,
        (0.06, 0.06, 0.06),
        (0.259, 0.522, 0.957),
    ),
    'sphere': (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (0.06, 0, 0),
        (0.918, 0.263, 0.208),
    ),
    'cylinder': (
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (0.05, 0.06, 0),
        (0.984, 0.737, 0.016),
    ),
    'ellipsoid': (
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        (0.07, 0.05, 0.045),
        (0.204, 0.659, 0.325),
    ),
}
# One drop cell per toy so they never spawn inside each other.
CELLS = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0))
SPREAD = 0.4  # Half-width of the square drop area, in meters.
# Where the Bake button writes the scene, next to this file.
SCENE_XML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'scene.xml'
)


class SimPlugin(plugin.SimPlugin):
  """Builds the toy-drop scene and reports when the toys settle."""

  # Scene parameters, set from the panel (see messages.SceneConfig).
  seed: int = 0
  drop_height: float = 0.5  # Height of the lowest toy above the floor.
  toy_scale: float = 1.0  # Overall toy size.
  # Edit while running: each toy's size varies by up to this fraction.
  size_jitter: float = 0.2
  # Edit while running: anything slower than this counts as settled.
  settle_speed: float = 0.05

  spec: mujoco.MjSpec | None = None
  drop_requested: bool = False
  settled_at: float | None = None

  def build_spec(self) -> mujoco.MjSpec:
    """Returns the scene for the current parameters."""
    rng = np.random.default_rng(self.seed)
    spec = mujoco.MjSpec()
    spec.modelname = 'toy_drop'
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1]
    spec.visual.global_.azimuth = 135
    spec.visual.global_.elevation = -20

    spec.add_texture(
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0, 0, 0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name='groundplane',
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    ground = spec.add_material(
        name='groundplane', texrepeat=[5, 5], texuniform=True, reflectance=0.2
    )
    ground.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'groundplane'

    # condim 6 adds torsional and rolling friction so round toys stop.
    spec.default.geom.friction = [0.6, 0.005, 0.005]
    spec.default.geom.condim = 6
    spec.default.geom.density = 400

    world = spec.worldbody
    world.add_light(
        pos=[0, 0, 2.5],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    world.add_geom(
        name='floor',
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material='groundplane',
    )
    for i, (toy, (geom_type, size, rgb)) in enumerate(TOYS.items()):
      scale = self.toy_scale * rng.uniform(
          1 - self.size_jitter, 1 + self.size_jitter
      )
      xy = np.array(CELLS[i]) * SPREAD / 2
      xy += rng.uniform(-0.25, 0.25, size=2) * SPREAD
      quat = rng.normal(size=4)
      body = world.add_body(
          name=toy,
          pos=[*xy, self.drop_height + 0.15 * i],
          quat=quat / np.linalg.norm(quat),
      )
      body.add_freejoint(name=f'{toy}_free')
      body.add_geom(
          name=f'{toy}_geom',
          type=geom_type,
          size=np.array(size) * scale,
          rgba=[*rgb, 1],
      )
    return spec

  @messages.handler
  def on_load_model(self, event: messages.LoadModelEvent) -> bool:
    if event.path:
      return False  # Show a model file instead (scene.xml, or a dropped file).
    self.spec = self.build_spec()
    self.handle.set_model(self.spec)
    return True

  @messages.handler
  def on_post_model(self, event: messages.PostModelEvent) -> None:
    # The daemon keeps the toys' poses when the scene is rebuilt after an edit.
    # A new drop instead starts them from where build_spec() places them.
    if self.drop_requested:
      self.drop_requested = False
      mujoco.mj_resetData(event.model, event.data)
      mujoco.mj_forward(event.model, event.data)
    self.settled_at = None

  @messages.handler
  def on_hot_reload(self, event: messages.HotReloadEvent) -> None:
    if self in event.plugins:
      self.handle.reload()

  @messages.handler
  def on_scene_config(self, snapshot: user_messages.SceneConfig) -> None:
    self.seed = snapshot.seed
    self.drop_height = snapshot.drop_height
    self.toy_scale = snapshot.toy_scale
    self.drop_requested = True
    self.handle.reload()

  @messages.handler
  def on_bake(self, event: user_messages.BakeEvent) -> None:
    """Writes the current scene to scene.xml and shows the file from now on."""
    del event
    if self.spec is None:
      return
    with open(SCENE_XML, 'w', encoding='utf-8') as f:
      f.write(self.spec.to_xml())
    self.handle.dispatch(messages.LoadModelEvent(path=SCENE_XML))

  @messages.handler
  def on_post_step(self, event: messages.PostStepEvent) -> None:
    data = event.data
    moving = bool(np.abs(data.qvel).max() >= self.settle_speed)
    if moving:
      self.settled_at = None
    elif self.settled_at is None:
      self.settled_at = float(data.time)
    self.handle.send_to_viewer(
        user_messages.DropTelemetry(
            seed=self.seed,
            sim_time=float(data.time),
            settled_at=self.settled_at,
        )
    )
