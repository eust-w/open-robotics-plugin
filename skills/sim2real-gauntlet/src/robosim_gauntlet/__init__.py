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
"""Sim-to-real gauntlet on reaf-sim and mujoco_playground DeviceLayers."""

import importlib.machinery
import importlib.util
import os
import sys


def ensure_gl_backend() -> str:
  """Selects a headless GL backend on Linux without importing `mujoco`."""
  cur = os.environ.get("MUJOCO_GL", "").strip()
  if cur or sys.platform != "linux":
    return cur
  spec = importlib.util.find_spec("mujoco")
  locs = list(spec.submodule_search_locations or ()) if spec else []
  if not locs:
    return ""
  finder = importlib.machinery.PathFinder.find_spec
  has_egl = finder("mujoco.egl", locs) is not None
  has_osmesa = finder("mujoco.osmesa", locs) is not None
  has_gpu = os.path.exists("/dev/dri/renderD128") or os.path.exists("/dev/nvidiactl")
  if has_egl and (not has_osmesa or has_gpu):
    os.environ["MUJOCO_GL"] = "egl"
  elif has_osmesa:
    os.environ["MUJOCO_GL"] = "osmesa"
  return os.environ.get("MUJOCO_GL", "")


ensure_gl_backend()
