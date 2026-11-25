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
"""Starter plugin files written into a new workspace.

The starter scene is built in Python by `sim_plugin.py`, so its parameters can
be driven from UI controls and rebuilt live. A `scene.xml` only appears when
the user asks for one, or bakes the Python scene into one.
"""

import os
import pathlib

# Do not `resolve()` symlinks here: under Bazel runfiles `__file__` links into
# a content-addressed store where the sibling `templates/` files do not exist.
TEMPLATES_DIR = pathlib.Path(os.path.abspath(__file__)).parent / 'templates'
TEMPLATE_FILES = (
    'messages.py',
    'sim_plugin.py',
    'viewer_plugin.py',
)


def read_template(name: str) -> str:
  """Returns the scaffold template `name` shipped in `studiod/templates/`."""
  if name not in TEMPLATE_FILES:
    raise ValueError(f'Unknown scaffold template: {name!r}')
  return (TEMPLATES_DIR / name).read_text(encoding='utf-8')


def scaffold_workspace(
    workspace_dir: pathlib.Path, scene_file: str = ''
) -> list[str]:
  """Writes the starter plugin files into `workspace_dir`.

  Does nothing once the workspace has any scene XML or plugin file, so user
  files are never overwritten.

  Args:
    workspace_dir: The simulation workspace, created if missing.
    scene_file: Optional scene file the user named; if it exists, the
      workspace already has a scene and nothing is written.

  Returns:
    The names of the files that were written.
  """
  workspace_dir.mkdir(parents=True, exist_ok=True)
  has_existing_files = (
      (bool(scene_file) and (workspace_dir / scene_file).exists())
      or any(workspace_dir.glob('*.xml'))
      or any((workspace_dir / f).exists() for f in TEMPLATE_FILES)
  )
  if has_existing_files:
    return []

  for filename in TEMPLATE_FILES:
    (workspace_dir / filename).write_text(
        read_template(filename), encoding='utf-8'
    )
  return list(TEMPLATE_FILES)
