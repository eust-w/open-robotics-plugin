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
"""Command-line options for `studiod.py`."""

import argparse
from collections.abc import Sequence
import dataclasses


@dataclasses.dataclass(frozen=True)
class DaemonOptions:
  """Command-line configuration for studiod."""

  workspace_dir: str = ''
  scene_file: str = ''
  http_port: int = 0
  control_port: int = 0
  session_file: str = ''
  runtime_dir: str = ''
  sidecar_notify_url: str = ''
  scaffold: bool = True
  compact_ui: bool = True
  open_browser: bool = False
  headless_test_steps: int = 0


def _add_bool_flag(
    arg_parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
  """Adds `--flag`, `--no-flag`, and `--noflag` boolean flag variants."""
  arg_parser.add_argument(
      f'--{name}',
      dest=name,
      action='store_true',
      default=default,
      help=help_text,
  )
  arg_parser.add_argument(
      f'--no-{name}',
      f'--no{name}',
      dest=name,
      action='store_false',
      help=argparse.SUPPRESS,
  )


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the CLI parser for `studiod.py`."""
  arg_parser = argparse.ArgumentParser(
      description='Agent-controlled MuJoCo Studio Web daemon.'
  )
  arg_parser.add_argument(
      '--workspace_dir',
      default='',
      help=(
          'Visible simulation directory containing optional scene.xml, '
          'sim_plugin.py, viewer_plugin.py, and messages.py. Defaults to '
          'CWD (or ~/mujoco_studio_default when invoked from ~).'
      ),
  )
  arg_parser.add_argument(
      '--scene_file',
      default='',
      help='Optional explicit XML/URDF/MJB scene file name (e.g. scene.xml).',
  )
  arg_parser.add_argument(
      '--http_port',
      type=int,
      default=0,
      help='Port for MuJoCo WebViewer HTTP/WebSocket server (0 = free port).',
  )
  arg_parser.add_argument(
      '--control_port',
      type=int,
      default=0,
      help='Loopback port for Studio control & context API (0 = free port).',
  )
  arg_parser.add_argument(
      '--session_file',
      default='',
      help='Optional explicit path for session.json IPC discovery.',
  )
  arg_parser.add_argument(
      '--runtime_dir',
      default='',
      help=(
          'Out-of-workspace directory for runtime status and context snapshot '
          'files. Defaults to PLUGIN_DATA, ANTIGRAVITY_EXECUTABLE_DATA_DIR, '
          'or the OS temp directory ($TMPDIR / /tmp).'
      ),
  )
  arg_parser.add_argument(
      '--sidecar_notify_url',
      default='',
      help='Optional sidecar /_studio/notify URL to notify on startup/updates.',
  )
  _add_bool_flag(
      arg_parser,
      'scaffold',
      True,
      'Whether to scaffold starter plugin files when the workspace'
      ' has no scene XML or plugin files yet.',
  )
  _add_bool_flag(
      arg_parser,
      'compact_ui',
      True,
      'Collapse Options and Inspector sidebars on startup for narrow AuxPanes.',
  )
  _add_bool_flag(
      arg_parser,
      'open_browser',
      False,
      'Whether WebViewer should open a standalone browser tab on startup.',
  )
  arg_parser.add_argument(
      '--headless_test_steps',
      type=int,
      default=0,
      help='If > 0, runs this many sync iterations and exits cleanly.',
  )
  return arg_parser


def parse_daemon_options(argv: Sequence[str] | None = None) -> DaemonOptions:
  """Parses `DaemonOptions` from `argv`."""
  arg_parser = build_arg_parser()
  ns = arg_parser.parse_args(argv)
  if ns.scene_file.endswith('.py'):
    arg_parser.error(
        '--scene_file must be a model file (XML, URDF, or MJB). Build Python'
        ' scenes in sim_plugin.py by handling LoadModelEvent instead.'
    )
  return DaemonOptions(
      workspace_dir=ns.workspace_dir,
      scene_file=ns.scene_file,
      http_port=ns.http_port,
      control_port=ns.control_port,
      session_file=ns.session_file,
      runtime_dir=ns.runtime_dir,
      sidecar_notify_url=ns.sidecar_notify_url,
      scaffold=bool(ns.scaffold),
      compact_ui=bool(ns.compact_ui),
      open_browser=bool(ns.open_browser),
      headless_test_steps=int(ns.headless_test_steps),
  )
