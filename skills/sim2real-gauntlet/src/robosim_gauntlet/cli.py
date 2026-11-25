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
"""Turnkey CLI and dynamic `--task-py` loader for `robosim_gauntlet`."""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import importlib.machinery
import importlib.util
import json
import os
import pathlib
import shlex
import sys
from typing import Any

from absl import app
from absl import flags

_SRC = pathlib.Path(__file__).absolute().parents[1]
if str(_SRC) not in sys.path:
  sys.path.insert(0, str(_SRC))


# pylint: disable=g-import-not-at-top,g-bad-import-order
import robosim_gauntlet
from robosim_gauntlet import episode
from robosim_gauntlet import preflight
from robosim_gauntlet import scorecard
from robosim_gauntlet import task as task_lib
# pylint: enable=g-import-not-at-top,g-bad-import-order

ensure_gl_backend = robosim_gauntlet.ensure_gl_backend

_TASK_PY = flags.DEFINE_string("task_py", "", "Path `<file.py>[:callable]`.")
_ROBOT = flags.DEFINE_string("robot", "yam", "Robot name passed to task factory.")
_TASKS = flags.DEFINE_list("tasks", [], "Optional subset of task_ids to run.")
_CONTROLLER = flags.DEFINE_string("controller", "", "Controller command line.")
_COMPARE = flags.DEFINE_string("compare_controller", "", "Optional v0 baseline.")
_MODE = flags.DEFINE_enum("mode", "sim", ["sim", "real"], "Execution mode.")
_TIERS = flags.DEFINE_list("tiers", ["T0", "T1", "T2", "T4"], "Tiers to run.")
_EPISODES = flags.DEFINE_integer("episodes", 1, "Episodes per tier.")
_SEED = flags.DEFINE_integer("seed", 0, "CRN base seed.")
_MCAP_DIR = flags.DEFINE_string("mcap_log_dir", "", "Optional .mcap output dir.")
_PREFLIGHT_ONLY = flags.DEFINE_bool("preflight_only", False, "Audit tasks only.")
_OUT_JSON = flags.DEFINE_string("output_json", "", "Optional JSON output path.")
_BASE_JSON = flags.DEFINE_string(
    "baseline_json", "", "Optional saved baseline JSON report for CRN compare."
)
_CAND_JSON = flags.DEFINE_string(
    "candidate_json",
    "",
    "Optional saved candidate JSON report for offline CRN.",
)


def _as_tasks(raw: Any) -> list[task_lib.Task]:
  if isinstance(raw, Mapping):
    return list(raw.values())
  if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
    return list(raw)
  return [raw]


def load_tasks(
    spec_str: str, *, robot: str = "", task_ids: Sequence[str] = ()
) -> list[task_lib.Task]:
  """Loads `Task` objects from `<path.py>[:callable]` with zero gauntlet deps."""
  path_part, _, attr = spec_str.partition(":")
  path = pathlib.Path(path_part).expanduser()
  if not path.is_file():
    raise FileNotFoundError(f"Task module not found: {path}")
  parent = str(path.parent.resolve())
  if parent not in sys.path:
    sys.path.insert(0, parent)
  loader = importlib.machinery.SourceFileLoader(f"_task_{path.stem}", str(path))
  mod_spec = importlib.util.spec_from_loader(loader.name, loader)
  if mod_spec is None or mod_spec.loader is None:
    raise ImportError(f"Cannot load module spec for {path}")
  mod = importlib.util.module_from_spec(mod_spec)
  sys.modules[mod_spec.name] = mod
  mod_spec.loader.exec_module(mod)
  ns = vars(mod)
  if attr:
    target = ns[attr]
    raw = target(robot) if callable(target) and robot else (
        target() if callable(target) else target
    )
    items = _as_tasks(raw)
  elif "build_tasks" in ns and callable(ns["build_tasks"]):
    items = _as_tasks(ns["build_tasks"](robot) if robot else ns["build_tasks"]())
  elif "get_task" in ns and "TASK_IDS" in ns:
    ids = task_ids or tuple(ns["TASK_IDS"])
    items = [ns["get_task"](tid, robot) if robot else ns["get_task"](tid) for tid in ids]
  elif "TASKS" in ns:
    items = _as_tasks(ns["TASKS"])
  else:
    raise ValueError(
        f"{path} must define `:callable`, `build_tasks`, `get_task`+`TASK_IDS`, or `TASKS`."
    )
  if task_ids:
    want = set(task_ids)
    items = [t for t in items if t.task_id in want]
  return items


def normalize_argv(argv: Sequence[str]) -> list[str]:
  """Accepts both `--task-py` and `--task_py` flag spellings."""
  out = []
  for arg in argv:
    if arg.startswith("--") and len(arg) > 2 and not arg.startswith("---"):
      head, sep, tail = arg[2:].partition("=")
      arg = "--" + head.replace("-", "_") + sep + tail
    out.append(arg)
  return out


def execute(
    *, task_py: str = "", robot: str = "yam", task_ids: Sequence[str] = (),
    controller: str = "", compare_controller: str = "",
    baseline_json: str = "", candidate_json: str = "", mode: str = "sim",
    tiers: Sequence[str] = ("T0",), episodes_per_tier: int = 1, seed: int = 0,
    mcap_log_dir: str = "", preflight_only: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Runs preflight, a gauntlet sweep, or an offline CRN JSON comparison."""
  if baseline_json and candidate_json and not controller and not preflight_only:
    base, cand = scorecard.load_json(baseline_json), scorecard.load_json(candidate_json)
    return {
        "mode": "compare_json", "ok": True,
        "baseline_summary": scorecard.summary(base),
        "summary": scorecard.summary(cand),
        "comparison": dataclasses.asdict(scorecard.compare(base, cand)),
    }
  tasks = load_tasks(task_py, robot=robot, task_ids=task_ids)
  audits = {t.task_id: preflight.run(t) for t in tasks} if mode == "sim" else {}
  report: dict[str, Any] = {
      "mode": mode, "robot": robot, "tasks": [t.task_id for t in tasks],
      "preflight": audits, "ok": all(not v for v in audits.values()),
  }
  if preflight_only:
    return report
  if not controller:
    raise ValueError("--controller is required unless --preflight_only is set.")
  eps = {tier: episodes_per_tier for tier in tiers} if mode == "sim" else {"T0": 1}
  run_env = env or {
      k: v for k, v in os.environ.items()
      if k.startswith(("RUNFILES", "TEST_")) or k == "MUJOCO_GL"
  }
  res = episode.run_suite(
      tasks, shlex.split(controller), episodes=eps, base_seed=seed,
      mode=mode, env=run_env, mcap_dir=mcap_log_dir or None,
  )
  report["summary"] = scorecard.summary(res) if mode == "sim" else {}
  report["episodes"] = scorecard.to_json_list(res)
  if baseline_json and mode == "sim":
    base = scorecard.load_json(baseline_json)
    report["comparison"] = dataclasses.asdict(scorecard.compare(base, res))
  elif compare_controller and mode == "sim":
    base = episode.run_suite(
        tasks, shlex.split(compare_controller), episodes=eps,
        base_seed=seed, mode=mode, env=run_env,
    )
    report["comparison"] = dataclasses.asdict(scorecard.compare(base, res))
  return report


def main(argv: Sequence[str]) -> None:
  del argv
  report = execute(
      task_py=_TASK_PY.value, robot=_ROBOT.value, task_ids=_TASKS.value,
      controller=_CONTROLLER.value, compare_controller=_COMPARE.value,
      baseline_json=_BASE_JSON.value, candidate_json=_CAND_JSON.value,
      mode=_MODE.value, tiers=_TIERS.value, episodes_per_tier=_EPISODES.value,
      seed=_SEED.value, mcap_log_dir=_MCAP_DIR.value,
      preflight_only=_PREFLIGHT_ONLY.value,
  )
  text = json.dumps(report, indent=2)
  if _OUT_JSON.value:
    pathlib.Path(_OUT_JSON.value).write_text(text + "\n", encoding="utf-8")
  print(text, flush=True)
  if _PREFLIGHT_ONLY.value and not report["ok"]:
    sys.exit(1)


def run() -> None:
  app.run(main, flags_parser=lambda a: flags.FLAGS(normalize_argv(a)))


if __name__ == "__main__":
  run()
