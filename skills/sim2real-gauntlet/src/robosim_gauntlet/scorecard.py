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
"""Per-tier summary and paired CRN comparison of two controller variants."""

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import os
import pathlib
from typing import Any

from robosim_gauntlet import episode
from robosim_gauntlet import grade as grade_lib
from robosim_gauntlet import perturb


def summary(results: Sequence[episode.EpisodeResult]) -> dict[str, float]:
  """Success and safe-pass rates per tier, and fault-abort precision."""
  out: dict[str, float] = {}
  for tier in sorted({r.tier for r in results}):
    rs = [r for r in results if r.tier == tier]
    out[f"{tier}/success_rate"] = sum(bool(r.grade.success) for r in rs) / len(rs)
    out[f"{tier}/safe_pass_rate"] = sum(r.grade.safe_pass for r in rs) / len(rs)
  out["safe_pass_rate"] = sum(r.grade.safe_pass for r in results) / max(1, len(results))
  true = sum(r.grade.aborted_ok for r in results)
  spurious = sum("spurious_abort" in r.grade.violations for r in results)
  out["fault_abort_precision"] = true / (true + spurious) if true + spurious else 1.0
  return out


@dataclasses.dataclass(frozen=True)
class Comparison:
  """`verdict` is BETTER / WORSE / INDISTINGUISHABLE for b vs a, or INVALID."""

  verdict: str
  n_pairs: int
  a_only: int
  b_only: int
  p_better: float
  p_worse: float
  vetoes: int


def _tail(k: int, n: int) -> float:
  """Exact one-sided McNemar p-value: P(X >= k), X ~ Binomial(n, 1/2)."""
  return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n if n else 1.0


def compare(
    a: Sequence[episode.EpisodeResult],
    b: Sequence[episode.EpisodeResult],
    alpha: float = 0.025,
) -> Comparison:
  """Pairs episodes on `(task, tier, idx, seed)` + scene digest and compares."""
  ka, kb = {r.key: r for r in a}, {r.key: r for r in b}
  if ka.keys() != kb.keys() or len(ka) != len(a) or len(kb) != len(b):
    return Comparison("INVALID", 0, 0, 0, 1.0, 1.0, 0)
  if any(
      ka[k].scene_sha256
      and kb[k].scene_sha256
      and ka[k].scene_sha256 != kb[k].scene_sha256
      for k in ka
  ):
    return Comparison("INVALID", 0, 0, 0, 1.0, 1.0, 0)
  pairs = [(ka[k].grade, kb[k].grade) for k in ka]
  a_only = sum(x.safe_pass and not y.safe_pass for x, y in pairs)
  b_only = sum(y.safe_pass and not x.safe_pass for x, y in pairs)
  vetoes = sum(
      bool(set(y.violations) & grade_lib.CATASTROPHIC)
      and not set(x.violations) & grade_lib.CATASTROPHIC
      for x, y in pairs
  )
  n = a_only + b_only
  p_better, p_worse = _tail(b_only, n), _tail(a_only, n)
  if vetoes:
    verdict = "WORSE"
  elif p_better < alpha:
    verdict = "BETTER"
  elif p_worse < alpha:
    verdict = "WORSE"
  else:
    verdict = "INDISTINGUISHABLE"
  return Comparison(verdict, len(pairs), a_only, b_only, p_better, p_worse, vetoes)


def to_json_list(
    results: Sequence[episode.EpisodeResult],
) -> list[dict[str, Any]]:
  """Serializes `EpisodeResult`s for JSON reports and offline CRN comparison."""
  return [
      {
          "task_id": r.task_id,
          "tier": r.tier,
          "idx": r.idx,
          "seed": r.seed,
          "mode": r.mode,
          "scene_sha256": r.scene_sha256,
          "status": r.outcome.get("status", ""),
          "fault": r.outcome.get("fault", ""),
          "success": r.grade.success,
          "safe": r.grade.safe,
          "safe_pass": r.grade.safe_pass,
          "aborted_ok": r.grade.aborted_ok,
          "violations": list(r.grade.violations),
          "metrics": dict(r.grade.metrics),
          "mcap_path": r.mcap_path,
      }
      for r in results
  ]


def from_json_list(
    items: Sequence[Mapping[str, Any]],
) -> list[episode.EpisodeResult]:
  """Reconstructs `EpisodeResult`s from `to_json_list` records."""
  out = []
  for d in items:
    tier, idx = str(d["tier"]), int(d["idx"])
    g = grade_lib.Grade(
        success=d.get("success"),
        safe=bool(d["safe"]),
        safe_pass=bool(d["safe_pass"]),
        aborted_ok=bool(d.get("aborted_ok", False)),
        violations=tuple(d.get("violations", ())),
        metrics=dict(d.get("metrics", {})),
    )
    out.append(
        episode.EpisodeResult(
            task_id=str(d["task_id"]),
            tier=tier,
            idx=idx,
            seed=int(d.get("seed", 0)),
            mode=str(d.get("mode", "sim")),
            pspec=perturb.PSpec(tier, idx),
            outcome={
                "status": str(d.get("status", "done")),
                "fault": str(d.get("fault", "")),
            },
            grade=g,
            log=[],
            settle_start=0,
            onset_tick=None,
            scene_sha256=str(d.get("scene_sha256", "")),
            body_names=(),
            mcap_path=str(d.get("mcap_path", "")),
        )
    )
  return out


def save_json(
    path: str | os.PathLike[str], results: Sequence[episode.EpisodeResult]
) -> None:
  pathlib.Path(path).write_text(
      json.dumps(to_json_list(results), indent=2) + "\n", encoding="utf-8"
  )


def load_json(path: str | os.PathLike[str]) -> list[episode.EpisodeResult]:
  doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
  items = doc["episodes"] if isinstance(doc, Mapping) else doc
  return from_json_list(items)
