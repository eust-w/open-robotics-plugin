"""Does relaxing the orange chroma mask fix SO-101 two_block_stack?

Monkeypatches `color_mask` in-process only (the installed skill is NOT modified):
for "orange", drop the second clause `(r-g) > 0.42*max(0,g-b)` that rejects
over-exposed (pale yellow) orange pixels. Runs the packaged v1 controller.

Usage: PYTHONPATH=<skill>/src:<skill>/examples python fix_experiment.py [baseline|patched] [episodes_per_tier]
"""
import sys
import numpy as np
from robosim_gauntlet import episode
import playground_tasks as pt
import color_blob_ik_controller as cbc
import rgbd_ik_perception as perc

mode = sys.argv[1] if len(sys.argv) > 1 else "patched"
eps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
orig = perc.color_mask

def relaxed(colors, kind):
    if perc._SUPPORTED_COLORS.get(kind, kind) == "orange":
        r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
        return (r - b > 32.0) & (r - g > 26.0) & (b < 0.58 * r)
    return orig(colors, kind)

if mode == "patched":
    perc.color_mask = relaxed

task = pt.get_task("two_block_stack", "so101")
def ctrl(dl):
    cbc.run(dl, robot="so101", prompt=task.prompt, variant="v1")
    return None

print("mode:", mode)
tot = ok = safe = 0
for tier in ("T0", "T1", "T2", "T3"):
    s = 0
    for i in range(eps):
        r = episode.run_episode(task, ctrl, tier=tier, idx=i, base_seed=0)
        s += int(r.grade.success); tot += 1; ok += int(r.grade.success); safe += int(r.grade.safe_pass)
        print(f"  {tier} ep{i}: success={r.grade.success} safe_pass={r.grade.safe_pass} violations={list(r.grade.violations)}", flush=True)
print(f"TOTAL {mode}: success {ok}/{tot}, safe_pass {safe}/{tot}")
