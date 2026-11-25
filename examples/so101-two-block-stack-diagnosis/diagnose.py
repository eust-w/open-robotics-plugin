"""Replay one SO-101 two_block_stack episode and print block poses over time.

Usage (env as in the sim2real-gauntlet run record):
  PYTHONPATH=<skill>/src:<skill>/examples python diagnose.py <variant> <tier> <idx> [<task>]
"""
import math, shlex, sys, json, pathlib
import numpy as np
from robosim_gauntlet import episode, grade
import playground_tasks as pt

variant, tier, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
task_id = sys.argv[4] if len(sys.argv) > 4 else "two_block_stack"
robot = "so101"
skd = pathlib.Path(pt.__file__).parent
cmd = shlex.split(f"{sys.executable} {skd/'color_blob_ik_controller.py'} --robot={robot} --task_id={task_id} --variant={variant}")
task = pt.get_task(task_id, robot)
r = episode.run_episode(task, cmd, tier=tier, idx=idx, base_seed=0)
names = r.body_names
print("outcome:", r.outcome, "| success:", r.grade.success, "| violations:", r.grade.violations)
n = len(r.log)
print("ticks:", n, "settle_start:", r.settle_start)
def tilt(q):
    w, x, y, z = q
    return math.degrees(math.acos(max(-1, min(1, 1 - 2 * (x * x + y * y)))))
rows = []
for i in sorted(set(list(range(0, n, max(1, n // 24))) + [r.settle_start - 1, n - 1])):
    p = grade.poses_at(r.log[i], names)
    t, b = p[task.target_body], p[task.support_body]
    rows.append(dict(tick=i, top=[round(float(v), 4) for v in t[:3]], base=[round(float(v), 4) for v in b[:3]],
                     dxy_mm=round(1000 * float(np.hypot(*(t[:2] - b[:2]))), 1), dz_mm=round(1000 * float(t[2] - b[2]), 1),
                     top_tilt=round(tilt(t[3:7]), 1), base_tilt=round(tilt(b[3:7]), 1)))
for row in rows: print(json.dumps(row))
p0 = grade.poses_at(r.log[0], names); pf = grade.poses_at(r.log[-1], names)
print("initial top/base xyz:", np.round(p0[task.target_body][:3], 4), np.round(p0[task.support_body][:3], 4))
print("final   top/base xyz:", np.round(pf[task.target_body][:3], 4), np.round(pf[task.support_body][:3], 4))
print("stacked() predicate:", task.goal_met(pf) if hasattr(task, "goal_met") else "n/a")

# --- gripper trace -----------------------------------------------------
print("\nmeas keys:", [k for k in r.log[1].meas.keys()][:14])
jk = [k for k in r.log[1].meas.keys() if "joint_position" in k]
print("joint keys:", jk)
grip_key = [k for k in jk if "grip" in k or "jaw" in k]
print("gripper key candidates:", grip_key)
if grip_key:
    gk = grip_key[0]
    g = [float(np.asarray(t.meas[gk]).ravel()[0]) for t in r.log]
    for i in range(0, n, max(1, n // 40)):
        print(i, "gripper", round(g[i], 3))

# --- end-effector vs target at close time --------------------------------
cand = [nm for nm in names if nm.startswith("robot/") and ("jaw" in nm or "gripper" in nm or "wrist" in nm)]
print("\nrobot link candidates:", cand)
tk = next((i for i in range(n) if g[i] < 60), None) if grip_key else None
print("gripper first < 60 at tick", tk)
if tk is not None:
    p = grade.poses_at(r.log[tk], names)
    tgt = p[task.target_body][:3]
    print("target xyz at close:", np.round(tgt, 4))
    for nm in cand:
        q = p[nm][:3]
        print(f"{nm:40s} xyz={np.round(q,4)}  dxy_to_target_mm={1000*np.hypot(*(q[:2]-tgt[:2])):.1f}  dz_mm={1000*(q[2]-tgt[2]):.1f}")
