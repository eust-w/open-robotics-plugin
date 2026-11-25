"""In-process run of the packaged controller with perception/motion tracing.

Usage: PYTHONPATH=<skill>/src:<skill>/examples python trace_perception.py <task_id> [variant]
Prints what the controller believed (target/dest xy, heights) next to ground truth.
"""
import sys
import numpy as np
from robosim_gauntlet import episode, grade
import playground_tasks as pt
import color_blob_ik_controller as cbc
import rgbd_ik_perception as perc

task_id = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else "v1"
task = pt.get_task(task_id, "so101")
print("prompt:", task.prompt)

orig_cm = perc.color_mask
_seen = {}
def cm(colors, kind):
    m = orig_cm(colors, kind)
    if kind == "orange" and not _seen.get(kind):
        _seen[kind] = True
        r, g, b_ = colors[:, 0], colors[:, 1], colors[:, 2]
        fam = (r - b_ > 32.0) & (r - g > 26.0)          # shared red-family test
        cA = b_ < 0.58 * r                              # orange clause 1
        cB = (r - g) > 0.42 * np.maximum(0.0, g - b_)   # orange clause 2
        print(f"[color_mask:orange] n_fg={len(colors)} red_family={int(fam.sum())} "
              f"fam&clauseA={int((fam & cA).sum())} fam&clauseB={int((fam & cB).sum())} final_pass={int(m.sum())}")
        if fam.any():
            rf = colors[fam]
            print("   red-family px (r,g,b) p10/p50/p90:", [list(np.percentile(rf[:, i], [10, 50, 90]).round(0)) for i in range(3)])
            print("   fraction of red-family px failing clause A (b<0.58r):", round(float((~cA[fam]).mean()), 2),
                  "| failing clause B:", round(float((~cB[fam]).mean()), 2))
    return m
perc.color_mask = cm
orig_rx = perc.robust_xy
def rx(pts, *a, **k):
    out = orig_rx(pts, *a, **k)
    if len(pts):
        print(f"[robust_xy] n={len(pts)} x[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}] -> {None if out is None else np.round(out,4)}")
    else:
        print("[robust_xy] n=0")
    return out
perc.robust_xy = rx

orig_ch = perc.cluster_height
def ch(fg_p, fg_dz, t_xy, *a, **k):
    ch.last = (fg_p, fg_dz)
    return orig_ch(fg_p, fg_dz, t_xy, *a, **k)
perc.cluster_height = ch

orig_po = cbc._Arm.perceive_overhead
def po(self, plan):
    out = orig_po(self, plan)
    t_xy, d_xy, obj_h, sup_dz, plane = out
    print(f"[perceive_overhead] target_xy={None if t_xy is None else np.round(t_xy,4)} dest_xy={None if d_xy is None else np.round(d_xy,4)} "
          f"obj_h={obj_h} sup_dz={sup_dz} plane={np.round(plane,4) if plane is not None else None}")
    return out
cbc._Arm.perceive_overhead = po

orig_rw = cbc._Arm.refine_wrist
def rw(self, t_xy, plane, key, color):
    out = orig_rw(self, t_xy, plane, key, color)
    print(f"[refine_wrist] in_xy={np.round(t_xy,4)} -> out_xy={np.round(out[0],4)}")
    return out
cbc._Arm.refine_wrist = rw

orig_mx = cbc._Arm.move_xyz
def mx(self, p, jaw, **kw):
    print(f"[move_xyz] rule={kw.get('rule')} target={np.round(np.asarray(p,float),4)}")
    return orig_mx(self, p, jaw, **kw)
cbc._Arm.move_xyz = mx

def ctrl(dl):
    return cbc.run(dl, robot="so101", prompt=task.prompt, variant=variant)

r = episode.run_episode(task, ctrl, tier="T0", idx=0, base_seed=0)
p0 = grade.poses_at(r.log[0], r.body_names)
print("GROUND TRUTH target:", np.round(p0[task.target_body][:3], 4), "support:", np.round(p0[task.support_body][:3], 4))
print("success:", r.grade.success, "outcome:", r.outcome["status"])
