"""Render a third-person clip of a packaged sim2real-gauntlet episode.

Runs the packaged v1 controller in-process (tier T0, idx 0, seed 0), hooks the
controller's per-tick `act` to render the simulator state with a free camera, and
writes PNG frames. The installed skill is not modified; everything is patched in
the Python process only.

Usage:
  PYTHONPATH=<skill>/src:<skill>/examples python render_clip.py \
      <robot yam|so101> <task_id> <out_dir> [--patched-orange] [--every 2] [--size 640x360]
"""
import argparse, pathlib
import numpy as np
import mujoco
from mujoco_playground import wrapper
from robosim_gauntlet import episode
import playground_tasks as pt
import color_blob_ik_controller as cbc
import rgbd_ik_perception as perc

ap = argparse.ArgumentParser()
ap.add_argument("robot"); ap.add_argument("task_id"); ap.add_argument("out_dir")
ap.add_argument("--patched-orange", action="store_true")
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--size", default="640x360")
ap.add_argument("--azimuth", type=float, default=140.0)
ap.add_argument("--elevation", type=float, default=-24.0)
ap.add_argument("--distance", type=float, default=None)
ap.add_argument("--tier", default="T0")
a = ap.parse_args()
W, H = map(int, a.size.split("x"))
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

if a.patched_orange:
    orig = perc.color_mask
    def relaxed(colors, kind):
        if perc._SUPPORTED_COLORS.get(kind, kind) == "orange":
            r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
            return (r - b > 32.0) & (r - g > 26.0) & (b < 0.58 * r)
        return orig(colors, kind)
    perc.color_mask = relaxed

task = pt.get_task(a.task_id, a.robot)
state = {}
cls = type(task)
orig_build = cls.build_device_layer
def build(self, *args, **kw):
    dl, reset, close = orig_build(self, *args, **kw)
    raw = dl.unwrapped if isinstance(dl, wrapper.DeviceLayerWrapper) else dl
    state["sim"] = raw.sim
    return dl, reset, close
cls.build_device_layer = build  # PlaygroundTask is frozen, so patch the class

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth, cam.elevation = a.azimuth, a.elevation
cam.distance = a.distance or (1.2 if a.robot == "yam" else 0.62)
cam.lookat[:] = (0.0, -0.01, 0.88) if a.robot == "yam" else (0.0, 0.0, 0.80)
opt = mujoco.MjvOption()
count = {"i": 0, "saved": 0}
orig_act = cbc._Arm.act
def act(self, q_arm, jaw):
    res = orig_act(self, q_arm, jaw)
    count["i"] += 1
    sim = state.get("sim")
    if sim is not None and count["i"] % a.every == 0:
        if "r" not in state:
            state["r"] = mujoco.Renderer(sim.mj_model, H, W)
            state["d"] = mujoco.MjData(sim.mj_model)
        mujoco.mj_copyData(state["d"], sim.mj_model, sim.mj_data)
        mujoco.mj_forward(sim.mj_model, state["d"])
        state["r"].update_scene(state["d"], camera=cam, scene_option=opt)
        from PIL import Image
        Image.fromarray(state["r"].render()).save(out / f"{count['saved']:05d}.png")
        count["saved"] += 1
    return res
cbc._Arm.act = act

def ctrl(dl):
    cbc.run(dl, robot=a.robot, prompt=task.prompt, variant="v1")
    return None

r = episode.run_episode(task, ctrl, tier=a.tier, idx=0, base_seed=0)
print(f"frames={count['saved']} ticks={count['i']} success={r.grade.success} safe_pass={r.grade.safe_pass} violations={list(r.grade.violations)}")
