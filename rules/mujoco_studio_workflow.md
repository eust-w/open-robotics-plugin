---
name: mujoco-studio-workflow
trigger: model_decision
description: >-
  Workflow rules for interactive MuJoCo simulations, scenes, and Studio UI
  panels. Use only when authoring, modifying, controlling, debugging, or
  visualizing an interactive MuJoCo simulation or MuJoCo Studio panel. Don't
  use for non-MuJoCo simulations or generic web/desktop viewers that do not use
  MuJoCo.
---

# MuJoCo Studio Workflow Rules

When asked to author, modify, control, debug, or visualize an interactive
MuJoCo simulation, MuJoCo MJCF/MjSpec scene, or MuJoCo Studio UI panel (never
apply this rule or load `mujoco-studio-sidecar` or `mujoco-studio` for
non-MuJoCo simulations or generic web viewers that do not use MuJoCo; only load
`mujoco-gui` when building Dear ImGui / ImPlot UIs for MuJoCo Studio, MuJoCo
viewers, or robotics simulation/control tools, not for unrelated
general-purpose Dear ImGui desktop apps):

1.  Load the `mujoco-studio-sidecar` skill (the daemon and aux-pane workflow)
    together with the `mujoco-studio` skill (Studio plugins, messaging,
    lifecycle events, and hot-reload rules), and follow the live hot-reload
    workflow. If `mujoco-studio` or `mujoco-gui` is not pre-loaded in your
    skills list (for example, when installed from GitHub without the Marketplace
    bundle), read it from `Path(mujoco.__file__).parent / 'doc/skills'` or
    `https://github.com/google-deepmind/mujoco/tree/main/doc/skills`.
2.  Place simulation source files (`messages.py`, `sim_plugin.py`,
    `viewer_plugin.py`, and `scene.xml` only when the user wants XML) in a
    visible workspace directory (always respecting any location specified by
    the user) so the user can inspect, edit, and commit them directly. Never
    write runtime status or snapshot files inside the version-controlled
    workspace directory.
3.  Build new scenes in Python in `sim_plugin.py`, with UI controls in
    `viewer_plugin.py` for the parameters worth tweaking. Use an XML scene only
    if the user asks for one or brings one. When the user is done iterating on
    a Python scene, ask once whether to bake it into `scene.xml`.
4.  Launch the MuJoCo Studio Web daemon in the background if not already
    running for that workspace, and include
    `[MuJoCo Studio](sidecar://robotics/mujoco_studio/)` in your reply so the
    user can open the 3D viewer in one click.
5.  After editing any scene or plugin file, check `/_studio/status` (or
    `status_file` from the session file) to confirm hot-reload succeeded
    without errors.
