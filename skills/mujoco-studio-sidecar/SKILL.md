---
name: mujoco-studio-sidecar
description: >-
  Runs the MuJoCo Studio Web sidecar (studiod daemon) to stream a live,
  hot-reloading MuJoCo simulation into the Antigravity aux pane. Use
  only when MuJoCo or mujoco_studio is used to launch studiod, scaffold a Studio
  workspace (messages.py, sim_plugin.py, viewer_plugin.py, scene.xml), check
  /_studio/status, or read captured SimHistory frames; load with mujoco-studio.
  Don't use for non-MuJoCo simulations or web viewers.
metadata:
  publisher: google
---

# MuJoCo Studio Web Sidecar Workflow

This skill covers only what the sidecar adds on top of MuJoCo Studio: the
`studiod` daemon, its per-user session file and control endpoints, the
scaffolded workspace layout, scene hot-reload with state preservation, and the
aux-pane handoff of captured frames. Everything about Studio itself lives in the
MuJoCo repo's skills; load them alongside this one:

*   **`mujoco-studio`** — **required whenever you write or edit
    `sim_plugin.py`, `viewer_plugin.py`, or `messages.py`**: `SimPlugin` /
    `ViewerPlugin`, `@messages.handler`, `Snapshot` vs `Event`, the lifecycle
    event tables, `viewer.extra_geoms`, how plugin hot reload swaps classes and
    what survives it, and the threading gotchas.
*   **`mujoco-gui`** — Dear ImGui / ImPlot widget layout and tooltips for
    `viewer_plugin.py`.
*   **`mujoco-python`** and **`mujoco-spec-editing`** — `MjModel` / `MjData`
    access and procedural `MjSpec` scenes built in `sim_plugin.py`.

All seven MuJoCo skills are bundled in the Antigravity Marketplace plugin from
the MuJoCo repo's `doc/skills/`: the four above plus `mujoco-rendering`
(offscreen images), `mujoco-accelerated` (MJX / MJWarp batch simulation), and
`mujoco-parity` (CPU vs accelerated backend checks). If you installed this
plugin from GitHub (or if the installed `mujoco` version differs from the
bundle), read the skills and the manual chapters they link to
(`../../<path>.rst`) from the installed `mujoco` package under
`pathlib.Path(mujoco.__file__).parent / 'doc'` (wheels built after 5 November
2026), or install them from
`https://github.com/google-deepmind/mujoco/tree/main/doc/skills`. For older
wheels, the same files are in the MuJoCo GitHub repo under `doc/`, at a release
tag or at the commit whose `PiperOrigin-RevId` matches the `<N>` of a
`4.0.0.dev<N>` wheel.

## 1. Visible Workspace Layout

If the user specifies a location (directory path) for where the simulation
should be authored, **always** respect that location and use it as
`<workspace_dir>`. Otherwise, place simulation source files in a **visible
workspace directory** (never in `/tmp` or ephemeral scratch directories) so the
user can inspect, edit, and commit their files directly:

```text
<workspace_dir>/
├── messages.py                # Shared @dataclass(frozen=True) Event / Snapshot classes
├── sim_plugin.py              # Simulation-thread SimPlugin: builds the MjSpec scene, runs controllers
├── viewer_plugin.py           # Viewer-thread Dear ImGui ViewerPlugin: UI controls for scene parameters
└── scene.xml                  # Only when the user asks for XML, or bakes the Python scene
```

Runtime status (`status.json`) and captured context snapshots
(`latest_snapshot.json`) are stored **outside `<workspace_dir>`** under
`runtime_dir` (whose `status_file`, `snapshot_file`, and `control_port` are
recorded in the per-user session file
`${TMPDIR:-/tmp}/mujoco_studio_session_$(id -u).json`, stripping any trailing
slash on `TMPDIR`) so version-controlled workspaces stay clean.

### Build the Scene in Python First

Build new scenes in Python, in `sim_plugin.py`, by handling
`messages.LoadModelEvent`: build an `MjSpec`, call
`self.handle.set_model(spec)`, and return `True`; call `self.handle.reload()`
on `HotReloadEvent` (see `mujoco-studio`, section "Scenes Built in Python").
Keep the scene's parameters on the sim plugin and drive them from controls in
`viewer_plugin.py`, so the user can tweak the model live: each change rebuilds
the scene in place. An XML file cannot do that, because nothing writes UI
values back into it.

Use an XML scene instead only when the user asks for XML or brings an existing
scene file. Edit that file directly; saving it hot-reloads. If they then want
UI controls for some of its parameters, follow the scaffold, tune, and bake
workflow in `mujoco-studio`.

Robot models and other assets, such as a robot from MuJoCo Menagerie, stay in
their own files but are not the scene: load them in `sim_plugin.py` with
`mujoco.MjSpec.from_file(...)`, attach them to the Python scene, and pass the
files to `self.handle.set_model(spec, watch=[...])` so editing them still
rebuilds the scene.

### Offer to Bake When the Scene Settles

Once the user stops iterating on a Python scene, ask once whether to keep it in
Python or bake it into `scene.xml` (for sharing, or for tools that only read
XML); do not bake without asking. The starter panel's **Bake to scene.xml**
button writes `spec.to_xml()` next to `sim_plugin.py` and shows that file from
then on; the sim plugin yields to it because it returns `False` when
`event.path` is set. After baking, remove or slim down the scene builder in
`sim_plugin.py`, since its controls no longer change the scene.

## 2. Launching the Agent-Controlled Studio Daemon

### Prerequisite: `mujoco>=4.0.1` (`mujoco.studio`)

`mujoco.studio` and its web viewer (`mujoco-studio-wasm`) ship in
`mujoco>=4.0.1` on PyPI (Linux, macOS, and Windows, Python 3.11+). Unless a
workspace-specific launch skill provides a bundled binary, verify
`mujoco.studio` before starting `studiod.py`:

1.  Check whether `mujoco.studio` and `mujoco_studio_wasm` are already
    importable in `<venv>` (`<workspace_dir>/.venv` if present, otherwise
    `~/.venvs/mujoco-studio` or `~\.venvs\mujoco-studio` on Windows):

    ```bash
    # POSIX (Linux / macOS):
    <venv>/bin/python -c "import mujoco.studio, mujoco_studio_wasm"
    # Windows (PowerShell / cmd):
    <venv>\Scripts\python.exe -c "import mujoco.studio, mujoco_studio_wasm"
    ```

2.  If `mujoco.studio` or `mujoco_studio_wasm` is missing, install
    `mujoco>=4.0.1` into a virtual environment — reusing `<workspace_dir>/.venv`
    if one already exists, otherwise creating `~/.venvs/mujoco-studio` outside
    `<workspace_dir>` with `python3 -m venv` (or `python -m venv` on Windows) or
    `uv venv` so version-controlled simulation folders stay clean and PEP 668
    externally-managed system Python is not modified (request network access /
    `BypassSandbox: true` if sandboxed):

    ```bash
    pip install --upgrade "mujoco>=4.0.1"
    # or with uv:
    uv pip install --upgrade "mujoco>=4.0.1"
    ```

> [!CAUTION]
> **Never bridge `mujoco.experimental.studio` (3.x wheels) or import `mujoco` or
> `glfw` in `.pth` / `sitecustomize` startup hooks.**
> Do not fall back to legacy `mujoco` 3.x wheels from a stale index or mirror,
> and never write an `import ...` line in a `.pth` file or `sitecustomize.py`
> that imports `mujoco` or `glfw`. On Linux when `MUJOCO_GL` is unset,
> `import mujoco` invokes `glfw.library._glfw_get_version()`, which spawns
> `subprocess.Popen([sys.executable, '-c', ...])` at import time; a `.pth`
> startup import executes again in every spawned child before `-c` runs,
> creating an unbounded recursive process chain that exhausts all system RAM
> and swap.

### Session Discovery & Daemon Startup

Resolve the active session file portably across Linux, macOS, and Windows:

*   POSIX (`Linux` / `macOS`):
    ```bash
    SESSION_DIR="${TMPDIR:-/tmp}"
    SESSION_FILE="${SESSION_DIR%/}/mujoco_studio_session_$(id -u).json"
    ```
*   Cross-platform (including Windows `PowerShell` / `cmd` / `Git Bash`):
    ```bash
    python -c "import sys; sys.path.insert(0, r'<skill_dir>/scripts'); from studiod import discovery; print(discovery.default_session_paths()[0])"
    ```

There is one live daemon per user: starting a new one for another workspace
shuts the previous one down, and every MuJoCo Studio pane (in any conversation)
shows the current daemon. Inspect `$SESSION_FILE` (and its `status_file` or
`http://127.0.0.1:<control_port>/_studio/status` via `curl --noproxy '*'` on
POSIX or `curl.exe --noproxy "*"` on Windows) via a shell command before
launching; if a daemon is already running for the `<workspace_dir>` you need,
reuse it. Prefer shell commands over workspace-scoped file-read tools when
checking `$SESSION_FILE` or `runtime_dir` paths outside `<workspace_dir>`.

Start the daemon as a background process (`IsDaemon: true`, passing
`BypassSandbox: true` when sandboxed so loopback IPC with the sidecar is not
isolated) using the Python interpreter that has `mujoco.studio` installed,
passing `<workspace_dir>` explicitly (always using the user-specified directory
when one is given):

```bash
<python_with_mujoco_studio> <skill_dir>/scripts/studiod.py --workspace_dir=<workspace_dir>
```

When `<workspace_dir>` has no scene XML or plugin files yet, `studiod` scaffolds
starter `messages.py`, `sim_plugin.py`, and `viewer_plugin.py` files (copied
from `scripts/studiod/templates/`): a toy-drop scene built in Python, with panel
controls for its seed, drop height, and toy size, and a bake button. With no
`scene.xml`, `studiod` dispatches `LoadModelEvent(path='')` on startup and the
sim plugin builds the scene; when a scene XML exists, `studiod` passes its path
and shows the file.

Include the sidecar pill in your reply so the user can open the interactive 3D
viewer in one click:

```markdown
[MuJoCo Studio](sidecar://robotics/mujoco_studio/)
```

## 3. Live In-Process Hot-Reload

`studiod` runs Studio with `--hot_reload` enabled for the model (`model`) and
the workspace plugins (`sim_plugin.py`, `viewer_plugin.py`, and local helper
imports in `<workspace_dir>` such as `messages.py`), while adding named state
preservation across reloads. Do **not** restart the daemon when editing files
in `<workspace_dir>`:

*   **Python Scene & Model Reloads** — Saving a Python-scene `SimPlugin` (or a
    local helper module it imports) that calls `self.handle.reload()` on
    `HotReloadEvent` rebuilds its `MjSpec`; so does saving any asset file
    passed to `self.handle.set_model(spec, watch=...)`, or `scene.xml` and its
    `<include>`s when the scene is XML. On every `LoadModelEvent` ->
    `PostModelEvent` cycle, `studiod` preserves compatible named joint `qpos`
    and `qvel`, actuator `ctrl`, and simulation `time`, so tweaking geometry or
    controller code does not reset the pose.
*   **Plugins and `messages.py`** — Studio's own `--hot_reload`: the plugin
    class is swapped in place on the live instance and
    `messages.HotReloadEvent` is dispatched. The rules for what survives a
    swap (class-level defaults, helper objects keeping their old class,
    keeping `Begin`/`End` and `Push*`/`Pop*` balanced) are in the
    `mujoco-studio` skill, section "Live Hot-Reload Iteration"; follow them.

After editing any file, inspect `status_file` from `$SESSION_FILE` via a shell
command (works even in network-isolated sandboxes) or run
`curl --noproxy '*' -s http://127.0.0.1:<control_port>/_studio/status`
(`BypassSandbox: true` when sandboxed) to verify `"ok": true` across `scene`,
`messages`, `sim_plugin`, and `viewer_plugin`. A failed reload keeps the last
working code and reports the traceback there. Changes that need a restart, such
as renaming a plugin class, are only logged by Studio, so restart the daemon
after those.

## 4. Reading User Interactions & Scrubbed Physics Context

When the user scrubs the `SimHistory` timeline, selects or perturbs a body in
the 3D viewport, or clicks **Simulation -> Capture Frame for Agent Context**
(and optionally clicks **Send to Agent Chat** in the sidecar toast), inspect
`snapshot_file` from `$SESSION_FILE` via a shell command or query
`curl --noproxy '*' -s http://127.0.0.1:<control_port>/_studio/context`.

To capture a fresh snapshot programmatically from the running daemon at any
time, read `control_port` from `$SESSION_FILE` and call (passing `--noproxy '*'`
with `curl`, and `BypassSandbox: true` if sandboxed):

*   `GET http://127.0.0.1:<control_port>/_studio/context?fresh=1`
*   `POST http://127.0.0.1:<control_port>/_studio/context/capture` with
    `{"note": "..."}`
