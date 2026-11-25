---
name: mujoco-asset-standards
trigger: model_decision
description: >-
  Authoritative standards and best practices for constructing, normalizing, and
  validating MuJoCo 3D assets (MJCF XML and MjSpec). Use only when the user
  explicitly names MuJoCo, MJCF, .mjz, or MjSpec/mjSpec while building,
  importing, editing, or reviewing rigid and articulated objects, containers,
  tools, furniture, mechanisms, collision geometry, materials, affordance sites,
  or physical parameters in MuJoCo. Don't use for non-MuJoCo 3D assets or
  STEP/SDF/URDF/OpenUSD/USDZ/GLB/CAD-to-XML converters, furniture/ergonomics
  linters, or generic Blender scripts where MuJoCo is not explicitly named.
---
<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# MuJoCo Asset Construction & Best Practices

> [!IMPORTANT]
>
> **Context-Window Router**:
>
> *   **Creating a new asset from scratch in Blender?** Read
>     [`blender-creation/SKILL.md`](../skills/blender-creation/SKILL.md)
>     instead—it embeds the canonical MJCF skeleton and `audit_scenarios`
>     checks. Consult this `rules/mujoco_asset_standards.md` reference **only on
>     demand** if a specific physics or mechanism check fails.
> *   **Auditing & fixing an existing asset (`mujoco-audit/SKILL.md`)?** Run
>     `audit_scenarios` first
>     ([`mujoco-audit/SKILL.md`](../skills/mujoco-audit/SKILL.md) §2) and look
>     up only the specific section(s) of `rules/mujoco_asset_standards.md`
>     corresponding to any failing check.

> [!TIP]
>
> **Related MuJoCo Documentation & Plugin Skills:**
>
> *   [MuJoCo Python Bindings][mj-python] & [`MjSpec` Editing API][mjspec-docs]
>     — Procedural model authoring, simulation lifecycle (`MjSpec` $\rightarrow$
>     `MjModel` $\rightarrow$ `MjData`), `mj_step`, `mj_forward`, named access
>     (`model.geom('name')`, `data.body('name')`), `bind()`, default classes
>     (`MjsDefault`), and sub-spec attachments (`add_frame`, `attach`,
>     `add_freejoint`).
> *   [MuJoCo Offscreen Rendering][mj-vis] — Headless offscreen rendering
>     (`mujoco.Renderer`, RGB, depth, segmentation masks, `MjvOption`,
>     `MUJOCO_GL=egl` / `osmesa`) and Filament PBR (`mjrf`).
> *   **Running Bundled & Scratch Python Scripts (`uv run`)** — Bundled skill
>     scripts ([`audit_scenarios.py`][audit-script] and
>     [`render_showcase_shot.py`][s-shot]) include PEP 723 `# /// script`
>     dependency headers, so `uv run <path/to/script.py> ...` automatically
>     provisions and caches an isolated environment in `~/.cache/uv` without
>     modifying the plugin directory or system Python. For custom scratch Python
>     scripts (e.g. `<ArtifactDir>/scratch/audit.py`), either include a PEP 723
>     `# /// script` header at the top of the script and invoke
>     `uv run <ArtifactDir>/scratch/audit.py`, or pass `--with` (e.g. `uv run
>     --with mujoco --with numpy --with Pillow --with scipy --with trimesh
>     <script.py>`). *(If `uv` is not installed, install via `brew install uv`
>     or `curl -LsSf https://astral.sh/uv/install.sh | sh`, or create a user
>     venv outside the repo at `~/.venvs/mujoco` and install `mujoco`,
>     `numpy`, `Pillow`, `scipy`, `trimesh`, `absl-py`, and `av`.)*
> *   [`blender-creation`](../skills/blender-creation/SKILL.md) — **Read this
>     skill whenever authoring or modifying 3D visual/collision meshes from
>     scratch** (from reference images, text descriptions, or specs). Covers
>     sculptural & procedural Blender 4.2+ (`bpy` / `bmesh`) modeling with the
>     bundled `blender_mesh_utils.py` (`bmu`) and `asset_build_utils.py` (`abu`)
>     helpers in `skills/blender-creation/scripts/`, Cycles 3D procedural shader
>     baking, PBR texture synthesis, co-authored convex-hull collision meshes,
>     standalone MJZ-ready variants, and the `render_6view_inspection_sheet`,
>     `render_side_by_side_comparison`, and `render_wireframe_and_uv_sheet`
>     helpers in [`render_showcase_shot.py`][s-shot].
> *   [`mujoco-audit`](../skills/mujoco-audit/SKILL.md) — **Read this skill
>     whenever auditing, validating, triaging, or remediating an asset** (or
>     verifying a newly built asset). Covers prompt-driven test selection
>     (`--scenarios=...` and `--static_only`), the 7-point structural checklist,
>     the 10-scenario auto-scaling physical test suite
>     ([`audit_scenarios.py`][audit-script]), engineering-judgment rules, and
>     the Before-vs-After remediation workflow.
> *   [`mujoco-showcase-shot`](../skills/mujoco-showcase-shot/SKILL.md) —
>     Studio-lit 360° Filament PBR turntable video (`.mp4`), `2×2` keyframe
>     sheet, and multi-view inspection/comparison sheet generator
>     ([`render_showcase_shot.py`][s-shot]).

Follow these standards whenever constructing, converting, editing, or reviewing
a 3D asset in MuJoCo (MJCF XML or `mujoco.MjSpec`). Every compliant asset is
self-contained, physically plausible, visually inspectable in MuJoCo Studio, and
ready for drop-in composition across scenes and manipulation tasks.

--------------------------------------------------------------------------------

## 1. Quick-Reference Standards Matrix

| Dimension        | Standard Requirement                 | Common             |
:                  :                                      : Anti-Pattern to    :
:                  :                                      : Avoid              :
| :--------------- | :----------------------------------- | :----------------- |
| **Units,         | Strictly SI: meters (`m`), kilograms | Leaving CAD meshes |
: Real-World       : (`kg`), seconds (`s`), and           : in millimeters     :
: Dimensions &     : `<compiler angle="radian"/>`. Verify : (`1000x`           :
: Angles**         : AABB dimensions against product      : oversized),        :
:                  : specs or real-world plausibility and : keeping            :
:                  : apply `<mesh scale="sx sy sz"/>`     : uncalibrated raw   :
:                  : (plus primitive geom scaling) when   : 3D scan scales     :
:                  : meshes are mis-scaled.               : (e.g. `2x` too     :
:                  :                                      : large), or mixing  :
:                  :                                      : degree/radian      :
:                  :                                      : conventions.       :
| **Up & Front     | **`+Z` is up**, **`+X` is front      | Leaving            |
: Axes**           : (forward)**, **`-X` is back**,       : GLTF/USD/Blender   :
:                  : **`+Y` is left**, **`-Y` is right**  : orientations       :
:                  : (`UP_POSZ_FORWARD_POSX_RIGHT_NEGY`). : unbaked (`-Y`      :
:                  : Origin `(0, 0, 0)` sits sensibly at  : front or $+Y$ up), :
:                  : the base, centroid, or pivot (not    : or a stray CAD     :
:                  : wildly offset outside the AABB).     : offset far from    :
:                  :                                      : `(0, 0, 0)`.       :
| **Top-Level Body | `worldbody.first_body()` has a       | Leaving exporter   |
: & Semantic       : **descriptive semantic name** (e.g., : names              :
: Naming**         : `wooden_end_table`), **zero          : (`blender_export`, :
:                  : transform** (`pos="0 0 0"`, `quat="1 : `root`, `model`,   :
:                  : 0 0 0"`), and **no `<freejoint/>`**  : `object`,          :
:                  : in `model.xml`.                      : `visual_00`),      :
:                  :                                      : non-zero `pos` /   :
:                  :                                      : non-identity       :
:                  :                                      : `quat`, or         :
:                  :                                      : `<freejoint/>` on  :
:                  :                                      : the reusable       :
:                  :                                      : asset's root body. :
| **Visual Geoms** | **`class="visual"` (`group="2"`)**,  | Leaving `density`  |
:                  : `contype="0"`, `conaffinity="0"`,    : unset on           :
:                  : **`density="0" mass="0"`**. Visible  : `contype="0"`      :
:                  : by default (`2` in Studio).          : visual meshes      :
:                  :                                      : (defaults to       :
:                  :                                      : $1000\text{        :
:                  :                                      : kg/m}^3$ and       :
:                  :                                      : double-counts      :
:                  :                                      : mass).             :
| **Collision      | **`class="collision"`                | Setting `rgba="... |
: Geoms**          : (`group="3"`)**, `contype="1"`,      : 0"` (invisible     :
:                  : `conaffinity="1"`, carries physical  : even when toggling :
:                  : `density`/`mass`, **visible          : group `3`) or      :
:                  : semi-transparent `rgba`** ($\alpha   : leaving colliders  :
:                  : \in [0.35, 0.6]$).                   : in                 :
:                  :                                      : `group="0"`/`"1"`. :
| **Affordance &   | Preserve physical affordances        | Capping            |
: Semantic         : (hollow cavities, handle loops,      : container/handle   :
: Geometry**       : articulated joints); optional        : openings with a    :
:                  : semantic `<site>`s in `group="4"`    : single convex      :
:                  : when task-relevant ($\alpha \in      : hull, or hiding    :
:                  : [0.25, 0.5]$).                       : `<site>`s          :
:                  :                                      : (`group="4"`) with :
:                  :                                      : `rgba` alpha $=    :
:                  :                                      : 0$.                :
| **Mass, CoM &    | Distribute mass explicitly across    | Using `<compiler   |
: Inertia**        : `group="3"` colliders proportional   : settotalmass=      :
:                  : to volume (or via explicit           : "..."/>`, putting  :
:                  : `<inertial/>`); per-link mass, CoM,  : `mass` on visual   :
:                  : and principal inertias must match    : geoms, or adding   :
:                  : real-world geometry.                 : invisible/interior :
:                  :                                      : ballast `box`      :
:                  :                                      : geoms.             :
| **Collision      | Analytic primitives for genuinely    | Colliding raw      |
: Fidelity**       : geometric parts; natively authored   : high-poly visual   :
:                  : convex hulls (`maxhullvert="64"`)    : meshes, capping    :
:                  : for rounded, contoured, or curved    : cavities, or       :
:                  : parts matching the true silhouette.  : approximating      :
:                  :                                      : rounded/curved     :
:                  :                                      : shapes with blocky :
:                  :                                      : `box` primitives.  :
| **Contacts &     | MJCF `condim` (`1`, `3`, `4`, or     | Relying on         |
: Friction**       : `6`) and explicit `friction`         : near-zero default  :
:                  : (`[sliding, torsional, rolling]`)    : torsional/rolling  :
:                  : calibrated to the real material &    : friction when      :
:                  : rolling regime; `solref` with        : grasp/roll damping :
:                  : `timeconst >= 2 * timestep` and      : is needed, or      :
:                  : `dampratio="1"`, `margin="0"`.       : `timeconst < 2*dt`.:
| **Joints,        | Pivots (`pos`) and axes (`axis`)     | Wrong hinge/slide  |
: Articulation &   : aligned with true mechanics;         : axis, missing      :
: Integrator**     : explicit `range`, `damping`,         : joint limits, zero :
:                  : `armature`, `frictionloss`; use      : damping/armature,  :
:                  : `integrator="discrete"` in scenes.   : or self-colliding  :
:                  :                                      : adjacent links.    :

--------------------------------------------------------------------------------

## 2. Directory Layout & File Separation

### 2.1 Self-Contained Asset Package & Separate-Folder Variants (`.mjz`-Ready)

Store each asset (and each color/material variant) in its own self-contained
directory with local relative paths:

```text
<asset_name>/
├── model.xml     # Pure asset definition (NO floor, skybox, or scene lights)
├── meshes/       # Visual (.obj) and convex hull (.obj/.stl) meshes
└── textures/     # 8-bit PNG maps (basecolor, normal, roughness/orm)
```

*   **Separate MJZ-ready directories for color/material and mechanical variants
    (never `model_<variant>.xml`)**: When generating color, finish, or trim
    variants of an asset (e.g., green, blue, pink, white) or a coupled
    mechanical variant (`<asset_name>_mechanical/`), emit each variant as its
    own **separate, self-contained asset directory** named
    `<asset_name>_<variant>/` containing `model.xml` (with
    `<mujoco model="<asset_name>_<variant>">` and root body
    `<body name="<asset_name>_<variant>">`), `meshes/`, and `textures/`. Never
    place multiple `model_<variant>.xml` files inside a single shared asset
    folder, so every variant directory can be zipped directly into a standalone
    `.mjz` archive (`<asset_name>.mjz`).
*   **Pure `model.xml` (no `scene.xml` or `<include/>`)**: `model.xml` defines
    **only** the asset's `<compiler>`, `<default>`, `<asset>`, `<worldbody>`
    (single root body), `<contact>`, `<equality>`, `<tendon>`, `<actuator>`, and
    `<sensor>`—never ground planes, skyboxes, or global `<visual>` blocks. Build
    test and evaluation scenes programmatically in Python with `mujoco.MjSpec`
    (`add_frame` + `attach` + `add_freejoint`; see §3.3 and
    [`MjSpec` Sub-model Composition & Attachments][mjspec-docs]).
*   **Compiler paths & no external traversal**: Declare `<compiler angle="radian"
    meshdir="meshes" texturedir="textures" autolimits="true"/>` and never use
    absolute paths (`/path/to/...`) or parent traversals (`../`) inside
    `model.xml`.

### 2.2 Semantic Naming, Metadata & Formatting

*   **Descriptive semantic `snake_case` names on `<mujoco model="..."/>` and
    every element (no exporter placeholders)**:
    *   `<mujoco model="<asset_name>">` must match the asset's canonical
        `snake_case` identifier—never leave `model="untitled"` or a raw numeric
        ID.
    *   Every physical `body`, `geom`, `joint`, `site`, `mesh`, `material`,
        `texture`, `actuator`, `sensor`, and `equality` must have a meaningful
        `snake_case` name describing the object or physical part (e.g., root
        body `name="wooden_end_table"`, child body `name="drawer"`, joint
        `name="drawer_slide"`, geoms `name="table_top_visual"`,
        `name="drawer_handle_visual"`).
    *   **Forbidden placeholder names**: Never leave elements unnamed (`None` or
        `geom_0`), and **never** leave generic exporter/CAD placeholder names on
        bodies or geoms such as `untitled`, `blender_export`, `root`,
        `world_root`, `main`, `model`, `object`, `body`, `geom`, `default`,
        `node_0`, `mesh_0`, `visual_00`, or `collision_00`.
*   **Preserve existing `<custom>` provenance blocks**: When editing an existing
    `model.xml`, preserve any existing `<custom>` metadata elements (such as
    `asset_proto` or session metadata) unless explicitly asked to modify them.
*   **Focus on physical affordances & semantics, not robot grasp synthesis**:
    When reviewing or authoring an asset, evaluate whether its **collision
    geometry and semantics support its real-world affordances** (e.g., collision
    shapes match the true functional geometry of the object; a mug has an open
    interior cavity and an unbridged handle loop; a hinged lid or sliding drawer
    moves along its mechanical axis). Do **not** get sidetracked running robotic
    gripper grasp planners or requiring artificial
    `<site name="..._grasp_site"/>` tags on simple props that have no dedicated
    handle or semantic target.
*   **Formatting**: Use `2` spaces per indentation level, wrap attributes at 120
    characters, and order top-level sections consistently:
    1.  `<compiler>`
    2.  `<option>` (in a scene or task wrapper)
    3.  `<visual>` (in a scene or task wrapper)
    4.  `<default>`
    5.  `<asset>`
    6.  `<worldbody>`
    7.  `<contact>`
    8.  `<equality>`
    9.  `<tendon>`
    10. `<actuator>`
    11. `<sensor>`
    12. `<keyframe>`

--------------------------------------------------------------------------------

## 3. Coordinate Frame, Units & Root Body Rules

### 3.1 Canonical Orientation, Real-World Scale & Origin Alignment

All standalone assets must be oriented in the canonical right-handed MuJoCo
metric frame (`UP_POSZ_FORWARD_POSX_RIGHT_NEGY`):

1.  **Meters & Real-World Dimension Verification (`<mesh scale="..."/>` allowed
    and expected)**:
    *   If source CAD meshes are in millimeters or centimeters, apply
        `scale="0.001 0.001 0.001"` or `scale="0.01 0.01 0.01"` on `<mesh>` (or
        bake the scale into vertices).
    *   **Verify real-world bounding-box dimensions (without overwriting real
        scans from "similar" web listings)**: Raw 3D scans and vendor meshes are
        frequently mis-scaled by arbitrary factors (e.g., a `45 cm` dinner plate
        or `13.5 cm` stress ball that is `2x` oversized) or exported for a
        different product size variant (e.g., Large vs. Small). Agents **are
        explicitly allowed and expected to apply `<mesh scale="sx sy sz"/>`** to
        both visual and collision `<mesh>` elements (and proportionally scale
        any primitive collider `size`, `pos`, and joint `pos`) whenever an
        asset's bounding box fails to match **verified exact-SKU product specs**
        or exhibits an obvious unit/scan-scale error (`2×`, `10×`, `100×`,
        `1000×`). However, when auditing a real scanned object whose exact
        SKU/brand is unknown and whose existing dimensions are already
        physically plausible for its object class, **do not rescale or distort
        its aspect ratio based on a generic web search for a merely "similar"
        object**—treat the physical scan and reference photo as ground truth.
    *   **Scale Calibration Hierarchy (In-Scene Rulers & Assembled Product Specs
        Beat Coarse Rounded Hints)**: When calibrating dimensions from reference
        photos and intake tables, prioritize **(1) direct in-image physical
        rulers, tapes, or scale grids** (accounting for camera perspective depth
        between the ruler plane and object mid-plane) and **(2) verified
        exact-SKU manufacturer *Assembled Product* dimensions (`mm` / `in`)**
        cross-checked against least-foreshortened photo `L : W : H` aspect
        ratios (strictly separating product dimensions from retail
        packaging/shipping box dimensions). Treat coarse whole-`cm` spreadsheet
        metadata as approximate sanity bounds—never distort true visual
        `L : W : H` aspect ratios to match a rounded `cm` integer, and never
        trace the 2D bounding box of a tilted/oblique photo directly as a `0°`
        orthographic dimension.
    *   **Recalibrate mass when scaling dimensions**: Volume scales as $s_x s_y
        s_z$ (e.g., scaling linear dimensions by $0.5\times$ reduces volume by
        $8\times$). Always verify and recalibrate `<geom mass="..."/>` after
        applying `<mesh scale="..."/>` so effective density remains physically
        realistic.
2.  **Axis orientation (`UP_POSZ_FORWARD_POSX_RIGHT_NEGY`)**:
    *   **`+Z`**: Up (`-Z` is Down; opposes gravity $g = (0, 0, -9.81)\text{
        m/s}^2$). The asset must be oriented upright in its natural resting pose
        (not sideways or upside-down).
    *   **`+X`**: Front / Forward (primary face, door, drawer pull, spout,
        display, or control panel faces **`+X`**).
    *   **`-X`**: Back.
    *   **`+Y` / `-Y`**: Left / Right in the **object's own forward (`+X`)
        coordinate frame** (bilateral symmetry plane is $Y = 0$).
        *   **Viewer-Facing Front (`+X`) Left/Right Handedness Trap**: When a
            user sits or stands in front of the asset at **`+X`** looking toward
            **`-X`** (e.g., typing on a laptop keyboard, viewing a monitor, or
            operating a front control panel), the **viewer's left hand is at
            `-Y`** and the **viewer's right hand is at `+Y`**. Never lay out
            user-facing keyboards, text, or left/right I/O ports from `+Y` to
            `-Y` (which horizontally mirrors the object).
3.  **Origin alignment (no forced $z_{\min} = 0$ shift on existing assets)**:
    *   When authoring a brand-new asset from scratch in Blender, place the
        lowest supporting surface flush at **$z = 0.000\text{ m}$** centered at
        $(x, y) = (0, 0)$.
    *   When auditing or remediating an existing asset, the local origin
        `(0, 0, 0)` may sit at the bottom resting surface ($z_{\min} \approx
        0$), at the object's geometric centroid / center of mass ($z_{\min} <
        0$), or at a primary mechanical pivot (such as the hinge clevis of
        tongs). Scene composers inspect the compiled bounding box or attachment
        sites to align objects to support surfaces automatically.
    *   **Do NOT shift child `<geom>`s or `<body>`s just to force $z_{\min} =
        0$** when the origin is already near the asset's base, centroid, or
        pivot—shifting child geoms alters authored grasp/pivot coordinates and
        can break relative transforms in articulated assets. Only remediate
        origin translations if the geometry is wildly offset outside its own
        bounding box (e.g., a stray CAD export offset $> 0.1\text{ m}$ away from
        `(0, 0, 0)`).

### 3.2 Zero Transform on the Top-Level Asset Body

The root body of the asset (`spec.worldbody.first_body()`) **must not have any
`pos` or `quat` offset** (it must be at `pos="0 0 0"` and `quat="1 0 0 0"`, or
omit `pos`/`quat` entirely) and must have a semantic name matching the asset:

```xml
<body name="coffee_mug">
  <!-- visual & collision geoms -->
</body>
```

*   **Why**: Scene assembly (`MjSpec.attach`), attachment frames, and
    `<freejoint/>` keyframes control the pose of the root body directly. Leaving
    a non-zero `pos` or non-identity `quat` on the root body (such as `<body
    name="blender_export" pos="0 0 0.307054" quat="0.707107 0 0 0.707107">`) is
    **not allowed** because attaching the asset to a frame/site or setting
    `qpos` will shift or rotate the object unexpectedly.
*   **How to fix offsets**: Bake coordinate frame rotations (e.g., rotating
    Blender's `-Y` front to MuJoCo's `+X` front) and vertical resting offsets
    into the mesh vertex coordinates, or place them on child `<geom>`s / child
    `<frame>`s inside the root body—never on the root `<body>` itself.
*   **Kinematic Tree Root Selection**: In any articulated asset, choose the
    **root `<body>`** to be the primary stationary base or primary grasped
    housing during normal operation, and make components that articulate
    relative to that housing the child `<body>` elements.

### 3.3 Attaching Assets & Adding a `<freejoint/>`

*   **Omit `<freejoint/>` from reusable `model.xml`**: Static furniture
    (counters, wall cabinets) is welded to the world or a room frame; movable
    objects receive a `<freejoint/>` only when spawned into a scene or test
    harness.
*   **How to attach an asset with a `<freejoint/>` in a scene**:
    *   In Python (`MjSpec`), load the asset spec, add the freejoint to its root
        body (`child_spec.worldbody.first_body().add_freejoint()`), create an
        attachment `<frame>` at the desired spawn pose on the parent scene
        (`frame = scene_spec.worldbody.add_frame(pos=..., quat=...)`), and call
        `scene_spec.attach(child_spec, frame=frame, prefix="...")`. See
        [`MjSpec` Attachment Documentation][mjspec-docs].
    *   At runtime, a free-floating body's pose is stored in its 7 `qpos`
        entries (`[x, y, z, qw, qx, qy, qz]`), not `model.body_pos`; modify
        `qpos` via named access (`data.joint('...').qpos`) and call
        `mujoco.mj_forward(model, data)` as documented in
        [MuJoCo Python Bindings][mj-python].

--------------------------------------------------------------------------------

## 4. Groups, Visibility & `rgba` Rules (Never Make Assets Invisible)

MuJoCo organizes `<geom>` and `<site>` visibility into integer groups (`0`–`5`).
Consistent group assignment and explicit `rgba` alpha values ensure assets look
clean by default while remaining fully inspectable when debugging in MuJoCo
Studio.

### 4.1 The Standard Group Hierarchy

| Group   | Default Class /     | Studio    | Default     | Physics Role &      |
:         : Element             : Hotkey    : Visibility  : `rgba` Rule         :
| :-----: | :------------------ | :-------: | :---------: | :------------------ |
| **`0`,  | Reserved for global | `0`, `1`  | Visible     | **Do not leave      |
: `1`**   : scene / robot base  :           :             : asset geoms in      :
:         :                     :           :             : group `0` or        :
:         :                     :           :             : `1`**—never use a   :
:         :                     :           :             : single unseparated  :
:         :                     :           :             : `group="0"` mesh    :
:         :                     :           :             : geom as both visual :
:         :                     :           :             : and collider.       :
:         :                     :           :             : Always split into   :
:         :                     :           :             : `class="visual"`    :
:         :                     :           :             : (`group="2"`) and   :
:         :                     :           :             : `class="collision"` :
:         :                     :           :             : (`group="3"`), even :
:         :                     :           :             : if a convex object  :
:         :                     :           :             : reuses the same     :
:         :                     :           :             : mesh with           :
:         :                     :           :             : `maxhullvert="64"`. :
| **`2`** | `<default           | `2`       | **Visible** | `contype="0"        |
:         : class="visual">`    :           :             : conaffinity="0"     :
:         : (`<geom>`)          :           :             : density="0"         :
:         :                     :           :             : mass="0"`. Renders  :
:         :                     :           :             : PBR                 :
:         :                     :           :             : materials/textures  :
:         :                     :           :             : or opaque `rgba`    :
:         :                     :           :             : ($\alpha = 1.0$).   :
| **`3`** | `<default           | `3`       | **Hidden**  | `contype="1"        |
:         : class="collision">` :           :             : conaffinity="1"`.   :
:         : (`<geom>`)          :           :             : Carries physical    :
:         :                     :           :             : `density` or        :
:         :                     :           :             : `mass`. **Must have :
:         :                     :           :             : visible             :
:         :                     :           :             : semi-transparent    :
:         :                     :           :             : `rgba`** ($\alpha   :
:         :                     :           :             : \in [0.35, 0.6]$).  :
| **`4`** | Affordance & task   | `Shift+4` | **Hidden**  | Non-colliding       |
:         : `<site>`s           :           :             : semantic volumes    :
:         :                     :           :             : (`box`, `cylinder`, :
:         :                     :           :             : `capsule`,          :
:         :                     :           :             : `sphere`) with      :
:         :                     :           :             : **visible           :
:         :                     :           :             : semi-transparent    :
:         :                     :           :             : `rgba`** ($\alpha   :
:         :                     :           :             : \in [0.25, 0.5]$).  :

### 4.2 Why `rgba` Alpha Must Never Be `0`

> **Rule: Never set `rgba="... 0"` on collision geoms or sites to hide them.**
>
> Visibility is controlled by `group="3"` (for colliders) and `group="4"` (for
> sites). If you set `rgba="0 0 0 0"` or `rgba="1 1 1 0"`, then pressing `3` or
> `Shift+4` in MuJoCo Studio shows **nothing**—reviewers and users cannot see
> whether colliders match the visual mesh, whether hollow openings are bridged,
> or where grasp sites are located.

Use distinct, semi-transparent colors by functional role so overlapping parts
are immediately distinguishable when group `3` (`3` + `T`) or group `4`
(`Shift+4`) is toggled on:

*   **Primary body / base colliders (`group="3"`)**: Green
    `rgba="0.20 0.75 0.35 0.50"`
*   **Wall / rim / slat colliders (`group="3"`)**: Blue
    `rgba="0.25 0.55 0.90 0.45"`
*   **Articulated lid / door / jaw colliders (`group="3"`)**: Orange
    `rgba="0.95 0.55 0.15 0.50"`
*   **High-friction grasp pad colliders (`group="3"`)**: Gold
    `rgba="0.90 0.80 0.15 0.55"`
*   **Grasp / handle sites (`group="4"`)**: Red/Coral
    `rgba="0.95 0.25 0.25 0.35"`
*   **Support / resting / cavity sites (`group="4"`)**: Cyan
    `rgba="0.15 0.80 0.85 0.30"`

### 4.3 Zero Visual Mass, No `<compiler settotalmass/>` & Collider Mass

1.  **Always set `density="0" mass="0"` in `<default class="visual">` (and never
    override `mass` on visual `<geom>`s)**: In MuJoCo, setting
    `contype="0" conaffinity="0"` disables contact generation, **not** mass
    computation! Unless `density="0"` (and `mass="0"`) is set on
    `class="visual"` without per-geom overrides, MuJoCo computes mass and
    inertia from the visual mesh, double-counting mass or leaving inertia on the
    visual mesh while collision hulls have zero/dummy mass.
2.  **Never use `<compiler settotalmass="..."/>`**: `<compiler settotalmass/>`
    globally rescales all body masses and inertias at compile time—including any
    visual geom that was not explicitly zeroed—and obscures per-link mass
    distribution. Instead, omit `settotalmass` and assign explicit masses (or
    calibrated densities) directly on the `group="3"` collision geoms,
    distributing each link's target mass **proportional to collider volume** so
    the center of mass and principal inertias remain unbiased.
3.  **Never add invisible "mass-only" dummy geoms or interior ballast `box`
    colliders**: Do not create hidden geoms
    (`contype="0" conaffinity="0" rgba="0 0 0 0"`) or fake interior `box`
    colliders (e.g., an internal battery/weight block inside a solid housing) to
    shift the center of mass or add ballast weight. Instead:
    *   Assign realistic volume-proportional `mass` or per-part `density`
        directly to the actual exterior `class="collision"` geoms, **or**
    *   Provide an explicit `<inertial pos="x y z" quat="w x y z" mass="M"
        diaginertia="Ixx Iyy Izz"/>` on the `<body>`.
4.  **Product-Link & Online Search (`search_web`) Mass/Dimension Calibration &
    Multi-Pack / Shipping-Weight Traps**: Always check `<custom>` metadata,
    reference specification sheets, or user-provided product links first; **if
    none are provided, actively use `search_web` to look up retail/manufacturer
    specs (dimensions, single-unit net weight, and material composition) for the
    exact item or comparable real-world objects**. Calibrate total mass and
    dimensions to the real physical item rather than guessing, while guarding
    against four common e-commerce listing traps:
    *   **Multi-pack total weights**: E-commerce `Item Weight` /
        `Package Weight` fields frequently report the weight of the entire
        multi-pack (e.g., a 2-pack, 6-pack, 12-pack, or 30-pack). Always check
        the unit count in the listing title/specs and divide by the pack count.
    *   **Inhomogeneous multi-packs**: When a multi-pack contains different
        sizes (e.g., a 2-pack containing one 9-inch and one 12-inch kitchen
        tong), do **not** blindly divide the total pack weight by 2; calibrate
        the mass to the specific size variant modeled by the asset's bounding
        box.
    *   **Shipping weight vs. assembled net weight & component
        double-counting**: When a physical product is split into multiple
        component assets (e.g., a bento box tray + lid, or an outer lunch bag +
        removable hard insert), verify that the **sum of the component assets
        matches the unboxed net product weight**, not the packaged shipping
        weight.
    *   **Effective hull density sanity check ($\rho_{\text{eff}} =
        m_{\text{total}} / V_{\text{hull}}$)**: Always compute the implied
        effective density across the collision hulls. Flag and correct
        unphysical densities (for example, assigning biological fruit weight or
        full-pack weight to a miniature foam prop, yielding $\rho_{\text{eff}} >
        1.5\text{ g/cm}^3$ for polyurethane foam).
5.  **Prevent `Error: mesh volume is too small ... Try setting inertia to shell`
    on Thin Visual Decals, Stickers, Labels & Screens**:
    *   In MuJoCo (`mjCMesh::Compile`), MuJoCo computes 3D enclosed volume for
        **every `<mesh>` in `<asset>` at compile time**—even when the `<geom>`
        referencing that mesh is in `class="visual"` with
        `density="0" mass="0"`. Any flat 2D single-sided quad or sub-millimeter
        wafer mesh (e.g., barcode stickers, brand tags, labels, or LCD windows)
        will fail `MjSpec.compile()` with `ValueError: Error: mesh volume is too
        small: <mesh_name>. Try setting inertia to shell`.
    *   Model thin walls, rims, shells, clips, or applied tags at their **true
        physical thickness** as closed 2-manifold solids (`solidify_shell` or
        CAD thickness—never inflate delicate rims, glassware, fabric brims, or
        sheet metal to an artificial `2–3 mm` slab!) and set
        **`inertia="shell"`** on visual `<mesh>` elements in `<asset>`
        (`<mesh name="..._vis" file="..._vis.obj" inertia="shell"/>`).

### 4.4 PBR Materials, `<layer>` Declarations & Color Spaces

*   **Basecolor / albedo textures**: Always declare `colorspace="sRGB"` on 8-bit
    PNG color textures (`<texture type="2d" name="wood_basecolor"
    file="wood_basecolor.png" colorspace="sRGB"/>`). Declaring
    `colorspace="linear"` on an sRGB basecolor texture causes washed-out,
    gamma-doubled colors in Filament/Studio.
*   **Non-color data maps**: Normal maps, roughness, metallic, and occlusion
    (ORM) maps must use `colorspace="linear"`.
*   **Multi-layer PBR `<material><layer .../></material>` rules**:
    *   Valid `<layer>` `role` values are `"rgb"`, `"normal"`, `"orm"`,
        `"occlusion"`, `"roughness"`, `"metallic"`, and `"emissive"`—never use
        `role="rgba"` (which fails MJCF schema compilation).
    *   Every `texture="..."` referenced by a `<layer>` must exist in `<asset>`
        (remove dangling exporter placeholders like
        `<layer texture="grid" role="rgb"/>`), and a `<material>` must never
        declare duplicate `<layer>` elements with the same `role`.
    *   For clear glass, crystal, or translucent polymers, pair a translucent
        `<material rgba="r g b alpha"/>` ($\alpha < 1.0$) with low roughness and
        crisp specular reflections calibrated to the real material's optical
        finish, and never set an opaque `rgba="1 1 1 1"` override on the
        referencing visual `<geom>` (as geom `rgba` overrides the material's
        alpha).
*   **Explicit PBR attributes & multi-layer texture calibration**:
    *   Prefer **full multi-layer PBR texture sets** (`<layer role="rgb"/>`,
        `<layer role="normal"/>`, `<layer role="roughness"/>`, and where
        applicable `<layer role="metallic"/>`, `<layer role="occlusion"/>`, or
        `<layer role="orm"/>` — `roughness`, `metallic`, and `normal` can be 2D
        texture maps, not just scalar floats) across all major visual materials
        (fabrics, textured plastics/nylon, soft-touch polyurethane/rubber,
        brushed metals, wood) rather than falling back to flat untextured `rgba`
        colors.
    *   **Filament roughness multiplication rule**: In MuJoCo's Filament PBR
        backend (`mjrf`), the `<material roughness="r" metallic="m"/>` scalar
        attributes **multiply** the pixel values of `<layer role="roughness"/>`
        and `<layer role="metallic"/>` ($r_{\text{eff}} = r_{\text{mat}} \times
        r_{\text{tex}}$). Setting `roughness="0.45"` on `<material>` while also
        attaching a roughness texture whose mean is `0.45` (`115/255`) yields
        $r_{\text{eff}} \approx 0.20$ (turning matte plastic/nylon into glossy
        wet lacquer!). When attaching a `<layer role="roughness"/>` (or
        `<layer role="metallic"/>`) texture, set **`roughness="1.0"`** (and
        **`metallic="1.0"`**) on `<material>` and calibrate the texture PNG
        values directly to the true physical finish observed in the reference
        photos (or if no texture is attached, set the scalar attributes
        directly).

--------------------------------------------------------------------------------

## 5. Canonical MJCF Templates

### 5.1 Reusable `model.xml` Template

```xml
<mujoco model="storage_caddy">
  <compiler angle="radian" meshdir="meshes" texturedir="textures"
            autolimits="true"/>

  <default>
    <default class="storage_caddy">
      <joint damping="0.05" armature="0.002" frictionloss="0.01"/>
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"
              density="0" mass="0"/>
      </default>
      <default class="collision">
        <!-- Tune condim, density, and friction to the asset's real material and contact/rolling regime -->
        <geom group="3" contype="1" conaffinity="1" condim="4"
              density="800" friction="0.8 0.02 0.005" solref="0.005 1"
              solimp="0.95 0.99 0.001" rgba="0.20 0.75 0.35 0.50"/>
      </default>
    </default>
  </default>

  <asset>
    <texture name="caddy_basecolor" type="2d"
             file="caddy_basecolor.png" colorspace="sRGB"/>
    <texture name="caddy_normal" type="2d"
             file="caddy_normal.png" colorspace="linear"/>
    <texture name="caddy_roughness" type="2d"
             file="caddy_roughness.png" colorspace="linear"/>
    <material name="caddy_plastic" metallic="0.0" roughness="1.0">
      <layer texture="caddy_basecolor" role="rgb"/>
      <layer texture="caddy_normal" role="normal"/>
      <layer texture="caddy_roughness" role="roughness"/>
    </material>
    <mesh name="caddy_body_vis" file="caddy_body.obj"/>
    <mesh name="caddy_lid_vis" file="caddy_lid.obj"/>
  </asset>

  <worldbody>
    <!-- Root body: semantic name, zero transform, and NO freejoint -->
    <body name="storage_caddy" childclass="storage_caddy">
      <!-- Visual mesh (group 2, zero mass) -->
      <geom name="caddy_body_visual" class="visual"
            mesh="caddy_body_vis" material="caddy_plastic"/>

      <!-- Hollow container collision primitives (group 3, semi-transparent
           rgba; +X=front, -X=back, +Y=left, -Y=right) -->
      <geom name="caddy_bottom_col" class="collision" type="box"
            pos="0 0 0.005" size="0.060 0.080 0.005" mass="0.120"/>
      <geom name="caddy_wall_front_col" class="collision" type="box"
            pos="0.055 0 0.045" size="0.005 0.080 0.035" mass="0.045"
            rgba="0.25 0.55 0.90 0.45"/>
      <geom name="caddy_wall_back_col" class="collision" type="box"
            pos="-0.055 0 0.045" size="0.005 0.080 0.035" mass="0.045"
            rgba="0.25 0.55 0.90 0.45"/>
      <geom name="caddy_wall_left_col" class="collision" type="box"
            pos="0 0.075 0.045" size="0.050 0.005 0.035" mass="0.035"
            rgba="0.25 0.55 0.90 0.45"/>
      <geom name="caddy_wall_right_col" class="collision" type="box"
            pos="0 -0.075 0.045" size="0.050 0.005 0.035" mass="0.035"
            rgba="0.25 0.55 0.90 0.45"/>

      <!-- Affordance sites (group 4, visible semi-transparent rgba) -->
      <site name="interior_cavity_site" group="4" type="box"
            pos="0 0 0.045" size="0.050 0.070 0.035"
            rgba="0.15 0.80 0.85 0.30"/>

      <!-- Articulated lid child body hinged at back (-X) -->
      <body name="caddy_lid" pos="-0.055 0 0.080">
        <joint name="lid_hinge" type="hinge" axis="0 -1 0" range="0 1.8"
               solreflimit="0.005 1"/>
        <geom name="caddy_lid_visual" class="visual"
              mesh="caddy_lid_vis" material="caddy_plastic"/>
        <geom name="caddy_lid_col" class="collision" type="box"
              pos="0.055 0 0.004" size="0.060 0.080 0.004" mass="0.070"
              rgba="0.95 0.55 0.15 0.50"/>
        <site name="lid_handle_grasp_site" group="4" type="box"
              pos="0.105 0 0.012" size="0.008 0.025 0.006"
              rgba="0.95 0.25 0.25 0.35"/>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude name="body_lid_exclude" body1="storage_caddy" body2="caddy_lid"/>
  </contact>

  <sensor>
    <subtreecom name="storage_caddy_com" body="storage_caddy"/>
  </sensor>
</mujoco>
```

### 5.2 Building Test & Evaluation Scenes Programmatically with `MjSpec`

Do **not** author a static `scene.xml` file in the asset bundle. Because a
reusable `model.xml` omits `<freejoint/>` on its root body, build test harnesses
in Python with `mujoco.MjSpec`: load `child_spec = mujoco.MjSpec.from_file(...)`,
call `child_spec.worldbody.first_body().add_freejoint()` in memory when spawning
a movable 6-DoF object, create `spawn_frame = scene_spec.worldbody.add_frame(...)`,
and call `scene_spec.attach(child_spec, frame=spawn_frame, prefix="...")` (see
§3.3 and [`MjSpec` Sub-model Composition Documentation][mjspec-docs]).

--------------------------------------------------------------------------------

## 6. Collision Geometry & Affordance Preservation

Raw visual meshes from CAD or 3D scans must **never** be used directly as
collision geoms (`type="mesh"` in `class="collision"` without
decomposition/simplification). When MuJoCo compiles a `<geom type="mesh"/>` with
`contype > 0`, it replaces the mesh internally with its **single convex hull**,
sealing every hole, cavity, and concave feature.

> [!IMPORTANT]
>
> **Core Rule — Collision Geometry Must Serve the Function of the Asset**:
>
> *   **Function-first collision design**: Every asset's collision geometry must
>     faithfully support its physical function—preserving open container
>     cavities, unobstructed handle/clip through-holes, flat or footed resting
>     bases, smooth rolling contours, and clean grasp/articulation clearances.
> *   **Prefer hand-authored convex hulls and analytic primitives first**:
>     Agents should prefer creating per-part convex hulls (e.g. via
>     `bmu.export_convex_hull_col` or semantic sub-part slicing with $\le 64$
>     native vertices) and exact analytic primitives (`box`, `cylinder`,
>     `capsule`, `sphere`) by hand so every collider maps cleanly to a true
>     functional part.
> *   **Fallback to `coacd` with parameter tuning when hand decomposition is too
>     difficult**: If an asset's concave geometry is too complex or organic to
>     decompose by hand, fall back to **`coacd`** (via `uv run --with coacd
>     --with trimesh python3 ...`) and **actively tune its parameters**
>     (`threshold`, `max_convex_hull`, `preprocess_mode`, and post-decimation to
>     `maxhullvert="64"`)—inspecting the per-hull color-coded 6-view sheet to
>     verify that cavities and through-holes remain open and closed shells are
>     not shattered into thin wall slivers.

### 6.1 Choosing the Right Collision Representation

Match the collision representation to the true geometry and functional contact
topology of each rigid body:

*   **Preserve existing well-fitted colliders during audits, and replace poorly
    fitted or over-fragmented approximations**:
    *   If an existing asset already has convex-hull collision meshes or
        primitive colliders that closely follow the visual contour without
        protrusion or undercut, preserve functional cavities/openings without
        tenting or asymmetric pie-wedge cuts, avoid gratuitous sliver
        fragmentation across closed shells, and have $\le 64$ vertices natively
        in their `.obj` files (so `maxhullvert="64"` does not greedily drop
        vertices at compile time), **keep them as-is**. Never replace a
        well-fitted convex decomposition with a sharp-cornered analytic
        primitive that degrades surface fit.
    *   **Semantic Convex Decomposition & Functional Contact Parsimony**:
        Decompose each rigid body into as many clean analytic primitives or
        natively authored $\le 64$-vertex convex hulls (`maxhullvert="64"`
        engine ceiling) as needed so functional concavities, openings, rests,
        and graspable features are not bridged over, while flat or
        already-convex spans are not gratuitously over-fragmented:
        *   **Molded housings & closed shells**: When a closed housing mesh has
            internal wall thickness, recessed graphics windows, or open boundary
            seams, blind volumetric CoACD can shatter the outer wall into dozens
            of thin slivers. Instead, wrap the outer shell with clean convex
            hulls and add separate convex hulls or primitives only for
            protruding controls, rims, or feet that participate in physical
            contact.
        *   **Open brackets, clips, frames & loop handles**: Slice the visual
            mesh along the structural limb junctions and build clean
            $\le 64$-vertex convex limb hulls (or `capsule`/`box` primitives
            when genuinely geometric) around the perimeter so the interior
            opening remains completely unobstructed without sliver
            fragmentation.
        *   **Turned & stepped profiles (vessels, bottles, goblets, tools)**:
            Blind volumetric CoACD frequently bridges across concave necks or
            waists while slicing symmetric bases into asymmetric pie-wedges.
            Slice along concave radial profile inflections $r(z)$ and semantic
            feature boundaries so coaxial convex hulls follow curved shoulders
            and stepped collars without tenting across concave throats or
            splitting straight cylindrical spans unnecessarily.
        *   **Decorative detail & multi-material density**: Only carve out
            cavities, slots, or through-holes that a gripper finger, tool, or
            placed object can physically enter. Wrap fine non-container detail
            (e.g., plant foliage clusters, bristles, or shallow decorative trim)
            in smooth envelope hulls rather than hundreds of micro-hulls, and
            for multi-material objects assign realistic per-material densities
            directly to those exterior hulls rather than inserting hidden
            internal ballast geoms.
    *   **Author Hulls Natively Within the `maxhullvert="64"` Compiler
        Ceiling**: Never export high-vertex `.obj` hulls and rely on
        `<mesh maxhullvert="64"/>` to downsample them at compile time—MuJoCo's
        greedy 64-vertex compiler drops vertices unevenly, flattening curved
        profiles and skewing compiled hull volumes (`hull_density_cv`). Author
        every convex hull `.obj` with $\le 64$ vertices natively:
        *   On coaxial turned profiles, build coaxial convex frustums (`<= 64`
            vertices per hull) that share exact boundary rings at each cut plane
            $z = z_{\text{cut}}$ (so `compute_collision_alignment_summary`
            recognizes shared cut planes as internal seams without overlapping
            hulls), allocating axial cuts where profile curvature $r''(z)$ is
            high rather than splitting straight constant-radius sections.
        *   Never artificially shrink collision hull radii inside the visual
            mesh's outer contour—doing so lets the visual mesh poke outside the
            collider in overlay renders and degrades `v2c` undercut metrics.
    *   Conversely, if existing colliders are coarse, blocky, bloated, tent
        across concave necks/waists, split symmetric cylinders into pie-wedges,
        exceed 64 native vertices per hull, or seal functional
        cavities/through-holes, replace them with a clean semantic convex
        decomposition.
    *   Always inspect `model.xml` for existing colliders hidden in `group="0"`
        or with `rgba="... 0"` ($\alpha = 0$) before assuming collision geometry
        is missing; move them to `group="3"` with $\alpha \in [0.35, 0.6]$ and
        verify their surface fit and hole clearance.
*   **Use analytic primitives ONLY when authoring colliders for genuinely
    geometric parts**—they provide smooth, exact surface normals, zero
    vertex-hunting jitter, and fast MJX/GPU evaluation:
    1.  **`box`**: Flat walls, tabletops, shelves, sharp-cornered rectangular
        trays, straight rectangular rails/legs (`size="hx hy hz"` half-extents).
    2.  **`cylinder`**: Flat-bottomed cans, coasters, circular pucks/buttons,
        mug bases, straight cylindrical shafts/nozzles
        (`size="radius half_length"`).
    3.  **`capsule`**: Rounded handle bars, wire bail handles, utensil shafts,
        rounded legs (`size="radius half_length"` or
        `fromto="x1 y1 z1 x2 y2 z2"`).
    4.  **`sphere`**: Balls, spherical knobs, feet, bulbs (`size="radius"`).
*   **Use convex hull meshes (`type="mesh"` with `<mesh maxhullvert="64"/>`) for
    rounded, curved, contoured, or organic shapes** (e.g., filleted/rounded
    housings, curved brackets, ergonomic handles, molded shells, contoured
    furniture):
    *   **Never approximate rounded, curved, or contoured silhouettes with a
        blocky patchwork of `box` primitives**—sharp `box` corners protrude past
        filleted corners and step across curved surfaces.
    *   Prefer decomposing rounded housings, vessels, and articulated brackets
        by hand into clean, natively authored ($\le 64$-vertex) convex hull
        `.obj` meshes per convex region (e.g. via `bmu.export_convex_hull_col`
        or Python slicing); if a mesh is too complex to decompose by hand, run
        `coacd` (`uv run --with coacd --with trimesh python3 ...`) and tune
        `threshold` / `max_convex_hull` / `preprocess_mode` so functional
        openings stay clear and closed shells are not over-fragmented.
    *   Always declare `maxhullvert="64"` (MuJoCo's convex-hull compiler ceiling
        for fast MJX/CPU collision), and combine convex hull meshes for rounded
        bodies with exact analytic primitives (e.g., `cylinder` for circular
        rollers, pins, or buttons) within the same asset where appropriate.

### 6.2 Preserving Functional Affordances

Collision geometry must preserve every physical interaction the asset supports:

*   **Hollow containers (mugs, bowls, bins, drawers, baskets)**:
    *   A small marble dropped at $(x=0, y=0)$ above the opening must fall
        **inside** and rest on the interior floor ($z \approx
        t_{\text{bottom}}$), not hover on an invisible convex-hull lid at $z =
        z_{\text{max}}$.
    *   Build the base from a `cylinder` or `box` and the perimeter from thin
        wall `box` slats (or pre-decomposed wall hull meshes).
*   **Loop handles & grasp clearances (mug handles, kettle handles, drawer
    pulls, open brackets & clips)**:
    *   Keep the through-hole open so a gripper finger or mating part can pass
        through the opening without colliding.
    *   Decompose open handles, clips, and brackets along their structural limbs
        into separate convex hull or `capsule`/`box` segments rather than a
        single bounding hull that webs the loop shut.
*   **Articulated links, hinged mechanisms & rolling mobility**:
    *   Split moving parts into separate `<body>` elements connected by
        `<joint type="hinge"/>` or `<joint type="slide"/>` at the physical joint
        axis, using the minimum number of joints required to represent the
        mechanism's true independent degrees of freedom.
    *   **Always articulate functional wheels, casters, and rollers as child
        `<body>` links with revolute `<joint type="hinge"/>` axes**: When an
        object has visible wheels, casters, or rollers whose physical purpose is
        rolling mobility across a surface, **never** weld them into a static
        mesh or fixed colliders on the root body. Model each wheel assembly as
        articulated child `<body>` links (for swivel casters: a vertical swivel
        yoke `<body ...><joint type="hinge" axis="0 0 1"/>` plus a wheel
        `<body ...><joint type="hinge" axis="..."/>` with
        `<geom type="cylinder"/>` or `<geom type="sphere"/>` wheel colliders) so
        the wheels physically spin and steer when the object is pushed.
    *   Ensure contacting surfaces meet cleanly at joint limits without
        self-collision lockup, and add
        `<contact><exclude body1="..." body2="..."/></contact>` between adjacent
        articulated links (never use `<exclude/>` to mask geometric
        interpenetration across a joint's range of motion).

### 6.3 Common Geometric Decomposition Traps

1.  **Symmetric `'+'` Cross / Star Base PCA Trap**: A 4-way `'+'` pedestal base
    or 4-spoke knob has degenerate principal moments ($I_{xx} = I_{yy}$),
    causing PCA / minimum-volume OBB fitters to rotate $45^\circ$ into a giant
    square bounding box that webs across all four corners. Decompose `'+'`
    shapes into **two axis-aligned `box` geoms** (`arm_x` and `arm_y`).
2.  **Hollow Container & Handle Trap**: Never fit a single hull to a hollow
    container or let handle colliders protrude inside the interior cavity; model
    the floor base, perimeter wall segments, and outer handle arch separately so
    both the interior cavity and the handle grasp loop remain open.
3.  **Thin Curved Shells & Utensils (Spoons, Spatulas, Tongs, Brims)**:
    Volumetric voxelization (V-HACD / blind CoACD) can inflate thin curved
    shells far beyond their physical thickness. Partition the curved surface
    into connected patches sized to the local curvature and emit natively
    authored $\le 64$-vertex `<geom type="mesh"/>` convex hulls that hug the
    true thin shell without bridging across its concavity.
4.  **Axisymmetric / Turned Parts Split into Pie-Wedges & Neck Tents**: Avoid
    blind volumetric CoACD on turned vessels (which tents across concave necks
    and slices bases into asymmetric pie-wedges); slice along radial profile
    inflections $r(z)$ into coaxial $\le 64$-vertex convex frustums with shared
    boundary rings at each cut plane $z_{\text{cut}}$ (§6.1).

--------------------------------------------------------------------------------

## 7. Physics, Contacts, Joints & Actuators

> [!TIP]
>
> **Take an Experimental Approach to Tuning Physics Parameters**: Starting
> values for friction, contact impedance (`solref`/`solimp`), joint `damping`,
> `frictionloss`, `armature`, and `stiffness` should always be treated as
> initial baselines rather than rigid constants. Where appropriate, build a
> programmatic `MjSpec` test scene, **design any custom simulation experiments
> suited to the asset**, inspect the resulting trajectories, contact forces, and
> settling behavior, and iteratively tune parameters until the asset behaves
> like its real-world counterpart.

### 7.1 Contact Parameters (`condim`, `friction`, `solref`, `solimp`)

*   **MJCF `condim` & `friction` Contract vs. Material Calibration**:
    *   **Engine contract**: MuJoCo's `friction="sliding torsional rolling"`
        takes 3 coefficients, and `condim` selects which contact wrench
        dimensions are active: `condim="1"` (frictionless normal), `condim="3"`
        (normal + 2D tangential sliding), `condim="4"` (adds torsional spin
        resistance about the contact normal), and `condim="6"` (adds 2D rolling
        resistance).
    *   **Choosing `condim` & `friction` per asset**: Choose `condim` and
        calibrate `[sliding, torsional, rolling]` to the real surface material
        (e.g., low-friction sleek plastics/metals vs. higher-friction wood,
        textured polymers, or rubber/elastomer grips) and the object's geometry:
        use at least `condim="4"` on graspable props so pinched grasps resist
        effortless in-finger spin, and use `condim="6"` with non-zero rolling
        friction on spheres, cylinders, and round vessels that should decelerate
        and come to rest naturally on flat surfaces.
*   **Solver stiffness & damping (`solref="timeconst dampratio"`)**:
    *   Always satisfy **`timeconst >= 2 * timestep`** (e.g. `solref="0.005 1"`
        at `timestep="0.002"`). Because MuJoCo's `refsafe` clamp is enabled by
        default, setting `timeconst < 2 * timestep` is silently clamped to $2
        \cdot \Delta t$, masking misconfigured stiffness if `timestep` changes.
    *   Use critical damping (`dampratio="1"`) on rigid contacts and joint
        limits (`solreflimit="0.005 1"`) to prevent artificial restitution
        bounce.
*   **Zero contact margin**: Keep `margin="0"` and `gap="0"` on manipulation
    assets so objects rest directly on their true geometric surfaces rather than
    hovering on an invisible force cushion.

### 7.2 Joint Dynamics & Articulated Mechanisms

Every `<joint type="hinge"/>` or `<joint type="slide"/>` must correspond to a
true physical degree of freedom and define physical dissipation and reflected
inertia so mechanisms behave realistically and integrate stably:

*   **Minimal degrees of freedom (no uncoupled floppy pivot links)**: Do not add
    more joints than the physical mechanism has independent degrees of freedom.
    Avoid adding uncoupled joints that leave intermediate links
    underactuated/floppy (e.g., modeling 1-DoF spring tongs, scissors, pliers,
    or clothespins with 2 uncoupled hinge joints on a small floating clevis root
    body—instead merge one arm into the root body for a single 1-DoF hinge
    joint, or couple the two joints via `<equality><joint>`), or joints on
    purely cosmetic parts whose internal mechanism is not modeled.
*   **Two-stage mechanism workflow (clean baseline `<asset_name>/` first, then
    `<asset_name>_mechanical/`)**:
    1.  **Baseline (`<asset_name>/`)**: Always build and verify the direct-joint
        articulated model first using standard joint parameters (`damping`,
        `frictionloss`, `stiffness`, `armature`) without internal contact
        clutches or multi-tendon ratchets.
    2.  **Mechanical Variant (`<asset_name>_mechanical/`)**: When a mechanical
        linkage, latch, clutch, ratchet, or release lever is central to how the
        object works, create a separate self-contained sibling directory
        `<asset_name>_mechanical/` that models the functional mechanism
        passively (using `<equality><joint>`, limited `<tendon><fixed>`
        constraints, and spring-preloaded `<contact><pair>` surfaces) so
        operating the trigger/lever physically drives or releases the mechanism.
        Set sufficient joint `armature` on lightweight internal mechanism links
        so the solver's constraint stiffness ($k_{\text{efc}} \approx
        m_{\text{eff}} / \text{timeconst}^2$) dominates the preload springs, and
        note that `<pair>` `friction` takes 5 floats
        (`tan1 tan2 spin roll1 roll2`).
*   **`damping`** ($\text{N}\cdot\text{m}\cdot\text{s/rad}$ or
    $\text{N}\cdot\text{s/m}$): Viscous resistance. Prevents doors/lids from
    swinging perpetually (`0.02` to `0.5` for small lids/hinges; `1.0` to `5.0`
    for heavy cabinet doors or damped drawers).
*   **`frictionloss`** ($\text{N}\cdot\text{m}$ or $\text{N}$): Dry Coulomb
    stiction solved implicitly by the constraint solver. Allows a laptop screen,
    hinged support stand, friction hinge, valve knob, pneumatic
    height-adjustment column (`seat_height`), or notched/ratcheted telescoping
    post (`armrest_*_height`) to hold its pose against gravity anywhere in its
    range.
    *   **Position-holding threshold rule (`frictionloss > F_gravity`)**:
        Whenever a non-spring (`stiffness="0"`) vertical/inclined `slide` joint
        or friction-positioned `hinge` joint is intended to maintain its set
        height or angle against gravity, you **must** set `frictionloss`
        strictly greater than the gravitational load of the supported child
        subtree along that joint axis ($\ge 1.25 \times |G_i(q)| = 1.25 \times
        m_{\text{subtree}} \, g \, |\hat{\mathbf{a}} \cdot \hat{\mathbf{z}}|$
        for a slide joint, or $\ge 1.25 \times m_{\text{subtree}} \, g \,
        L_{\text{com}}$ for a hinge), paired with stiff friction constraint
        impedance
        (`solreffriction="0.004 1" solimpfriction="0.99 0.999 0.0001"`) to
        prevent elastic solver sag. Never leave `frictionloss` at a small
        default (e.g., `1.0 N`) on a vertical column supporting a `20 kg`
        (`~200 N`) assembly, and never verify height/pose retention with
        `mj_forward` alone—always step `mj_step` under gravity from mid-stroke
        to confirm zero sag.
*   **`armature`** ($\text{kg}\cdot\text{m}^2$ or $\text{kg}$): Added diagonal
    inertia (`0.001` to `0.01` on small mechanism joints), stabilizing light
    links under stiff springs or position servos.
*   **`stiffness` & `springref`**: For spring-return mechanisms (e.g., tactile
    push buttons, toaster lever, spring-loaded tongs, self-closing latch), set
    `stiffness="k"` and `springref="q_eq"` (in radians or meters).
*   **Scalar joint dynamics & radian limits**: On `hinge` and `slide` joints,
    `damping`, `armature`, `frictionloss`, and `stiffness` are **scalar**
    attributes (1 float, e.g., `damping="0.02"`—never a 3-float vector like
    `damping="0 1e-4 0"`), and `range="min max"` must be specified in
    **radians** for hinges (e.g., `range="-1.5708 0"`, never degrees
    `range="-90 0"`) or **meters** for slide joints under
    `<compiler angle="radian" autolimits="true"/>` with `solreflimit="0.005 1"`.

### 7.3 Actuators & Datasheet Conversion

*   **Unit conversion from hobby/industrial servo datasheets**:
    $$\tau_{\text{max}}\;(\text{N}\cdot\text{m}) = T_{\text{kg}\cdot\text{cm}}
    \times 0.0980665$$
*   **Position servo (`<position>`)**: Always bound both control range and
    actuator output force:

    ```xml
    <position name="jaw_servo" joint="jaw_hinge" kp="25.0" kv="1.5"
              ctrllimited="true" ctrlrange="0.0 0.85"
              forcelimited="true" forcerange="-3.5 3.5"/>
    ```
*   **Direct torque motor (`<motor>`)**:

    ```xml
    <motor name="wheel_drive" joint="wheel_spin" gear="1.0"
           ctrllimited="true" ctrlrange="-1.0 1.0"
           forcelimited="true" forcerange="-2.0 2.0"/>
    ```
*   **Recommended solver & integrator options (`integrator="discrete"`)**:

    ```xml
    <option timestep="0.002" integrator="discrete"/>
    ```

    **Use `integrator="discrete"` (`mjINT_DISCRETE`) as the default go-to
    integrator** (replacing `implicitfast`; see the
    [MuJoCo Generalized Integrators documentation][mj-integrators]). `discrete`
    is a strict improvement over `implicitfast` that lets you use larger
    timesteps and is especially stable when there is high stiffness in the model
    (e.g., high-gain position-controlled arms, stiff springs, or very stiff
    contacts).
*   **Closed kinematic loops (4-bar linkages & parallel mechanisms)**: Since
    MJCF `<worldbody>` is a kinematic tree, break any closed mechanical loop at
    one pivot and join the two branches with an equality constraint:

    ```xml
    <equality>
      <connect name="fourbar_closure" body1="coupler_link" body2="rocker_link"
               anchor="0.04 0 0" solref="0.004 1"/>
    </equality>
    ```

--------------------------------------------------------------------------------

## 8. Programmatic Authoring, Editing & Scene Attachment with `MjSpec`

When building, normalizing, or modifying assets programmatically in Python or
C++, always use `mujoco.MjSpec` (never fragile regex or string manipulation on
XML). Refer directly to the official MuJoCo documentation for authoritative API
patterns:

*   **Procedural `MjSpec` authoring, default classes (`MjsDefault`), bodies,
    geoms, joints, sites, actuators & sensors**: See
    [MuJoCo `MjSpec` Documentation][mjspec-docs].
*   **Attaching an asset into a scene with an attachment `<frame>` and adding a
    `<freejoint/>`**: See [MuJoCo Python Bindings][mj-python].

--------------------------------------------------------------------------------

## 9. Asset Auditing & Verification

### 9.1 Programmatic Audit & Remediation Skill (`mujoco-audit`)

Whenever you are asked to **audit, triage, validate, or remediate** an asset (or
verify a newly created asset before completion), **read and follow
[`mujoco-audit`](../skills/mujoco-audit/SKILL.md)**:

*   **Prompt-Driven Scope & Speed**: Adapts testing depth (`--scenarios=all` vs.
    a fast/targeted subset like `--scenarios=1,4` or `--static_only`) and
    deliverables to the user's prompt (Fast/Lightweight, User-Guided Symptom,
    Read-Only Triage, Rigid-Only, or Full Guarantee).
*   **7-Point Static Checklist & 10-Scenario Auto-Scaling Physics Suite**: Runs
    [`audit_scenarios.py`][audit-script] (including `validate_static_and_mesh_hygiene`,
    `compute_collision_alignment_summary`, and 3-axis orthographic
    ray/cavity/through-hole probing) to produce a structured `.json` scorecard
    and `.png` visual contact sheet, evaluated with engineering judgment for the
    object's shape.
*   **Custom `MjSpec` Experiments & Before-vs-After Remediation**: Use `uv run`
    for custom mechanism probes and in-memory fixes, and
    [`mujoco-showcase-shot`](../skills/mujoco-showcase-shot/SKILL.md) for 360°
    Filament PBR turntable videos.

[audit-script]: ../skills/mujoco-audit/scripts/audit_scenarios.py
[mj-integrators]: https://mujoco.readthedocs.io/en/latest/computation/index.html#geintegrators
[mj-python]: https://mujoco.readthedocs.io/en/latest/python.html
[mj-vis]: https://mujoco.readthedocs.io/en/latest/programming/visualization.html
[mjspec-docs]: https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjspec
[s-shot]: ../skills/mujoco-showcase-shot/scripts/render_showcase_shot.py
