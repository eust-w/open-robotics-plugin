---
name: blender-creation
description: >-
  Creates physically calibrated, articulated,
  PBR-textured 3D MuJoCo assets (MJCF
  model.xml, .mjz, mjSpec) from scratch
  using headless Blender (bpy, bmesh,
  blender_mesh_utils, asset_build_utils),
  convex collision synthesis, 10-scenario
  auditing, and Studio preview. Use only
  when asked to create, model, or build a
  3D MuJoCo object from a URL, image, or
  text description, or when blender-creation
  is named. Don't use for non-MuJoCo
  Blender/Cycles/CAD modeling.
metadata:
  publisher: google
---

# Authoring 3D MuJoCo Assets from Scratch in Blender (`bpy` / `bmesh` + PBR)

> [!IMPORTANT]
>
> **Companion to [`rules/mujoco_asset_standards.md`][asset-standards]**: Read
> [`rules/mujoco_asset_standards.md`][asset-standards] first for the mandatory
> MuJoCo coordinate frame (`UP_POSZ_FORWARD_POSX_RIGHT_NEGY`), SI unit rules,
> group separation (`group="2"` visual, `group="3"` collision, `group="4"`
> sites), and physics/audit requirements. This guide covers how to research,
> measure, model, sculpt, UV-unwrap, texture, bake, export, verify, and
> interactively preview new assets from scratch using headless Blender
> (`PYTHONDONTWRITEBYTECODE=1 "$BLENDER_BIN" -b -P ...`),
> [`scripts/blender_mesh_utils.py`](scripts/blender_mesh_utils.py) (`bmu`),
> [`scripts/asset_build_utils.py`](scripts/asset_build_utils.py) (`abu`),
> [`mujoco-audit/SKILL.md`](../mujoco-audit/SKILL.md),
> [`mujoco-showcase-shot/SKILL.md`](../mujoco-showcase-shot/SKILL.md)
> ([`render_showcase_shot.py`](../mujoco-showcase-shot/scripts/render_showcase_shot.py)),
> and [`mujoco-studio-sidecar/SKILL.md`](../mujoco-studio-sidecar/SKILL.md)
> so they look, behave, and inspect accurately in **MuJoCo's Filament PBR
> renderer** and **MuJoCo Studio**.

---

## 1. Entry Point, Subagent Orchestration, Local Blender Setup & Reference Intake

### 1.0 Invocation, Defaults & Subagent Orchestration

Users invoke this skill in natural language with a product URL, reference image(s), or text specifications:

> *"Build a 3D MuJoCo asset for these kitchen tongs: `<url>`"*
> *"Follow blender-creation to model the object in `<image/link>`"*

**Defaults & Optional Overrides**:
By default, the skill executes the full end-to-end pipeline: auto-detected articulations, intermediate diagnostic renders, 10-scenario dynamic physics simulation auditing (§7.3), and a 360° Filament turntable showcase video (§7.4). Users can optionally request overrides in plain English or bullet points:
- **`Fast` / `No Turntable`**: Skips Step 7.4 showcase video rendering (saves render time).
- **`Rigid-Only`**: Forces `0` joints, authoring the object as a single rigid body.
- **`Skip Physics Audit` / `Fast Audit`**: Skips the 10-scenario dynamic physics simulation suite in Step 7.3 (relying only on Step 7.1 static hygiene & collision alignment checks).

1. **If you are the top-level / parent conversation**:
   - Spawn a `self` subagent (`Model: "inherit"`) via `invoke_subagent` to build and verify the asset, passing all user-specified instructions, URLs, and overrides, `Parent Conversation ID: <parent_conv_id>`, and the target asset output parent directory, and instructing the subagent to read this `blender-creation` skill, [`rules/mujoco_asset_standards.md`][asset-standards], [`mujoco-audit/SKILL.md`](../mujoco-audit/SKILL.md), and [`mujoco-showcase-shot/SKILL.md`](../mujoco-showcase-shot/SKILL.md). Do **not** instruct the subagent to write scratch/intake files inside the parent conversation's `brain/<parent_conv_id>/` directory (macOS Seatbelt sandboxing blocks cross-conversation `brain/` access with `PermissionError`); let the subagent use its own `<ArtifactDir>/scratch/` and return the finished asset and preview paths.
   - **When the subagent finishes**: Follow **Step 7.6** ([`mujoco-studio-sidecar/SKILL.md`](../mujoco-studio-sidecar/SKILL.md)) in the top-level conversation to launch (or reuse) the MuJoCo Studio Web sidecar daemon (`studiod.py`) pointed at the finished asset directory (`--workspace_dir=<asset_dir>`) and verify `/_studio/status` (`"ok": true`, `"daemon_connected": true`). Launch `studiod.py` from the top-level conversation rather than inside the builder subagent so the background daemon is not reaped when the subagent exits. In your final response, first present the finished asset path, model summary (dimensions, mass, joints, visual/collision breakdown, audit status), side-by-side comparison, and turntable video (unless skipped via `Fast (No Turntable)`, briefly noting in a 1-line caption that the reel shows a full 360° textured visual spin followed by translucent physics collision hull inspection), and **place `[MuJoCo Studio](sidecar://robotics/mujoco_studio/)` (plus `http://127.0.0.1:<http_port>/` from `$SESSION_FILE` if the Node sidecar proxy is not active) as the very last item at the bottom of your response** so the interactive sidecar pill is never missed at the top.

2. **If you are already the spawned Main Builder Subagent**:
   - Do **not** spawn a nested build subagent—execute **Sections 1.1 through 7.5** directly (you may still invoke the read-only `browser` subagent during Step 1.1 if a product URL requires client-side JS rendering). Keep all scratch/intake files in **this subagent's own** `<ArtifactDir>/scratch/` so sandboxed `run_command` steps have local read/write access without `PermissionError`. Do **not** run **Step 7.6** (`studiod.py`) inside the subagent; instead, return the finished `<asset_dir>` path, model summary, comparison images, and turntable video path to the parent conversation so the parent conversation launches `studiod.py` and presents the Studio link.
   - **Ensure `uv` and Headless Blender (`4.2+`) Are Installed Locally (macOS & Linux)**:
     ```bash
     uv --version
     if [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
       BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
     else
       BLENDER_BIN="blender"
     fi
     "$BLENDER_BIN" --version || (sudo apt-get update && sudo apt-get install -y blender)
     ```
     Run Blender scripts with `PYTHONDONTWRITEBYTECODE=1 "$BLENDER_BIN" -b -P <ArtifactDir>/scratch/build_asset.py` and add `<PLUGIN_DIR>/skills/blender-creation/scripts` to `sys.path` inside the Blender script to `import blender_mesh_utils as bmu`.
   - **MuJoCo Compilation, Filament Showcase Renders & Reference Intake (`uv run`)**:
     ```bash
     uv run --with mujoco --with numpy --with scipy --with trimesh --with Pillow \
       python3 <ArtifactDir>/scratch/script.py
     ```
     Inside your scratch script, add the plugin script directories (`<PLUGIN_DIR>/skills/blender-creation/scripts`, `<PLUGIN_DIR>/skills/mujoco-audit/scripts`, and `<PLUGIN_DIR>/skills/mujoco-showcase-shot/scripts`) to `sys.path` to import `asset_build_utils as abu`, `audit_scenarios`, and `render_showcase_shot`.
   - **Scope & Workspace Hygiene**:
     - Run `audit_scenarios` 10-scenario physics drop/tip/grasp simulations (**Step 7.3**) by default, verifying all scenarios pass (or remediate failing checks following [`mujoco-audit/SKILL.md`](../mujoco-audit/SKILL.md)). Only skip dynamic simulations if the user explicitly specifies `Skip Physics Audit` or `Fast Audit`. Do not generate a second `<asset_name>_mechanical/` bundle unless requested. Verify clean static `MjModel` compilation, zero-backface normals (`abu.render_blender_diagnostic_sheet`), collision alignment (`audit_scenarios.compute_collision_alignment_summary`), kinematic joint poses (`Articulations: Auto`), and all dynamic physics drop/tip/settle scenarios.
     - Keep all scratch scripts, downloaded reference photos, intermediate meshes, diagnostic `.jpg`/`.png` sheets, `.blend` files, virtual environments (`.venv`), and caches in `<ArtifactDir>/scratch/`, `~/.venvs/`, and `~/.cache/mujoco_pbr_cache`. Only write final asset deliverables (`model.xml`, `meshes/*.obj`, `textures/*.png`) into the target asset directory, and never write `.pyc` / `__pycache__`, `.venv`, or `studiod` scaffold/runtime files into the asset directory.
   - Honor knob overrides: `Articulations: Rigid-Only` -> `0` joints; `Depth: Fast (No Turntable)` -> skip Step 7.4; `Skip Physics Audit` / `Fast Audit` -> skip Step 7.3.

### 1.1 Intake, Reference Contact/Detail Sheets & Form Brief

1. **Product URL & Reference Intake (`<= 8` Image Reads)**:
   - Never guess product identity, category, or dimensions from a URL path slug (e.g., `/Walnut-Elegant-Design-Coffee-Products/dp/...` may be a truncated SEO slug rather than a coffee table).
   - **When a product URL is provided**:
     1. Invoke the read-only `browser` subagent to navigate to `<product_url>` on behalf of the user. Inspect the live rendered page just as a human designer would:
        - Extract the verified product title, materials, mechanism descriptions, and dimensional specs (`L × W × H` in `mm` and bare-product mass in `kg`, cross-checking retail `Item Weight` vs packaged shipping weight).
        - Identify high-resolution product gallery / carousel image URLs.
     2. **If access is blocked by an interactive CAPTCHA or security gate**: stop without attempting to circumvent the gate and ask the user for the product dimensions, specifications, or reference photos directly.
     3. Download the collected high-resolution gallery images into `<ArtifactDir>/scratch/intake/ref_*.jpg` (e.g. using `curl` or a minimal inline download script).
   - **When no URL is provided (text-only query)**: Look up bare-product `L x W x H` (`mm`) and mass (`kg`) via `search_web` (or invoke the `browser` subagent if `search_web` returns only bare redirect links without snippets). Cross-check retail `Item Weight` against dimensions and material density.
   - **Composite Sheets (`<= 8` Image Reads)**: Never call `read_file` on dozens of individual frames. Stitch the downloaded `ref_*.jpg` photos and sub-feature crops into `ref_contact_sheet` and `ref_details_grid` via `abu.build_ref_contact_and_detail_sheets`, then inspect those composite sheets with `read_file`.
2. **Sample Photometric RGB Percentiles Directly from Reference Crops (`rgb_sample_boxes`)**:
   - Do **not** invent `RGB` values from verbal category names (*"red plastic"*, *"black rubber"*, *"green plush"*). Pass `rgb_sample_boxes=[('main_shell', ref_path, (x0, y0, x1, y1)), ...]` to `abu.build_ref_contact_and_detail_sheets` to measure `[p25, p50, p75]` sRGB tuples directly from uniform interior surface patches (excluding specular hotspots and cast shadows).
3. **Write a `<= 25`-Line Form Brief & Named Parameter Block `P` (`<= 40` Named Dimensions)**:
   - Before writing Blender `bmesh` code, write a concise **Form Brief** and define a named parameter dictionary `P = dict(...)` (`<= 40` named dimensions in `mm` / `m` and `deg` with provenance comments) rather than hard-coding hundreds of magic coordinate tuples:
     - **Named primary surfaces & how they meet**: side silhouette facets/angles, plan-view taper, crown cross-section, negative space / open daylight gaps between parts, and flush vs. stepped junctions.
     - **Left/right (`+Y` vs. `-Y`) and front/back (`+X` vs. `-X`) asymmetry**: explicitly note which levers, clips, buttons, labels, or ports appear on each side.
     - **Material regions & parting lines**: e.g., two-shot overmold where rigid ABS forms the inner core and exposed flush rims while matte TPE forms the outer grip skin, or recessed branding pockets.
     - **Mechanisms & travel**: moving sub-parts (hinged lids, sliders, clips, blades, buttons, knobs) with true physical pivot/axis and travel in `m` or `rad`.

---

## 2. Tool Reference & Target Renderer Rules

### 2.1 Quick Helper API Reference (`blender_mesh_utils`, `asset_build_utils`, `render_showcase_shot`)

- **`blender_mesh_utils as bmu`** *(inside headless Blender `"$BLENDER_BIN" -b -P`, from [`scripts/blender_mesh_utils.py`](scripts/blender_mesh_utils.py))*:
  - `bmu.clear_scene() -> None`
  - `bmu.assign_pbr_material(objects, name=None, *, base_color=None, base_color_srgb=(200, 200, 200), roughness=0.4, metallic=0.0, normal_bump_mm=0.0, bump_height_mm=0.0, bump_scale=250.0) -> bpy.types.Material`
  - `bmu.skin_skeleton(name, nodes_xyzr, edges=None, *, subsurf_levels=2) -> obj`
  - `bmu.metaball_sdf(name, elements, *, resolution=0.002) -> obj`
  - `bmu.swept_tube(name, points_xyzr, *, bevel_resolution=6) -> obj`
  - `bmu.voxel_smooth_union(name, objects, *, voxel_size=0.0015, smooth_iterations=6) -> obj`
  - `bmu.apply_boolean_cutter(target_obj, cutter_obj, *, operation='DIFFERENCE', solver='MANIFOLD') -> obj`
  - `bmu.smart_seam_unwrap(target_obj, *, angle_limit_deg=66.0, margin=0.005) -> obj`
  - `bmu.smart_uv_project(objects, *, angle_limit_deg=66.0, island_margin=0.01, margin=None) -> list[obj]`
  - `bmu.pack_shared_uv_atlas(objs, *, margin=0.005, rotate=True, align_minimal_bbox=True, match_3d_area=True) -> None`
  - `bmu.finalize_and_export_obj(target_obj, out_obj_path, *, crease_deg=None, world_pos=None, body_pos=None, ensure_uvs=True) -> {'verts', 'quads', 'tris', 'earclipped_ngons'}`
  - `bmu.export_convex_hull_col(source_obj_or_points, out_obj_path, *, clip_bounds_world_m=None, clip_bounds_m=None, max_verts=64, name=None, world_pos=(0,0,0), body_pos=None) -> col_obj`
  - `bmu.bake_pbr_maps(out_dir, *, material_groups=None, resolution=1024, samples=8) -> {key: {'basecolor', 'roughness', 'metallic', 'normal'}}`
- **`asset_build_utils as abu`** *(inside `uv run` scripts, from [`scripts/asset_build_utils.py`](scripts/asset_build_utils.py))*:
  - `abu.build_ref_contact_and_detail_sheets(ref_paths, detail_crops, out_contact_png, out_details_png, *, raw_crops_dir=None, rgb_sample_boxes=None, cols=3, cell_size=440, include_shadow_lift=False) -> dict`
  - `abu.fetch_ambientcg_maps(asset_id, *, cache_dir=None, resolution='1K') -> {'color', 'roughness', 'normal_gl', 'metallic', 'ao', 'opacity', ...}`
  - `abu.rasterize_uv_world_fields(obj_paths, canvas_size=1024, *, part_offsets=None) -> dict`
  - `abu.dilate_uv_gutters(arr, valid_mask, *, max_dist_px=12) -> np.ndarray`
  - `abu.triplanar_project_map(map_2d, xyz, world_normal, *, texels_per_meter=4000.0, sharpness=4.0) -> np.ndarray`
  - `abu.height_to_normal_rgb(height_map, strength=3.5, *, smooth_sigma=0.0, valid_mask=None) -> (H,W,3) uint8`
  - `abu.normal_rgb_from_3d_height_fn(height_fn, uv_world_fields, *, step_mm=0.15, strength=1.0) -> (H,W,3) uint8`
  - `abu.fbm_noise_3d(xyz, *, base_freq=25.0, wavelength_mm=None, octaves=4, persistence=0.5, seed=42) -> (...,) float32`
  - `abu.compute_volume_weighted_hull_masses(mesh_dir, col_names, total_mass_kg, *, density_weights=None) -> {col_name: mass_kg}`
  - `abu.render_blender_diagnostic_sheet(blend_path, out_png, *, xml_path=None, part_pivots=None, panel_size=420) -> {'out_png', 'has_backfaces', 'backface_pixels_per_view'}`
  - `abu.build_variant_bundles(base_asset_dir, parent_assets_dir, base_name, variants, render_variant_textures_fn) -> list[str]`
- **`render_showcase_shot`** *(inside `uv run` scripts, from [`../mujoco-showcase-shot/scripts/render_showcase_shot.py`](../mujoco-showcase-shot/scripts/render_showcase_shot.py))*:
  - `render_side_by_side_comparison(asset_path, ref_specs, out_png, *, detail_crops_out_png=None, six_view_out_png=None, six_view_panel_size=420, six_view_qpos=None, fovy=35.0, white_bg=True, clay_only=False) -> {'comparison_png', 'detail_crops_png', 'six_view_png'}`
  - `render_filament_still(asset_path, views, *, width=800, height=800, fovy=None, white_bg=False, show_plinth=False) -> list[np.ndarray]`
  - `render_6view_inspection_sheet(asset_path, out_png, *, panel_size=420, qpos=None) -> out_png`
  - `render_wireframe_and_uv_sheet(asset_path, out_png, *, panel_size=440) -> out_png`
  - `render_showcase_shot_video(asset_path, output_mp4, width=960, height=720, fps=30, duration=8.0, articulation_cycles=2.0) -> dict`

---

### 2.2 Target Renderer: Optimize for MuJoCo Filament (Not Standalone Cycles)

1. **Get into MuJoCo Filament Early & Render Intermediate Diagnostics**:
   - Do **not** build an entire asset blind and only render at the very end. After blocking out primary 3D geometry in Blender (saving `<ArtifactDir>/scratch/<asset_name>.blend` and exporting `.obj` meshes), immediately run **intermediate diagnostic renders**:
     - `abu.render_blender_diagnostic_sheet(blend_path, out_png=..., xml_path=...)` to inspect 6-view clay/wireframe shading, backface orientation (`has_backfaces == False`), and joint pivots.
     - `render_showcase_shot.render_side_by_side_comparison(..., clay_only=True)` to compare your untextured 3D form directly against the reference photos at matched camera angles before spending time on textures and decals.
2. **MuJoCo Filament Normal Handling (`model_resources.cc`) — Smooth Curves vs. Crisp Creases**:
   - In MuJoCo's Filament backend (`FillMeshBuffer`), if the angle between a triangle's geometric `face_normal` and its exported vertex normal `mesh_normal` exceeds $\sim 36.87^\circ$ ($\cos\theta < 0.8$), Filament may fall back to the flat `face_normal` (`UseFaceNormal`).
   - **For smooth curved surfaces, bevels & domes**:
     1. Always call `bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])` on every closed mesh before setting smooth shading or exporting `.obj`.
     2. Set `f.smooth = True` on smoothly curved/beveled faces and keep adjacent bevel/fillet step angles $\le 15^\circ\text{–}22^\circ$ so `dot(face_normal, vertex_normal) > 0.8`.
   - **For crisp industrial-design creases, chamfers & faceted parts**:
     1. Do **not** blindly set `f.smooth = True` across sharp ridges, flat-topped prismatic facets, or distinct chamfer bevels.
     2. Use `bmu.finalize_and_export_obj(obj, out_obj_path, crease_deg=35.0, body_pos=...)` (which marks crease edges above `crease_deg` sharp and applies `WEIGHTED_NORMAL(keep_sharp=True)`) or keep flat facets `f.smooth = False` so Filament preserves crisp specular transitions.

---

## 3. Pure-Blender Geometry Authoring, Modular `bmu` Helpers & Topology Rules

### 3.1 Freedom of Geometry & Modular `blender_mesh_utils` (`bmu`) Builders

You have full freedom in Python (`bpy` / `bmesh` / `mathutils` / modifiers) plus the zero-magic builders in [`scripts/blender_mesh_utils.py`](scripts/blender_mesh_utils.py) (`bmu`) to construct whatever geometry faithfully captures each component:

- **Profile Lofting, Extrusion, Bevels & Boolean Split Shells (`bmesh` + `bmu.apply_boolean_cutter`)**:
  - For faceted or contoured industrial housings (e.g., an ergonomic tool housing with shoulder chamfers, plan-view taper, and two-shot elastomer/ABS overmold parting lines), drive cross-sections from your named parameter dictionary `P`.
  - Use `bmu.apply_boolean_cutter(target_obj, cutter_obj, operation='DIFFERENCE' | 'INTERSECT' | 'UNION')` (which safely runs Blender's `MANIFOLD` boolean solver with an automatic welded `EXACT` fallback) to carve recessed pockets, moat grooves, actuator slots, and flush two-shot material splits from a shared outer envelope.
- **Declarative Organic & Tubular Builders (`bmu`)**:
  - `bmu.skin_skeleton(name, nodes_xyzr, edges=None, *, subsurf_levels=2)`: branching organic limbs, frames, or handles from `(x, y, z, r)` or `(x, y, z, (rx, ry))` skeleton graphs.
  - `bmu.metaball_sdf(name, elements, *, resolution=0.002)`: analytical implicit SDF blends (`BALL`, `CAPSULE`, `ELLIPSOID`, `CUBE`, with `use_negative=True` for ergonomic thumb indents or cavities).
  - `bmu.swept_tube(name, points_xyzr, *, bevel_resolution=6)`: 3D `NURBS` tubes, cords, straps, wire loops, or bent rods with per-point radius and end caps.
  - `bmu.voxel_smooth_union(name, objects, *, voxel_size=0.0015, smooth_iterations=6)`: watertight voxel-remeshed fillets across intersecting sub-parts.

### 3.2 Kinematic Root Selection & Mesh Splitting by Link / Material Role

1. **Kinematic Root Body Selection**:
   - Choose the **root `<body>`** to be the **primary stationary base or primary grasped housing** during normal operation, and make components that move relative to that housing the child `<body>` elements.
2. **Split into Separate Meshes by Articulation Link & Material Role**:
   - Export separate `.obj` files in `meshes/` for each articulated `<body>` (link) and each distinct `<material>` within a body.

### 3.3 Sensible Polycount Budgets & `bmesh` Topology Rules

- **Polycount Budgets**:
  - Simple flat/beveled component: $100\text{–}1,500\text{ tris}$.
  - Contoured housing with rounded corners, fillets & cutouts: $1,500\text{–}6,500\text{ tris}$.
  - Complete multi-part prop/appliance: **$5,000\text{–}20,000\text{ tris}$ total** across all visual meshes.
- **`bmesh` Topology Rules**:
  1. **Subdivide straight edges before carving localized edge features** and add an inner rim loop so cutouts stay confined cleanly to the outer lip.
  2. **Keep planar hole loops strictly inside the outer perimeter ($\ge 2\text{ mm}$ clearance)** before calling `bmesh.ops.triangle_fill`.
  3. **Prevent coplanar Z-fighting between layered parts**: separate layered surfaces by $\ge 0.2\text{ mm}$ or cut an interior pocket.
  4. **Uniform metric edge offsets**: inset perimeter bevels by a constant metric distance $d$ (in meters) along $X$ and $Y$ rather than radial fractions on elongated parts, and never pinch top/bottom rings into a zero-thickness knife-edge equator.
  5. **Watertight thickness + `inertia="shell"` on thin visual meshes**: extrude thin plaques, labels, blades, and lenses to $\ge 1.5\text{–}3\text{ mm}$ watertight thickness and declare `<mesh name="..._vis" file="..._vis.obj" inertia="shell"/>` so `mjCMesh::Compile` never fails with `mesh volume is too small`.

---

## 4. Canonical MuJoCo Coordinate Frame, Camera Angles & `model.xml`

1. **World Coordinate Frame (`UP_POSZ_FORWARD_POSX_RIGHT_NEGY`, `1 unit = 1.0 m`)**:
   - **`+Z`** Up (resting base at $z = 0.000\text{ m}$), **`+X`** Front, **`-X`** Back, **`+Y`** Left, **`-Y`** Right.
   - Looking at **`+X` Front** toward `-X`, viewer-left is **`-Y`** and viewer-right is **`+Y`**. Looking at **`-Y` Right** toward `+Y`, viewer-left is **`-X`** and viewer-right is **`+X`**.
2. **MuJoCo `mjvCamera` Azimuth & Elevation Convention**:
   - **`azimuth = 180°`**: Views **`Front (+X)`** | **`azimuth = 0°`**: Views **`Back (-X)`** | **`azimuth = 90°`**: Views **`Right (-Y)`** | **`azimuth = 270°`**: Views **`Left (+Y)`**.
   - **`elevation < 0°`**: Camera above looking downward (`-25°..-45°` = tabletop 3/4 view; `-88°` = overhead `Top (+Z)` view).
3. **Articulated Child Bodies & Non-Destructive Identity-Axis Export (`bmu.finalize_and_export_obj`)**:
   - Keep scene objects at their assembled world poses in `.blend`, and pass `body_pos=(px, py, pz)` to `bmu.finalize_and_export_obj(obj, out_obj_path, crease_deg=..., body_pos=(px, py, pz))` so `.obj` vertices export around each child body's local joint origin `(0, 0, 0)` with `forward_axis='Y', up_axis='Z', export_materials=False` without mutating live `.blend` positions.
4. **Canonical `model.xml` Skeleton** (`<asset_name>/model.xml`, no `scene.xml`, no `<option/>`, no root `<freejoint/>`, visual geoms in `group="2"` with `mass="0" density="0"`, physical mass on `group="3"` colliders):

```xml
<mujoco model="asset_name">
  <compiler angle="radian" meshdir="meshes" texturedir="textures" autolimits="true"/>

  <default>
    <default class="visual">
      <geom group="2" type="mesh" contype="0" conaffinity="0" density="0" mass="0"/>
    </default>
    <default class="collision">
      <geom group="3" type="mesh" contype="1" conaffinity="1" condim="4"
            friction="0.7 0.02 0.005" solref="0.004 1" margin="0"
            rgba="0.25 0.75 0.45 0.45"/>
    </default>
  </default>

  <asset>
    <texture name="body_basecolor" type="2d" file="body_basecolor.png" colorspace="sRGB"/>
    <texture name="body_roughness" type="2d" file="body_roughness.png" colorspace="linear"/>
    <texture name="body_normal" type="2d" file="body_normal.png" colorspace="linear"/>
    <texture name="body_metallic" type="2d" file="body_metallic.png" colorspace="linear"/>

    <!-- Set scalar roughness="1.0" (and metallic="1.0") when wiring texture layers -->
    <material name="body_mat" rgba="1 1 1 1" metallic="1.0" roughness="1.0">
      <layer texture="body_basecolor" role="rgb"/>
      <layer texture="body_roughness" role="roughness"/>
      <layer texture="body_normal" role="normal"/>
      <layer texture="body_metallic" role="metallic"/>
    </material>

    <mesh name="body_vis" file="body_vis.obj" inertia="shell"/>
    <mesh name="body_hull_0_col" file="body_hull_0_col.obj" maxhullvert="64"/>
  </asset>

  <worldbody>
    <body name="asset_name" pos="0 0 0" quat="1 0 0 0">
      <geom name="body_vis" class="visual" mesh="body_vis" material="body_mat"/>
      <geom name="body_hull_0_col" class="collision" mesh="body_hull_0_col" mass="0.250"/>
    </body>
  </worldbody>
</mujoco>
```

---

## 5. PBR Textures, Procedural Normal Maps & Cycles Baking

### 5.1 Cycles PBR Baking (`bmu.assign_pbr_material` + `bmu.bake_pbr_maps`) & 3D UV-World Field Synthesis

1. **Baking Procedural Materials in Blender (`bmu.assign_pbr_material` + `bmu.bake_pbr_maps`)**:
   - Assign procedural PBR materials (`bmu.assign_pbr_material` or custom Blender `ShaderNodeTexNoise`/`ShaderNodeBump` node trees) and bake 4-channel PBR maps (`basecolor`, `roughness`, `metallic`, `normal`) via `bmu.bake_pbr_maps`:
   ```python
   bmu.assign_pbr_material(
       ["housing_vis"], base_color_srgb=(200, 28, 34), roughness=0.32, bump_height_mm=0.08
   )
   bmu.bake_pbr_maps(f"{asset_dir}/textures", resolution=1024, samples=8)
   ```
2. **Fetching Real CC0 PBR Scans (`abu.fetch_ambientcg_maps`) & 3D UV-World Field Synthesis (`abu.rasterize_uv_world_fields`)**:
   - Use `abu.fetch_ambientcg_maps(asset_id, resolution='1K')` to fetch real CC0 PBR texture maps (cached in `~/.cache/mujoco_pbr_cache`).
   - For seam-free 3D procedural normal maps or triplanar projection across UV islands in `uv run`, rasterize the exported `.obj` UV layout with `fields = abu.rasterize_uv_world_fields(obj_path, canvas_size=1024)` and synthesize tangent-space normal maps via `abu.normal_rgb_from_3d_height_fn(height_fn, fields)` or `abu.height_to_normal_rgb(..., valid_mask=fields['valid_mask'])`.
   - **Filament Scalar-vs-Texture Multiplication Rule**: Always set `roughness="1.0"` (and `metallic="1.0"` when wiring a metallic map) on `<material>` so `<layer role="roughness"/>` and `<layer role="metallic"/>` represent true physical values without double-multiplying.
3. **Outward-Normal Planar UV Rule for Procedural Legends (`PIL.ImageDraw`)**:
   - When projecting planar UVs manually in `bmesh`, orient $u$ along the viewer's local rightward tangent $\hat{t}_{\text{right}} = \text{normalize}(\vec{n} \times \hat{z})$ using the face's outward normal $\vec{n}$ so opposing faces (`+Y` vs. `-Y`, `+X` vs. `-X`) are never horizontally mirrored.

---

## 6. Collision Hulls & Mass/Inertia (`bmu.export_convex_hull_col` & `coacd` Fallback)

1. **Collision Geometry Must Match the Function of the Asset**:
   - The `group="3"` collision representation must faithfully preserve the functional affordances of the asset:
     - **Open cavities & receptacles** (mugs, bowls, bins, trays, drawers) must remain open and unobstructed so smaller objects can be dropped or placed inside.
     - **Through-holes & grasp loops** (scissors finger rings, kettlebell/mug handles, trigger guards) must remain open so fingers or hooks can pass through.
     - **Stable resting bases & rolling contours**: Flat resting feet must sit flush at $z = 0.000\text{ m}$ without rocking on polyhedral facets, and cylindrical/spherical rolling surfaces must remain smooth and concentric.
     - **Articulation clearances**: Colliders on parent and child links must not self-collide or jam across the full joint range.
     - **Gap-free silhouette coverage (no needle penetration)**: Every visual surface must be backed by collision geometry so no ray/needle can pierce the visual mesh (`group="2"`) without intersecting a collider (`group="3"`).
2. **Prefer Hand-Authored Convex Hulls (`bmu.export_convex_hull_col`) & Analytic Primitives**:
   - **Agents should prefer creating convex hulls and analytic primitives by hand** directly in the Blender script where exact part boundaries and functional splits are known:
     - Do not approximate rounded housings or curved parts with blocky `<geom type="box"/>` primitives. Use `bmu.export_convex_hull_col(source_obj_or_points, out_obj_path, clip_bounds_world_m=(cmin, cmax), max_verts=64, body_pos=...)` to plane-clip (`bmesh.ops.bisect_plane`), downsample (`<= 64` vertices), validate 3D enclosed volume, and export tight convex hulls (`maxhullvert="64"`) directly from your visual meshes.
     - **Overlap adjacent slices & ring segments at seams**: When splitting a part across multiple `clip_bounds_world_m` boxes or arranging wall/ring primitives around a cavity or loop, overlap adjacent bounds/segments by `1–2 mm` (or share exact cut planes before downsampling) so no open cracks or corner gaps remain between neighboring colliders.
     - Use analytic primitives (`cylinder`, `capsule`, `sphere`, `box`) for genuinely geometric shafts, rods, wheels, and flat plates, or arrange ring segments around hollow cavities/loops.
     - Use `abu.compute_volume_weighted_hull_masses(mesh_dir, col_names, total_mass_kg, density_weights=...)` in `uv run` to distribute physical mass across `group="3"` collision geoms proportionally to hull volume and material density.
3. **Fallback to `coacd` for Complex Concave Geometry**:
   - If an asset's geometry is too complex to decompose by hand into plane-clipped hulls or primitives, **fall back to `coacd`** (the standard `coacd` package/CLI via `uv run --with coacd --with trimesh`) and **tune its parameters** (`threshold`, `max_convex_hull`, `preprocess_mode`, and `maxhullvert="64"`) so the decomposition matches the asset's function without plugging cavities or shattering simple walls into sliver hulls:
   ```python
   # Run via: uv run --with coacd --with trimesh --with numpy python3 <ArtifactDir>/scratch/run_coacd.py
   import coacd
   import trimesh

   mesh = trimesh.load("/path/to/asset/meshes/part_vis.obj", force="mesh")
   coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
   parts = coacd.run_coacd(
       coacd_mesh,
       threshold=0.05,        # Tune (e.g. 0.02..0.08) to preserve cavities without over-shattering
       max_convex_hull=16,    # Cap hull count per link to match functional features
       preprocess_mode="auto",
   )
   ```

---

## 7. Intermediate Verification, Alignment Check, Audit, Turntable & Studio Preview

### 7.1 Static Hygiene, Collision Alignment & 2-Column Side-by-Side Comparison (`render_side_by_side_comparison`)

Combine static `MjModel` compilation, mesh/UV hygiene, collision alignment (`audit_scenarios.compute_collision_alignment_summary`), intermediate Blender backface diagnostics (`abu.render_blender_diagnostic_sheet`), and showcase comparison renders (`ref_vs_filament_comparison.jpg` + `render_details_grid.jpg` + `six_view_inspection_sheet.jpg`) into a **single `verify_and_render.py` invocation**:

```bash
uv run --with mujoco --with numpy --with scipy --with trimesh --with Pillow \
  python3 <ArtifactDir>/scratch/verify_and_render.py
```

```python
import os
import sys

plugin_dir = os.environ.get("PLUGIN_DIR", "/path/to/plugins/robotics")
for sub in (
    "skills/blender-creation/scripts",
    "skills/mujoco-audit/scripts",
    "skills/mujoco-showcase-shot/scripts",
):
  p = os.path.join(plugin_dir, sub)
  if p not in sys.path:
    sys.path.insert(0, p)

import asset_build_utils
import audit_scenarios
import mujoco
import render_showcase_shot

xml_path = "/path/to/asset/model.xml"
blend_path = "/path/to/scratch/asset_name.blend"
scratch_dir = "/path/to/scratch"

if hasattr(audit_scenarios, "validate_static_and_mesh_hygiene"):
  hygiene = audit_scenarios.validate_static_and_mesh_hygiene(xml_path)
  assert hygiene["passed"], f"Hygiene errors: {hygiene['errors']}"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
align_res = audit_scenarios.compute_collision_alignment_summary(m, d)
assert align_res["status"] != "FAIL", f"Alignment failed: {align_res['issues']}"
if m.njnt > 0:
  for _ in range(100):
    mujoco.mj_step(m, d)
  assert d.qacc.size == 0 or all(abs(float(x)) < 1e6 for x in d.qacc), "Unstable joint settle!"
  mujoco.mj_resetData(m, d)
  mujoco.mj_forward(m, d)

diag = asset_build_utils.render_blender_diagnostic_sheet(
    blend_path=blend_path, out_png=f"{scratch_dir}/blender_diag_sheet.jpg", xml_path=xml_path
)
assert not diag["has_backfaces"], f"Flipped faces: {diag['backface_pixels_per_view']}"

ref_specs = [
    {
        "ref_path": f"{scratch_dir}/intake/ref_00.jpg",
        "label": "View 1 (Hero View)",
        "azimuth": 135.0,
        "elevation": -25.0,
        "white_bg": True,
        "crop_boxes_norm": [("Primary Form & Surface", (0.25, 0.20, 0.75, 0.70))],
    },
]

if "--clay" in sys.argv:
  render_showcase_shot.render_side_by_side_comparison(
      xml_path,
      ref_specs=ref_specs,
      out_png=f"{scratch_dir}/ref_vs_clay.jpg",
      detail_crops_out_png=f"{scratch_dir}/clay_details_grid.jpg",
      clay_only=True,
  )
  sys.exit(0)

render_showcase_shot.render_wireframe_and_uv_sheet(
    xml_path, out_png=f"{scratch_dir}/wireframe_and_uv_sheet.jpg"
)
render_showcase_shot.render_side_by_side_comparison(
    xml_path,
    ref_specs=ref_specs,
    out_png=f"{scratch_dir}/ref_vs_filament_comparison.jpg",
    detail_crops_out_png=f"{scratch_dir}/render_details_grid.jpg",
    six_view_out_png=f"{scratch_dir}/six_view_inspection_sheet.jpg",
)
```

### 7.2 Difference-Driven Self-Review (Iterative Intermediate & Final Critique)

- Inspect `blender_diag_sheet.jpg`, `ref_vs_clay.jpg` (during geometry blockout), `ref_vs_filament_comparison.jpg`, `render_details_grid.jpg`, and `six_view_inspection_sheet.jpg` directly with `read_file`.
- Before every geometry or material revision, explicitly list the **top 3 visual discrepancies** between the reference photo and the render:
  1. **View** (e.g., `View 1 Right (-Y)` or `View 2 Front-Right 3/4`),
  2. **Feature** (e.g., front-top chamfer slope, upper rim thickness, actuator button height),
  3. **Estimated error in `mm` or `deg`** (e.g., upper rim is `~4 mm` too high; nose plan taper is `~3 mm` too wide),
  4. **Owning key in `P` or material parameter** (e.g., `P["rim_z_frac"] = 0.72 -> 0.66`), then re-run your Blender script and `verify_and_render.py` to confirm the fix.

### 7.3 Programmatic Physics Scenario Audit (`audit_scenarios`)

By default (unless overridden by `Skip Physics Audit` / `Fast Audit`; see [`mujoco-audit/SKILL.md`](../mujoco-audit/SKILL.md) and [`audit_scenarios.py`](../mujoco-audit/scripts/audit_scenarios.py)), run the 10-scenario physical drop/tip/grasp simulation test suite:

```bash
uv run "$PLUGIN_DIR/skills/mujoco-audit/scripts/audit_scenarios.py" \
  --xml_path=/path/to/asset/model.xml \
  --out_dir=<ArtifactDir>/scratch/audit_out \
  --label=<asset_name>
```

Verify that all scenarios pass. If a scenario fails (e.g., penetration, excessive bounce, or unstable resting contact), follow the engineering-judgment remediation rules in [`mujoco-audit/SKILL.md`](../mujoco-audit/SKILL.md) to adjust mass/inertial diagonal, contact friction, or collision geoms before finalizing the asset.

### 7.4 Final 360° Filament PBR Turntable Showcase (`render_showcase_shot`)

By default (unless overridden by `Fast (No Turntable)`; see [`mujoco-showcase-shot/SKILL.md`](../mujoco-showcase-shot/SKILL.md) and [`render_showcase_shot.py`](../mujoco-showcase-shot/scripts/render_showcase_shot.py)):

```bash
uv run "$PLUGIN_DIR/skills/mujoco-showcase-shot/scripts/render_showcase_shot.py" \
  --asset=/path/to/asset/model.xml \
  --output=<ArtifactDir>/showcase_shots/<asset_name>_turntable.mp4 \
  --width=960 --height=720 --fps=30 --duration=8.0
```

### 7.5 Color/Finish Variants in Separate MJZ-Ready Directories (`build_variant_bundles`)

Every variant must be emitted into its own self-contained sibling directory (`<asset_name>_<variant>/model.xml` with its own `meshes/` and `textures/`—never `model_<variant>.xml` inside a shared folder):

```python
import os
import sys

skill_dir = os.environ.get("SKILL_DIR", "/path/to/skills/blender-creation")
sys.path.insert(0, os.path.join(skill_dir, "scripts"))
import asset_build_utils

variant_dirs = asset_build_utils.build_variant_bundles(
    base_asset_dir=base_asset_dir,
    parent_assets_dir=parent_assets_dir,
    base_name=base_name,
    variants={"<variant_name>": {...}},
    render_variant_textures_fn=render_variant_textures,
)
```

### 7.6 Present the Asset in Interactive MuJoCo Studio (`mujoco-studio-sidecar`)

Once the asset bundle (`<asset_dir>/model.xml`, `meshes/*.obj`, `textures/*.png`) is complete and verified, present it loaded into an interactive **MuJoCo Studio** window following [`mujoco-studio-sidecar/SKILL.md`](../mujoco-studio-sidecar/SKILL.md) (and the bundled `mujoco-studio` skill):

1. **Install `mujoco.studio` (if not already present) without polluting
   `<asset_dir>`**:
   - Check whether `python -c "import mujoco.studio, mujoco_studio_wasm"`
     succeeds in your active environment; if not, install `mujoco>=4.0.1` from
     PyPI (`pip install --upgrade "mujoco>=4.0.1"` or
     `uv pip install --upgrade "mujoco>=4.0.1"`) into an existing workspace
     `.venv` or an external venv such as `~/.venvs/mujoco-studio` — **never**
     create a `.venv` inside `<asset_dir>`.
2. **Reuse an active session or launch `studiod.py` for `<asset_dir>`**:
   - Inspect `${TMPDIR:-/tmp}/mujoco_studio_session_$(id -u).json` first to see if a daemon is already running. When `<asset_dir>/model.xml` already exists, `studiod.py --workspace_dir=<asset_dir>` automatically detects `model.xml`, skips creating any starter Python scaffold files inside `<asset_dir>`, and writes its runtime files (`status.json`, `latest_snapshot.json`) to an out-of-workspace runtime directory:

   ```bash
   SESSION_DIR="${TMPDIR:-/tmp}"
   SESSION_FILE="${SESSION_DIR%/}/mujoco_studio_session_$(id -u).json"
   SIDECAR_FILE="${SESSION_DIR%/}/mujoco_studio_sidecar_$(id -u).json"
   ```

   - Check whether the Antigravity Node sidecar proxy (`main.mjs`) is active
     (`$PLUGIN_DATA/sidecar.json`,
     `$ANTIGRAVITY_EXECUTABLE_DATA_DIR/sidecar.json`, or `$SIDECAR_FILE`). If no
     sidecar discovery file exists (for example, on Antigravity clients where
     `enable-ui-sidecars` is disabled), add `--open_browser` so `studiod.py`
     also opens the local WebViewer tab automatically.
   - Launch [`studiod.py`](../mujoco-studio-sidecar/scripts/studiod.py) in the background (`IsDaemon: true`, and `BypassSandbox: true` if running inside a sandboxed shell so loopback IPC is accessible) from the **top-level / parent conversation**:

   ```bash
   ~/.venvs/mujoco-studio/bin/python "$PLUGIN_DIR/skills/mujoco-studio-sidecar/scripts/studiod.py" \
     --workspace_dir=<asset_dir>
   ```

   - If you created multiple variants in Step 7.5, point `--workspace_dir` at the primary asset (or pass `--scene_file=...` / restart `--workspace_dir=<variant_dir>` when switching to inspect a specific variant).
3. **Verify daemon connection & hot-reload**:
   - Read `control_port` and `http_port` from `$SESSION_FILE` and query `curl --noproxy '*' -s http://127.0.0.1:<control_port>/_studio/status` to confirm `"ok": true` and `"daemon_connected": true` (note: `"daemon_connected"` and `"http_port"` are reported by `/_studio/status` and `$SESSION_FILE`, while `status_file` on disk records `"ok"` and scene load status).
   - If you refine `model.xml`, `meshes/*.obj`, or `textures/*.png` after launching, MuJoCo Studio automatically watches and hot-reloads the model in place — re-check `/_studio/status` or `status_file` after edits to confirm reload succeeded without errors.
   - If the user wants custom interactive controls or a live drop/manipulate testbench alongside the asset, keep `<asset_dir>/model.xml` pure and load it via `mujoco.MjSpec.from_file("<asset_dir>/model.xml")` (`spec.worldbody.add_body(...).add_freejoint()` + `self.handle.set_model(spec, watch=[...])`) in a separate Studio workspace outside `<asset_dir>`.
4. **Surface the interactive link(s) at the very end of the final response**:
   - **Placement requirement**: Put `[MuJoCo Studio](sidecar://robotics/mujoco_studio/)` (and any fallback `[MuJoCo Studio Web](http://127.0.0.1:<http_port>/)` link) as the **very last thing at the end of your final response** — **after** the finished asset path, model summary, side-by-side comparison images, and turntable video. Never place the `sidecar://` link at the top or middle of the response, where the rendered pill is easy to overlook.
   - Always include `[MuJoCo Studio](sidecar://robotics/mujoco_studio/)` (do not suppress it even if the host client does not auto-open `sidecar://` panes).
   - If no active Node sidecar discovery file is present (so the sidecar aux-pane proxy is not running on the user's client), also include the direct local WebViewer URL `[MuJoCo Studio Web](http://127.0.0.1:<http_port>/)` (using `http_port` from `$SESSION_FILE` / `/_studio/status`) right alongside it at the end so the user can open the interactive 3D viewer in one click.

[asset-standards]: ../../rules/mujoco_asset_standards.md
