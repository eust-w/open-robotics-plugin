---
name: mujoco-showcase-shot
description: >-
  Renders a studio-lit 360-degree turntable showcase video (.mp4) and keyframe
  sheet for a MuJoCo asset (MJCF model.xml, .mjz, or mjSpec/MjModel) using
  MuJoCo's Filament PBR renderer (mjrf), sweeping articulated joints, pulsing
  collision geoms, and drawing a HUD overlay. Use only when rendering a showcase
  turntable video for a MuJoCo MJCF, .mjz, mjSpec, or MjModel asset. Don't use
  for non-MuJoCo turntable, Filament, OpenGL, or CAD renders.
metadata:
  publisher: google
---

# MuJoCo Filament Showcase Shot Turntable Video Generator

Use this skill whenever you want to generate a polished, studio-lit **360°
turntable showcase video (`.mp4`)** for a MuJoCo MJCF asset (`model.xml`) using
**MuJoCo's Filament PBR renderer (`mjrf`)**.

## What It Showcases

1.  **Filament Studio Softbox IBL, Gradient Skybox & Reflective Turntable
    Plinth**:
    *   Automatically wraps the asset in a studio scene with a procedural
        6-face studio softbox cubemap IBL (`studio_ibl_tex`) paired with a
        balanced 4-point physical studio lighting rig (shadow-casting key
        spotlight, cool fill, warm rim, and upward floor-bounce directional
        light), ACES tone mapping, high-quality SSAO, and subtle bloom.
    *   Places the asset on a smooth 96-slice two-tier chamfered turntable
        plinth with a brushed metallic accent ring resting on an infinite
        reflective studio checker floor (`reflectance="0.08"`).
2.  **2-Orbit 360° Turntable Sequence + Articulation Playback**:
    *   Completes **Orbit 1 (`0%–50%`, full 360°)** showing pure textured visual
        meshes so users can inspect every side uninterrupted, followed by
        **Orbit 2 (`50%–100%`, second 360°)** cycling through physics collision
        verification while smoothly sweeping all `<joint type="hinge"/>` and
        `<joint type="slide"/>` degrees of freedom via a cosine trajectory.
3.  **Cross-Dissolved Physics Collision Geometry Inspection**:
    *   Transitions via a smooth alpha cross-fade (~8–10 frames, avoiding hard
        1-frame pop-ins) across three plain-English chapters:
        *   **`0%–50%` (Orbit 1 — Full 360°)**: **Textured Visual (`group="2"`)**
        *   **`50%–78%` (Orbit 2a)**: **Visual + Translucent Ghost Collider
            Overlay** (uniform mint-cyan ghost shell over the textured mesh)
        *   **`78%–100%` (Orbit 2b)**: **Physics Collision Hulls Only** (visual
            mesh hidden; per-hull color-coded convex shapes)
4.  **Self-Explanatory HUD Overlay**:
    *   Top-left translucent card showing **Asset Name**, **Bounding Box Size
        (`L × W × H` in `mm` / `cm`)**, **Total Mass (`g` / `kg`)**, **Visual /
        Collider counts**, and **Live Joint Angles/Positions**.
    *   Top-right chapter badge (`1/3 · TEXTURED VISUAL (360°)` /
        `2/3 · VISUAL + COLLIDER OVERLAY` / `3/3 · COLLISION HULLS ONLY`) and
        orbit angle readout (`Orbit 1/2` / `Orbit 2/2`).
    *   Bottom-center plain-English mode callout banner and **3-segment chapter
        timeline bar** showing all three phases at a glance.

--------------------------------------------------------------------------------

## Command-Line Usage

Resolve `SKILL_DIR` to the directory containing this `SKILL.md` file. Bundled
[`scripts/render_showcase_shot.py`](scripts/render_showcase_shot.py) includes a
PEP 723 `# /// script` header (`mujoco`, `av`, `numpy`, `Pillow`, `absl-py`), so
`uv run` automatically resolves and caches an isolated environment on first use
(ensure `ffmpeg` is also installed when passing `--before_xml`). Always write
output videos and images to `<ArtifactDir>/` (never inside a source repository):

```bash
SKILL_DIR="/path/to/skills/mujoco-showcase-shot"
uv run "$SKILL_DIR/scripts/render_showcase_shot.py" \
  --asset=/path/to/model.xml \
  --output=<ArtifactDir>/showcase_shots/<asset_name>_turntable.mp4 \
  --width=960 \
  --height=720 \
  --fps=30 \
  --duration=8.0
```

This produces both:

*   `<ArtifactDir>/showcase_shots/<asset_name>_turntable.mp4` (H.264 turntable
    video)
*   `<ArtifactDir>/showcase_shots/<asset_name>_turntable_keyframes.png` (2×2
    keyframe summary image across the 4 showcase phases)

When imported from Python (`import render_showcase_shot` after adding
`$SKILL_DIR/scripts` to `sys.path`), this module also provides:

*   `render_6view_inspection_sheet`: Renders a `4×7` multi-view inspection sheet
    (`Front (+X)`, `Back (-X)`, `Right (-Y)`, `Left (+Y)`, `Top (+Z)`,
    `Bottom (-Z)`, and `3/4 Front-Right`) across PBR Visual (`group="2"`),
    Untextured Clay 3D Form (`group="4"`), Visual + Collision Overlay
    (`group="2" + "3"`), and Per-Hull Collision (`group="3"`) modes.
*   `render_side_by_side_comparison`: Renders Reference vs. Filament PBR vs.
    Untextured Clay comparison sheets (including `clay_only=True` for fast early
    geometry iteration, `detail_crops_out_png` for 1:1 high-resolution zoom
    crops, and `six_view_out_png` for single-pass multi-view output).
*   `render_wireframe_and_uv_sheet`: Renders a `2×3` raw `.obj` quad/triangle
    topology wireframe and per-material UV layout inspection sheet.
*   `render_filament_still`: Renders batched Filament PBR, Clay, Overlay, or
    Collision still views.
