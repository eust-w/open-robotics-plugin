# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mujoco>=3.2.0",
#     "numpy>=1.24.0",
#     "Pillow>=10.0.0",
#     "scipy>=1.10.0",
#     "trimesh>=4.0.0",
# ]
# ///

"""Zero-magic reference intake, 3D UV/normal math, CC0 scan fetch, and MJCF helpers.

Provides deterministic utilities for reference intake, 3D UV/normal synthesis,
and intermediate Blender diagnostics:
- `build_ref_contact_and_detail_sheets`: True un-lifted sRGB reference contact
  sheets and 1:1 raw crops.
- `fetch_ambientcg_maps`: Fetches and caches CC0 scanned PBR material maps from
  AmbientCG into `~/.cache/mujoco_pbr_cache` as `uint8` arrays.
- `rasterize_uv_world_fields`, `normal_rgb_from_3d_height_fn`,
  `height_to_normal_rgb`, `fbm_noise_3d`, `triplanar_project_map`,
  `dilate_uv_gutters`: Orthonormal per-texel 3D `(X, Y, Z, tangent_u, tangent_v,
  normal)` UV rasterization (supporting positive and negative `.obj` face
  indices) and seam-free 3D normal-map synthesis.
- `compute_volume_weighted_hull_masses` & `build_variant_bundles`: Convex-hull
  volume mass allocation and XML-preserving multi-variant asset bundling.
- `render_blender_diagnostic_sheet`: Non-destructive 2x4 Blender Workbench
  cavity + `Raw` `0.5 * N + 0.5` world-normal / backface sheet.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from collections.abc import Callable, Mapping, Sequence  # pylint: disable=g-import-not-at-top
import io  # pylint: disable=g-import-not-at-top
import json  # pylint: disable=g-import-not-at-top
import math  # pylint: disable=g-import-not-at-top
import os  # pylint: disable=g-import-not-at-top
import re  # pylint: disable=g-import-not-at-top
import shutil  # pylint: disable=g-import-not-at-top
import subprocess  # pylint: disable=g-import-not-at-top
import tempfile  # pylint: disable=g-import-not-at-top
from typing import Any  # pylint: disable=g-import-not-at-top
import urllib.request  # pylint: disable=g-import-not-at-top
import xml.etree.ElementTree as ET  # pylint: disable=g-import-not-at-top
import zipfile  # pylint: disable=g-import-not-at-top

import numpy as np  # pylint: disable=g-import-not-at-top
from PIL import Image  # pylint: disable=g-import-not-at-top
from PIL import ImageDraw  # pylint: disable=g-import-not-at-top
from PIL import ImageFilter  # pylint: disable=g-import-not-at-top
from PIL import ImageFont  # pylint: disable=g-import-not-at-top

# Register short (`asset_build_utils`) module name so callers share a single
# loaded module instance regardless of import path.
_THIS_MOD = sys.modules.get(__name__)
if _THIS_MOD is not None:
  sys.modules.setdefault("asset_build_utils", _THIS_MOD)

_NDIMAGE_MOD: Any = None
_SPATIAL_MOD: Any = None


def _get_ndimage() -> Any:
  """Lazily imports `scipy.ndimage` on first use (~0.15s faster 2D sheet intake)."""
  global _NDIMAGE_MOD  # pylint: disable=global-statement
  if _NDIMAGE_MOD is None:
    from scipy import ndimage as _ndi  # pylint: disable=g-import-not-at-top

    _NDIMAGE_MOD = _ndi
  return _NDIMAGE_MOD


def _get_spatial() -> Any:
  """Lazily imports `scipy.spatial` on first use."""
  global _SPATIAL_MOD  # pylint: disable=global-statement
  if _SPATIAL_MOD is None:
    from scipy import spatial as _sp  # pylint: disable=g-import-not-at-top

    _SPATIAL_MOD = _sp
  return _SPATIAL_MOD


def __getattr__(name: str) -> Any:
  if name == "ndimage":
    return _get_ndimage()
  if name == "spatial":
    return _get_spatial()
  if name == "mujoco":
    try:
      import mujoco as _mj  # pylint: disable=g-import-not-at-top

      return _mj
    except ImportError:
      return None
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_label_font(
    size: int = 13,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  """Loads a readable TrueType label font or falls back to default."""
  for font_path in (
      "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
  ):
    if os.path.exists(font_path):
      try:
        return ImageFont.truetype(font_path, size=size)
      except OSError:
        pass
  return ImageFont.load_default()


def _save_inspection_image(
    sheet: Image.Image,
    out_path: str,
    *,
    max_long_edge: int | None = None,
    jpeg_quality: int = 92,
) -> None:
  """Saves an inspection sheet at 1:1 caller resolution (unless `max_long_edge` is set)."""
  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  w, h = sheet.size
  if max_long_edge is not None and max(w, h) > max_long_edge:
    scale = float(max_long_edge) / float(max(w, h))
    sheet = sheet.resize(
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        Image.Resampling.LANCZOS,
    )
  if out_path.lower().endswith((".jpg", ".jpeg")):
    sheet.convert("RGB").save(
        out_path,
        quality=int(jpeg_quality),
        subsampling=0,
        optimize=True,
    )
  else:
    sheet.save(out_path, optimize=True)


def height_to_normal_rgb(
    height_map: np.ndarray,
    strength: float = 3.5,
    *,
    smooth_sigma: float = 0.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
  """Converts a 2D height map into an 8-bit OpenGL tangent-space normal map.

  Args:
    height_map: 2D float array of surface elevation values.
    strength: Gradient scale factor controlling normal-map relief intensity.
    smooth_sigma: Optional Gaussian smoothing sigma in pixels applied before
      gradient differentiation (`0.0` default preserves fine intentional grain
      and crisp micro-relief without blurring).
    valid_mask: Optional `(H, W)` boolean UV-island mask. When provided, dilates
      valid island heights into empty UV gutters before smoothing and gradient
      differentiation so `0.0` gutter backgrounds do not create artificial seam
      cliffs.

  Returns:
    `(H, W, 3)` `uint8` RGB array with flat normal encoded as `(128, 128, 255)`.
  """
  height_f32 = np.asarray(height_map, dtype=np.float32)
  if valid_mask is not None:
    height_f32 = dilate_uv_gutters(height_f32, valid_mask, max_dist_px=12)
  if smooth_sigma > 0.0:
    height_f32 = _get_ndimage().gaussian_filter(
        height_f32, sigma=float(smooth_sigma), mode="wrap"
    )
  grad_y, grad_x = np.gradient(height_f32)
  norm_x = -grad_x * float(strength)
  norm_y = grad_y * float(strength)
  norm_z = np.ones_like(height_f32)
  inv_len = 1.0 / np.maximum(
      np.sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z), 1e-8
  )
  rgb = np.stack(
      [
          (norm_x * inv_len * 0.5 + 0.5) * 255.0,
          (norm_y * inv_len * 0.5 + 0.5) * 255.0,
          (norm_z * inv_len * 0.5 + 0.5) * 255.0,
      ],
      axis=-1,
  )
  return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _parse_obj_vertices(obj_path: str) -> np.ndarray:
  """Parses 3D vertex coordinates (`v x y z`) from a Wavefront `.obj` file."""
  verts: list[list[float]] = []
  with open(obj_path, "r", encoding="utf-8") as obj_file:
    for raw_line in obj_file:
      line = raw_line.split("#", 1)[0].strip()
      if not line:
        continue
      parts = line.split()
      if parts[0] == "v" and len(parts) >= 4:
        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
  if not verts:
    return np.zeros((0, 3), dtype=np.float64)
  return np.asarray(verts, dtype=np.float64)


def compute_volume_weighted_hull_masses(
    mesh_dir: str,
    col_names: Sequence[str],
    total_mass_kg: float,
    *,
    density_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
  """Computes volume- (and optional material-density-) weighted hull masses.

  Args:
    mesh_dir: Directory containing the collision `.obj` files.
    col_names: Sequence of collision mesh names (with or without `.obj` suffix).
    total_mass_kg: Target total mass in kilograms across all listed hulls.
    density_weights: Optional mapping from hull name (or stem without `.obj`) to
      positive relative material density multiplier (`1.0` default per hull).

  Returns:
    Dict mapping each entry in `col_names` to its assigned mass in kilograms.
  """
  weighted_vols: dict[str, float] = {}
  for name in col_names:
    fname = name if name.endswith(".obj") else f"{name}.obj"
    stem = fname[:-4] if fname.endswith(".obj") else fname
    obj_path = os.path.join(mesh_dir, fname)
    verts = _parse_obj_vertices(obj_path)
    vol = 1e-9
    if len(verts) >= 4:
      sp = _get_spatial()
      try:
        hull = sp.ConvexHull(verts)
        vol = max(float(hull.volume), 1e-9)
      except sp.QhullError:
        span = np.ptp(verts, axis=0)
        vol = max(float(np.prod(np.maximum(span, 1e-4))), 1e-9)
    d_mult = 1.0
    if density_weights:
      if name in density_weights:
        d_mult = max(float(density_weights[name]), 1e-6)
      elif stem in density_weights:
        d_mult = max(float(density_weights[stem]), 1e-6)
    weighted_vols[name] = vol * d_mult

  total_wvol = sum(weighted_vols.values())
  if total_wvol <= 0.0:
    even_mass = float(total_mass_kg) / max(len(col_names), 1)
    return {name: round(even_mass, 6) for name in col_names}

  return {
      name: float(total_mass_kg) * (wvol / total_wvol)
      for name, wvol in weighted_vols.items()
  }


def build_variant_bundles(
    base_asset_dir: str,
    parent_assets_dir: str,
    base_name: str,
    variants: Mapping[str, Mapping[str, Any]],
    render_variant_textures_fn: Callable[[str, Mapping[str, Any]], None],
) -> list[str]:
  """Emits a self-contained `<base_name>_<variant>/` directory for each variant.

  Skips redundant `.obj`/`.stl` file copies when the destination
  file already exists with matching byte size and modification time.

  Args:
    base_asset_dir: Path to the primary `<base_name>/` asset directory.
    parent_assets_dir: Parent `assets/` directory where sibling variants live.
    base_name: Base asset name.
    variants: Mapping from variant suffix to a palette dict.
    render_variant_textures_fn: Callback `fn(v_textures_dir, palette)` that
      writes the variant's PNG textures into `v_textures_dir`.

  Returns:
    List of created variant directory paths.
  """
  base_xml = os.path.join(base_asset_dir, "model.xml")
  base_meshes = os.path.join(base_asset_dir, "meshes")
  out_dirs: list[str] = []

  for variant_key, palette in variants.items():
    variant_name = f"{base_name}_{variant_key}"
    v_dir = os.path.join(parent_assets_dir, variant_name)
    v_meshes = os.path.join(v_dir, "meshes")
    v_textures = os.path.join(v_dir, "textures")
    os.makedirs(v_meshes, exist_ok=True)
    os.makedirs(v_textures, exist_ok=True)

    if os.path.abspath(base_meshes) != os.path.abspath(v_meshes):
      for fname in os.listdir(base_meshes):
        if fname.endswith(".obj") or fname.endswith(".stl"):
          src_p = os.path.join(base_meshes, fname)
          dst_p = os.path.join(v_meshes, fname)
          need_copy = True
          if os.path.exists(dst_p):
            if os.path.getsize(src_p) == os.path.getsize(
                dst_p
            ) and os.path.getmtime(src_p) <= os.path.getmtime(dst_p):
              need_copy = False
          if need_copy:
            shutil.copy2(src_p, dst_p)

    render_variant_textures_fn(v_textures, palette)

    tree = ET.parse(base_xml)
    root = tree.getroot()
    root.set("model", variant_name)
    worldbody = root.find("worldbody")
    if worldbody is not None:
      first_body = worldbody.find("body")
      if first_body is not None:
        first_body.set("name", variant_name)

    v_xml = os.path.join(v_dir, "model.xml")
    ET.indent(tree, space="  ")
    tree.write(v_xml, encoding="unicode", xml_declaration=False)
    with open(v_xml, "a", encoding="utf-8") as xml_file:
      xml_file.write("\n")
    out_dirs.append(v_dir)

  return out_dirs


def _resolve_box_to_px(
    box: Sequence[float],
    width: int,
    height: int,
    *,
    min_span_px: float = 1.0,
) -> tuple[float, float, float, float]:
  """Resolves a pixel `(> 1.5)` or normalized `[0..1]` `(x0, y0, x1, y1)` box to pixels."""
  b0, b1, b2, b3 = [float(c) for c in box]
  if max(abs(b0), abs(b1), abs(b2), abs(b3)) > 1.5:
    raw_x0, raw_y0, raw_x1, raw_y1 = b0, b1, b2, b3
  else:
    raw_x0, raw_y0 = b0 * float(width), b1 * float(height)
    raw_x1, raw_y1 = b2 * float(width), b3 * float(height)
  xa, xb = min(raw_x0, raw_x1), max(raw_x0, raw_x1)
  ya, yb = min(raw_y0, raw_y1), max(raw_y0, raw_y1)
  span_x = min(float(min_span_px), float(max(width, 1)))
  span_y = min(float(min_span_px), float(max(height, 1)))
  ox0 = max(0.0, min(float(width) - span_x, xa))
  oy0 = max(0.0, min(float(height) - span_y, ya))
  ox1 = max(ox0 + span_x, min(float(width), xb))
  oy1 = max(oy0 + span_y, min(float(height), yb))
  return (ox0, oy0, ox1, oy1)


def build_ref_contact_and_detail_sheets(
    ref_paths: Sequence[
        str
        | tuple[str, str]
        | tuple[str, str, tuple[float, float, float, float]]
    ],
    detail_crops: Sequence[tuple[Any, ...]],
    out_contact_png: str,
    out_details_png: str,
    *,
    raw_crops_dir: str | None = None,
    rgb_sample_boxes: (
        Mapping[str, tuple[str, tuple[float, float, float, float]]]
        | Sequence[tuple[str, str, tuple[float, float, float, float]]]
        | None
    ) = None,
    cols: int = 3,
    cell_size: int = 440,
    include_shadow_lift: bool = False,
) -> dict[str, Any]:
  """Generates `ref_contact_sheet`, true-sRGB `ref_details_grid`, and optional 1:1 `raw_crops_dir`.

  Outputs true un-lifted sRGB detail crops on `out_details_png` by default
  (`include_shadow_lift=False`) so shadow contrast inside recessed pockets and
  grooves is never washed out.

  Args:
    ref_paths: Sequence of reference image paths (`str`) or `(label, img_path)`
      tuples to tile into the contact sheet.
    detail_crops: Sequence of `(label, img_path, (u0, v0, u1, v1))` normalized
      `[0..1]` or source-pixel crops to tile into `out_details_png`.
    out_contact_png: Output path for the overview contact sheet.
    out_details_png: Output path for the detail crop grid.
    raw_crops_dir: Optional directory where each `detail_crops` box is also
      saved at 1:1 native source resolution (zero resize, zero tone-curve
      filtering).
    rgb_sample_boxes: Optional mapping `name -> (img_path, (u0, v0, u1, v1))` or
      sequence of `(name, img_path, (u0, v0, u1, v1))` tuples for sampling `[25,
      50, 75]` RGB percentiles.
    cols: Number of grid columns per sheet.
    cell_size: Pixel width and height of each cell.
    include_shadow_lift: If True, also emits `[SHADOW-LIFT]` companion tiles for
      dark crops (`False` default preserves true un-lifted sRGB only).

  Returns:
    Dict with `"contact_png"`, `"details_png"`, `"raw_crop_paths"`, and
    `"rgb_percentiles"`.
  """
  font = _load_label_font(13)
  pad = 8
  hdr_h = 26

  def _render_grid(
      items: Sequence[tuple[str, Image.Image]], out_path: str
  ) -> None:
    if not items:
      return
    n_cols = min(cols, len(items))
    n_rows = (len(items) + n_cols - 1) // n_cols
    sheet_w = n_cols * cell_size + (n_cols + 1) * pad
    sheet_h = n_rows * (cell_size + hdr_h) + (n_rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 22, 30))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(items):
      row_i = idx // n_cols
      col_i = idx % n_cols
      x0 = pad + col_i * (cell_size + pad)
      y0 = pad + row_i * (cell_size + hdr_h + pad)
      draw.rectangle([x0, y0, x0 + cell_size, y0 + hdr_h], fill=(32, 48, 68))
      draw.text((x0 + 8, y0 + 5), label[:48], fill=(242, 246, 252), font=font)
      iw, ih = img.size
      scale = min(cell_size / max(iw, 1), cell_size / max(ih, 1))
      nw = max(1, int(round(iw * scale)))
      nh = max(1, int(round(ih * scale)))
      resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
      cell_bg = Image.new("RGB", (cell_size, cell_size), (245, 245, 247))
      cell_bg.paste(resized, ((cell_size - nw) // 2, (cell_size - nh) // 2))
      sheet.paste(cell_bg, (x0, y0 + hdr_h))
    _save_inspection_image(sheet, out_path)

  contact_items: list[tuple[str, Image.Image]] = []
  contact_meta: list[dict[str, Any]] = []
  for idx, entry in enumerate(ref_paths):
    crop_norm = None
    if isinstance(entry, (tuple, list)) and len(entry) == 3:
      lbl, ref_p, crop_norm = str(entry[0]), str(entry[1]), entry[2]
    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
      first, second = str(entry[0]), str(entry[1])
      if os.path.exists(second):
        lbl, ref_p = first, second
      else:
        ref_p, lbl = first, second
    else:
      ref_p = str(entry)
      lbl = f"[{idx}] {os.path.basename(ref_p)}"
    if os.path.exists(ref_p):
      raw_im = Image.open(ref_p)
      im = raw_im.convert("RGB")
      orig_w, orig_h = im.width, im.height
      box_px = None
      if crop_norm is not None and len(crop_norm) == 4:
        ox0, oy0, ox1, oy1 = _resolve_box_to_px(
            crop_norm, orig_w, orig_h, min_span_px=8.0
        )
        x0 = max(0, min(orig_w - 1, int(round(ox0))))
        y0 = max(0, min(orig_h - 1, int(round(oy0))))
        x1 = min(orig_w, max(x0 + 1, int(round(ox1))))
        y1 = min(orig_h, max(y0 + 1, int(round(oy1))))
        im = im.crop((x0, y0, x1, y1))
        box_px = [x0, y0, x1, y1]
      contact_items.append((f"{lbl} ({orig_w}x{orig_h})", im))
      contact_meta.append({
          "label": lbl,
          "path": os.path.abspath(ref_p),
          "image_size_px": [orig_w, orig_h],
          "has_explicit_box": box_px is not None,
          "obj_bbox_px": box_px,
      })
  _render_grid(contact_items, out_contact_png)

  detail_items: list[tuple[str, Image.Image]] = []
  raw_crop_paths: list[str] = []
  if raw_crops_dir:
    os.makedirs(raw_crops_dir, exist_ok=True)
  for idx, crop_entry in enumerate(detail_crops):
    a, b, (u0, v0, u1, v1) = crop_entry[0], crop_entry[1], crop_entry[2]
    if os.path.exists(str(b)):
      lbl, img_p = str(a), str(b)
    else:
      img_p, lbl = str(a), str(b)
    if os.path.exists(img_p):
      im = Image.open(img_p).convert("RGB")
      iw, ih = im.size
      x0, y0, x1, y1 = _resolve_box_to_px(
          (u0, v0, u1, v1), iw, ih, min_span_px=1.0
      )
      slug_chars = [ch.lower() if ch.isalnum() else "_" for ch in lbl]
      slug = "_".join(s for s in "".join(slug_chars).split("_") if s)
      slug = re.sub(r"^(\d+_)+", "", slug)[:42]
      if raw_crops_dir:
        rx0 = max(0, min(iw - 1, int(round(x0))))
        ry0 = max(0, min(ih - 1, int(round(y0))))
        rx1 = max(rx0 + 1, min(iw, int(round(x1))))
        ry1 = max(ry0 + 1, min(ih, int(round(y1))))
        raw_out = os.path.join(
            raw_crops_dir, f"{idx + 1:02d}_{slug or 'crop'}.jpg"
        )
        im.crop((rx0, ry0, rx1, ry1)).save(
            raw_out, quality=94, subsampling=0, optimize=True
        )
        raw_crop_paths.append(raw_out)
      cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
      side = min(max(abs(x1 - x0), abs(y1 - y0), 16.0), float(min(iw, ih)))
      sx0 = float(np.clip(cx - 0.5 * side, 0.0, max(0.0, iw - side)))
      sy0 = float(np.clip(cy - 0.5 * side, 0.0, max(0.0, ih - side)))
      iside = max(1, int(round(side)))
      ix0, iy0 = int(round(sx0)), int(round(sy0))
      box = (ix0, iy0, min(iw, ix0 + iside), min(ih, iy0 + iside))
      crop_im = im.crop(box).resize(
          (cell_size, cell_size), Image.Resampling.LANCZOS
      )
      detail_items.append((f"#{idx + 1}: {lbl}", crop_im))

      if include_shadow_lift:
        crop_arr = np.asarray(crop_im, dtype=np.float32)
        luma = (
            0.299 * crop_arr[..., 0]
            + 0.587 * crop_arr[..., 1]
            + 0.114 * crop_arr[..., 2]
        )
        p10, p22, p50, p90 = np.percentile(luma, [10, 22, 50, 90])
        if p22 < 100.0 or p50 < 105.0 or (p90 - p10) < 95.0:
          arr01 = np.clip(crop_arr / 255.0, 0.0, 1.0)
          v_max = np.max(arr01, axis=-1, keepdims=True)
          lifted = np.clip(arr01 * (1.26 / (v_max + 0.26)), 0.0, 1.0)
          enh_im = Image.fromarray(
              np.round(lifted * 255.0).astype(np.uint8), mode="RGB"
          )
          detail_items.append((f"#{idx + 1}b: {lbl} [SHADOW-LIFT]", enh_im))
  _render_grid(detail_items, out_details_png)

  rgb_stats: dict[str, dict[str, tuple[int, int, int]]] = {}
  if rgb_sample_boxes:
    if isinstance(rgb_sample_boxes, Mapping):
      sample_iter = [(k, v[0], v[1]) for k, v in rgb_sample_boxes.items()]
    else:
      sample_iter = list(rgb_sample_boxes)
    for mat_name, img_p, (u0, v0, u1, v1) in sample_iter:
      if os.path.exists(img_p):
        arr = np.asarray(Image.open(img_p).convert("RGB"), dtype=np.float32)
        ih, iw = arr.shape[:2]
        ox0, oy0, ox1, oy1 = _resolve_box_to_px(
            (u0, v0, u1, v1), iw, ih, min_span_px=1.0
        )
        ix0 = max(0, min(iw - 1, int(round(ox0))))
        iy0 = max(0, min(ih - 1, int(round(oy0))))
        ix1 = max(ix0 + 1, min(iw, int(round(ox1))))
        iy1 = max(iy0 + 1, min(ih, int(round(oy1))))
        patch = arr[iy0:iy1, ix0:ix1]
        if patch.size > 0:
          p25, p50, p75 = np.percentile(patch, [25, 50, 75], axis=(0, 1))
          rgb_stats[str(mat_name)] = {
              "p25": (
                  int(round(p25[0])),
                  int(round(p25[1])),
                  int(round(p25[2])),
              ),
              "p50": (
                  int(round(p50[0])),
                  int(round(p50[1])),
                  int(round(p50[2])),
              ),
              "p75": (
                  int(round(p75[0])),
                  int(round(p75[1])),
                  int(round(p75[2])),
              ),
          }
          print(
              f"[RGB Sample] {mat_name}: p25={rgb_stats[str(mat_name)]['p25']},"
              f" p50={rgb_stats[str(mat_name)]['p50']},"
              f" p75={rgb_stats[str(mat_name)]['p75']}"
          )
  result: dict[str, Any] = {
      "contact_png": out_contact_png,
      "details_png": out_details_png,
      "raw_crop_paths": raw_crop_paths,
      "rgb_percentiles": rgb_stats,
  }
  result.update(rgb_stats)
  return result


_BLENDER_DIAG_SCRIPT = """
import sys
sys.dont_write_bytecode = True

import json, math, os
import bpy, mathutils

cfg = json.loads(sys.argv[-1])
out_dir = cfg["tmp_dir"]
res = int(cfg.get("panel_size", 420))

vis_objs = []
for obj in list(bpy.data.objects):
    if obj.type != "MESH":
        obj.hide_render = True
        continue
    low_name = obj.name.lower()
    is_col = (
        low_name.endswith("_col")
        or "_col_" in low_name
        or "_hull_" in low_name
        or low_name.endswith("_hull")
    ) and not low_name.endswith("_vis") and "_vis_" not in low_name
    if is_col:
        obj.hide_render = True
        continue
    obj.hide_render = False
    vis_objs.append(obj)

bpy.context.view_layer.update()
bmin = mathutils.Vector((1e9, 1e9, 1e9))
bmax = mathutils.Vector((-1e9, -1e9, -1e9))
for obj in vis_objs:
    for corner in obj.bound_box:
        wc = obj.matrix_world @ mathutils.Vector(corner)
        for i in range(3):
            bmin[i] = min(bmin[i], wc[i])
            bmax[i] = max(bmax[i], wc[i])
if bmin[0] > bmax[0]:
    bmin = mathutils.Vector((-0.05, -0.05, 0.0))
    bmax = mathutils.Vector((0.05, 0.05, 0.10))

center = 0.5 * (bmin + bmax)
span = max((bmax - bmin).length, 1e-3)
half_span = 0.5 * span

cam_data = bpy.data.cameras.new("_DiagCam")
cam_data.lens = 50.0
fov = max(float(cam_data.angle), 0.1)
dist = (half_span / math.tan(0.5 * fov)) * 1.18 + 0.5 * span
cam_data.clip_start = max(1e-4, min(0.001, dist * 0.01))
cam_data.clip_end = max(100.0, dist * 10.0)
cam_obj = bpy.data.objects.new("_DiagCam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

scene = bpy.context.scene
scene.render.resolution_x = res
scene.render.resolution_y = res
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = True

views = [
    ("front", 0.0, 12.0),
    ("right", -90.0, 12.0),
    ("top", -90.0, 88.0),
    ("hero_iso", -52.0, 26.0),
]

def place_cam(az_deg, el_deg):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    pos = center + mathutils.Vector((
        dist * math.cos(el) * math.cos(az),
        dist * math.cos(el) * math.sin(az),
        dist * math.sin(el),
    ))
    cam_obj.location = pos
    direction = center - pos
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

# Row 1: BLENDER_WORKBENCH with Per-Object Colors + Ridge/Valley Cavity Shading
_OBJ_PALETTE = [
    (0.84, 0.76, 0.68, 1.0),
    (0.64, 0.76, 0.86, 1.0),
    (0.72, 0.84, 0.72, 1.0),
    (0.86, 0.70, 0.74, 1.0),
    (0.82, 0.80, 0.62, 1.0),
    (0.74, 0.72, 0.86, 1.0),
]
for idx_obj, obj in enumerate(vis_objs):
    obj.color = _OBJ_PALETTE[idx_obj % len(_OBJ_PALETTE)]

scene.render.engine = "BLENDER_WORKBENCH"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.display.shading.light = "MATCAP"
scene.display.shading.color_type = "OBJECT"
scene.display.shading.show_cavity = True
scene.display.shading.cavity_type = "BOTH"
scene.display.shading.cavity_ridge_factor = 1.6
scene.display.shading.cavity_valley_factor = 1.6
scene.display.shading.show_object_outline = True

for vname, az, el in views:
    place_cam(az, el)
    scene.render.filepath = os.path.join(out_dir, f"wb_{vname}.png")
    bpy.ops.render.render(write_still=True)

# Row 2: CYCLES 1-Sample True Raw World Normal RGB (`0.5 * N + 0.5`) + Red Backfacing
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.view_settings.view_transform = "Raw"
scene.view_settings.look = "None"

mat = bpy.data.materials.new("_DiagNormalFaceMat")
if mat.node_tree is None:
    mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
n_out = nt.nodes.new("ShaderNodeOutputMaterial")
n_em = nt.nodes.new("ShaderNodeEmission")
n_geo = nt.nodes.new("ShaderNodeNewGeometry")
n_vec = nt.nodes.new("ShaderNodeVectorMath")
n_vec.operation = "MULTIPLY_ADD"
n_vec.inputs[1].default_value = (0.5, 0.5, 0.5)
n_vec.inputs[2].default_value = (0.5, 0.5, 0.5)
n_mix = nt.nodes.new("ShaderNodeMix")
n_mix.data_type = "RGBA"
n_mix.inputs["B"].default_value = (1.0, 0.039, 0.059, 1.0)
nt.links.new(n_geo.outputs["Normal"], n_vec.inputs[0])
nt.links.new(n_vec.outputs["Vector"], n_mix.inputs["A"])
nt.links.new(n_geo.outputs["Backfacing"], n_mix.inputs["Factor"])
nt.links.new(n_mix.outputs["Result"], n_em.inputs["Color"])
nt.links.new(n_em.outputs["Emission"], n_out.inputs["Surface"])

for obj in vis_objs:
    obj.data.materials.clear()
    obj.data.materials.append(mat)

for vname, az, el in views:
    place_cam(az, el)
    scene.render.filepath = os.path.join(out_dir, f"nm_{vname}.png")
    bpy.ops.render.render(write_still=True)
"""


def render_blender_diagnostic_sheet(
    blend_path: str,
    out_png: str,
    *,
    xml_path: str | None = None,
    part_pivots: Mapping[str, tuple[float, float, float]] | None = None,
    panel_size: int = 420,
) -> dict[str, Any]:
  """Renders a non-destructive 2x4 Blender Workbench Cavity + Raw Normal/Backface sheet.

  Row 1 renders `BLENDER_WORKBENCH` (`show_cavity=True`) ridge/valley cavity
  shading across 4 views (`Front +X`, `Right -Y`, `Top +Z`, `3/4 Hero Iso`).
  Row 2 renders 1-sample `CYCLES` under `view_transform = 'Raw'` encoding true
  `0.5 * N + 0.5` world normals, with flipped backfaces (`Backfacing == 1`)
  highlighted in Bright Neon Red (`#FF0A0F`). Never mutates `obj.location`.

  Args:
    blend_path: Path to the saved `.blend` file.
    out_png: Output composite PNG path.
    xml_path: Unused compatibility argument (scene transforms in `blend_path`
      are rendered directly without mutating `obj.location`).
    part_pivots: Unused compatibility argument (scene transforms in `blend_path`
      are rendered directly without mutating `obj.location`).
    panel_size: Pixel width and height of each panel cell.

  Returns:
    Dict with `"out_png"`, `"backface_pixels_per_view"`,
    `"backface_pixel_counts"` per view, and `"has_backfaces"` boolean flag.
  """
  del xml_path, part_pivots
  views_meta = [
      ("front", "FRONT (+X)"),
      ("right", "RIGHT (-Y)"),
      ("top", "TOP (+Z)"),
      ("hero_iso", "3/4 HERO ISO"),
  ]
  with tempfile.TemporaryDirectory(prefix="blender_diag_") as tmp_dir:
    script_p = os.path.join(tmp_dir, "_run_diag.py")
    with open(script_p, "w", encoding="utf-8") as sf:
      sf.write(_BLENDER_DIAG_SCRIPT)
    payload = json.dumps({
        "tmp_dir": tmp_dir,
        "panel_size": panel_size,
    })
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    blender_bin = (
        os.environ.get("MUJOCO_BLENDER_BIN")
        or shutil.which("blender")
        or (
            "/Applications/Blender.app/Contents/MacOS/Blender"
            if os.path.exists(
                "/Applications/Blender.app/Contents/MacOS/Blender"
            )
            else "/usr/bin/blender"
        )
    )
    proc = subprocess.run(
        [blender_bin, "-b", blend_path, "-P", script_p, "--", payload],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
      err_tail = (proc.stderr or proc.stdout or "")[-2000:]
      raise RuntimeError(
          f"Blender diagnostic render failed (exit {proc.returncode}) via "
          f"{blender_bin!r}:\n{err_tail}"
      )

    font = _load_label_font(12)
    pad = 8
    hdr_h = 26
    sheet_w = 4 * panel_size + 5 * pad
    sheet_h = 2 * (panel_size + hdr_h) + 3 * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 18, 24))
    draw = ImageDraw.Draw(sheet)

    backface_counts: dict[str, int] = {}
    for col_i, (vkey, vlabel) in enumerate(views_meta):
      x0 = pad + col_i * (panel_size + pad)
      y0 = pad
      draw.rectangle([x0, y0, x0 + panel_size, y0 + hdr_h], fill=(32, 48, 68))
      draw.text(
          (x0 + 8, y0 + 5),
          f"WORKBENCH CAVITY: {vlabel}",
          fill=(242, 246, 252),
          font=font,
      )
      wb_p = os.path.join(tmp_dir, f"wb_{vkey}.png")
      if os.path.exists(wb_p):
        wb_rgba = Image.open(wb_p).convert("RGBA")
        bg = Image.new("RGBA", wb_rgba.size, (34, 39, 46, 255))
        comp = Image.alpha_composite(bg, wb_rgba).convert("RGB")
        sheet.paste(comp, (x0, y0 + hdr_h))

      y1 = pad + (panel_size + hdr_h + pad)
      nm_p = os.path.join(tmp_dir, f"nm_{vkey}.png")
      bf_px = 0
      if os.path.exists(nm_p):
        nm_rgba = Image.open(nm_p).convert("RGBA")
        arr = np.asarray(nm_rgba, dtype=np.int32)
        # Unit normal sphere satisfiesnx^2 + ny^2 + nz^2 = 1, so `0.5 * N + 0.5`
        # with R > 210 mathematically has max(G, B) >= 59; `< 45` isolates `#FF0A0F`.
        red_mask = (
            (arr[..., 3] > 200)
            & (arr[..., 0] > 210)
            & (arr[..., 1] < 45)
            & (arr[..., 2] < 45)
        )
        bf_px = int(np.sum(red_mask))
        backface_counts[vkey] = bf_px
        bg_nm = Image.new("RGBA", nm_rgba.size, (18, 22, 28, 255))
        comp_nm = Image.alpha_composite(bg_nm, nm_rgba).convert("RGB")
        sheet.paste(comp_nm, (x0, y1 + hdr_h))

      hdr_col = (145, 28, 38) if bf_px > 12 else (42, 34, 64)
      draw.rectangle([x0, y1, x0 + panel_size, y1 + hdr_h], fill=hdr_col)
      bf_tag = f" [BACKFACE RED={bf_px}px!]" if bf_px > 12 else " [OK: 0 Red]"
      draw.text(
          (x0 + 8, y1 + 5),
          f"NORMAL RGB + BACKFACE: {vlabel}{bf_tag}",
          fill=(242, 246, 252),
          font=font,
      )

  _save_inspection_image(sheet, out_png)
  total_bf = sum(backface_counts.values())
  has_bf = any(v > 12 for v in backface_counts.values())
  if has_bf:
    print(
        "[BLENDER DIAG WARNING] Detected BRIGHT RED backfacing/flipped faces "
        f"in {out_png}: {backface_counts}! Call bmesh.ops.recalc_face_normals "
        "and fix any folded non-convex polygon caps."
    )
  else:
    print(f"[BLENDER DIAG OK] Saved {out_png} (0 flipped backfaces detected).")
  return {
      "out_png": out_png,
      "backface_pixel_counts": backface_counts,
      "backface_pixels_per_view": backface_counts,
      "total_backface_pixels": total_bf,
      "has_backfaces": has_bf,
  }


def _to_uint8_image_array(raw_bytes: bytes, want_rgb: bool) -> np.ndarray:
  """Decodes 8-bit or 16-bit PNG/JPG bytes into a normalized `uint8` array."""
  with Image.open(io.BytesIO(raw_bytes)) as img:
    if want_rgb:
      if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(img, dtype=np.float32)
        scale = 257.0 if arr.max() > 255.0 else 1.0
        u8 = np.clip(np.round(arr / scale), 0, 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)
      rgb_im = img.convert("RGB")
      return np.asarray(rgb_im, dtype=np.uint8)

    if img.mode in ("I", "I;16", "I;16B", "I;16L"):
      arr = np.asarray(img, dtype=np.float32)
      return np.clip(np.round(arr / 257.0), 0, 255).astype(np.uint8)
    if img.mode == "F":
      arr = np.asarray(img, dtype=np.float32)
      scale = 255.0 if arr.max() <= 1.05 else 1.0
      return np.clip(np.round(arr * scale), 0, 255).astype(np.uint8)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def fetch_ambientcg_maps(
    asset_id: str,
    *,
    cache_dir: str | None = None,
    resolution: str = "1K",
) -> dict[str, np.ndarray]:
  """Fetches and caches CC0 AmbientCG PBR, imperfection, and decal maps.

  Args:
    asset_id: AmbientCG asset identifier (e.g. `"Rubber004"`, `"Metal009"`).
    cache_dir: Local directory for caching extracted 8-bit PNGs. If `None`, uses
      `~/.cache/mujoco_pbr_cache`.
    resolution: Preferred resolution prefix (`"1K"` default, or `"2K"`).

  Returns:
    Dict mapping available channel keys (`"color"`, `"roughness"`,
    `"normal_gl"`, `"opacity"`, `"metallic"`, `"ao"`, `"displacement"`) to
    `uint8` numpy arrays (`(H, W, 3)` for RGB/normal, `(H, W)` for scalar maps).
  """
  cache_root = os.path.abspath(
      cache_dir or os.path.expanduser("~/.cache/mujoco_pbr_cache")
  )
  pack_dir = os.path.join(cache_root, f"{asset_id}_{resolution}")
  manifest_path = os.path.join(pack_dir, "_manifest.json")

  if os.path.isfile(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as mf:
      manifest = json.load(mf)
    out_cached: dict[str, np.ndarray] = {}
    for ch_key, rel_file in manifest.items():
      img_p = os.path.join(pack_dir, rel_file)
      if os.path.isfile(img_p):
        want_rgb = ch_key in ("color", "normal_gl")
        with Image.open(img_p) as im:
          out_cached[ch_key] = np.asarray(
              im.convert("RGB" if want_rgb else "L"), dtype=np.uint8
          )
    if out_cached:
      return out_cached

  api_url = (
      f"https://ambientcg.com/api/v2/full_json?id={asset_id}"
      "&include=downloadData"
  )
  req = urllib.request.Request(
      api_url, headers={"User-Agent": "mujoco-asset-builder/1.0"}
  )
  with urllib.request.urlopen(req, timeout=25) as resp:
    payload = json.loads(resp.read().decode("utf-8"))

  found = payload.get("foundAssets", [])
  if not found:
    raise ValueError(f"AmbientCG asset {asset_id!r} not found")
  dl_folders = (
      found[0]
      .get("downloadFolders", {})
      .get("default", {})
      .get("downloadFiletypeCategories", {})
  )
  zip_downloads = dl_folders.get("zip", {}).get("downloads", [])
  by_attr = {
      d.get("attribute", ""): d.get("downloadLink", "") for d in zip_downloads
  }

  chosen_url = ""
  for cand in (
      f"{resolution}-PNG",
      f"{resolution}-JPG",
      "1K-PNG",
      "1K-JPG",
      "2K-PNG",
      "2K-JPG",
  ):
    if by_attr.get(cand):
      chosen_url = by_attr[cand]
      break
  if not chosen_url and zip_downloads:
    chosen_url = zip_downloads[0].get("downloadLink", "")
  if not chosen_url:
    raise RuntimeError(f"No zip download URL found for AmbientCG {asset_id!r}")

  zip_req = urllib.request.Request(
      chosen_url, headers={"User-Agent": "mujoco-asset-builder/1.0"}
  )
  with urllib.request.urlopen(zip_req, timeout=45) as zresp:
    zip_bytes = zresp.read()

  channel_suffixes = (
      ("color", ("_color.", "_albedo.")),
      ("roughness", ("_roughness.",)),
      ("normal_gl", ("_normalgl.",)),
      ("opacity", ("_opacity.", "_mask.")),
      ("metallic", ("_metalness.", "_metallic.")),
      ("ao", ("_ambientocclusion.", "_ao.")),
      ("displacement", ("_displacement.", "_height.")),
  )

  os.makedirs(pack_dir, exist_ok=True)
  extracted: dict[str, np.ndarray] = {}
  manifest_out: dict[str, str] = {}
  with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
    for member in zf.namelist():
      low = member.lower()
      if not low.endswith((".png", ".jpg", ".jpeg")):
        continue
      for ch_key, suffixes in channel_suffixes:
        if any(sfx in low for sfx in suffixes) and ch_key not in extracted:
          raw = zf.read(member)
          want_rgb = ch_key in ("color", "normal_gl")
          arr_u8 = _to_uint8_image_array(raw, want_rgb=want_rgb)
          extracted[ch_key] = arr_u8
          out_name = f"{ch_key}.png"
          Image.fromarray(arr_u8).save(os.path.join(pack_dir, out_name))
          manifest_out[ch_key] = out_name
          if ch_key == "color" and "opacity" not in extracted:
            with Image.open(io.BytesIO(raw)) as cim:
              if "A" in cim.getbands():
                alpha_arr = np.asarray(cim.getchannel("A"), dtype=np.uint8)
                if alpha_arr.min() < 250:
                  extracted["opacity"] = alpha_arr
                  Image.fromarray(alpha_arr).save(
                      os.path.join(pack_dir, "opacity.png")
                  )
                  manifest_out["opacity"] = "opacity.png"

  with open(manifest_path, "w", encoding="utf-8") as mf:
    json.dump(manifest_out, mf, indent=2)
  return extracted


def dilate_uv_gutters(
    arr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_dist_px: int = 12,
) -> np.ndarray:
  """Dilates valid UV island values into adjacent empty gutter margins via EDT.

  Args:
    arr: `(H, W)` or `(H, W, C)` array to dilate.
    valid_mask: `(H, W)` boolean mask where `True` marks inside-island texels.
    max_dist_px: Maximum dilation radius in pixels (`12` default).

  Returns:
    Copy of `arr` with gutter pixels within `max_dist_px` filled from nearest
    valid island texel.
  """
  mask = np.asarray(valid_mask, dtype=bool)
  if not np.any(mask) or np.all(mask):
    return np.array(arr, copy=True)
  dists, nearest_idx = _get_ndimage().distance_transform_edt(
      ~mask, return_distances=True, return_indices=True
  )
  out = np.array(arr, copy=True)
  fill = (~mask) & (dists <= float(max_dist_px))
  out[fill] = arr[nearest_idx[0][fill], nearest_idx[1][fill]]
  return out


def _resolve_obj_index(tok_part: str, current_pool_len: int) -> int:
  """Resolves a positive (`1`-based) or negative (relative) Wavefront `.obj` index."""
  if not tok_part:
    return -1
  raw_i = int(tok_part)
  if raw_i > 0:
    return raw_i - 1
  if raw_i < 0:
    return current_pool_len + raw_i
  return -1


def rasterize_uv_world_fields(
    obj_paths: str | Sequence[str],
    canvas_size: tuple[int, int] | int = 1024,
    *,
    part_offsets: Mapping[str, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
  """Rasterizes per-texel 3D world `(X, Y, Z)` and orthonormal `(tangent_u, tangent_v, normal)` in UV space.

  Args:
    obj_paths: Single `.obj` path or sequence of `.obj` paths sharing a UV map.
    canvas_size: Target `(width, height)` or square `size` in pixels.
    part_offsets: Optional mapping from mesh stem or filename to `(px, py, pz)`
      world body offset.

  Returns:
    Dict with `"xyz"` (`(H, W, 3)` float32), `"normal"` (`(H, W, 3)` float32),
    `"tangent_u"` (`(H, W, 3)` float32), `"tangent_v"` (`(H, W, 3)` float32),
    `"mm_per_texel"` (`(H, W)` float32), `"valid_mask"` (`(H, W)` bool),
    `"part_id"` (`(H, W)` int32), and `"part_masks"` (`dict[str, np.ndarray]`).
  """
  if isinstance(canvas_size, int):
    width, height = int(canvas_size), int(canvas_size)
  else:
    width, height = int(canvas_size[0]), int(canvas_size[1])

  paths = [obj_paths] if isinstance(obj_paths, str) else list(obj_paths)
  xyz_buf = np.zeros((height, width, 3), dtype=np.float32)
  nrm_buf = np.zeros((height, width, 3), dtype=np.float32)
  nrm_buf[..., 2] = 1.0
  tu_buf = np.zeros((height, width, 3), dtype=np.float32)
  tu_buf[..., 0] = 1.0
  tv_buf = np.zeros((height, width, 3), dtype=np.float32)
  tv_buf[..., 1] = 1.0
  mm_px_buf = np.ones((height, width), dtype=np.float32)
  valid = np.zeros((height, width), dtype=bool)
  part_id_buf = np.full((height, width), -1, dtype=np.int32)
  part_stems: list[str] = []

  for obj_p in paths:
    if not os.path.isfile(obj_p):
      continue
    stem = os.path.splitext(os.path.basename(obj_p))[0]
    p_idx = len(part_stems)
    part_stems.append(stem)
    offset = np.zeros(3, dtype=np.float32)
    if part_offsets:
      if stem in part_offsets:
        offset = np.asarray(part_offsets[stem], dtype=np.float32)
      elif os.path.basename(obj_p) in part_offsets:
        offset = np.asarray(
            part_offsets[os.path.basename(obj_p)], dtype=np.float32
        )

    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    norms: list[list[float]] = []
    tris: list[
        tuple[
            tuple[int, int, int],
            tuple[int, int, int],
            tuple[int, int, int],
        ]
    ] = []

    with open(obj_p, "r", encoding="utf-8", errors="ignore") as f:
      for raw_line in f:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
          continue
        sp = line.split()
        cmd = sp[0]
        if cmd == "v" and len(sp) >= 4:
          verts.append([float(sp[1]), float(sp[2]), float(sp[3])])
        elif cmd == "vt" and len(sp) >= 3:
          uvs.append([float(sp[1]), float(sp[2])])
        elif cmd == "vn" and len(sp) >= 4:
          norms.append([float(sp[1]), float(sp[2]), float(sp[3])])
        elif cmd == "f":
          tokens = sp[1:]
          if len(tokens) < 3:
            continue
          corner_idx: list[tuple[int, int, int]] = []
          n_v_cur, n_vt_cur, n_vn_cur = len(verts), len(uvs), len(norms)
          for tok in tokens:
            inds = tok.split("/")
            vi = _resolve_obj_index(inds[0], n_v_cur)
            ti = _resolve_obj_index(inds[1], n_vt_cur) if len(inds) > 1 else -1
            ni = _resolve_obj_index(inds[2], n_vn_cur) if len(inds) > 2 else -1
            corner_idx.append((vi, ti, ni))
          for k in range(1, len(corner_idx) - 1):
            tris.append((corner_idx[0], corner_idx[k], corner_idx[k + 1]))

    if not verts or not uvs:
      continue
    v_arr = np.asarray(verts, dtype=np.float32) + offset[None, :]
    uv_arr = np.asarray(uvs, dtype=np.float32)
    n_arr = (
        np.asarray(norms, dtype=np.float32)
        if norms
        else np.zeros((0, 3), dtype=np.float32)
    )
    num_v, num_uv, num_n = len(v_arr), len(uv_arr), len(n_arr)

    for c0, c1, c2 in tris:
      if not (
          0 <= c0[0] < num_v and 0 <= c1[0] < num_v and 0 <= c2[0] < num_v
      ) or not (
          0 <= c0[1] < num_uv and 0 <= c1[1] < num_uv and 0 <= c2[1] < num_uv
      ):
        continue
      uv0, uv1, uv2 = uv_arr[c0[1]], uv_arr[c1[1]], uv_arr[c2[1]]
      x0, y0 = uv0[0] * (width - 1), (1.0 - uv0[1]) * (height - 1)
      x1, y1 = uv1[0] * (width - 1), (1.0 - uv1[1]) * (height - 1)
      x2, y2 = uv2[0] * (width - 1), (1.0 - uv2[1]) * (height - 1)

      ix0 = max(0, int(np.floor(min(x0, x1, x2))))
      ix1 = min(width - 1, int(np.ceil(max(x0, x1, x2))))
      iy0 = max(0, int(np.floor(min(y0, y1, y2))))
      iy1 = min(height - 1, int(np.ceil(max(y0, y1, y2))))
      if ix0 > ix1 or iy0 > iy1:
        continue

      denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
      if abs(denom) < 1e-8:
        continue
      inv_den = 1.0 / denom

      gx = np.arange(ix0, ix1 + 1, dtype=np.float32)[None, :]
      gy = np.arange(iy0, iy1 + 1, dtype=np.float32)[:, None]
      w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) * inv_den
      w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) * inv_den
      w2 = 1.0 - w0 - w1
      inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
      if not np.any(inside):
        continue

      p0, p1, p2 = v_arr[c0[0]], v_arr[c1[0]], v_arr[c2[0]]
      p_interp = w0[..., None] * p0 + w1[..., None] * p1 + w2[..., None] * p2
      if (
          num_n > 0
          and 0 <= c0[2] < num_n
          and 0 <= c1[2] < num_n
          and 0 <= c2[2] < num_n
      ):
        vn0, vn1, vn2 = n_arr[c0[2]], n_arr[c1[2]], n_arr[c2[2]]
        n_interp = (
            w0[..., None] * vn0 + w1[..., None] * vn1 + w2[..., None] * vn2
        )
      else:
        fn = np.cross(p1 - p0, p2 - p0)
        fn_norm = fn / max(float(np.linalg.norm(fn)), 1e-8)
        n_interp = np.broadcast_to(fn_norm, p_interp.shape)

      n_len = np.maximum(np.linalg.norm(n_interp, axis=-1, keepdims=True), 1e-8)
      n_unit = n_interp / n_len

      dx1, dy1 = x1 - x0, y1 - y0
      dx2, dy2 = x2 - x0, y2 - y0
      det_xy = dx1 * dy2 - dx2 * dy1
      if abs(det_xy) > 1e-8:
        inv_det_xy = 1.0 / det_xy
        dp1, dp2 = p1 - p0, p2 - p0
        dp_dx = (dy2 * dp1 - dy1 * dp2) * inv_det_xy
        dp_dy = (-dx2 * dp1 + dx1 * dp2) * inv_det_xy
        len_dx = float(np.linalg.norm(dp_dx))
        len_dy = float(np.linalg.norm(dp_dy))
        tri_mm_per_px = float(np.sqrt(max(len_dx * len_dy, 1e-18)) * 1000.0)
        tu_raw = dp_dx / max(len_dx, 1e-8)
        tv_raw = -dp_dy / max(len_dy, 1e-8)
      else:
        tri_mm_per_px = 1.0
        tu_raw = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        tv_raw = np.array([0.0, 1.0, 0.0], dtype=np.float32)

      tu_proj = tu_raw[None, None, :] - (
          np.sum(tu_raw[None, None, :] * n_unit, axis=-1, keepdims=True)
          * n_unit
      )
      tu_len = np.linalg.norm(tu_proj, axis=-1, keepdims=True)
      if np.any(tu_len < 1e-6):
        ref_axis = np.where(
            np.abs(n_unit[..., 2:3]) < 0.9,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        tu_fb = np.cross(ref_axis, n_unit)
        tu_proj = np.where(tu_len >= 1e-6, tu_proj, tu_fb)
        tu_len = np.linalg.norm(tu_proj, axis=-1, keepdims=True)
      tu_unit = tu_proj / np.maximum(tu_len, 1e-8)

      ortho_v = np.cross(n_unit, tu_unit)
      ortho_v /= np.maximum(
          np.linalg.norm(ortho_v, axis=-1, keepdims=True), 1e-8
      )
      handedness = np.where(
          np.sum(tv_raw[None, None, :] * ortho_v, axis=-1, keepdims=True) < 0.0,
          np.float32(-1.0),
          np.float32(1.0),
      )
      tv_unit = ortho_v * handedness

      xyz_buf[iy0 : iy1 + 1, ix0 : ix1 + 1][inside] = p_interp[inside]
      nrm_buf[iy0 : iy1 + 1, ix0 : ix1 + 1][inside] = n_unit[inside]
      tu_buf[iy0 : iy1 + 1, ix0 : ix1 + 1][inside] = tu_unit[inside]
      tv_buf[iy0 : iy1 + 1, ix0 : ix1 + 1][inside] = tv_unit[inside]
      mm_px_buf[iy0 : iy1 + 1, ix0 : ix1 + 1][inside] = tri_mm_per_px
      valid[iy0 : iy1 + 1, ix0 : ix1 + 1][inside] = True
      part_id_buf[iy0 : iy1 + 1, ix0 : ix1 + 1][inside] = p_idx

  xyz_dil = dilate_uv_gutters(xyz_buf, valid, max_dist_px=12)
  nrm_dil = dilate_uv_gutters(nrm_buf, valid, max_dist_px=12)
  tu_dil = dilate_uv_gutters(tu_buf, valid, max_dist_px=12)
  tv_dil = dilate_uv_gutters(tv_buf, valid, max_dist_px=12)
  mm_dil = dilate_uv_gutters(mm_px_buf, valid, max_dist_px=12)
  part_id_dil = dilate_uv_gutters(part_id_buf, valid, max_dist_px=12)
  part_masks = {
      stem: part_id_dil == p_idx for p_idx, stem in enumerate(part_stems)
  }
  return {
      "xyz": xyz_dil,
      "normal": nrm_dil,
      "tangent_u": tu_dil,
      "tangent_v": tv_dil,
      "mm_per_texel": mm_dil,
      "valid_mask": valid,
      "part_id": part_id_dil,
      "part_masks": part_masks,
  }


_IN_3D_HEIGHT_DIFF = False


def normal_rgb_from_3d_height_fn(
    height_fn: Any,
    uv_world_fields: Mapping[str, Any],
    *,
    step_mm: float = 0.15,
    strength: float = 1.0,
) -> np.ndarray:
  """Evaluates a 3D height function into an OpenGL tangent-space normal map.

  Differentiates `height_fn(xyz)` directly in 3D along the local orthonormal UV
  surface tangents (`tangent_u`, `tangent_v` from `rasterize_uv_world_fields`),
  avoiding 2D UV-island gutter cliffs, seam discontinuities, and UV stretch.

  Args:
    height_fn: Callable `fn(pts_m: np.ndarray) -> np.ndarray` mapping `(..., 3)`
      3D coordinates in meters to scalar surface elevation (when elevation is in
      `mm`, `strength=1.0` corresponds to exact 1:1 geometric slope `dh_mm /
      ds_mm`).
    uv_world_fields: Dict returned by `rasterize_uv_world_fields`.
    step_mm: Central finite-difference step along the surface tangent in `mm`.
    strength: Slope multiplier controlling normal-map relief intensity.

  Returns:
    `(H, W, 3)` `uint8` OpenGL tangent-space normal map (`(128, 128, 255)`
    flat).
  """
  global _IN_3D_HEIGHT_DIFF
  xyz = np.asarray(uv_world_fields["xyz"], dtype=np.float32)
  tu = np.asarray(uv_world_fields["tangent_u"], dtype=np.float32)
  tv = np.asarray(uv_world_fields["tangent_v"], dtype=np.float32)
  step_v = max(float(step_mm), 1e-4)
  step_m = np.float32(step_v * 1e-3)
  inv_2s = np.float32(1.0 / (2.0 * step_v))

  prev_diff = _IN_3D_HEIGHT_DIFF
  _IN_3D_HEIGHT_DIFF = True
  try:
    dh_du = (
        np.asarray(height_fn(xyz + step_m * tu), dtype=np.float32)
        - np.asarray(height_fn(xyz - step_m * tu), dtype=np.float32)
    ) * inv_2s
    dh_dv = (
        np.asarray(height_fn(xyz + step_m * tv), dtype=np.float32)
        - np.asarray(height_fn(xyz - step_m * tv), dtype=np.float32)
    ) * inv_2s
  finally:
    _IN_3D_HEIGHT_DIFF = prev_diff

  norm_x = -dh_du * float(strength)
  norm_y = -dh_dv * float(strength)
  norm_z = np.ones_like(dh_du, dtype=np.float32)
  inv_len = 1.0 / np.maximum(
      np.sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z), 1e-8
  )
  rgb = np.stack(
      [
          (norm_x * inv_len * 0.5 + 0.5) * 255.0,
          (norm_y * inv_len * 0.5 + 0.5) * 255.0,
          (norm_z * inv_len * 0.5 + 0.5) * 255.0,
      ],
      axis=-1,
  )
  vm = uv_world_fields.get("valid_mask")
  if vm is not None:
    rgb = dilate_uv_gutters(rgb, vm, max_dist_px=12)
  return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def fbm_noise_3d(
    xyz: np.ndarray,
    *,
    base_freq: float = 25.0,
    wavelength_mm: float | None = None,
    octaves: int = 4,
    persistence: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
  """Evaluates seam-free 3D fractional Brownian motion on `(..., 3)` XYZ in meters."""
  rng = np.random.default_rng(seed)
  pts = np.asarray(xyz, dtype=np.float32)
  total = np.zeros(pts.shape[:-1], dtype=np.float32)
  amp = 1.0
  amp_sum = 0.0
  freq = (
      472.0 / max(float(wavelength_mm), 1e-4)
      if wavelength_mm is not None
      else float(base_freq)
  )
  grid_n = 64

  ndi = _get_ndimage()
  for oct_i in range(max(1, int(octaves))):
    lattice = rng.uniform(-1.0, 1.0, (grid_n, grid_n, grid_n)).astype(
        np.float32
    )
    lattice = ndi.gaussian_filter(lattice, sigma=0.85, mode="wrap")
    shift = np.array(
        [11.3 * oct_i, -7.9 * oct_i, 5.7 * oct_i], dtype=np.float32
    )
    coords = (pts * freq + shift) * (grid_n / 8.0)
    flat_c = coords.reshape(-1, 3).T
    sampled = ndi.map_coordinates(
        lattice, flat_c, order=1, mode="wrap"
    ).reshape(pts.shape[:-1])
    total += amp * sampled
    amp_sum += amp
    amp *= float(persistence)
    freq *= 2.03

  normed = total / max(amp_sum * 0.45, 1e-6)
  return np.clip(normed, -1.0, 1.0)


def triplanar_project_map(
    map_2d: np.ndarray,
    xyz: np.ndarray,
    world_normal: np.ndarray,
    *,
    texels_per_meter: float = 4000.0,
    sharpness: float = 4.0,
    preserve_float: bool = False,
) -> np.ndarray:
  """Projects any 2D grayscale or RGB map onto 3D `(X, Y, Z)` without UV seams."""
  src = np.asarray(map_2d, dtype=np.float32)
  mh, mw = src.shape[:2]
  pts = np.asarray(xyz, dtype=np.float32) * float(texels_per_meter)
  nrm = np.abs(np.asarray(world_normal, dtype=np.float32))
  weights = np.power(np.maximum(nrm, 1e-4), float(sharpness))
  weights /= np.maximum(np.sum(weights, axis=-1, keepdims=True), 1e-6)

  def _sample_plane(u_px: np.ndarray, v_px: np.ndarray) -> np.ndarray:
    u_fl = np.floor(u_px)
    v_fl = np.floor(v_px)
    fu = (u_px - u_fl).astype(np.float32)
    fv = (v_px - v_fl).astype(np.float32)
    ix0 = np.mod(u_fl.astype(np.int32), mw)
    iy0 = np.mod(v_fl.astype(np.int32), mh)
    ix1 = (ix0 + 1) % mw
    iy1 = (iy0 + 1) % mh
    s00 = src[iy0, ix0]
    s10 = src[iy0, ix1]
    s01 = src[iy1, ix0]
    s11 = src[iy1, ix1]
    if src.ndim == 3:
      fu = fu[..., None]
      fv = fv[..., None]
    top = s00 * (1.0 - fu) + s10 * fu
    bot = s01 * (1.0 - fu) + s11 * fu
    return top * (1.0 - fv) + bot * fv

  proj_x = _sample_plane(pts[..., 1], -pts[..., 2])
  proj_y = _sample_plane(pts[..., 0], -pts[..., 2])
  proj_z = _sample_plane(pts[..., 0], -pts[..., 1])

  if src.ndim == 3:
    blended = (
        weights[..., 0:1] * proj_x
        + weights[..., 1:2] * proj_y
        + weights[..., 2:3] * proj_z
    )
  else:
    blended = (
        weights[..., 0] * proj_x
        + weights[..., 1] * proj_y
        + weights[..., 2] * proj_z
    )
  if map_2d.dtype == np.uint8 and not preserve_float and not _IN_3D_HEIGHT_DIFF:
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)
  return blended
