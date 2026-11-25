---
name: pick-and-place
description: >-
  Cross-embodiment robotic pick-and-place guide and gripper/wrist helpers for
  grasping, moving, rotating, stacking, binning, or placing objects in
  simulation or on hardware. Use when authoring or debugging robotic
  manipulation tasks. Don't use for pure robot bring-up, non-manipulation
  navigation, or non-robotics tasks.
metadata:
  publisher: google
---

# Pick and Place ([`grasp_helpers.py`](scripts/grasp_helpers.py))

Compose [`grasp_helpers.py`](scripts/grasp_helpers.py) (`numpy`) with
[`physical_robot_safety.md`](../../rules/physical_robot_safety.md),
[`workspace-calibration`](../workspace-calibration/SKILL.md),
[`rgb-object-localization`](../rgb-object-localization/SKILL.md),
[`visual-servoing`](../visual-servoing/SKILL.md), and
[`motion-planning`](../motion-planning/SKILL.md) (when obstacles or walls sit
near the swept path). Whenever `surface_z` or wrist hand-eye is not yet cached,
read [`workspace-calibration`](../workspace-calibration/SKILL.md) and import
[`calibration.py`](../workspace-calibration/scripts/calibration.py)
(`probe_surface_z`, `fit_wrist_jacobian`, `jacobian_step_xy`) — **never
hand-roll a Z-probe**.

1. **Live Perception & Pre-Localize Movable Destinations**:
   - Keep robot, camera, and IK handles live in one re-runnable script—unless
     fixed coordinates are given, **never hardcode object or destination
     `(x, y)`** from an earlier probe.
   - Localize any movable place target (cup, coaster, stack base, bin) **before
     picking**, while the gripper is empty and unoccluded.
   - Prefer wrist-camera closed-loop alignment when the target is in view
     (masking out frame borders/base clutter with `roi_mask`); add an overhead
     coarse seed when targets may lie outside the wrist FOV.
2. **Separate `z_top` from `z_grasp` — Descend Flush on Top-Down Jaw Grasps**:
   - Use `z_top = surface_z + visible_height` for 3D ray-plane localization;
     never pass `z_top` as jaw `z_grasp` (which closes in thin air).
   - For top-down jaw grasps, **descend all the way to the support plane**
     (`surface_z_cmd` on compliant arms, `surface_z + 1–2 mm` on stiff arms)
     unless the object is taller than the finger pads or flanged—this maximizes
     pad contact area. If the object rests on a raised tray, book, or coaster
     rather than the probed table, use that surface's height or depth. For
     side, rim, suction, or wall-singulation grasps, see
     [`references/grasp_modes.md`](references/grasp_modes.md).
3. **`0°` Top-Down Pitch, Wrist Limits & Gentle Closure**:
   - **Keep `0°` vertical pitch on top-down grasps whenever IK reaches it:**
     tilting even `10–15°` puts one jaw on the table while the other stays in
     mid-air, pinching nothing or sweeping the object.
   - Align jaw closing across the narrow width (perpendicular to the long axis)
     at hover, check wrist limits with `plan_wrist_yaw(...)` (`±π` pre-wind or
     midpoint regrasp), and gently nudge wall-flush objects into open space
     before grasping.
   - **Import [`grasp_helpers.py`](scripts/grasp_helpers.py)** rather than
     hand-rolling grip loops: close with `close_until_contact(...)`
     (polarity-agnostic contact stall to `g_hold`, keeping `tol_frac` small for
     thin shafts) and verify twice with `verify_grasp(...)` (**post-close** and
     **post-lift**); if `grasped` is `False`, **abort and retry** rather than
     continuing to transit.
4. **Place, Vertical Retreat & Retry Cap**:
   - Descend vertically to placement height, release gently, **retreat
     vertically first**, and confirm settled placement in a compact `.jpg`.
   - Cap retries at `<= 3` (flushing frames and re-localizing live each try).
