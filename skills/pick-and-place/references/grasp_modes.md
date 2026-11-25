# Non-Default Grasp Modes

Load these patterns when standard vertical top-down parallel-jaw grasping does
not fit the object geometry, clutter layout, or end-effector type.

- **Wall / neighbor singulation (blocked jaw clearance `< 1 cm`)**: Close the
  jaws to form a narrow rigid finger, descend to mid-object height beside the
  blocked side, gently nudge `2–4 cm` toward open workspace (`<= 0.05 m/s`),
  retreat vertically, flush stale frames, and re-localize before grasping.
- **Side / angled grasps (tall cylinders, bottles, mugs, upright boxes)**: When
  `height > 2 * width` or the top face exceeds jaw aperture while the body
  fits, approach horizontally (or `45–75°` over bin walls) near mid-height
  center of mass, verify clearance with `motion-planning`, close to `g_hold`,
  and **lift vertically first** before retracting laterally.
- **Rim / handle pinch (bowls, wide cups, trays)**: Target a point on the rim
  or handle tangent, align `closing_axis_ee` radially across the wall,
  pre-open to `wall_thickness + 1–2 cm`, and place one finger inside and one
  outside.
- **Suction / magnetic & multi-finger hands**: Engage planar non-porous faces
  along the surface normal at `z_top` with guarded contact and verify vacuum
  seal before lifting. For multi-finger hands, use thumb-index pinch on small
  items and proximal-first power wrap on heavy cylinders/spheres.
