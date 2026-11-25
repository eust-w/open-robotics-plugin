---
name: visual-servoing
description: >-
  Aligns an eye-in-hand physical or simulated robot arm end-effector (SO-101,
  I2RT YAM, or MuJoCo robot grippers) and grasp orientation to sub-millimeter
  precision using robot IBVS (analytical hover-plane ray alignment or CPU OpenCV
  feature-matching IBVS). Use only for closed-loop eye-in-hand robot gripper
  visual servoing before grasping or manipulation. Don't use for non-robot
  microscope, telescope/PTZ, AR glasses, or homography tasks.
metadata:
  publisher: google
---

# Eye-in-Hand Visual Servoing ([`Ibvs`](scripts/ibvs.py))

Use bundled [`Ibvs`](scripts/ibvs.py) (`opencv-python numpy`, CPU only, no GPU
or `torch`) with calibrated camera intrinsics `(fx_px, fy_px, cx_px, cy_px)` and
hand-eye extrinsics `T_ee_cam` (`T_ee_cam` and `T_world_cam` use OpenCV pinhole
axes `+X` right, `+Y` down, `+Z` forward; from MJCF use
`mp.get_camera_poses_opencv`, never raw `cam_xmat`; if exact `K`/`T_ee_cam` are
absent, estimate `K` from approximate FOV and align closed-loop with a 2-nudge
empirical Jacobian at `1-2 cm` hover clearance per `workspace-calibration`):

1. **Choose the Visual Frontend by Target Texture**:
   - **Smooth / Low-Texture Objects**: Segment the target foreground mask (passing an optional `roi_mask` to exclude any robot/gripper geoms visible in the FOV) and extract pose features via `Ibvs.ray_plane_contour_pose` or `Ibvs.pca_contour_target(..., min_aspect=1.0, roi_mask=roi_mask, prev_theta_rad=...)`. Use `det["anisotropy"]` (`eigval_max / eigval_min`) to decide whether wrist yaw alignment is required vs. keeping neutral wrist roll on symmetric objects.
   - **Textured Objects (CPU `match_keypoints`)**: Call `m = match_keypoints(img_cur, img_goal, detector="sift", mask_cur=target_mask)` (`"orb"` is faster; pass masks to exclude gripper fingers). It keeps Lowe-ratio matches with RANSAC homography inliers and returns `m["u_px"], m["v_px"]` -> `m["u_goal_px"], m["v_goal_px"]` for `compute_step`, or `None` when fewer than `min_inliers` survive (re-detect the target instead of servoing on bad matches).
2. **Analytical Hover-Plane Alignment (Known Support Plane `z_plane_m`, 2 Steps to `< 1 mm`)**:
   - Compute the optimal unoccluded hover clearance analytically from `T_ee_cam` via `Ibvs.principal_ray_hover_distance(T_ee_cam, approach_axis_ee)["hover_distance_m"]` (clamped to safe workspace clearance `h_hover`).
   - At hover, compute `T_world_cam = T_world_ee(q_actual) @ T_ee_cam` from measured joint states and call `target = Ibvs.ray_plane_contour_pose(mask, fx_px, fy_px, cx_px, cy_px, T_world_cam, z_plane_m, roi_mask=roi_mask, prev_yaw_rad=...)` (which back-projects the contour onto `z_plane_m` first with arc-length weighting so tilted-camera perspective never biases the 3D midpoint).
   - Command the robot's IK solver to `(target["x_m"], target["y_m"])` (aligning jaw yaw to `target["yaw_rad"]` when `target["anisotropy"] >= 2.0`) until metric XY error `||(x_ee, y_ee) - (target["x_m"], target["y_m"])|| < pos_tol_m` (or pixel error against `servo.project_ee_point(h_actual * approach_axis_ee, T_ee_cam)` converges).
3. **Full 6-DOF `IBVS` Velocity Step (Unknown Plane or Non-Planar Approach)**:
   - Pair each frame timestamp `t_img` with `q_actual(t_img)`, instantiate `servo = Ibvs(fx_px, fy_px, cx_px, cy_px)` with depth `z_depth_m` (from aligned RGB-D or two-view triangulation), call `step = servo.compute_step(u_px, v_px, z_depth_m, u_goal_px, v_goal_px, T_ee_cam)`, and integrate `T_target = T_world_ee @ Ibvs.se3_exp(step["v_ee"], dt_s)` into differential IK until `step["converged"]`.
4. **Loop Protocol**:
   - **Fresh frame per step**: Discard buffered frames before each capture, recompute `T_world_cam` from `q_actual` at capture time, and wait for each commanded step to finish before the next capture.
   - **Iteration cap**: Stop after `~10` hover-plane corrections or `~200` IBVS velocity steps without convergence; re-localize and re-plan instead of continuing.
   - **Target lost**: If detection returns `None` or the contour touches the image border, never extrapolate the last target. Retract `3–5 cm` along `-approach_axis_ee` to widen the view and re-detect; if still lost, return to the look pose and re-localize with an object-localization skill.
   - **Camera occlusion near the goal**: If the gripper, tool, or held object will block the camera before the goal, stop servoing at the last pose where the target is fully visible and finish with a short, task-appropriate open-loop move. If the camera stays clear, servo all the way to the goal.
