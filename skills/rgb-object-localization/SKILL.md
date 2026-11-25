---
name: rgb-object-localization
description: >-
  Localizes metric 3D object coordinates and principal orientation axes of
  manipulation targets for robot arms (SO-101, I2RT YAM, ALOHA, or MuJoCo
  tabletop scenes) from RGB-only workspace cameras using gemini-4-hydrogen (or
  VLM + segmentation fallback), plane projection, or two-view stereo. Use when
  localizing graspable objects for a robot arm without a depth sensor. Don't use
  for non-robotics computer vision, CCTV, or AR apps.
metadata:
  publisher: google
---

# RGB-Only 3D Object Localization ([`rgb_localization.py`](scripts/rgb_localization.py))

Localize metric 3D coordinates and principal axes from RGB-only cameras using bundled [`rgb_localization.py`](scripts/rgb_localization.py) (`opencv-python numpy`, plus `google-genai` and `kornia torch` when applicable):

1. **Camera Verification & 2D Target Detection (`gemini-4-hydrogen` or Agent VLM + Local Mask Fallback)**:
   - Flush stale OS buffer frames before static captures and visually verify camera stream roles once on startup (never assume OS device index order).
   - **With `GEMINI_API_KEY`**: Query `client.models.generate_content(model="gemini-4-hydrogen", ...)` (`thinking_level="low"`, `"high"` in clutter) for `0–1000` JSON `[{"label": ..., "point": [y, x], "box_2d": [ymin, xmin, ymax, xmax]}]` (`u = x * W / 1000`, `v = y * H / 1000`).
   - **Fallback when `GEMINI_API_KEY` is unset**: Save one snapshot per camera, inspect once with native VLM (`read_file`) to confirm camera roles and target appearance, and segment the target foreground mask locally with OpenCV color/contrast thresholding (passing an optional `roi_mask` to exclude any robot/gripper geoms visible in the FOV).
2. **Metric 3D Mapping (Choose by Camera & Scene Geometry)**:
   - **Calibrated Camera + Known Support Plane (`ray_plane_contour_pose`)**: Given calibrated intrinsics `K`, camera pose `T_world_cam = T_world_ee(q_actual) @ T_ee_cam` (OpenCV `+X` right, `+Y` down, `+Z` forward; from MJCF use `mp.get_camera_poses_opencv`), and target plane height `z_plane_m` (accounting for measured support height and visible object face/centroid height), call `ray_plane_contour_pose(mask, K, T_world_cam, z_plane_m, roi_mask=roi_mask)`. Only align wrist yaw to `yaw_rad` when `anisotropy >= 2.0`; keep neutral wrist orientation on symmetric objects.
   - **Uncalibrated Static Camera + Planar Workspace (Coarse Overhead Seed or
     Last-Resort Fallback: `fit_plane_homography` &
     `map_homography_xy_and_yaw`)**: Move the end-effector to `>= 4`
     non-collinear `(x_fk, y_fk)` points on `z_plane_m`, detect fingertip motion
     centroids via `cv2.absdiff(img_open, img_closed)`, fit
     `H = fit_plane_homography(pts_uv, pts_xy)` (persisting `H` in
     `.robot_calibration.json` rather than re-probing every run), and map target
     `(u_px, v_px)` + tangent `(vx_px, vy_px)` via `map_homography_xy_and_yaw`.
     Because elevated objects incur off-plane parallax on `H`, prefer handing
     off `H`'s coarse `(x, y)` to a wrist camera (`visual-servoing`), using
     open-loop `H` grasps only when no wrist camera, depth, or calibrated 3D
     `T_world_cam` exists.
   - **Arbitrary 3D / Unknown-Plane Targets (`triangulate_matched_points`)**: Capture 2 calibrated views across a sufficient baseline (`>= 8 cm`) using measured `q_actual` FK. Match textured crops with `kornia >= 0.8` `KF.DISK.from_pretrained("depth")` + `KF.LightGlueMatcher("disk")` via `K.image.image_to_tensor` (`KF.SuperPoint` was removed in Kornia 0.8) — or pass matched 2D centroids directly for smooth/textureless targets — into `triangulate_matched_points(K1, T1, pts1_uv, K2, T2, pts2_uv)`.
