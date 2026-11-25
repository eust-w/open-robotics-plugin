# Franka Research 3 Duo Hardware Limits, Mount Geometry & Sensor Specs

## 1. FR3 7-DOF Arm Joint & Dynamic Limits (Per Arm)

| Joint | Position Min (`rad`) | Position Max (`rad`) | IK Safety Margin (`rad`) | HW Max Vel (`rad/s`) | Bring-Up Vel Cap (`kIgnore` / `kEnforce`) | Max Torque (`Nm`) | Max Accel (`rad/s^2`) | Max Jerk (`rad/s^3`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fr3_joint1` | `-2.7437` | `+2.7437` | `0.025` | `2.62` | `0.262` (`15 deg/s`) / `0.524` (`30 deg/s`) | `87.0` | `7.5` | `3500` |
| `fr3_joint2` | `-1.7837` | `+1.7837` | `0.025` | `2.62` | `0.262` / `0.524` | `87.0` | `7.5` | `3500` |
| `fr3_joint3` | `-2.9007` | `+2.9007` | `0.025` | `2.62` | `0.262` / `0.524` | `87.0` | `9.5` | `4500` |
| `fr3_joint4` | `-3.0421` | `-0.1518` | **`0.100`** (`[-2.9421, -0.2518]`) | `2.62` | `0.262` / `0.524` | `87.0` | `9.5` | `4500` |
| `fr3_joint5` | `-2.8065` | `+2.8065` | `0.025` | `5.26` | `0.262` / `0.524` | `12.0` | `9.5` | `4500` |
| `fr3_joint6` | `+0.5445` | `+4.5169` | `0.025` | `4.18` | `0.262` / `0.524` | `12.0` | `9.5` | `4500` |
| `fr3_joint7` | `-3.0159` | `+3.0159` | `0.025` | `5.26` | `0.262` / `0.524` | `12.0` | `9.5` | `4500` |

*   **Mandatory `fr3_joint4` `0.100 rad` padding:** Clamp `fr3_joint4` to `[-2.9421, -0.2518] rad` (`0.100 rad` margin) with nullspace posture regularization (`posture_cost >= 1e-2`, `posture_gain >= 0.5`). *Why:* Gravity droop (`20-35 mrad`) near `-3.0421 rad` trips FR3's `-3.0321 rad` soft reflex and risks elbow-flip singularities.
*   **J1/J2 acceleration & jerk caps:** `FR3_HW_MAX_ACC_RAD_S2` caps J1–J2 at `7.5 rad/s^2` (`3500 rad/s^3` jerk) and J3–J7 at `9.5 rad/s^2` (`4500 rad/s^3` jerk) to avoid `communication_constraints_violation` during 1 kHz position streaming.
*   **Nominal joint impedance gains:** `Kp = (600, 600, 600, 600, 250, 150, 50) Nm/rad`, `Kd = (30, 30, 30, 30, 10, 10, 5) Nm*s/rad`.
*   **Canonical neutral rest poses (`rad`):**
    *   Default FR3 Duo bimanual home (rotated outward with vertical TCPs):
        *   `left`: `(0.90, -0.30, -0.4330, -2.1180, 0.8727, 1.9509, -0.7647)`
        *   `right`: `(-0.90, -0.30, 0.4330, -2.1180, -0.8727, 1.9509, 0.7647)`

---

## 2. Coordinate Frame & Menagerie Mount Geometry

All controller Cartesian coordinates, TCP readouts, and external wrenches are expressed in the Duo **`base` frame (`franka_fr3_duo_world`)**:

*   `+X`: forward across the tabletop (`[0.15, 0.95] m`).
*   `+Y`: robot-left / operator-right when facing the pedestal (`[-0.55, +0.55] m`).
*   `+Z`: vertically upward (`z_table_m` typically `[-0.03, +0.05] m`; minimum inter-arm clearance `>= 0.030 m`).
*   **Floor-mounted pedestal & tabletop defaults (`config/robot_config.json`):** See [fr3_duo_dimensions_reference.png](../assets/fr3_duo_dimensions_reference.png). The `base` origin sits `0.163887 m` (`+0.1639 m`) below the flat center bracket underside between shoulder cheeks (`base_height_m`, default `0.913887 m`), whose front lip is at `x = +0.08437 m` (`+0.0844 m`). Tabletop Z derives from floor measurements as `z_table_m = 0.163887 - (base_height_m - table_height_m)` (default ALOHA `table_height_m = 0.75 m` -> `z_table_m = 0.0 m`). The default ALOHA `1.22 m` wide (`table_width_m`) x `0.74 m` deep (`table_depth_m`) tabletop (`0.04 m` thick) sits `table_front_offset_m = 0.25 m` (`25 cm`) ahead of that bracket underside front lip (`x_front = 0.33437 m`, `x_center = 0.70437 m`; exposed via `cfg.table_box_bounds()`).

Canonical MJCF composition (`fr3_mjcf.build_fr3_duo_mjcf()`):

*   **Source:** Built directly from `mujoco_menagerie` (`franka_fr3_duo/fr3_duo_with_spine.xml` + `robotiq_2f85/2f85.xml`).
*   **Naming normalization:** Renames `{left,right}_fr3v2_*` joints/bodies to `{left,right}/fr3_*`.
*   **Attachments:** Attaches Robotiq 2F-85 grippers at `{left,right}_attachment_site` (rotated `90 deg` about `+Z`) and mounts cameras on `head_link` and `{side}/fr3_link7`.

*   **Tilted shoulder mounts (`base` frame in `fr3_duo_with_spine.xml`):**
    *   `left/fr3_link0`: `pos = (0.037, 0.05018, 0.05)` m, `quat = (0.865807, -0.436878, 0.0222879, -0.242939)` (`wxyz`)
    *   `right/fr3_link0`: mirrored symmetric tilted mount on the Duo pedestal
*   **Head & wrist camera extrinsics:**
    *   `head_camera_left` / `head_camera_right` on `head_link` (`pos = (0.069440, 0.0 / -0.063, 0.230128)` m, `quat = (0.643441, 0.293230, -0.293230, -0.643441)` `wxyz`, `63.0 mm` stereo baseline).
    *   `wrist_camera_{left,right}` (`{side}/wrist_camera/wrist_camera_{side}`) on `{side}/fr3_link7` (`pos = (0.0639045, 0.0, 0.173407)` m, `quat = (0.122788, 0.696364, 0.696364, 0.122788)` `wxyz`).
*   **Note on `pylibfranka` `O_T_EE`:** `RobotState.O_T_EE` is a 16-element **column-major (Fortran-order)** flat array in each arm's local `link0` frame (`np.asarray(o_t_ee, dtype=np.float64).reshape((4, 4), order="F")`). Transform by `T_base_arm` before comparing against `base`-frame MJCF FK!

---

## 3. Robotiq 2F-85 USB/RS-485 Modbus RTU Specifications

*   **Serial transport:** `115200` baud, `8N1`, parity `None`, default Modbus slave ID `9`; bind via `/dev/serial/by-id/usb-...` symlinks.
*   **Registers:**
    *   Write (`0x10`, `0x03E8`..`0x03EA`, 3 registers / 6 bytes): byte 0 `rACT | (rGTO << 3)`, byte 3 `rPR` (`0..255`), byte 4 `rSP` (`0..255`), byte 5 `rFR` (`0..255`).
    *   Read (`0x04`, `0x07D0`..`0x07D2`, 3 registers / 6 bytes): byte 0 `gACT`, `gGTO`, `gSTA`, `gOBJ` (`(b0 >> 6) & 0x03`: `0`=moving, `1`=outer contact, `2`=inner object stall, `3`=at requested target), byte 2 `gFLT`, byte 3 `gPR`, byte 4 `gPO` (`0..255`), byte 5 `gCU` (`10 * raw mA`).
*   **Timing & closure physics:**
    *   Normalized command/status: `0.0` = fully open (`85 mm`, `raw = 0`), `1.0` = fully closed (`0 mm`, `raw = 255`).
    *   Serial onset deadtime: `~160 ms`; full-stroke close time: `~950 ms`.
    *   **Empty pad-to-pad contact:** Physical silicone pads touch at `raw = 210..231 / 255` (`ratio ~0.82-0.91`, never `1.0`), generating `~26 N` internal pinch force. Detect object grasps via **jaw travel shortfall** (`gOBJ == 2`, `position_ratio < 0.92`; stall overdrive cap `0.032` -> `0.832` max object command) confirmed by load change or camera view.
    *   **Flange & TCP offset:** Stock 2F-85 `pinch_site` sits at `z = 0.145 m` in gripper base frame (`~11.4 mm` 4-bar arc plunge during closure; extended finger variants such as `0.185 m` can be set via `tcp_offset_z_m`).

---

## 4. 4-Camera UVC Perception Specs & Optical Frame Conversion

| Canonical Stream | Hardware Sensor | Resolution & FPS | Nominal Intrinsics (`HD720`) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `head_camera_left` | Stereolabs ZED Mini (left half of `2560x720` UVC) | `1280x720 @ 30 Hz` | `fx=fy=710.0, cx=640, cy=360` | `63.0 mm` stereo baseline; optional `.conf` rectification |
| `head_camera_right` | Stereolabs ZED Mini (right half of `2560x720` UVC) | `1280x720 @ 30 Hz` | `fx=fy=710.0, cx=640, cy=360` | Right stereo view (`-0.063 m` baseline offset) |
| `wrist_camera_left` | Intel RealSense D405 (RGB UVC interface) | `1280x720 @ 30 Hz` | `fx=fy=635.0, cx=640, cy=360` | Short-range passive sensor (`0.07-0.50 m`); RGB-only over pure UVC |
| `wrist_camera_right` | Intel RealSense D405 (RGB UVC interface) | `1280x720 @ 30 Hz` | `fx=fy=635.0, cx=640, cy=360` | Opposing rest orientation comes from tilted ±45° shoulder mounts via FK—never rotate raw pixels in software |

*   **Camera frame convention:** MuJoCo cameras look along `-Z_cam` (`+X` right, `+Y` up), whereas OpenCV pinhole helpers (`rgb-object-localization`, `visual-servoing`) look along `+Z_cam` (`+X` right, `+Y` down). Convert camera orientation matrices via `R_world_cv = R_world_mj @ np.diag([1.0, -1.0, -1.0])`.
