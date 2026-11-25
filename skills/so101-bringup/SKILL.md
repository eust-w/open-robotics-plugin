---
name: so101-bringup
description: >-
  Guides physical SO-101 6-DOF robot arm bring-up from environment setup through
  Hello World wave: LeRobot environment check with lightweight Feetech SDK
  fallback (Stage 0), arm discovery and STS3215 joint calibration (Stage 1),
  optional camera discovery (Stage 2), and Hello World wave smoke test (Stage
  3). Use only when setting up, calibrating, checking, or operating a physical
  SO-101 robot arm. Don't use for non-SO-101 webcam, CNC, or 3D printer tasks.
metadata:
  publisher: google
---

# SO-101 Hardware Bring-Up

Bring up an **SO-101 follower arm** (`so101_follower`, 6× Feetech STS3215
servos) and any optional workspace cameras:

1.  **Stage 0 — Environment Check & Setup**:
    -   **LeRobot-first**: Check whether `lerobot` (`SO101Follower`) and `cv2`
        are already importable in the active Python environment. If LeRobot is
        already installed (e.g., from LeRobot's starter guide), use it natively
        (`SO101Follower`, `lerobot-calibrate`, `lerobot-setup-motors`,
        `RobotKinematics`, and `HF_LEROBOT_HOME`).
    -   **If not yet installed**: Either install `lerobot[feetech,kinematics]`
        (`--prefer-binary`; latest `lerobot` requires Python `>= 3.12`, while
        `v0.4.x` supports `3.10–3.11`) for full teleop/dataset/policy
        workflows, **or** use the fast LeRobot-compatible quick-start stack
        (`feetech-servo-sdk`, `pyserial`, `numpy`, `opencv-python` for cameras,
        and `placo` for FK/IK with `--prefer-binary`) when running on Python
        `< 3.12` or doing quick hardware bring-up.
    -   For FK/IK (`so101_new_calib.urdf`), reuse an existing local copy or
        sparse-checkout `Simulation/SO101` from
        `https://github.com/TheRobotStudio/SO-ARM100`.

2.  **Stage 1 — Arm Discovery & Joint Calibration (Torque OFF)**:
    -   **Fresh DIY kits only**: If servos are brand-new unconfigured kits (all
        shipping at factory `ID 1`), assign IDs `1–6` first with
        `lerobot-setup-motors --robot.type=so101_follower --robot.port=<port>`.
    -   **Read-only check & calibration reuse**: Run bundled
        [`calibrate.py`](scripts/calibrate.py) `--check` (cross-platform USB
        port detection, direct Feetech bus check with `Torque_Enable = 0`) to
        verify all 6 STS3215 servos (`IDs 1–6`, `1,000,000` baud) respond with
        healthy supply voltage, validate onboard EEPROM calibration, sync it
        into LeRobot's calibration cache (`$HF_LEROBOT_HOME` or
        `~/.cache/huggingface/lerobot`), and record the arm's natural limp
        `rest_pose`.
    -   **Manual calibration (only if uncalibrated or explicitly requested)**:
        Keep torque OFF and give the user the calibration command to run in
        their own interactive terminal: `lerobot-calibrate
        --robot.type=so101_follower --robot.port=<port> --robot.id=<id>` when
        LeRobot is installed, or `python3 scripts/calibrate.py --port <port>
        --robot-id <id>` (writes the exact same STS3215 EEPROM registers and
        LeRobot JSON cache):
        1.  Hold the arm at the middle of each joint's range and press `Enter`.
        2.  Sweep all 6 joints through their full min-to-max range and press
            `Enter` to save.
        3.  Set the arm back straight/centered in its folded resting spot
            before powered motions.
    -   **Span heads-up**: Flag any joint whose calibrated span falls outside
        `~70%–130%` of its nominal URDF span (`220°, 200°, 194°, 190°, 320°,
        110°`) as a non-blocking heads-up (too narrow suggests an incomplete
        sweep; too wide suggests wrap-around or a slipped horn).

3.  **Stage 2 — Camera Discovery (Optional)**:
    -   Skip if running a bare arm without cameras.
    -   Enumerate OS cameras and capture a quick still from each (waiting
        `~1 s` for auto-exposure to settle). Map OS camera names to indices to
        distinguish external USB cameras from built-in laptop webcams or macOS
        Continuity Cameras, and label roles (`wrist` if jaws are visible;
        stationary `fixed`/`overhead` otherwise).
    -   Request `MJPG` when streaming two cameras on USB 2.0 so they do not
        saturate bus bandwidth, and flush a few buffered frames from OpenCV
        after arm motion before reading frames for perception or visual
        servoing.
    -   Save the discovered port, calibration path, `rest_pose`, and any camera
        indices in a simple workspace config (e.g., `config/robot_config.json`).

4.  **Stage 3 — "Hello World" Wave Smoke Test ([`wave.py`](scripts/wave.py))**:
    -   Run bundled [`wave.py`](scripts/wave.py) (`--dry-run` offline, or
        `--port auto` / `--port <port>` on hardware; uses LeRobot
        `SO101Follower` when available and falls back seamlessly to
        `feetech-servo-sdk`, `1.00 rad/s` cap) to unfold safely via a cleared
        pre-park pose, perform a symmetric wide greeting wave with a gentle
        gripper flutter, and fold smoothly back to `rest_pose` with torque
        disabled.
    -   If a camera is configured, optionally capture before/mid/after frames
        (outside the motion loop) to confirm the motion visually.
    -   If the lightweight fallback was used because `lerobot` is not installed
        yet, note in your final summary that calibration is already saved in
        LeRobot's format and ask whether the user would like to install full
        `lerobot` (`~5–10 min`) for teleop, dataset recording, or policy
        workflows.

## SO-101 Hardware, Kinematics & Limits Notes

-   **Calibrated limits**: hard speed cap `1.00 rad/s` (`~57°/s`); stay within
    calibrated joint ranges (`~5°` margin when moving, allowing departure
    from/return to `rest_pose`); keep default STS3215 gripper torque/current
    limits.
-   **Safe folding & unfolding (avoid self-collision)**: Never fold straight
    from a working pose into a tightly tucked `rest_pose` (especially when
    `wrist_flex` rests near `+70°`). Pass through a cleared intermediate
    pre-park pose first (base centered near rest, `wrist_roll ≈ 0°`,
    `shoulder_lift ≈ -48°`, `elbow_flex ≈ +48°`, `wrist_flex ≈ +20°`), then
    fold straight down into `rest_pose` — and unfold through that same pose
    before panning or twisting the wrist.
-   **Torque & register gotchas**: `SO101Follower.connect()` enables torque
    immediately — always use a read-only Feetech bus connection for inspection
    and keep `Torque_Enable = 0` whenever a human handles the arm. STS3215
    `Present_Load` uses 10-bit sign-magnitude encoding (bit 10 is direction —
    mask `raw & 0x3FF` for load magnitude).
-   **Motor map & base-frame signs** (`gripper_frame_link`, `+X` forward, `+Y`
    left, `+Z` up):
    1.  `shoulder_pan` (`+` right / `-Y`, `-` left / `+Y`)
    2.  `shoulder_lift` (`-` folded back, `+` pitches forward/down)
    3.  `elbow_flex` (`+` folds in, `-` extends up/out)
    4.  `wrist_flex` (`+` pitches down toward table)
    5.  `wrist_roll` (jaw roll)
    6.  `gripper` (`0–100`, higher = more open; physical closed stop may sit
        above `0%` depending on calibration — do not blindly command `0%`)
-   **Kinematics quirks**: Always clamp IK solutions (`RobotKinematics` /
    `placo`) to the arm's calibrated joint bounds (the solver only enforces
    wider generic URDF limits), and ignore the 6 benign adjacent-link
    self-collision warnings printed at neutral pose when loading
    `so101_new_calib.urdf`.
