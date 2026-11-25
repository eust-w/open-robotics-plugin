# Franka Research 3 Duo Bring-Up Checklist (Stages 0–4)

Use this runbook when setting up a **Franka Research 3 Duo (`fr3_duo`)** cell for the first time, after hardware/desk changes, or when diagnosing hardware (`config/robot_config.json` missing/unverified). Once verified, follow `R2` in [safety_rules.md](safety_rules.md) and `--dry-run` your target task directly without re-running bring-up motion steps.

All stages run through the single CLI entrypoint [bringup.py](../scripts/bringup.py) (`uv run scripts/bringup.py ...`), which orchestrates internal stage modules `discover_hardware.py`, `check_cameras.py`, and `check_robot.py`.

---

## Stage 0 — Environment & `pylibfranka` Pin Check (`R12–R16`)

1. Summarize `R1–R16` from [safety_rules.md](safety_rules.md), initialize `NOTES.md`, and run zero-motion offline Stage 0 + safety preview (`uv run scripts/bringup.py --step env --dry-run`). Do **not** bundle Stage 1 geometry or Desk payload checks into an initial prompt yet.
2. **Environment & asset setup:**
   * **Python (`3.9–3.12`):** `pylibfranka` Linux x86_64 wheels require Python `3.9–3.12` (`requires-python = ">=3.9,<3.13"`). `uv run scripts/bringup.py ...` ([bringup.py](../scripts/bringup.py)) handles this automatically; for custom task scripts or an explicit virtualenv, pass `--python 3.12`.
   * **MuJoCo robot models (auto-cached):** Standalone runs of [fr3_mjcf.py](../scripts/fr3_mjcf.py) automatically sparse-clone and cache `mujoco_menagerie` (`franka_fr3_duo` + `robotiq_2f85`) into `$XDG_CACHE_HOME/mujoco_menagerie` (or `~/.cache/mujoco_menagerie`)—no manual clone or env var needed:
   ```bash
   # Zero-motion offline Stage 0 + safety preview:
   uv run scripts/bringup.py --step env --dry-run

   # Custom task script or explicit Python 3.12 venv:
   uv run --python 3.12 my_task.py --dry-run
   # Or: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python pylibfranka pyserial opencv-python numpy mujoco scipy
   ```
   * **Firmware / `pylibfranka` compatibility:** FR3 system `>= 5.7.0` pairs with `pylibfranka >= 0.14` (`libfranka >= 0.14`); older firmware returns `IncompatibleLibraryVersionException` on `read_once()`.
   * **Linux `video` / `dialout` permissions:** If udev rules do not already grant R/W access to `/dev/video*` and `/dev/ttyUSB*`:
     ```bash
     sudo usermod -aG video,dialout "$USER"
     ```
   * **Where to run:** `--dry-run` and `Fr3DuoRobot(cfg, mode="sim")` run offline anywhere. Live hardware (FCI Ethernet, `/dev/ttyUSB*`, `/dev/video*`) must run directly (unsandboxed) on the Linux control PC.

---

## Stage 1 — Read-Only Hardware Discovery & Geometry Calibration (Two-Turn Prompt, `R4`, `R8`, `R16`)

### Turn 1 — Minimal Connection Prerequisite -> Immediate Read-Only Preflight

1. Ask the operator for **only** the minimal prerequisite to probe read-only:
   * **Left and right FR3 FCI IP addresses** (common defaults: left `172.16.0.2`, right `172.16.1.2`; or verify saved IPs in `config/robot_config.json`), explicitly noting that `left` (`+Y`, operator's right) and `right` (`-Y`, operator's left) are from the **robot's perspective (not the operator's when facing the pedestal)**.
   * Confirm **joints are unlocked, FCI mode is enabled, and physical E-stop is within reach (`R16`)**.
   * *Do not ask the operator to manually verify tilted Desk gravity/payload or conflicting processes up front*—let `--step preflight --read-only` inspect those automatically and only report concrete failures if any occur.
2. Run Stage 1 preflight immediately (read-only, zero physical motion):
   ```bash
   uv run scripts/bringup.py --step preflight --read-only --output-dir logs/stage1
   uv run scripts/bringup.py --up-to preflight --left-ip 172.16.0.2 --right-ip 172.16.1.2 --output-dir logs/stage1
   ```
   * `discover_hardware.run_discovery` checks (zero motion):
     * **Host & USB/serial:** `PREEMPT_RT` status, Robotiq RS-485 ports (`0403:6015` preferred, `0403:6014` teleop ignored, auto-binds `/dev/robotiq_gripper_{left,right}`), ZED Mini / D405 V4L2 nodes, `video`/`dialout` R/W access, and conflicting host processes holding serial, camera, or FCI connections (ask the operator to stop conflicts only if detected).
     * **Desk & wrench:** `robot_mode == kIdle`, payload (`m_ee`, `F_x_Cee`, `I_ee`), tilted gravity `O_ddP_O` (`<= 8.0 deg` vs MJCF), free-air wrench `O_F_ext_hat_K <= 6.5 N`, and joint torque residual `tau_ext_hat_filtered <= 4.5 Nm` (why: upright gravity causes `> 15 N` false `tau_ext` and arm droop).
     * **Kinematics match:** `fk_pos_error_m <= 0.005` (`5.0 mm`) vs MJCF FK.
   * Once gripper/camera paths are saved to `config/robot_config.json`, subsequent runs reuse saved bindings unless `--no-write-config` / rediscovery is requested.
   * Non-`PREEMPT_RT` kernels are supported out of the box under `--realtime auto` (`RealtimeConfig.kIgnore`, `<= 15 deg/s` bring-up cap).

### Turn 2 — Physical Cell Geometry & Fingertip Calibration (`--step geometry`)

3. Calibrate tabletop height (`z_table_m`), front edge (`table_front_offset_m`), footprint (`table_width_m`, `table_depth_m`), and closed fingertip offset (`tcp_offset_z_m`) using **single-arm FR3 cuff touch (`--step geometry` / `--touch-geometry`)** (if a calibrated config already exists on disk, interactive `--step geometry` prompts `[1] Reuse existing config | [2] Recalibrate (start fresh)`):

   * **Zero-motion single-arm FR3 cuff touch (`--step geometry` / `--touch-geometry`):**
     * **Display/attach [cuff_touch_explainer.gif](../assets/cuff_touch_explainer.gif) and [fr3_duo_dimensions_reference.png](../assets/fr3_duo_dimensions_reference.png) inline before asking the operator to touch the closed fingertips to the table front edge** (when supported by the UI/harness) so the operator sees the required pose and tabletop footprint values.
     * **Why use FK touch:** Reads live FCI joint angles (`q_rad`) read-only via `probe_arm_readonly` and evaluates `fr3_kinematics.compute_arm_fk`—**zero software arm motion commands are sent**. Eliminates floor/pedestal `base_height_m` & `table_height_m` tape-measure discrepancies.
     * **Single-arm default (`--touch-arm auto`, `left` -> `right`):** Only **one** arm needs to touch the table (`left` `+Y` / operator's right when enabled, else `right`). Pass `--touch-arm both` only if you explicitly want to average both arms (rejects `|z_left - z_right| > 0.025 m`, exit code `2`).
     * **Closed-gripper & Desk `Idle` check:** Select `tcp_offset_z_m` (`0.145 m` stock 2F-85 pads vs `0.185 m` extended ALOHA fingers) and tabletop footprint (`[1]` ALOHA `1.22 x 0.74 m` vs `[2]` custom), ensure the gripper is **CLOSED** (pass `--close-grippers` to activate and close enabled Robotiq grippers before cuff touch; skipped under `--dry-run` / `--read-only`), hold the FR3 cuff button to rest the closed fingertips on the **table front edge**, release the cuff button so Desk returns to `Idle`, and press `ENTER`. `bringup.py` warns and offers a retry if Desk is not `kIdle` or `fk_pos_error_m > 5.0 mm`.
     ```bash
     # Dry-run preview of single-arm cuff touch (no live FCI sockets opened)
     uv run scripts/bringup.py --step geometry --touch-geometry --dry-run

     # Interactive calibration (prompts [1] Reuse / [2] Recalibrate if already calibrated, else cuff touch)
     uv run scripts/bringup.py --step geometry --output-dir logs/stage1

     # Auto-activate & close enabled Robotiq grippers before cuff touch
     uv run scripts/bringup.py --step geometry --touch-geometry --close-grippers --output-dir logs/stage1

     # Explicit single-arm cuff touch with extended closed tips
     uv run scripts/bringup.py --step geometry --touch-geometry --tcp-offset-z-m 0.185 --output-dir logs/stage1

     # Opt-in dual-arm averaging
     uv run scripts/bringup.py --step geometry --touch-geometry --touch-arm both --output-dir logs/stage1
     ```

   * **Non-interactive CLI overrides (optional):** Explicit measurement flags (`--base-height-m`, `--table-height-m`, `--table-front-offset-m`, `--table-width-m`, `--table-depth-m`, `--tcp-offset-z-m`) + `--confirm-geometry` are also accepted for automated runs (see [fr3_duo_dimensions_reference.png](../assets/fr3_duo_dimensions_reference.png) for frame definitions).
     ```bash
     # Preview current geometry & config completeness offline
     uv run scripts/bringup.py --step geometry --dry-run

     # Explicit non-interactive calibration + confirmation
     uv run scripts/bringup.py --step geometry --base-height-m 0.913887 --table-height-m 0.75 --table-front-offset-m 0.25 --table-width-m 1.22 --table-depth-m 0.74 --tcp-offset-z-m 0.145 --confirm-geometry
     ```
   * Even with `--yes`, standard `--step geometry` (without `--touch-geometry`) will **not** silently mark unverified default dimensions confirmed; pass explicit measurement flags + `--confirm-geometry` or run `--touch-geometry`.
   * **Completeness gate before live Stage 3/4 motion:** `bringup.py` enforces `fr3_config.validate_robot_config_complete(cfg)` before `gripper`, `nudge`, `abort-test`, and `hello-world` (exiting with code `2` and listing missing `arms.{left,right}.ip`, `arms.{left,right}.gripper_port`, `cameras.{head_stereo,wrist_camera_left,wrist_camera_right}`, or `geometry_confirmed=True`).

---

## Stage 2 — 4-Camera Freshness, Contact Sheet & MP4 Check (`R6`, `R7`)

Verify all 4 canonical streams (`head_camera_left`, `head_camera_right`, `wrist_camera_left`, `wrist_camera_right`) via `check_cameras.run_camera_check`:

```bash
uv run scripts/bringup.py --step cameras --dry-run
uv run scripts/bringup.py --up-to cameras --read-only --output-dir logs/stage2
```

1. Confirm `age <= 0.30 s`, changing frames (non-frozen digests), and adequate lighting.
2. Inspect `contact_sheet_2x2.png`, `fk_projection_overlay_2x2.png`, and `cameras_recording.mp4`.

---

## Stage 3 — Progressive Robot Verification & Abort Unwind (`R1–R6`, `R11`)

Before running live motion steps, optionally attach a persistent `tmux` session (`tmux new-session -d -s robot && tmux attach -t robot`, `R11`) so the operator can observe live telemetry and trigger `q` + `Enter` aborts. Run `--dry-run` safety preview first (`R2`), then step progressively through live checks after explicit human `"go"` (`R1`):

```bash
# Offline kinematics & collision preview (no hardware contact)
uv run scripts/bringup.py --up-to abort-test --dry-run

# Interactive progressive bring-up (prompts before live actuation; pass --yes only after workspace is verified clear)
uv run scripts/bringup.py --step gripper --output-dir logs/stage3
uv run scripts/bringup.py --step nudge --realtime auto --output-dir logs/stage3
uv run scripts/bringup.py --step abort-test --realtime auto --output-dir logs/stage3
```

* `gripper`: activates and cycles both Robotiq 2F-85 grippers `0.0 -> 1.0 -> 0.0`, confirming empty-pinch stall (`0.80 <= ratio < 1.0`).
* `nudge`: executes a smooth ($C^2$ quintic) `5 deg` Joint 1 (J1) sweep and return on `left` then `right`, enforcing `>= 30 mm` inter-arm clearance and `z_min >= z_table_m`.
* `abort-test`: verifies `q` + `Enter` / `SIGINT` decelerates smoothly and unwinds backwards along `travelled_waypoints` to `rest_pose` **without opening grippers** (`R3`, `R4`).

---

## Stage 4 — Bimanual "Hello World" Motion & Homing (`R1–R8`, `R14`)

Run the full bimanual greeting (`check_robot.run_hello_world` via `bringup.py`) or return both arms cleanly to canonical rest pose (`--step home`):

```bash
# Dry-run preview, then full interactive bring-up through Stage 4 hello-world
uv run scripts/bringup.py --up-to hello-world --dry-run
uv run scripts/bringup.py --up-to all --realtime auto --output-dir logs/stage4

# Return both arms smoothly to canonical rest_pose (or dry-run first;
# --step home enables upward-only <= 0.015 m lift-off from initial table contact)
uv run scripts/bringup.py --step home --dry-run
uv run scripts/bringup.py --step home --yes
```

`hello-world` performs and records:
* **Motion sequence:** Moves `left` and `right` arms through a collision-checked greeting pose, cycles both grippers once, returns to `rest_pose`, and cleanly closes 1 kHz FCI control.
* **Saved outputs:** Writes `stage4_hello_world.mp4` (continuous 4-camera video) and `bringup_summary.json`.
* **Report:** Emits metrics for your 6-field `R14` report.
