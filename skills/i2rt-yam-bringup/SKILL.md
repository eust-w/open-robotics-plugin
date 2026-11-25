---
name: i2rt-yam-bringup
description: >-
  Checks and installs the I2RT YAM SDK (i2rt), configures Linux socketcan or
  macOS gs_usb CAN bringup for I2RT YAM robot arms, and provides a wave smoke
  test (wave.py). Use only before bringing up or programming a physical or
  simulated I2RT YAM robot arm, or when asked to run a wave gesture on an I2RT
  YAM robot. Don't use for non-YAM CAN bus debugging (e.g. automotive OBD-II or
  industrial PLC CAN networks) or non-robotics tasks.
metadata:
  publisher: google
---

# I2RT YAM SDK & CAN Bringup

Ensure the official I2RT YAM Python SDK (`i2rt`) and host CAN transport are ready before running YAM robot code (`i2rt` drives physical YAM Damiao motors over 1 Mbps CAN via `python-can`; only `get_yam_robot(..., sim=True)` runs without CAN):

1. **SDK Check & Install**:
   - Check `python3 -c "from i2rt.robots.utils import ArmType, GripperType; print('i2rt ready')"`.
   - Prefer Python >= 3.10 (`uv venv --python 3.11` or `python3.11`/`3.12`)
     when available; on Python 3.9-only hosts, pass `--prefer-binary` for wheel
     deps (pinning `mink<1.0`, plus `libusb-package` on macOS if Homebrew
     `libusb` is absent) and `--ignore-requires-python --no-deps
     --no-build-isolation` for `i2rt` so `pip` does not attempt slow source
     builds of `mujoco`/`scipy` or fail on Python `>=3.10` metadata.
   - If missing (`i2rt` is not on PyPI), clone `git clone https://github.com/i2rt-robotics/i2rt.git` and install with `pip install -e ./i2rt` (or `pip install --no-deps --no-build-isolation -e ./i2rt` plus `numpy mujoco mink ruckig python-can pyyaml portal crcmod-plus dm-env` if `pyproject.toml` pins conflict with host Python).

2. **Real-Hardware CAN Bringup (Linux `socketcan` vs. macOS [`macos_can.py`](scripts/macos_can.py))**:
   - **Linux**: Bring up `socketcan` at 1 Mbps (`sudo ip link set can0 up type can bitrate 1000000`) and pass `channel="can0"` to `get_yam_robot`.
   - **macOS (Apple Silicon & Intel)**: `i2rt`'s `get_yam_robot` and `run_startup_checks` require `"can"` in `channel` and hardcode `bustype="socketcan"`, which macOS lacks; moreover, `python-can`'s built-in `GsUsbBus` calls `detach_kernel_driver(0)` (`Access denied` without `sudo`) and `dev.reset()` (wedging `gs_usb` STM32 `EP 0x81` when TX echoes are unread).
   - On macOS, ensure `libusb` and `gs-usb` are installed (`brew install libusb && pip install gs-usb pyusb` for candleLight / CANable / I2RT USB-CAN adapters) and call `install_mac_can()` from bundled [`macos_can.py`](scripts/macos_can.py) once before `get_yam_robot(channel="can0", ...)`:
     - Safe no-op on Linux; auto-discovers `libusb-1.0.dylib` across `/opt/homebrew/lib`, `/usr/local/lib`, and `/opt/local/lib` inside isolated `venv`/`conda` environments without `DYLD_LIBRARY_PATH`.
     - Maps `"can0"`, `"can1"`, ... deterministically across single or bimanual `gs_usb` adapters without `detach_kernel_driver` or `dev.reset()`, filters TX echoes (`echo_id != GS_USB_NONE_ECHO_ID`), drains `clean_error` bursts, runs a silent motor-independent `GS_CAN_MODE_LOOP_BACK` health check (with DFU auto-recovery only if `EP 0x81` is wedged), and also supports PCAN (`interface="pcan"`) or SLCAN (`interface="slcan", backend_channel="/dev/cu.usbmodem..."`).


3. **YAM Speed Limits & Hardware Teardown** (follows
   [`physical_robot_safety.md`](../../rules/physical_robot_safety.md)):
   - **Speeds**: **`0.75 rad/s` default / `1.50 rad/s` ceiling** (smooth ramps;
     see [`wave.py`](scripts/wave.py)).
   - **YAM-specific failure modes**: `robot.close()` immediately zeroes all
     Damiao torques; multi-second GIL/blocking stalls trigger Damiao
     `0xD LOST_COMM` watchdog drops; Motor 7 (gripper) over-temp during long
     idle clamps kills the entire `i2rt` CAN thread; never park elevated in
     `zero_gravity_mode=True`. When streaming a USB camera on the same USB hub
     as a `gs_usb` CAN adapter, request `MJPG` at low FPS (e.g., `640x480` at
     `10–15 fps`) or move the camera to a separate USB root port—uncompressed
     UVC video can starve bulk CAN reads and drop the motor chain mid-motion.
   - **Session pattern** (`SIGTERM` → `finally:`, smooth ramp from live
     `get_joint_pos()` to `Q_HOME` at `<= 0.50 rad/s` before `robot.close()`):

     ```python
     from contextlib import contextmanager
     import signal, sys, time
     import numpy as np

     @contextmanager
     def yam_session(robot, q_home, park_vel_rad_s=0.50, dt_s=0.01):
       prev = signal.signal(
           signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM)
       )
       try:
         yield robot
       finally:
         try:
           q0 = np.asarray(robot.get_joint_pos(), dtype=np.float64)
           q1 = q0.copy()
           q1[:len(q_home)] = q_home
           dq = float(np.max(np.abs(q1[:6] - q0[:6])))
           dur = max(0.5, 0.5 * np.pi * dq / park_vel_rad_s)
           for u in np.linspace(0.0, 1.0, max(2, int(dur / dt_s)) + 1):
             s = 0.5 * (1.0 - np.cos(np.pi * u))
             robot.command_joint_pos(q0 + s * (q1 - q0))
             time.sleep(dt_s)
         finally:
           robot.close()
           signal.signal(signal.SIGTERM, prev)

     with yam_session(robot, Q_HOME) as arm:
       # Motor heartbeat is live: fast control (<50 ms/step) in this process
       # only; run heavy encoding/VLM in a separate process or after this block.
       ...
     ```

4. **Bring-Up Wave Task ([`wave.md`](references/tasks/wave.md), [`wave.py`](scripts/wave.py))**:
   - Run [`wave.py`](scripts/wave.py) (`--sim` offline or `--channel can0` on
     hardware) for a free-space wave smoke test; see
     [`references/tasks/wave.md`](references/tasks/wave.md) for keyframes and
     flags.
