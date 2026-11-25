---
name: physical-robot-safety
trigger: model_decision
description: >-
  Mandatory safety, joint speed limits, drop-proof homing/teardown, and motor
  heartbeat rules for physical robots. Use only when controlling, calibrating,
  probing, or scripting physical robot arms, grippers, or motor buses. Don't
  use for pure software, web apps, or simulation-only tasks.
---

# Physical Robot Safety & Speed Rules

When commanding or scripting any physical robot arm, gripper, or motor bus:

1.  **Speed Limits**
    *   Check the robot's bring-up or control skill for its calibrated default
        and ceiling joint speeds.
    *   For **any unknown or uncalibrated robot**, default to **`<= 0.50
        rad/s`** and cap at **`<= 1.00 rad/s`** unless the user or task
        explicitly requires faster or slower motion.
    *   Ramp joint trajectories smoothly with zero endpoint velocity (e.g.,
        cosine interpolation); never command raw large single-step jumps.

2.  **Park Before Disabling Torque (`try / except / finally` +
    `SIGINT`/`SIGTERM`)**
    *   Never disable motor torque, exit a script, or call `robot.close()` while
        the arm is elevated—always return smoothly to a safe resting/park pose
        inside `finally:` **and** `SIGINT`/`SIGTERM` handlers first.
    *   If the arm is holding an object or inside a bin/near obstacles when an
        error or abort occurs, retreat vertically or clear the obstacle first
        (releasing if safe) before parking.
    *   **Guard-proof homing**: inside `finally:`, read the *actual* current
        joint positions and interpolate smoothly back to the park pose in
        bounded steps (`<= 0.50 rad/s`) rather than sending a single jump that
        a velocity or step guard could reject.
    *   Never leave an elevated arm in `zero_gravity_mode` / pure
        gravity-compensation mode where it can sag or drop.

3.  **Keep the Motor Heartbeat Alive (Prevent Watchdog / `LOST_COMM` Drops)**
    *   Fast local computation (`< 50 ms` OpenCV, NumPy, or IK) in the control
        process is fine.
    *   Avoid **multi-second GIL or blocking stalls** (software MP4 encoding,
        slow VLM/network requests, or `input()` waits) inside the motor-owning
        process while the arm is elevated—run heavy encoding or VLM inference
        in a separate subprocess or *after* parking.
    *   Never exit a process while the arm is elevated expecting a subsequent
        script to take over; keep a single long-lived owning process per motor
        bus/serial port, and never open two processes on the same CAN/serial
        interface simultaneously.

4.  **Actuator Thermal Protection**
    *   Holding a grasp during normal pick-and-place transit (seconds) is
        expected and required; avoid leaving a gripper or motor hard-clamped
        against a rigid stop during long idle waits (tens of seconds or
        longer), as a single motor thermal fault can kill the entire bus thread
        and drop all joints.
    *   If the user explicitly asks to hold the arm elevated or a gripper
        clamped for an extended period, warn them about the thermal/watchdog
        drop risk and confirm before proceeding.
