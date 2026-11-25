# Task: Bring-Up Cute Wave (`scripts/wave.py`)

Expressive free-space greeting wave and bring-up smoke test for I2RT YAM and YAM
Pro arms (`numpy` + `i2rt`).

## Usage

```bash
# Offline MuJoCo dry run (no CAN hardware):
python3 skills/i2rt-yam-bringup/scripts/wave.py --sim

# Hardware execution (Linux socketcan or macOS USB-CAN via macos_can.py):
python3 skills/i2rt-yam-bringup/scripts/wave.py --channel can0 --cycles 3.5
```

Optional flags: `--arm-type yam|yam_pro|yam_ultra|yam_ultra_2`,
`--gripper-type linear_4310|linear_3507|crank_4310|flexible_4310|no_gripper`,
and macOS `--interface` / `--backend-channel`.

## Choreography & Safety

| Stage | `[J1, J2, J3, J4, J5, J6]` (rad) | Gripper |
| :--- | :--- | :--- |
| `Q_HOME` | `[-0.02, 0.01, 0.01, -0.07, -0.02, 0.00]` | `1.00` |
| `Q_WAVE_READY` | `[0.00, 0.38, 0.62, 0.24, 0.00, 0.00]` | `0.68-0.72` |
| `Q_NOD_DOWN` / `UP` | `[0.00, 0.40, 0.65, 0.13, 0.00, 0.16]` / `[0.00, 0.37, 0.61, 0.28, 0.00, -0.14]` | `0.28` / `0.82` |
| `Q_WINK` | `[0.04, 0.37, 0.60, 0.29, 0.16, 0.28]` | `0.25` |

*   **Overlapping-Action Wave**: After a cheerful double-nod & pinch at compact
    `Q_WAVE_READY` (`grasp_site ~ [0.22, 0.00, 0.43] m`), streams a `100 Hz`
    `sin^2(pi * u)`-enveloped `3.5`-cycle wave combining waist sway (`J1`),
    forearm bounce (`J2/J3/J4`), wrist wave (`J5`), phase-lagged wrist roll
    follow-through (`J6`), and finger flutter, followed by a closing wink/tilt.
*   **Safety**: Clamps resting `J2 = 0, J3 = 0` stop jitter (rejecting states
    outside `+/- 0.08 rad` of YAM limits), keeps `>= 0.20 rad` limit margin
    throughout the elevated wave, holds on `> 0.45 rad` tracking error, and
    smoothly cosine-ramps back to `Q_HOME` before `robot.close()`.
