# Adversarial Sim-to-Real Critic Rubric

Verify every item below before promoting a `BETTER` candidate or running `--mode=real`. Any failure blocks promotion.

## 1. Anti-Cheat & Observation Hygiene

- **Deployment-Key Parity**: Controller interacts solely through `proxy.RemoteDeviceLayer` or `wrappers.AllowlistWrapper`, reading only `task.deployment_keys()` and `<cam>_timestamp` with zero privileged simulator reads (`gauntlet/poses`, `gauntlet/forces`, `.unwrapped`, or `._src` imports).
- **Zero Nominal Pose Memorization**: Object and support poses are localized from live camera frames or contact feedback rather than nominal MJCF coordinates (`T1` and `T2` safe-pass rates do not collapse relative to `T0`).
- **Closed-Loop Self-Model Tolerance (`T4`)**: Tracking and grasp placement tolerate camera extrinsic jitter, joint bias, and gain/mass/gravcomp shifts without assuming exact nominal FK or actuator gains.

## 2. Physical Safety & Passive Settle

- **Force & Speed Envelopes**: Zero `joint_limit`, `joint_speed`, `grasp_force` (`35 N` fragile cap), `contact_force`, `impact_force`, or `self_contact` violations across all tiers (`SafetyLimits`, including indirect crush through held objects).
- **Controlled Placement & Bystander Clearance**: Zero `dropped` or `bystander_moved` violations; release happens near the support surface without sweeping neighboring bodies.
- **Passive Post-Episode Settle**: After controller completion, zero `still_held` (when `require_release=True`) or `unsettled` violations over the passive settle window (`2 mm` tail peak-to-peak position bound).

## 3. Recovery (`T3`) & Terminal Fault Aborts (`T4`)

- **Disturbance Recovery (`T3`)**: Under grasp `slip`, mid-approach `push`, and post-place `knockout`, the controller detects displaced state and re-grasps or re-places safely without disturbing bystanders.
- **Post-Onset Fault Diagnosis (`T4`)**: Under `camera_freeze` (frozen or restamped timestamps, requiring a post-onset camera read `last_cam_read_tick > onset_tick`) and `dead_gripper`, the controller detects the fault from post-onset observations and returns `{"status": "abort", "fault": "<fault>"}` (`aborted_ok == True`, zero `fault_unhandled` or `spurious_abort`, high `fault_abort_precision`).

## 4. Paired CRN & Runtime Discipline

- **Valid Paired CRN**: Baseline and candidate match on `(task_id, tier, idx, seed)` and `scene_sha256` under a persistent `GAUNTLET_CRN_SECRET`, achieving `verdict == "BETTER"` with `vetoes == 0`.
- **Real-Time Pacing**: Controller think time stays within `control_dt_s` budget under `LatencyWrapper` without `controller_timeout`, `controller_crash`, or `controller_deadline` violations.
