---
name: mujoco-playground-rl
description: >-
  Authors, inspects in Studio, audits, trains, monitors in TensorBoard, and
  deploys GPU-accelerated locomotion, dexterous, and vision RL tasks on MuJoCo
  Playground and Menagerie robots with PyTorch/Warp reaf-sim flat_scene and
  RSL-RL. Use when creating a MuJoCo Playground RL environment, tuning visual
  domain randomization, diagnosing RSL-RL TensorBoard curves, or deploying RL
  checkpoints. Don't use for non-RL controllers (use mujoco-playground) or
  non-MuJoCo RL.
metadata:
  publisher: google
---

# MuJoCo Playground RL Task Authoring & Training ([`rl_task_check.py`](scripts/rl_task_check.py))

Author, optionally inspect in MuJoCo Studio, audit, GPU-train (`Impl.WARP` +
RSL-RL), diagnose in TensorBoard, and deploy locomotion, dexterous in-hand, or
vision manipulation RL tasks in `mujoco_playground`.

## 1. Fast Discovery, Public APIs, & Zero-Churn Workflow

- **Workspace & Accelerator Setup**:
  - Clone and sync (`make sync` on CUDA 13, `make sync-cu12` on Linux CUDA 12,
    or `make sync-cpu` on macOS/CPU):

    ```bash
    git clone https://github.com/google-deepmind/mujoco_playground.git && cd mujoco_playground && make sync
    ```
  - Invoke `.venv/bin/python` / `.venv/bin/playground` (or `uv run --no-sync`)
    directly so `uv` never re-syncs over your chosen accelerator group. On
    headless Linux NVIDIA GPUs, set `MUJOCO_GL=egl` (omit on macOS), and pass
    `--wp_kernel_cache_dir=""` to `playground train rsl_rl` so
    `rl_task_check.py` and training share Warp's default user cache (compiling
    kernels once).
- **One-Pass Survey (No Piecemeal Introspection Churn)**:
  - Check `mujoco_playground/robots/` (`g1`, `so101`, `trossen_wxai`, `yam`) and
    one reference task in `mujoco_playground/tasks/`
    (`manipulation/pick_cube/{task,mdp,reset}.py` or `whole_body/tracking/`),
    reusing existing robot scenes/constants instead of duplicating
    `robots/<robot>/`.
  - Load Menagerie MJCFs via
    `asset.get("github:google-deepmind/mujoco_menagerie/<robot>/<file>.xml").spec()`.
    When binding an unfamiliar Menagerie robot, inspect all bodies, joints,
    ranges, actuators (`gainprm`/`biasprm`/`forcerange`), sites, cameras,
    keyframes, and active collision geoms in **one** Python command, and rely on
    Section 3's `reaf_sim` reference rather than reading
    `third_party/reaf_sim/` internals or running `inspect.getsource`.
- **Public Imports & AST Audit Rules in `tasks/` & `robots/`**:
  - `rl_task_check.py` rejects private `._src` imports, `__all__`, and **any**
    module-scope `.compile()` call (including `MjSpec.compile()` or
    `re.compile()`) inside `tasks/` and `robots/`. Import public facades only:

    ```python
    from mujoco_playground import asset, rl, wrapper
    from reaf.simulation import torch as reaf_sim
    from reaf.simulation.torch import (
        action_history, environment, feature_producer, flat_scene,
        math as reaf_math, model_randomization, mujoco_types, noise,
        processors, resetter, tensors, termination_checker,
    )
    ```

  - Build ground planes, boxes, and mocap targets directly on `mujoco.MjSpec`
    (`spec.worldbody.add_body(...)`) instead of
    `mujoco_playground._src.objects`.
- **Single-Command Audit (Skip Redundant Test/Formatter Churn)**:
  - [`rl_task_check.py`](scripts/rl_task_check.py) (`--action=all`) validates
    AST sync/import rules, CPU `Impl.CPP` reset/step, CUDA `Impl.WARP`
    `torch.cuda.set_sync_debug_mode("error")` reset/step, finite
    observations/rewards, and `flat_scene.make_deployment_config`. Do not write
    redundant `task_test.py` suites or run formatters unless asked.
  - **MuJoCo Studio Inspection**: Load **`mujoco-studio-sidecar`** +
    **`mujoco-studio`** when interactive visual inspection in MuJoCo Studio
    (`sidecar://robotics/mujoco_studio/`) is requested; for headless CLI/batch
    training requests, go directly from `rl_task_check.py` to `rsl_rl.train`.

## 2. High-Throughput GPU `MjSpec`, Collisions, & `warn_overflow = 0`

- **Solver Caps & Mandatory `warn_overflow = 0` (`10x`–`40x` Faster Training)**:
  - Configure solver & buffer caps on `flat_scene.Sim` (applied *after* all
    `MjSpec.attach` calls, avoiding parent `100`/`50` overrides; `nconmax` and
    `njmax` are **per-world** capacities, never multiplied by `nworld`):
    - **Locomotion & dexterous hands (`50 Hz`)**: `simulation_timestep=0.005`,
      `control_timestep=0.02` (`decimation=4`), `iterations=6`,
      `ls_iterations=10`, `nconmax=32`–`48`, `njmax=128`–`192`,
      `integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST`,
      `cone=mujoco.mjtCone.mjCONE_PYRAMIDAL`,
      `solver=mujoco.mjtSolver.mjSOL_NEWTON`.
    - **Tabletop arm manipulation (`20 Hz`)**: `simulation_timestep=0.005`,
      `control_timestep=0.05` (`decimation=10`, `2.5x` fewer substeps/iter than
      `0.1s`/`0.004s`), `iterations=8`, `ls_iterations=10`,
      `ccd_iterations=35`, `nconmax=32`, `njmax=128`.
  - **CRITICAL — Silence `mujoco_warp` `wp.printf` Overflow Floods**:
    `mjw.put_model` defaults `warp_model.opt.warn_overflow` to
    `OverflowType.ALL`. With `iterations=6`–`8`, random RL exploration hits the
    iteration cap across thousands of GPU threads and prints millions of
    `"solver/linesearch iterations limit reached"` lines via CUDA `wp.printf`,
    stalling training from `100k+` FPS to `1.4k` FPS! Always disable
    `warn_overflow` inside `EnvironmentReset.on_startup(self)` (which executes
    *before* `Simulator` captures its CUDA step graph, stripping `wp.printf` at
    Warp compile time):

    ```python
    sim = self._env.device_layer.sim
    if sim is not None and sim.impl == reaf_sim.Impl.WARP:
        sim.warp_model.opt.warn_overflow = 0
    ```

    *(If using only `DomainRandomizationResetter` without a custom resetter, add
    a `"startup"` term `lambda model, spec, dev: setattr(model.obj.opt,
    "warn_overflow", 0) if hasattr(model.obj.opt, "warn_overflow") else None`.)*
- **Primitive Colliders & Self-Collision Bitmasks (`4x` Faster MJWarp + Instant Compile)**:
  - Name every unnamed geom (`f"{body.name}_geom_{i}"`) before attaching, and
    ensure floating bases have a named freejoint (`"root"`).
  - Menagerie hands and grippers often have dozens of `mjGEOM_MESH` /
    `mjGEOM_CYLINDER` colliders and internal rail/housing boxes that trigger
    multi-minute GJK `ccd_kernel` compilations, self-collision `njmax`
    overflows, and `4x` slower stepping:
    1. Convert active (`contype != 0 or conaffinity != 0`) `mjGEOM_MESH` and
       `mjGEOM_CYLINDER` colliders to local AABB `mjGEOM_BOX` primitives using a
       one-time `m0 = spec.compile()` (`center, half = m0.geom_aabb[gid, :3],
       np.maximum(m0.geom_aabb[gid, 3:], 1e-3)`, shifting `g.pos` by
       `R_geom @ center`, setting `g.type = mujoco.mjtGeom.mjGEOM_BOX`,
       `g.size = half`, `g.meshname = ""`; merge multi-piece palm hull splits
       into a single union AABB box on the palm).
    2. Disable collisions (`contype = conaffinity = 0`) on non-contacting
       internal gripper carriages, rails, and camera housings, keeping only
       finger pads, wrist/forearm capsules, manipuland, and table/terrain
       active.
    3. Separate robot vs. environment collision bitmasks (`robot: contype=1,
       conaffinity=2`; `object/table/terrain: contype=2, conaffinity=3`) when
       robot self-collision sensing is not needed.
- **Actuator Damping Sign (`biasprm[2] < 0`!) & Palm-Up Hand Mounting**:
  - In MuJoCo `mjBIAS_AFFINE`, `force = gainprm[0] * ctrl + biasprm[0] +
    biasprm[1] * qpos + biasprm[2] * qvel`, so **`biasprm[2]` MUST be strictly
    negative (`-kd`)**! Preserve Menagerie servos only when `biasprm[2] < 0`.
    Whenever an asset has pure torque `<motor>`s (`biastype == mjBIAS_NONE`),
    undamped servos (`biasprm[2] == 0`), **or positive anti-damping
    `biasprm[2] > 0`** (a common Menagerie hand bug where `+kd * qvel` injects
    energy and causes violent bang-bang finger chatter!), set negative velocity
    damping (`act.biasprm[2] = -0.05 * kp` and `j.damping[0] = max(float(
    j.damping[0]), 0.02)` on finger joints, or replace torque motors on legs
    with `kp=25.0, kd=0.5`, `j.armature[0] = 0.01`, `j.damping[0] = 0.1`,
    `j.frictionloss[0] = 0.0` — index `[0]` on `MjsJoint` 3-vector fields!) by
    mutating attributes on the returned `MjsActuator`
    (`spec.add_actuator(name=j, target=j)` raises `TypeError` on kwargs):

    ```python
    act = spec.add_actuator(name=j, target=j)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.dyntype = mujoco.mjtDyn.mjDYN_NONE
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm[0] = kp
    act.biasprm[1:3] = [-kp, -kd]
    act.ctrllimited = False
    act.forcelimited = True
    act.forcerange[:] = [-effort, effort]
    ```

  - **Dexterous Hand Palm-Up Orientation**: Standalone Menagerie hands often
    mount upright (`+z` along fingers) at `qpos=0`, dropping objects
    immediately. Check fingertip motion from open to flexed `qpos`
    (`mj_forward`) to find the palmar curl direction, mount the hand root
    **palm-up (`+z`)** at `z ≈ 0.25 m` in a mid-flexed cupped `home_qpos` with
    the thumb opposed over the palm center, and spawn a high-friction (`1.2`)
    `4.5–5.0 cm` object `3–4 cm` above the palm so all fingers can reach and
    roll it.

## 3. Exact `flat_scene` & `reaf_sim` Reference (Zero Introspection Needed)

- **Top-Level `flat_scene.Environment` Structure**:
  - Fields: `base_scene: mujoco.MjSpec`, `device_layer: flat_scene.DeviceLayer`,
    `task_layer: flat_scene.TaskLayer`, `observation_adapter: dict[str,
    flat_scene.ObsGroup]`, `resetters: dict[str, flat_scene.Component]`, `sim:
    flat_scene.Sim`, `episode_length_secs: float = 20.0`, `nan_termination:
    bool = True`.
  - `flat_scene.Component(Cls, **runtime_kwargs)` auto-injects `name` (dict
    key), `nworld`, and `xpu_device` for `devices`, `command_processors`,
    `command_feature_producers`, `feature_producers`, and
    `termination_checkers`, **except `resetters`**, which instantiates
    `Cls(**runtime_kwargs)` directly and injects `self._env`, `self._nworld`,
    and `self._xpu_device` via `set_env(env)`.
  - Every `flat_scene.Obs` in `ObsGroup(terms={...}, enable_noise=...)` needs
    `.configure(output_shape=(D,))` (string shortcut `Obs("feat/key")` takes no
    kwargs before `.configure(...)`).
  - `task_layer.terminations` `Term` functions take `(features, **params)` and
    return `torch.where(cond, termination_checker.TerminationResult.TERMINATE,
    termination_checker.TerminationResult.DO_NOT_TERMINATE)`; sim-only
    `resetter.TerminationResetter(terms=[reaf_sim.Term(fn, ...)])` takes
    `(device_layer, **params)` and returns boolean `(nworld,)`.
- **Core `DeviceLayer` & `TaskLayer` Components (Exact Keys & Shapes)**:
  - `reaf_sim.ActuatorDevice(targets=list(JOINTS), measurements=["joint_pos",
    "joint_vel"], home_ctrl=list(HOME))` -> consumes `"{key}/ctrl": (B, nu)`,
    emits `"{key}/joint_pos": (B, nu)`, `"{key}/joint_vel": (B, nu)` (resets
    `data.ctrl` only; write `data.qpos`/`data.qvel` in `on_reset` using
    `tensors.masked_fill(target, env_mask, val)`).
  - `reaf_sim.ArticulatedDevice(joint_names=list(JOINTS), home_qpos=list(HOME),
    measurements=["joint_pos", "joint_vel", "home_qpos", "joint_range"])` ->
    emits `"{key}/joint_pos": (B, nq)`, `"{key}/joint_range": (B, nq, 2)`.
  - `reaf_sim.FreeDevice(body_name=..., joint_name=..., measurements=["xpos",
    "xquat", "joint_vel"], reference_site_name=None)` -> emits `"{key}/xpos":
    (B, 3)`, `"{key}/xquat": (B, 4)` `[w,x,y,z]`, `"{key}/joint_vel": (B, 6)`
    (`[:, :3]` **world** lin vel -> rotate via
    `reaf_math.quat_rotate_inverse(quat, vel[:, :3])`; `[:, 3:6]` **body** ang
    vel); setters: `set_pos(env_mask, pos)`, `set_quat(env_mask, quat)`,
    `set_vel(env_mask, vel)`.
  - `reaf_sim.MocapDevice(body_name=...)` -> emits `"{key}/mocap_pos": (B, 3)`,
    `"{key}/mocap_quat": (B, 4)`; setters: `set_pos`, `set_quat`.
    `reaf_sim.SiteDevice(site_name=...)` -> `"{key}/xpos"`, `"{key}/xmat"`.
  - `flat_scene.Component(reaf_sim.ContactSensorDevice, fields=("found",
    "force"), history_length=decimation).configure_spec(objtype="geom"|"body"|
    "xbody", objname=..., reftype=..., refname=..., reduce="maxforce", num=1)`
    -> emits `"{key}/{f}": (B, n_sensors, dim_f)` (`found`: 1, `force`: 3) and
    `"{key}/{f}_history": (B, n_sensors, history_length, dim_f)`.
  - `action_history.ActionHistory(key="action", feature_shape=(nu,),
    history_length=1)` (in `task_layer.command_feature_producers`) -> consumes
    `"action"`, emits `"action": (B, nu)` and `"action_history": (B, 1, nu)`.
    Pair with `processors.TransformActions(input_key="action",
    output_key="robot/ctrl", scale=..., offset=..., consume=True)` (**pass
    `consume=True`** when `input_key != output_key` so `"action"` is consumed!).
  - `feature_producer.FeatureProducer` subclass methods: `name` property,
    `produced_features_spec() -> Mapping[str, specs.Array]`,
    `required_features_keys() -> set[str]`, pure `produce_features(features) ->
    Mapping[str, Tensor]`, plus stateful `on_step(features)`,
    `on_begin_stepping(features, env_mask=None)`, and `reset(env_mask=None)`.
    **Execution-order rule**: `TaskLayer` calls `on_step(features)` **before**
    `produce_features(features)` and `compute_reward()` on every step! Therefore
    in any progress/goal `FeatureProducer`, compute and cache `self._rot_delta =
    self._prev_err - cur_err` and `self._goal_reached = (cur_err < thr).float()`
    inside `on_step()` *before* resampling `self._goal_quat` on reached worlds
    and updating `self._prev_err`, then return those cached tensors from
    `produce_features()` (otherwise `prev_err - cur_err` is identically `0`!).
    Mark critic/reward-only producers with `.mark_privileged()`.
  - **Vision Cameras (`32x32` Raytracing + `Impl.CPP` Smoke Support)**:
    Render at `cam_resolution=[(32, 32)]` (`~85k` FPS on A100, `3x` faster than
    `64x64` without extra downsampling) with `use_precomputed_rays=False` and
    **1** `castshadow=True` key light (set fill lights `castshadow=False`).
    Because `env_registry.register()`'s `build()` pops `impl` from `kwargs`
    before calling `env_cfg_fn()`, forward `impl` in `env_registry.py` when
    `"impl" in inspect.signature(env_cfg_fn).parameters` (defaulting to `"warp"
    if device_str.startswith("cuda") else "cpp"`) and branch in
    `env_cfg_fn(impl="warp")`:

    ```python
    if impl == "cpp":
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        cfg.device_layer.devices["camera"] = flat_scene.Component(
            reaf_sim.MujocoCameraDevice, camera_name=cam, width=32, height=32,
            render_rate_hz=rate_hz, scene_option=opt,
            image_output_name_prefix=cam, rgb_output_name_suffix="_rgb")
    else:
        cfg.device_layer.devices["camera"] = flat_scene.Component(
            reaf_sim.CameraDevice, cam_name=[cam], cam_resolution=[(32, 32)],
            render_rate_hz=rate_hz, render_rgb=True, render_depth=False,
            enabled_geom_groups=(0, 2), use_precomputed_rays=False, use_shadows=True)
    ```

## 4. Zero-Sync CUDA Rules & High-Score Domain RL Formulations

- **Zero-Sync CUDA Rules (Enforced in Both `step()` and `reset()` / DR)**:
  1. Never call `.item()`, `.cpu()`, `.numpy()`, `.tolist()`, `.nonzero()`,
     `.new_tensor()`, `bool()`, or `.any()`/`.all()` without `dim=`.
  2. **No boolean mask indexing on CUDA tensors** (`x[env_mask] = ...` or
     `model.jnt_qposadr[ltd]` triggers `nonzero()` D2H sync!). Precompute index
     tensors in `on_startup()` and write masked updates via
     `tensors.masked_fill(dst, env_mask, val)` or `torch.where`.
  3. **No `torch.tensor([...], device=cuda)` or `torch.as_tensor(val,
     device=cuda)` inside hot/reset functions** (host literal upload syncs
     CUDA!). Note: `noise.sample_so3_uniform` calls `torch.as_tensor(low,
     device=device)` internally and strips `shape[-1]`; sample uniform SO(3)
     unit quaternions zero-sync via `q = torch.randn((n, 4), device=device);
     q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)`.
- **Domain Formulation Recipes (`FunctionalRewardProvider` scales all rewards by `control_dt`)**:
  - **Locomotion (Quadrupeds & Bipedal Humanoids)**:
    - Actions: `TransformActions(input_key="action", output_key="robot/ctrl",
      scale=SCALE, offset=DEFAULT_POS, consume=True)` (`0.25` for quadrupeds;
      per-joint `(0.25 * effort / kp).clamp(0.15, 0.45)` around bent-knee pose
      on bipeds).
    - Asymmetric `ObsGroup`s: `"policy"` (`enable_noise=True`) gets body
      `ang_vel`, `projected_gravity`, `command` `(vx, vy, wz)`, `joint_pos -
      default_pos`, `joint_vel`, `action`; `"critic"` (`enable_noise=False`)
      adds clean body `lin_vel` and foot contacts.
    - Rewards (avoid the standing-frozen local optimum!): Make velocity tracking
      dominant — `track_lin_vel_xy = torch.exp(-err_xy_sq / 0.25)`
      (`weight=3.0`) and `track_ang_vel_z = torch.exp(-err_wz_sq / 0.25)`
      (`weight=1.5`), keep `alive` modest (`0.2` on bipeds, never `>= 1.0`), and
      make `feet_air_time` **strictly non-negative** (`torch.clamp(last_air_time
      - 0.15, min=0.0, max=0.35) * first_contact / control_dt`, `weight=0.75` —
      a negative `min` punishes early baby steps and freezes the robot!). Add
      upright `||g_proj_xy||^2` (`-2.5`), `v_z^2` (`-2.0`), `||w_xy||^2`
      (`-0.05`), `||a_t - a_{t-1}||^2` (`-0.015`), `||qvel||^2` (`-5e-4`),
      non-sagittal posture (`hip_roll`, `hip_yaw`, waist, arms, `-0.25`),
      stand-still (`-0.5`), and undesired contacts (`-1.0`).
  - **Dexterous In-Hand SO(3) Reorientation**:
    - Actions & exploration: Fix any `biasprm[2] >= 0` anti-damping first! Use
      wide joint authority `TransformActions(input_key="action",
      output_key="robot/ctrl", scale=0.65, offset=CUPPED_HOME,
      clip=(Q_LOW, Q_HIGH), consume=True)` (so fingers sweep their full range to
      roll the object) at `50 Hz`, `episode_length_secs=10.0`, paired with
      **`init_std=0.5` and `entropy_coef=0.002`** in `RslRlOnPolicyRunnerCfg`.
    - Observations: `reaf_math.quat_box_minus(goal_quat, obj_quat)` (3D tangent
      rotation error), `obj_pos - palm_pos`, fingertip site positions relative
      to `obj_pos`, `joint_pos - home_qpos`, `joint_vel`, and `action`; critic
      adds `obj_lin_vel` and `obj_ang_vel`.
    - Active rotation + finger engagement rewards: Combine (1) **fingertip
      engagement** `mean_i exp(-||tip_i - obj_pos|| / 0.05)` (`weight=1.0`, so
      all fingers immediately contact the object), (2) **rotation progress**
      `(prev_rot_err - rot_err) / control_dt` (`weight=3.0`, where `rot_err =
      reaf_math.quat_error_magnitude(obj_quat, goal_quat)`), (3) dense alignment
      `1.0 / (rot_err + 0.1)` (`weight=1.5`), (4) palm centering
      `exp(-||obj_pos - palm_center||^2 / 0.05^2)` (`weight=1.0`), (5)
      **goal-reached impulse** `(rot_err < 0.4).float() / control_dt`
      (`weight=10.0` — divide by `control_dt` so `* control_dt` preserves the
      full `+10.0` bonus!), resampling a new Haar-uniform quaternion on
      goal-reached worlds in `on_step()` and tracking `consecutive_successes`,
      plus (6) drop termination (`||obj_pos - palm_center|| > 0.10` or `obj_z <
      palm_z - 0.04`).
  - **Vision & Sim-to-Real Tabletop Manipulation**:
    - Smooth delta servo control (`20 Hz`, `control_timestep=0.05`): wire
      `processors.DeltaActions(output_key="robot/ctrl", scale=DELTA_SCALE,
      offset=HOVER_OPEN_POSE, action_clip=(-1.0, 1.0), smoothing=0.5,
      clip=(JOINT_MIN, JOINT_MAX))` and `processors.StochasticSubstepCommands(
      keys="robot/ctrl", delay=(0.0, 0.005))`. Center-crop `(B, C, H, W)` images
      before any resize.
    - Calibration `bias` & visual DR: Add per-episode `bias=noise.Noise(
      dist=noise.Uniform(low=..., high=...), op="add")` on `"policy"`
      `joint_pos` (`±0.01` rad) and `ee_pos` (`±0.0025` m). Dim headlight
      (`diffuse=[0.1, 0.1, 0.1]`), add multi-directional room lights targeting
      `workspace_center`, strip textures (`override_flat_materials`), randomize
      robot `geom_rgba` and high-contrast HSV object vs. floor/table `geom_rgba`
      (`trigger="reset"`; squeeze `rgb` when `bound.geom_rgba.ndim == 2`), and
      randomize `light_pos`, `light_dir`, `light_diffuse`, `cam_pos` (`±5` mm),
      `cam_quat` (`4` deg), and `cam_fovy` (`0.9`–`1.1x`).
    - Grounded 4-stage rewards: `reaching` (`exp(-d^2 / 0.06^2)`, `weight=1.0`)
      -> proximity-gated `grasp` (`exp(-d^2 / 0.03^2) * ((open_pos - q_grip) /
      (open_pos - closed_pos)).clamp(0.0, 1.0)`, `weight=1.5`) -> `lift` &
      `bringing` (`weight=3.0` each; **set `table_z = surface_z +
      cube_half_size`**, e.g. `0.75 + 0.02 = 0.77` on elevated tables vs `0.018`
      on `z=0` floors, and `lift_height_threshold = table_z + 0.03`) ->
      `success` (`weight=2.0`), plus `floor_contact` (`-1.0`), `action_rate`
      (`-0.02`), and `joint_pos_limits`.

## 5. Compiled RSL-RL Training, TensorBoard Diagnostics, & Safe Deploy

- **Runner Registration (`_src/rl/rsl_rl/env_rl_config.py` & `_src/rl/env_registry.py`)**:
  - Set `torch_compile_mode: str | None = "default"` on `RslRlOnPolicyRunnerCfg`
    subclasses (`~2x` faster PPO updates on CUDA; auto-disabled on CPU) and
    `save_interval: int = 50`. Never set `clip_actions` (`RslRlVecEnvWrapper`
    raises `ValueError("clip_actions is not supported.")`).
  - State tasks: `nworld=4096`, `num_steps_per_env=24`, `hidden_dims=(512, 256,
    128)`, `obs_normalization=True`, `num_learning_epochs=5`,
    `num_mini_batches=4`, `init_std=1.0` (`0.8` on 29-DOF bipeds; `0.5` with
    `entropy_coef=0.002` on `20+` DOF hands).
  - Vision tasks: `obs_groups={"actor": ["policy", "camera"], "critic":
    ["critic"]}`, `nworld=4096` (`2048` if GPU is shared),
    `num_steps_per_env=16`, `num_learning_epochs=4`, `actor =
    RslRlCNNModelCfg(hidden_dims=(256, 256, 128), obs_normalization=True,
    cnn_cfg={"camera": RslRlCNNCfg(output_channels=(16, 32), kernel_size=(5, 3),
    stride=(2, 2), activation="elu", max_pool=False)})`, and asymmetric state
    `critic = RslRlMLPModelCfg(hidden_dims=(512, 256, 128),
    obs_normalization=True)`. Register an **uncalled** `env_cfg_fn` in
    `_src/rl/env_registry.py`.
- **Audit, Pre-Train Sanity Rollout, & One-Shot GPU Training**:

  ```bash
  # On macOS, omit MUJOCO_GL=egl.
  MUJOCO_GL=egl .venv/bin/python <skill_dir>/scripts/rl_task_check.py --action=all --repo_root=. --task_dir=mujoco_playground/tasks/<category>/<task> --env_id=<EnvId>
  MUJOCO_GL=egl .venv/bin/playground train rsl_rl --env=<EnvId> --wp_kernel_cache_dir="" --logdir=<logdir> --overrides='{"max_iterations": 350, "save_interval": 50}'
  .venv/bin/tensorboard --logdir=<logdir> --port=6006
  ```

  - **Pre-train 5-step sanity check -> ONE training run**: Before launching
    `rsl_rl.train`, step `nworld=4` for 5 random actions in a 2-second snippet
    to confirm stateful progress rewards are non-zero. Then run **one** GPU
    training job (`300`–`400` iterations, matching the requested minimum plus a
    small margin) using `NotificationTimeoutSeconds` instead of polling loops.
  - **Do NOT run speculative 2nd/3rd retraining loops** once `<logdir>` has
    completed the requested iterations with finite, rising `Train/mean_reward`
    (on `20+` DOF dexterous hands, `300`–`400` iterations establishes stable
    cradling, rising reward/episode length, and initial `consecutive_successes >
    0`, whereas full `rot_err < 0.5` rad takes `2,000+` iterations; launching
    late re-trains risks hitting evaluation wall-clock timeouts!). Never
    `rm -rf` a completed `<logdir>`.
  - **TensorBoard Scalar Diagnostics**:
    - `Train/mean_reward`, `reward_episode/<term>`, & `Metrics/<metric>`:
      Climbing `reaching` with flat `lift` -> miscalibrated `table_z` or missing
      proximity-gated grasp; flat `rot_err ≈ 2.0` rad on hands -> `biasprm[2] >=
      0` anti-damping, small action `scale`, or missing fingertip engagement;
      frozen `lin_vel_xy_error` -> negative `feet_air_time` clamp or high
      `alive`.
    - `Loss/entropy`, `Loss/value`, `Perf/total_fps`, &
      `Train/mean_episode_length`: Low `Perf/total_fps` with huge stdout logs
      indicates missing `warp_model.opt.warn_overflow = 0`.
- **Fast-Fail Checkpoint Deployment (`deploy.py` with `absl.flags`)**:
  - Use `absl.app` + `absl.flags` (`--checkpoint`, `--mode`, `--num_steps`) and
    validate `--checkpoint` (`epath.Path(ckpt).expanduser()`, verifying the
    `.pt` file or a `model_*.pt` inside a directory exists) **at the top of
    `main()` before lazy `mujoco`/`warp`/`rsl_rl` imports or `create_env()`**,
    so path typos fail in `<0.2s` while physical servos stay idle.
  - Build single-world inference (`nworld=1`) via
    `flat_scene.make_deployment_config(cfg, observation_groups=actor_groups)`
    (`flat_scene.build_env(deploy_cfg)` in sim, or
    `wrapper.TorchDeviceLayerWrapper` with masked `SIGINT` return-to-rest on
    hardware), setting `runner_cfg.nworld = 1` and `runner_cfg.obs_groups =
    {"actor": list(actor_groups), "critic": [actor_groups[0]]}` so checkpoint
    loading never requires stripped privileged critic observations.
