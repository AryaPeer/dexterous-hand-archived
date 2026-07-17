from dataclasses import dataclass, field


@dataclass
class DomainRandomization:
    enabled: bool = True
    mass_range: tuple[float, float] = (0.7, 1.3)
    friction_range: tuple[float, float] = (0.7, 1.3)
    actuator_gain_range: tuple[float, float] = (0.85, 1.15)


@dataclass
class SceneConfig:
    mount_x: float = -0.10
    mount_y: float = 0.0
    mount_height: float = 0.78
    table_height: float = 0.4
    table_half_size: float = 0.25
    object_mass: float = 0.1
    object_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    action_smoothing_alpha: float = 0.2
    # 0.005 x 8 substeps = 25 Hz control, the MuJoCo Playground recipe for
    # MJX hand tasks (Panda sim dt 0.005, LEAP 0.01, all with implicitfast —
    # set in the builders). Every substep pays the full Newton solve on GPU,
    # so halving the substep count is a direct ~2.5x throughput win over
    # 0.002 x 20. The cube's form-closure grip is stable at this dt; the peg
    # task is NOT (see PegSceneConfig.sim_timestep) and keeps 0.002. Re-run
    # the CPU test suite, check_reward_gradient, the renders, and
    # scripts/mjx_parity_check.py after any change here.
    sim_timestep: float = 0.005
    frame_skip: int = 8
    # Newton solver iteration caps, compiled into the scene by the builder.
    # MuJoCo's defaults (100/50) are sized for CPU, where Newton early-exits
    # on tolerance; MJX pays the configured worst case per substep, and
    # Playground-class MJX hand tasks run 4-8. Re-run the CPU test suite and
    # scripts/mjx_parity_check.py after any change here.
    solver_iterations: int = 8
    ls_iterations: int = 8
    # MJX contact culling — Playground-style custom numerics read by
    # mujoco-mjx's collision driver ("max_geom_pairs"/"max_contact_points").
    # None = off; CPU MuJoCo ignores the numerics either way. Set on the pod
    # AFTER measuring max ncon over the parity trajectories + a random-policy
    # rollout (~2x the observed max) and re-running mjx_parity_check with the
    # MJX backend: culling too low silently drops real contacts, which the
    # parity bars catch as grip failure.
    mjx_max_geom_pairs: int | None = None
    mjx_max_contact_points: int | None = None


@dataclass
class RewardWeights:
    # Proportions: reaching is a hint, grasping is meaningful, lifting
    # dominates, holding pays comparably to lifting so the policy holds at
    # height instead of oscillating at the lift cap.
    reaching: float = 0.5
    grasping: float = 2.5
    lifting: float = 6.0
    holding: float = 6.0
    drop: float = 1.0
    action_penalty: float = 1.0
    success: float = 1.0
    idle: float = 1.0


@dataclass
class RewardConfig:
    weights: RewardWeights = field(default_factory=RewardWeights)
    reach_tanh_k: float = 5.0
    # 0.10 = a real, visible pick-up (the Apr-10 value). This eroded to 0.012
    # over rounds 11-13 because the scene had no vertical arm DOF and finger
    # curl capped physical lift at ~1cm — the bar was lowered to match a broken
    # scene instead of fixing the scene. With slide_z in the grasp scene
    # (scene_builder.py) a 10cm lift is mechanically direct, so the task is
    # again "pick the cube up", not "twitch it 12mm".
    lift_target: float = 0.10
    hold_velocity_threshold: float = 0.05
    # Sharpness of the holding height-gate around lift_target. With
    # lift_target=0.10, k=50 keeps the gate ~0 at zero lift (sigmoid(-5) =
    # 0.7% — no grasp-and-sit subsidy) while giving a smooth gradient over the
    # last few cm of the lift. (k=200 was needed when lift_target was 0.012;
    # at 0.10 it would make the gate a step function.)
    hold_height_smoothness_k: float = 50.0
    # Sharpness of the holding velocity-gate. At the 0.05 m/s threshold, k=100
    # pays a perfectly still cube sigma(100*0.05) = 99.3% of the gate; k=20
    # capped it at 73%, silently under-paying the exact behavior the term
    # exists to reward.
    hold_velocity_smoothness_k: float = 100.0
    # Per-fingertip weights for the reaching distance, in fingertip-site order
    # [ff, mf, rf, lf, th]: emphasize the THUMB (index 4) — thumb opposition
    # is the binding constraint of the grip, so the reach shaping pulls it in
    # hardest.
    fingertip_weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 2.5)
    drop_penalty: float = -20.0
    # Per-step payment while the success condition holds (an annuity, like the
    # peg's `complete`): dropping the cube strictly loses income, so there is
    # no bonus-farming cycle to patch, and there is no one-shot spike for
    # VecNormalize's reward clip to attenuate. Adroit relocate pays its
    # proximity bonuses per-step the same way.
    success_bonus_per_step: float = 5.0
    # 25 steps = 1s of continuous at-height hold before the annuity starts.
    success_hold_steps: int = 25
    # Height at which the drop penalty arms. Below this the cube was never
    # meaningfully carried; above it, releasing (lift < 0.01) costs
    # drop_penalty. Must sit above spawn jitter but well below lift_target,
    # otherwise a cube carried to 9cm and dumped is penalty-free.
    drop_arm_height: float = 0.04
    # Quadratic action-magnitude penalty coefficient (applied to sum(a^2)).
    action_penalty_scale: float = 2e-4
    no_contact_idle_penalty: float = -0.08
    idle_grace_steps: int = 3


@dataclass
class PegRewardWeights:
    reach: float = 0.2
    grasp: float = 2.0
    # Reduced 15->10 so the insertion stack (depth weight 3, max ~30/step) is not
    # out-rewarded by lifting. At 15 the lift term peaked at +37.5/step — more
    # than the entire align+depth+insertion stack combined — so the policy was
    # paid most to "lift high and hold" rather than to insert (round-13/16
    # symptom). With lift=10 and the proportional term capped at 1.0 (see
    # peg_reward.py) lift maxes at +20/step, below depth's +30. NOTE: the precise
    # lift-vs-depth balance is the key remaining tuning knob — validate any
    # further change with a CPU pre-flight (check_reward_gradient.py) + a short
    # sanity run before a full run.
    lift: float = 10.0
    opposition: float = 1.0
    align: float = 2.0
    depth: float = 3.0
    complete: float = 1.0
    force: float = 1.0
    drop: float = 1.0
    action_penalty: float = 1.0
    idle_stage0: float = 1.0
    idle_stage1: float = 1.0
    # Round-16: rewards abs(dot(peg_axis, hole_axis)) * contact_scale at every
    # step while in contact, *before* lift. The last peg run (5M steps)
    # converged to peg held 85deg off vertical because no reward term
    # incentivized vertical grip until align_weight fired (peg lifted >2cm),
    # by which time the policy had already settled on a sideways grip.
    axis_in_grip: float = 1.0
    # 2026-07-14: keypoint shaping toward the RELEASE pose (peg vertical, tip
    # release_height above the entrance) — fills the post-containment-fix
    # gradient dead zone between lift saturation (5cm) and in-bore depth.
    # See the place term in peg_reward.py for why the target is the release
    # pose and not the inserted pose (wall-press local minimum). 8.0 makes the
    # transport gradient (lifted-at-spawn -> hover-at-release: place 0.40 ->
    # 1.0) ~+5/step against a ~27/step gripped-lifted baseline — the only x/y
    # gradient after grasp, so it must be visible over grip-noise.
    place: float = 8.0


@dataclass
class PegRewardConfig:
    weights: PegRewardWeights = field(default_factory=PegRewardWeights)
    drop_penalty: float = -20.0
    complete_bonus: float = 250.0
    depth_reward_scale: float = 10.0
    force_threshold: float = 15.0
    idle_stage0_penalty: float = -0.3
    # Lift reward = step bonus (1.0 if lift_height > lift_step_threshold) +
    # proportional term (min(lift_height/lift_target, 1.5) * contact_scale).
    # The step bonus breaks the round-11 grasp-and-sit local minimum by
    # giving PPO an immediate non-zero gradient at the moment of lift-off.
    lift_target: float = 0.05
    lift_step_threshold: float = 0.005
    # Mirror of idle_stage0_penalty but for "grasp without lift" — fires
    # when nfc >= idle_stage1_min_contacts AND lift_height < lift_step_threshold
    # AND stage == 1. Cap at idle_grace_steps to avoid double-jeopardy.
    idle_stage1_penalty: float = -0.1
    idle_stage1_min_contacts: int = 2
    lateral_gate_k: float = 5.0
    idle_stage_cutoff: int = 3
    idle_grace_steps: int = 3
    success_threshold: float = 0.7
    peg_hold_steps: int = 10
    reach_tanh_k: float = 5.0
    # Per-fingertip weights for the reaching distance, in fingertip-site order
    # [ff, mf, rf, lf, th]: emphasize the THUMB (index 4), matching the
    # thumb-opposition grasp terms.
    fingertip_weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 2.5)
    # Quadratic action-magnitude penalty coefficient (applied to sum(a^2)).
    action_penalty_scale: float = 2e-4
    # place-term shape: target tip height RELATIVE to the hole entrance
    # (negative = inside the bore), and the tanh sharpness on the summed
    # 2-keypoint distance. -0.015 targets the ENGAGED release pose: a peg
    # released with its tip above the entrance topples (measured: 7.6cm
    # capsule, 4mm-clearance bore -> ~6 deg self-alignment cone, every
    # scripted above-entrance release fell flat across the tube top), while
    # a tip 1-2cm inside the bore is laterally guided and slides down —
    # IndustReal's (Tang'23) engagement distinction. With hand<->wall
    # collision on, the finger proximal links rest on the tube rim and cap
    # the gripped tip depth at ~-0.011; the -0.015 target is deliberately
    # just past that stop so the shaping keeps pulling the grip fully onto
    # the rim (deeper engagement is what makes the release robust — shallow
    # releases get kicked out by the opening fingers; measured).
    release_height: float = -0.015
    place_k: float = 4.0


@dataclass
class PegSceneConfig:
    mount_x: float = -0.10
    mount_y: float = 0.0
    # Round-16: reverted 0.78 -> 0.82. The round-15 grasp-driven mount lower
    # cost peg an extra 2.8mm of slide_z descent before finger-table contact
    # saturated the actuator (audit: peg_tip stopped at 0.4068m@0.82 vs
    # 0.4096m@0.78). Grasp's SceneConfig keeps 0.78.
    mount_height: float = 0.82
    table_height: float = 0.4
    table_half_size: float = 0.25
    clearance: float = 0.004
    hole_depth: float = 0.06
    # Lift the hole body so its entrance sits this far above the table top,
    # forming a guide tube. The hand's middle/ring KNUCKLES bottom out on the
    # table and cap slide_z descent, leaving the peg tip ~8mm above the table
    # top — but with the entrance elevated the hand never needs to reach table
    # level: it releases the peg over the bore and gravity finishes insertion.
    #
    # True in-tube geometry (CPU mujoco drop test, guarded by
    # tests/test_geometry.py::test_peg_drop_insertion_reaches_success_depth):
    # a peg settled on the hole_bottom plate measures insertion fraction
    # 0.757 (depth 0.0575 = hole_depth - plate half-thickness), a +0.057
    # margin over success_threshold=0.70 (~4.3mm). The ceiling is set by
    # hole_depth, NOT by this elevation; do NOT set success_threshold above
    # the in-tube ceiling (test guards this).
    #
    # NOTE: an earlier comment here claimed "achievable 0.94 / +0.24 margin" —
    # that was an artifact of the pre-2026-06-10 insertion metric, which had
    # no lateral containment and measured open-air descent NEXT to the tube
    # (see get_insertion_depth_jax). Depth now only counts inside the bore.
    # Matches robosuite NutAssembly / TwoArmPegInHole convention of placing
    # the receptacle above the workspace surface.
    hole_top_above_table: float = 0.08
    hole_offset: tuple[float, float] = (0.0, 0.0)
    spawn_min_radius: float = 0.04
    spawn_max_radius: float = 0.05 * 1.4142135623730951
    peg_radius: float = 0.008
    peg_half_length: float = 0.03
    peg_mass: float = 0.02
    peg_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    action_smoothing_alpha: float = 0.2
    # 0.002 x 20 = 25 Hz control. The peg task CANNOT run the grasp task's
    # dt=0.005: the precision pinch on the smooth capsule loses orientation
    # at dt >= 0.004 (measured: peg tilt ratchets 20deg -> 55deg during
    # transport, independent of integrator, contact solref, and how gently
    # the hand moves), and the toppled peg never enters the bore. The 7cm
    # cube's form-closure grip has no such sensitivity, so SceneConfig keeps
    # dt=0.005. Re-run the peg proofs + renders after any change here.
    sim_timestep: float = 0.002
    frame_skip: int = 20
    # Newton solver iteration caps — see SceneConfig.solver_iterations.
    solver_iterations: int = 8
    ls_iterations: int = 8
    # MJX contact culling — see SceneConfig.mjx_max_geom_pairs.
    mjx_max_geom_pairs: int | None = None
    mjx_max_contact_points: int | None = None


@dataclass
class MjxGraspTrainConfig:
    num_envs: int = 768
    total_timesteps: int = 70_000_000
    # Milestone compute-saver. When True the run stops early if the task
    # metrics regress/collapse or a progress metric goes flat at a milestone
    # (10M grip-health, 50M lift-emergence for grasp) — see the gate list in
    # scripts/training/train_grasp.py and MilestoneGateCallback in _common.py.
    # A ~500k checkpoint always exists, so a stop is resumable. CLI: --no-gate.
    gate_enabled: bool = True
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # Small positive entropy bonus to actively maintain exploration. With
    # ent_coef=0 the clamped log_std could only decay toward its floor
    # (sigma->0.05), monotonically losing exploration with no recovery path —
    # a known cause of premature convergence to bad grips. 1e-3 keeps a gentle
    # counter-pressure; the log_std clamp (<=0 -> sigma<=1.0) still prevents
    # runaway. Reference: MuJoCo Playground LEAP reorient uses 1e-2.
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    # Round-13: enabled. With it off, value_loss grew 130× (469→61k) over
    # 43M steps as reward magnitudes climbed, value function diverged, and
    # task metrics regressed from peak at 28.9M.
    norm_reward: bool = True
    obs_noise_std: float = 0.005
    max_episode_steps: int = 200
    # log_std clamp on the Gaussian policy. Default: σ ∈ [0.05, 1.0].
    # Init at the ceiling so PG can only push σ down — see
    # dexterous_hand/policies/clamped_actor.py.
    log_std_init: float = 0.0
    log_std_min: float = -3.0
    log_std_max: float = 0.0
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    dr: DomainRandomization = field(default_factory=DomainRandomization)


def _mjx_peg_reward_config() -> PegRewardConfig:
    return PegRewardConfig()


@dataclass
class MjxPegTrainConfig:
    num_envs: int = 768
    total_timesteps: int = 150_000_000
    # Milestone compute-saver. When True the run stops early if the task
    # metrics regress/collapse or a progress metric goes flat at a milestone
    # (10M and 30M for peg) — see the gate list in scripts/training/train_peg.py
    # and MilestoneGateCallback in _common.py. The 2026-06-01 5M sanity bars
    # were invalidated by the insertion-metric containment fix (the old metric
    # scored never-inserted pegs as inserted); current floors are
    # first-principles collapse bars — re-derive from the first post-fix
    # sanity. A ~500k checkpoint always exists, so a stop is resumable.
    # CLI: --no-gate.
    gate_enabled: bool = True
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 10
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # Small positive entropy bonus to actively maintain exploration. With
    # ent_coef=0 the clamped log_std could only decay toward its floor
    # (sigma->0.05), monotonically losing exploration with no recovery path —
    # a known cause of premature convergence to bad grips. 1e-3 keeps a gentle
    # counter-pressure; the log_std clamp (<=0 -> sigma<=1.0) still prevents
    # runaway. Reference: MuJoCo Playground LEAP reorient uses 1e-2.
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    # Round-13: enabled. Same reasoning as grasp config — round-12 peg
    # regressed from peg_height +147mm at 13M back toward grasp-and-sit by
    # 48M with value_loss climbing similarly.
    norm_reward: bool = True
    obs_noise_std: float = 0.005
    max_episode_steps: int = 500
    log_std_init: float = 0.0
    log_std_min: float = -3.0
    log_std_max: float = 0.0
    scene_config: PegSceneConfig = field(default_factory=PegSceneConfig)
    reward_config: PegRewardConfig = field(default_factory=_mjx_peg_reward_config)
    # DR is deliberately OFF for peg: insertion with tight clearance is already
    # hard, and dynamics randomization slows convergence on a task that hasn't
    # yet reached a baseline success. obs_noise_std=0.005 is kept. Re-enable the
    # grasp-style mass/friction/gain ranges (ideally + contact stiffness, a la
    # MuJoCo Playground) once a baseline insertion policy exists and transfer is
    # a goal. NOTE: contrast with MjxGraspTrainConfig, which has DR enabled.
    dr: DomainRandomization = field(default_factory=lambda: DomainRandomization(enabled=False))
    curriculum_reference_timesteps: int = 100_000_000
    curriculum_stages: list[tuple[int, float, float]] = field(
        default_factory=lambda: [
            (0, 0.004, 1.0),
            (8_000_000, 0.004, 0.7),
            (16_000_000, 0.003, 0.5),
            (24_000_000, 0.002, 0.3),
            (32_000_000, 0.001, 0.2),
        ]
    )
