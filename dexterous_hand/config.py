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
    sim_timestep: float = 0.002
    frame_skip: int = 20


@dataclass
class RewardWeights:
    reaching: float = 1.0
    grasping: float = 1.0
    lifting: float = 12.0
    holding: float = 10.0
    drop: float = 1.0
    action_penalty: float = 1.0
    success: float = 1.0
    idle: float = 1.0
    opposition: float = 1.0


@dataclass
class RewardConfig:
    weights: RewardWeights = field(default_factory=RewardWeights)
    reach_tanh_k: float = 5.0
    lift_target: float = 0.012
    hold_velocity_threshold: float = 0.05
    # Sharpness of the holding height-gate around lift_target. Raised 50->200 so
    # the gate is ~0 below lift_target (no reward for holding an UNLIFTED cube)
    # and ~1 once lifted — see the holding term in grasp_reward.py.
    hold_height_smoothness_k: float = 200.0
    hold_velocity_smoothness_k: float = 20.0
    fingertip_weights: tuple[float, float, float, float, float] = (2.5, 1.0, 1.0, 1.0, 1.0)
    drop_penalty: float = -20.0
    success_bonus: float = 250.0
    success_hold_steps: int = 20
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
    insertion_drive: float = 3.0
    # Round-16: rewards abs(dot(peg_axis, hole_axis)) * contact_scale at every
    # step while in contact, *before* lift. The last peg run (5M steps)
    # converged to peg held 85deg off vertical because no reward term
    # incentivized vertical grip until align_weight fired (peg lifted >2cm),
    # by which time the policy had already settled on a sideways grip.
    axis_in_grip: float = 1.0


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
    fingertip_weights: tuple[float, float, float, float, float] = (2.5, 1.0, 1.0, 1.0, 1.0)


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
    # table (penetrating ~1mm) and cap slide_z descent at ~-0.0814 regardless
    # of actuator force, leaving the peg tip ~7-10mm above the table top. The
    # success criterion needs the tip ~53mm below the hole entrance, so the
    # required descent is set by the ENTRANCE elevation, not the actuator.
    #
    # Measured by CPU mujoco grip-descend; the invariant is now guarded by
    # tests/test_geometry.py::test_peg_insertion_physically_reachable, which
    # asserts achievable insertion >= success_threshold + 0.05 at this elevation:
    #   hole_top=0.06 -> achievable insertion fraction 0.68 (BELOW 0.70 -> task
    #                    was geometrically unwinnable; matches the round-16 FAIL)
    #   hole_top=0.07 -> 0.81
    #   hole_top=0.08 -> 0.94  (+0.24 margin over success_threshold=0.70)
    # hole_depth does not affect the ceiling here (the knuckle cap binds before
    # the hole floor). 0.08 gives a healthy margin; do NOT set success_threshold
    # above the measured achievable fraction at the chosen elevation.
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
    sim_timestep: float = 0.002
    frame_skip: int = 20


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
    # and MilestoneGateCallback in _common.py. Bars are derived from the
    # 2026-06-01 5M sanity, NOT the older doc bars. A ~500k checkpoint always
    # exists, so a stop is resumable. CLI: --no-gate.
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
