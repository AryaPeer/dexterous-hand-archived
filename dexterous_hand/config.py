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
    hold_height_smoothness_k: float = 50.0
    hold_velocity_smoothness_k: float = 20.0
    fingertip_weights: tuple[float, float, float, float, float] = (2.5, 1.0, 1.0, 1.0, 1.0)
    drop_penalty: float = -20.0
    success_bonus: float = 250.0
    success_hold_steps: int = 20
    no_contact_idle_penalty: float = -0.08
    idle_grace_steps: int = 3


@dataclass
class TrainConfig:
    n_envs: int = 256
    total_timesteps: int = 30_000_000
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class PegRewardWeights:
    reach: float = 0.2
    grasp: float = 2.0
    lift: float = 15.0
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
    mount_height: float = 0.78
    table_height: float = 0.4
    table_half_size: float = 0.25
    clearance: float = 0.004
    hole_depth: float = 0.06
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
class PegTrainConfig:
    n_envs: int = 32
    total_timesteps: int = 40_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 8
    ent_coef: str = "auto"
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    scene_config: PegSceneConfig = field(default_factory=PegSceneConfig)
    reward_config: PegRewardConfig = field(default_factory=PegRewardConfig)
    curriculum_reference_timesteps: int = 40_000_000
    # (timestep, clearance, p_pre_grasped) tuples for set_curriculum_params
    curriculum_stages: list[tuple[int, float, float]] = field(
        default_factory=lambda: [
            (0, 0.004, 1.0),
            (8_000_000, 0.004, 0.7),
            (16_000_000, 0.003, 0.5),
            (24_000_000, 0.002, 0.3),
            (32_000_000, 0.001, 0.2),
        ]
    )


@dataclass
class MjxGraspTrainConfig:
    num_envs: int = 768
    total_timesteps: int = 70_000_000
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
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
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 10
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
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
