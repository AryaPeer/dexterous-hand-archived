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
    object_half_extent: float = 0.025
    object_mass: float = 0.1
    object_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    action_smoothing_alpha: float = 0.2
    sim_timestep: float = 0.005
    frame_skip: int = 8
    solver_iterations: int = 8
    ls_iterations: int = 8
    mjx_max_geom_pairs: int | None = 256
    mjx_max_contact_points: int | None = 96


@dataclass
class RewardWeights:
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
    lift_target: float = 0.10
    hold_velocity_threshold: float = 0.05
    hold_height_smoothness_k: float = 50.0
    hold_velocity_smoothness_k: float = 100.0
    fingertip_weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 2.5)
    drop_penalty: float = -20.0
    success_bonus_per_step: float = 5.0
    success_hold_steps: int = 25
    drop_arm_height: float = 0.04
    action_penalty_scale: float = 2e-4
    no_contact_idle_penalty: float = -0.08
    idle_grace_steps: int = 3


@dataclass
class PegRewardWeights:
    reach: float = 0.2
    grasp: float = 2.0
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
    axis_in_grip: float = 1.0
    place: float = 8.0


@dataclass
class PegRewardConfig:
    weights: PegRewardWeights = field(default_factory=PegRewardWeights)
    drop_penalty: float = -20.0
    complete_bonus: float = 250.0
    depth_reward_scale: float = 10.0
    force_threshold: float = 15.0
    idle_stage0_penalty: float = -0.3
    lift_target: float = 0.05
    lift_step_threshold: float = 0.005
    idle_stage1_penalty: float = -0.1
    idle_stage1_min_contacts: int = 2
    lateral_gate_k: float = 5.0
    idle_stage_cutoff: int = 3
    idle_grace_steps: int = 3
    success_threshold: float = 0.7
    peg_hold_steps: int = 10
    reach_tanh_k: float = 5.0
    fingertip_weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 2.5)
    action_penalty_scale: float = 2e-4
    release_height: float = -0.015
    place_k: float = 4.0


@dataclass
class PegSceneConfig:
    mount_x: float = -0.10
    mount_y: float = 0.0
    mount_height: float = 0.82
    table_height: float = 0.4
    table_half_size: float = 0.25
    clearance: float = 0.004
    hole_depth: float = 0.06
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
    solver_iterations: int = 8
    ls_iterations: int = 8
    mjx_max_geom_pairs: int | None = 384
    mjx_max_contact_points: int | None = 48


@dataclass
class MjxGraspTrainConfig:
    num_envs: int = 768
    total_timesteps: int = 70_000_000
    gate_enabled: bool = True
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float = 0.05
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    obs_noise_std: float = 0.005
    max_episode_steps: int = 200
    log_std_init: float = -1.0
    log_std_min: float = -3.0
    log_std_max: float = 0.0
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    dr: DomainRandomization = field(default_factory=DomainRandomization)
    curriculum_reference_timesteps: int = 70_000_000
    curriculum_schedule_timesteps: int = 0
    curriculum_stages: list[tuple[int, float]] = field(
        default_factory=lambda: [
            (0, 0.0),
        ]
    )


def _mjx_peg_reward_config() -> PegRewardConfig:
    return PegRewardConfig()


@dataclass
class MjxPegTrainConfig:
    num_envs: int = 768
    total_timesteps: int = 150_000_000
    gate_enabled: bool = True
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float = 0.05
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    obs_noise_std: float = 0.005
    max_episode_steps: int = 500
    log_std_init: float = -1.0
    log_std_min: float = -3.0
    log_std_max: float = 0.0
    scene_config: PegSceneConfig = field(default_factory=PegSceneConfig)
    reward_config: PegRewardConfig = field(default_factory=_mjx_peg_reward_config)
    dr: DomainRandomization = field(default_factory=lambda: DomainRandomization(enabled=False))
    curriculum_reference_timesteps: int = 100_000_000
    curriculum_schedule_timesteps: int = 0
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
class PickPlaceSceneConfig:
    mount_x: float = -0.10
    mount_y: float = 0.0
    mount_height: float = 0.78
    table_height: float = 0.4
    table_half_size: float = 0.25
    object_half_extent: float = 0.025
    object_mass: float = 0.1
    object_friction: tuple[float, float, float] = (1.0, 0.005, 0.001)
    action_smoothing_alpha: float = 0.2
    sim_timestep: float = 0.005
    frame_skip: int = 8
    solver_iterations: int = 8
    ls_iterations: int = 8
    mjx_max_geom_pairs: int | None = 256
    mjx_max_contact_points: int | None = 96
    spawn_x_range: tuple[float, float] = (0.05, 0.10)
    spawn_y_range: tuple[float, float] = (-0.03, 0.03)
    goal_nominal_xy: tuple[float, float] = (0.075, 0.09)
    goal_x_range: float = 0.02
    goal_y_range: float = 0.025
    goal_marker_radius: float = 0.04


@dataclass
class PickPlaceRewardWeights:
    reaching: float = 0.5
    grasping: float = 2.5
    lifting: float = 4.0
    transport: float = 4.0
    placed: float = 14.0
    success: float = 1.0
    drop: float = 1.0
    action_penalty: float = 1.0
    idle: float = 1.0


@dataclass
class PickPlaceRewardConfig:
    weights: PickPlaceRewardWeights = field(default_factory=PickPlaceRewardWeights)
    reach_tanh_k: float = 5.0
    lift_target: float = 0.06
    carry_clear_height: float = 0.03
    transport_tanh_k: float = 5.0
    goal_tanh_k: float = 15.0
    goal_radius: float = 0.04
    on_table_tol: float = 0.02
    on_table_k: float = 100.0
    hold_velocity_threshold: float = 0.05
    at_rest_k: float = 100.0
    success_bonus_per_step: float = 5.0
    place_hold_steps: int = 25
    drop_penalty: float = -20.0
    drop_arm_height: float = 0.04
    action_penalty_scale: float = 2e-4
    no_contact_idle_penalty: float = -0.08
    idle_grace_steps: int = 3
    fingertip_weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 2.5)


@dataclass
class MjxPickPlaceTrainConfig:
    num_envs: int = 768
    total_timesteps: int = 70_000_000
    gate_enabled: bool = True
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps_per_env: int = 128
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float = 0.05
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    seed: int = 42
    norm_obs: bool = True
    norm_reward: bool = True
    obs_noise_std: float = 0.005
    max_episode_steps: int = 250
    log_std_init: float = -1.0
    log_std_min: float = -3.0
    log_std_max: float = 0.0
    scene_config: PickPlaceSceneConfig = field(default_factory=PickPlaceSceneConfig)
    reward_config: PickPlaceRewardConfig = field(default_factory=PickPlaceRewardConfig)
    dr: DomainRandomization = field(default_factory=DomainRandomization)
