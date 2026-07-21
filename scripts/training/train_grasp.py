
import argparse

from dexterous_hand.config import MjxGraspTrainConfig
from dexterous_hand.curriculum.callbacks import (
    GraspCurriculumCallback,
    scale_stage_starts,
)
from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv
from scripts.training._common import run_training

GRASP_GATES = [
    (
        10_000_000,
        [
            ("metrics/num_finger_contacts", 2.5, 3.42, "grip forms and stays formed"),
            ("reward/grasping", 0.60, 0.845, "grasp reward maintained"),
        ],
        "grasp 10M: grip health",
    ),
    (
        30_000_000,
        [
            ("reward/lifting", 0.15, 0.028,
             "lifting is being learned, not just spawned by the curriculum "
             "(the frozen-exploration 70M run plateaued at 0.093 by 30M)"),
            ("metrics/success_hold_steps", 0.01, 0.001,
             "the cube is held at target for consecutive steps; the frozen run read "
             "exactly 0.0 in all 713 rollouts, so any sustained value is real progress"),
        ],
        "grasp 30M: lift emergence",
    ),
]


def train(config: MjxGraspTrainConfig) -> None:
    curriculum_stages = scale_stage_starts(
        stages=config.curriculum_stages,
        total_timesteps=config.total_timesteps,
        reference_total_timesteps=config.curriculum_reference_timesteps,
    )
    run_training(
        config=config,
        env_cls=ShadowHandGraspMjxEnv,
        run_prefix="grasp_mjx",
        wandb_name=f"grasp-mjx-{config.num_envs}env",
        gates=GRASP_GATES,
        extra_callbacks=[GraspCurriculumCallback(stages=curriculum_stages, verbose=1)],
        extra_wandb_config={"effective_curriculum_stages": curriculum_stages},
    )


def parse_args() -> MjxGraspTrainConfig:
    defaults = MjxGraspTrainConfig()
    parser = argparse.ArgumentParser(description="Train Shadow Hand grasping (MJX + SBX PPO)")
    parser.add_argument("--num-envs", type=int, default=defaults.num_envs)
    parser.add_argument("--total-timesteps", type=int, default=defaults.total_timesteps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--n-steps-per-env", type=int, default=defaults.n_steps_per_env)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable the 10M/30M milestone compute-saver gate (let the run go to the end).",
    )
    args = parser.parse_args()

    return MjxGraspTrainConfig(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        gate_enabled=not args.no_gate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps_per_env=args.n_steps_per_env,
        seed=args.seed,
    )


if __name__ == "__main__":
    config = parse_args()
    train(config)
