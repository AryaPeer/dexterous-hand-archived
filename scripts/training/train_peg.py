
import argparse

from dexterous_hand.config import MjxPegTrainConfig
from dexterous_hand.curriculum.callbacks import (
    AssemblyCurriculumCallback,
    scale_stage_starts,
)
from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv
from scripts.training._common import run_training

PEG_GATES = [
    (
        10_000_000,
        [
            ("metrics/num_finger_contacts", 1.5, 2.60,
             "peg held, not parked (the pre-fix 5M park-farm read 0.19)"),
            ("metrics/axis_align", 0.70, 0.957,
             "peg held vertical; the frozen-exploration run collapsed to 0.10 here "
             "and this is the gate that correctly caught it"),
            ("metrics/stage", 1.2, 0.946, "task progressed past grasp-and-sit"),
            ("metrics/peg_height", 0.435, 0.4366,
             "peg upright, not knocked over (resting height is 0.438; the frozen "
             "run read 0.410 lying on its side)"),
        ],
        "peg 10M: vertical lifted grip (real insertion not expected yet)",
    ),
    (
        30_000_000,
        [
            ("metrics/axis_align", 0.80, 0.90, "vertical grip held"),
            ("metrics/insertion_depth", 0.001, 0.0,
             "in-bore insertion happening at all (exact 0 = never inserts)"),
            ("metrics/insertion_hold_steps", 0.05, 0.0,
             "some sustained in-bore holds occurring (exact 0 = never holds depth)"),
        ],
        "peg 30M: insertion exists",
    ),
]


def train(config: MjxPegTrainConfig) -> None:
    curriculum_stages = scale_stage_starts(
        stages=config.curriculum_stages,
        total_timesteps=config.total_timesteps,
        reference_total_timesteps=config.curriculum_reference_timesteps,
    )
    run_training(
        config=config,
        env_cls=ShadowHandPegMjxEnv,
        run_prefix="peg_mjx",
        wandb_name=f"peg-mjx-{config.num_envs}env",
        gates=PEG_GATES,
        extra_callbacks=[AssemblyCurriculumCallback(stages=curriculum_stages, verbose=1)],
        extra_wandb_config={"effective_curriculum_stages": curriculum_stages},
    )


def parse_args() -> MjxPegTrainConfig:
    defaults = MjxPegTrainConfig()
    parser = argparse.ArgumentParser(description="Train Shadow Hand peg-in-hole (MJX + SBX PPO)")
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

    return MjxPegTrainConfig(
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
