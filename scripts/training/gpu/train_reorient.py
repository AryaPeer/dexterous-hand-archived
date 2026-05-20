
import argparse
from dataclasses import asdict
from pathlib import Path

import flax.linen as nn
from sbx import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from dexterous_hand.config import MjxReorientTrainConfig
from dexterous_hand.curriculum.callbacks import (
    ReorientCurriculumCallback,
    scale_stage_starts,
)
from dexterous_hand.envs.gpu.reorient_env import ShadowHandReorientMjxEnv
from dexterous_hand.policies.clamped_actor import make_clamped_actor
from scripts.training._common import RewardInfoLoggerCallback, setup_sb3_logger


def train(config: MjxReorientTrainConfig) -> None:

    run_dir = Path("runs") / f"reorient_mjx_{config.num_envs}env_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    curriculum_stages = scale_stage_starts(
        stages=config.curriculum_stages,
        total_timesteps=config.total_timesteps,
        reference_total_timesteps=config.curriculum_reference_timesteps,
    )

    rollout_size = config.num_envs * config.n_steps_per_env
    if config.batch_size > rollout_size:
        new_bs = max(rollout_size // 4, 64)
        print(
            f"WARNING: batch_size {config.batch_size} > rollout_size {rollout_size}. "
            f"Auto-resized to {new_bs}."
        )
        config.batch_size = new_bs

    wandb_config = asdict(config)
    wandb_config["effective_curriculum_stages"] = curriculum_stages
    wandb.init(
        project="dexterous-hand",
        name=f"reorient-mjx-{config.num_envs}env",
        config=wandb_config,
    )

    vec_env = ShadowHandReorientMjxEnv.from_config(config)
    vec_env = VecMonitor(vec_env)

    if config.norm_obs or config.norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=10.0,
        )

    curriculum_callback = ReorientCurriculumCallback(
        stages=curriculum_stages,
        verbose=1,
    )


    activation_fn = {"elu": nn.elu, "relu": nn.relu, "tanh": nn.tanh}[config.activation]

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps_per_env,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=0.02,
        policy_kwargs={
            "net_arch": dict(pi=config.net_arch.copy(), vf=config.net_arch.copy()),
            "activation_fn": activation_fn,
            "log_std_init": config.log_std_init,
            "actor_class": make_clamped_actor(
                log_std_min=config.log_std_min,
                log_std_max=config.log_std_max,
            ),
        },
        verbose=1,
        seed=config.seed,
    )

    setup_sb3_logger(model, run_dir)

    callbacks = [
        curriculum_callback,
        RewardInfoLoggerCallback(),
        CheckpointCallback(
            save_freq=max(500_000 // config.num_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            save_vecnormalize=True,
        ),
        WandbCallback(
            model_save_path=str(run_dir),
            model_save_freq=max(100_000 // config.num_envs, 1),
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(str(run_dir / "final_model"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"Saved to {run_dir}")
    wandb.finish()
    vec_env.close()

def parse_args() -> MjxReorientTrainConfig:
    parser = argparse.ArgumentParser(description="Train Shadow Hand reorientation (MJX + SBX PPO)")
    parser.add_argument("--num-envs", type=int, default=768)
    parser.add_argument("--total-timesteps", type=int, default=500_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-steps-per-env", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--curriculum-reference-timesteps",
        type=int,
        default=None,
        help="Reference total for curriculum scaling. Default keeps config value (500M). "
        "Set to a small value like 30_000_000 with a 3M sanity to pin stage 0 (30°) for the full run.",
    )
    args = parser.parse_args()

    kwargs = dict(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps_per_env=args.n_steps_per_env,
        seed=args.seed,
    )
    if args.curriculum_reference_timesteps is not None:
        kwargs["curriculum_reference_timesteps"] = args.curriculum_reference_timesteps

    return MjxReorientTrainConfig(**kwargs)

if __name__ == "__main__":
    config = parse_args()
    train(config)
