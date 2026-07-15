
import argparse
from dataclasses import asdict
from pathlib import Path

import flax.linen as nn
from sbx import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from dexterous_hand.config import MjxGraspTrainConfig
from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv
from dexterous_hand.policies.clamped_actor import make_clamped_actor
from scripts.training._common import (
    MilestoneGateCallback,
    RewardInfoLoggerCallback,
    setup_sb3_logger,
)

# Compute-saver gates.
#
# NOTE (2026-07-14): the pre-slide_z baselines (nfc 4.92, grasping 0.985,
# object_height 0.4349 flat at 5M, ~11mm lift by 40M) came from the scene
# WITHOUT a vertical arm DOF, where lift was only possible via finger curl.
# With slide_z + lift_target restored to 0.10, lift has a direct actuator
# gradient and should emerge far earlier — but until a fresh 5M sanity exists
# these floors are first-principles collapse bars only (baseline column NaN).
# Re-derive real floors from the first post-slide_z sanity.
# info_key, floor, baseline, why:
GRASP_GATES = [
    (
        10_000_000,
        [
            ("metrics/num_finger_contacts", 2.5, float("nan"), "grip forms and stays formed"),
            ("reward/grasping", 0.60, float("nan"), "grasp reward maintained"),
        ],
        "grasp 10M: grip health",
    ),
    (
        30_000_000,
        [
            ("metrics/object_height", 0.445, float("nan"),
             "lift emerged (mean >= ~1cm over the window; flat 0.435 = never lifts "
             "despite the slide_z gradient — the run's bet has failed)"),
        ],
        "grasp 30M: lift emergence",
    ),
]


def train(config: MjxGraspTrainConfig) -> None:

    run_dir = Path("runs") / f"grasp_mjx_{config.num_envs}env_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rollout_size = config.num_envs * config.n_steps_per_env
    if config.batch_size > rollout_size:
        new_bs = max(rollout_size // 4, 64)
        print(
            f"WARNING: batch_size {config.batch_size} > rollout_size {rollout_size}. "
            f"Auto-resized to {new_bs}."
        )
        config.batch_size = new_bs

    wandb.init(
        project="dexterous-hand",
        name=f"grasp-mjx-{config.num_envs}env",
        config=asdict(config),
    )

    vec_env = ShadowHandGraspMjxEnv.from_config(config)
    vec_env = VecMonitor(vec_env)

    if config.norm_obs or config.norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=10.0,
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
        # Round-14: raised from 0.02 to 0.05. Round-13 grasp 50M run
        # plateaued at success_hold_steps≈5.6 with adaptive LR collapsed
        # to 5e-5 because target_kl=0.02 was being hit constantly under
        # normalized rewards. 0.05 keeps LR near 3e-4 so learning
        # continues past the early plateau.
        target_kl=0.05,
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
    if config.gate_enabled:
        callbacks.insert(0, MilestoneGateCallback(GRASP_GATES, verbose=1))

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

def parse_args() -> MjxGraspTrainConfig:
    parser = argparse.ArgumentParser(description="Train Shadow Hand grasping (MJX + SBX PPO)")
    parser.add_argument("--num-envs", type=int, default=768)
    parser.add_argument("--total-timesteps", type=int, default=70_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-steps-per-env", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable the 10M/50M milestone compute-saver gate (let the run go to the end).",
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
