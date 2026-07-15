
import argparse
from dataclasses import asdict
from pathlib import Path

import flax.linen as nn
from sbx import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from dexterous_hand.config import MjxPegTrainConfig
from dexterous_hand.curriculum.callbacks import (
    AssemblyCurriculumCallback,
    scale_stage_starts,
)
from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv
from dexterous_hand.policies.clamped_actor import make_clamped_actor
from scripts.training._common import (
    MilestoneGateCallback,
    RewardInfoLoggerCallback,
    setup_sb3_logger,
)

# Compute-saver gates.
#
# NOTE (2026-06-10): the 2026-06-01 5M sanity "baselines" (insertion_depth
# 0.060, complete 3.24, insertion_hold_steps 1.83, stage 2.65) were artifacts
# of the lateral-blind insertion metric — any peg at table level scored
# fraction 1.0, so those numbers measured the drop-the-peg exploit, not
# insertion. After the containment fix in get_insertion_depth_jax, honest mean
# insertion depth is expected to be ~0 well past 10M. The floors below are
# therefore first-principles bars, not sanity-derived: at 10M we check grip
# pose + progress (metrics whose semantics the fix did not change), at 30M we
# check that real in-bore insertion EXISTS at all (a mean of exactly 0 over
# the ~1.5M-step window means the policy never inserts — the run's bet has
# failed). Re-derive proper floors from the first post-fix 5M sanity; the
# baseline column is NaN until then. info_key, floor, baseline, why:
PEG_GATES = [
    (
        10_000_000,
        [
            ("metrics/axis_align", 0.70, float("nan"),
             "peg held vertical (round-16 collapsed to 0.07)"),
            ("metrics/stage", 1.5, float("nan"), "task progressed past grasp-and-sit"),
            ("metrics/peg_height", 0.45, float("nan"),
             "peg held lifted, not dropped (round-14 10M kill bar: mean >= +27mm)"),
        ],
        "peg 10M: vertical lifted grip (real insertion not expected yet)",
    ),
    (
        30_000_000,
        [
            ("metrics/axis_align", 0.80, float("nan"), "vertical grip held"),
            ("metrics/insertion_depth", 0.001, float("nan"),
             "in-bore insertion happening at all (exact 0 = never inserts)"),
            ("metrics/insertion_hold_steps", 0.05, float("nan"),
             "some sustained in-bore holds occurring (exact 0 = never holds depth)"),
        ],
        "peg 30M: insertion exists",
    ),
]


def train(config: MjxPegTrainConfig) -> None:

    run_dir = Path("runs") / f"peg_mjx_{config.num_envs}env_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rollout_size = config.num_envs * config.n_steps_per_env
    if config.batch_size > rollout_size:
        new_bs = max(rollout_size // 4, 64)
        print(
            f"WARNING: batch_size {config.batch_size} > rollout_size {rollout_size}. "
            f"Auto-resized to {new_bs}."
        )
        config.batch_size = new_bs

    curriculum_stages = scale_stage_starts(
        stages=config.curriculum_stages,
        total_timesteps=config.total_timesteps,
        reference_total_timesteps=config.curriculum_reference_timesteps,
    )

    wandb_config = asdict(config)
    wandb_config["effective_curriculum_stages"] = curriculum_stages
    wandb.init(
        project="dexterous-hand",
        name=f"peg-mjx-{config.num_envs}env",
        config=wandb_config,
    )

    vec_env = ShadowHandPegMjxEnv.from_config(config)
    vec_env = VecMonitor(vec_env)

    if config.norm_obs or config.norm_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.norm_obs,
            norm_reward=config.norm_reward,
            clip_obs=10.0,
        )

    curriculum_callback = AssemblyCurriculumCallback(
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
        # Round-14: raised from 0.02 to 0.05. With norm_reward=True the
        # advantage scale changes and KL grows faster per update, which
        # tripped the adaptive-LR throttle in round-13 (LR collapsed to
        # 5e-5 by 50M on grasp, stalling learning). 0.05 keeps a real
        # safety bound while letting LR stay near the 3e-4 starting point.
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
    if config.gate_enabled:
        callbacks.insert(1, MilestoneGateCallback(PEG_GATES, verbose=1))

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

def parse_args() -> MjxPegTrainConfig:
    parser = argparse.ArgumentParser(description="Train Shadow Hand peg-in-hole (MJX + SBX PPO)")
    parser.add_argument("--num-envs", type=int, default=768)
    parser.add_argument("--total-timesteps", type=int, default=150_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-steps-per-env", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
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
