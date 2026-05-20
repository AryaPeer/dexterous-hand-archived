
import argparse
from pathlib import Path
from types import SimpleNamespace

from sbx import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from dexterous_hand.config import MjxGraspTrainConfig
from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv
from scripts.training._common import RewardInfoLoggerCallback, setup_sb3_logger


def train(args: SimpleNamespace) -> None:

    model_path = Path(args.model_path).expanduser().resolve()
    vec_norm_path = Path(args.vec_normalize_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"model not found at {model_path}")
    if not vec_norm_path.exists():
        raise FileNotFoundError(f"vec_normalize not found at {vec_norm_path}")

    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        src = model_path.parent
        run_dir = src.with_name(src.name + "_resumed")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = MjxGraspTrainConfig(num_envs=args.num_envs, seed=args.seed)

    vec_env = ShadowHandGraspMjxEnv.from_config(config)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
    vec_env.training = True
    vec_env.norm_reward = config.norm_reward

    model = PPO.load(str(model_path), env=vec_env)
    model.target_kl = 0.02

    setup_sb3_logger(model, run_dir)

    callbacks = [
        RewardInfoLoggerCallback(),
        CheckpointCallback(
            save_freq=max(500_000 // config.num_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            save_vecnormalize=True,
        ),
    ]

    print(f"Resuming from {model_path} for {args.additional_timesteps:,} additional timesteps.")
    print(f"Output dir: {run_dir}")

    model.learn(
        total_timesteps=args.additional_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save(str(run_dir / "final_model"))
    vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print(f"Saved to {run_dir}")
    vec_env.close()

def parse_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description="Resume Shadow Hand grasping (MJX + SBX PPO)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to final_model.zip (or any checkpoint .zip)")
    parser.add_argument("--vec-normalize-path", type=str, required=True,
                        help="Path to vec_normalize.pkl saved alongside the model")
    parser.add_argument("--additional-timesteps", type=int, required=True,
                        help="How many MORE timesteps to train (additional, not cumulative)")
    parser.add_argument("--num-envs", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save resumed run (default: <input_dir>_resumed)")
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
