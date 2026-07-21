"""Render a deterministic rollout of a trained checkpoint to mp4."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np


def _render_task(
    task: str,
    model_path: Path,
    vec_normalize_path: Path,
    out_path: Path,
    steps: int | None,
    seed: int,
) -> dict[str, float]:
    from sbx import PPO
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    from scripts.training._common import load_saved_config

    config_cls: Any
    env_cls: Any
    build_fn: Any
    if task == "grasp":
        from dexterous_hand.config import MjxGraspTrainConfig
        from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv
        from dexterous_hand.envs.scene_builder import build_scene

        config_cls, env_cls, build_fn = MjxGraspTrainConfig, ShadowHandGraspMjxEnv, build_scene
    else:
        from dexterous_hand.config import MjxPegTrainConfig
        from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv
        from dexterous_hand.envs.peg_scene_builder import build_peg_scene

        config_cls, env_cls, build_fn = MjxPegTrainConfig, ShadowHandPegMjxEnv, build_peg_scene

    config = config_cls()
    load_saved_config(config, model_path)
    config.num_envs = 1
    config.seed = seed
    config.obs_noise_std = 0.0

    if steps is None:
        steps = config.max_episode_steps

    raw_env: Any = env_cls.from_config(config)
    env: Any = VecMonitor(raw_env)
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)

    cpu_model, cpu_data, _nm = build_fn(config.scene_config)
    renderer = mujoco.Renderer(cpu_model, height=480, width=640)

    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    obs = env.reset()
    frames = []
    for _ in range(steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, _rewards, _dones, infos = env.step(actions)
        for k, v in infos[0].items():
            if k.startswith("metrics/") or k.startswith("reward/"):
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                metric_counts[k] = metric_counts.get(k, 0) + 1

        qpos = np.asarray(raw_env._mjx_data_batch.qpos[0])
        cpu_data.qpos[:] = qpos
        mujoco.mj_forward(cpu_model, cpu_data)
        renderer.update_scene(cpu_data, camera="track_cam")
        frames.append(renderer.render().copy())

    renderer.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=25, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()

    return {k: metric_sums[k] / metric_counts[k] for k in sorted(metric_sums)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", choices=("grasp", "peg", "both"), default="both")
    ap.add_argument("--grasp-model", type=Path, default=None)
    ap.add_argument("--grasp-vec-normalize", type=Path, default=None)
    ap.add_argument("--peg-model", type=Path, default=None)
    ap.add_argument("--peg-vec-normalize", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/render_overnight"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = ["grasp", "peg"] if args.task == "both" else [args.task]
    for task in tasks:
        model_path = args.grasp_model if task == "grasp" else args.peg_model
        vn_path = args.grasp_vec_normalize if task == "grasp" else args.peg_vec_normalize
        if model_path is None or vn_path is None:
            raise SystemExit(f"--{task}-model and --{task}-vec-normalize are required for task={task}")
        out_path = args.out_dir / f"{task}_rollout.mp4"
        print(f"[{task}] rendering deterministic rollout -> {out_path}", flush=True)
        summary = _render_task(task, model_path, vn_path, out_path, args.steps, args.seed)
        print(f"[{task}] done. per-step metric means over the rollout:")
        for k, v in summary.items():
            print(f"    {k:36s} = {v:.4f}")


if __name__ == "__main__":
    main()
