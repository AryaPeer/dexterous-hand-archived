from __future__ import annotations

import argparse

import numpy as np

from dexterous_hand.config import (
    MjxGraspTrainConfig,
    MjxPegTrainConfig,
    MjxReorientTrainConfig,
)


def build_env(task: str, num_envs: int, seed: int):
    if task == "grasp":
        from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv
        cfg = MjxGraspTrainConfig(num_envs=num_envs, seed=seed)
        return ShadowHandGraspMjxEnv.from_config(cfg)
    if task == "peg":
        from dexterous_hand.envs.peg_env import ShadowHandPegMjxEnv
        cfg = MjxPegTrainConfig(num_envs=num_envs, seed=seed)
        env = ShadowHandPegMjxEnv.from_config(cfg)
        # peg align/depth are gated on curriculum stage; pin to stage 0 to mirror training start.
        stage0 = cfg.curriculum_stages[0]
        env.set_curriculum_params(clearance=stage0[1], p_pre_grasped=stage0[2])
        return env
    if task == "reorient":
        from dexterous_hand.envs.reorient_env import ShadowHandReorientMjxEnv
        cfg = MjxReorientTrainConfig(num_envs=num_envs, seed=seed)
        return ShadowHandReorientMjxEnv.from_config(cfg)
    raise ValueError(f"unknown task: {task!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Random-policy reward probe; flags components dominating |total|."
    )
    ap.add_argument("--task", choices=["grasp", "peg", "reorient"], required=True)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dominance-threshold", type=float, default=0.80,
                    help="flag any component whose |mean| / |total| exceeds this")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"building {args.task} env: num_envs={args.num_envs} seed={args.seed}")
    env = build_env(args.task, args.num_envs, args.seed)

    obs = env.reset()
    n_act = env.action_space.shape[0]
    print(f"obs shape={obs.shape} action_dim={n_act} max_episode_steps={env._max_episode_steps}")
    print(f"running random-policy probe for {args.steps} steps "
          f"({args.steps * args.num_envs:,} transitions)")

    reward_sums: dict[str, float] = {}
    reward_sumsq: dict[str, float] = {}
    reward_counts: dict[str, int] = {}
    total_reward_sum = 0.0
    total_reward_sumsq = 0.0
    total_reward_count = 0

    for step in range(args.steps):
        actions = rng.uniform(-1.0, 1.0, size=(args.num_envs, n_act)).astype(np.float32)
        _, rewards, _, infos = env.step(actions)

        total_reward_sum += float(rewards.sum())
        total_reward_sumsq += float((rewards ** 2).sum())
        total_reward_count += rewards.size

        for info in infos:
            for k, v in info.items():
                if not (k.startswith("reward/") or k.startswith("metrics/")):
                    continue
                val = float(np.asarray(v))
                reward_sums[k] = reward_sums.get(k, 0.0) + val
                reward_sumsq[k] = reward_sumsq.get(k, 0.0) + val * val
                reward_counts[k] = reward_counts.get(k, 0) + 1

        if (step + 1) % max(args.steps // 10, 1) == 0:
            pct = 100 * (step + 1) / args.steps
            running = total_reward_sum / total_reward_count
            print(f"  step {step + 1:>6}/{args.steps}  ({pct:5.1f}%)  mean reward={running:+.4f}")

    env.close()

    print()
    print(f"summary across {total_reward_count:,} transitions:")
    print()

    total_mean = total_reward_sum / total_reward_count
    total_var = max(total_reward_sumsq / total_reward_count - total_mean ** 2, 0.0)
    total_std = total_var ** 0.5
    print(f"  reward/total          mean={total_mean:+10.4f}  std={total_std:>10.4f}")
    print()

    rows: list[tuple[str, float, float, float]] = []
    for k in sorted(reward_counts.keys()):
        if k == "reward/total":
            continue
        n = reward_counts[k]
        mean = reward_sums[k] / n
        var = max(reward_sumsq[k] / n - mean ** 2, 0.0)
        std = var ** 0.5
        share = abs(mean) / max(abs(total_mean), 1e-9) if k.startswith("reward/") else 0.0
        rows.append((k, mean, std, share))

    print(f"  {'component':<40}  {'mean':>12}  {'std':>10}  {'|share|':>8}")
    print("  " + "-" * 76)
    for k, mean, std, share in sorted(rows, key=lambda r: -abs(r[1])):
        marker = ""
        if k.startswith("reward/") and share > args.dominance_threshold:
            marker = "  <-- dominant"
        if k.startswith("metrics/"):
            print(f"  {k:<40}  {mean:>+12.4f}  {std:>10.4f}    -    ")
        else:
            print(f"  {k:<40}  {mean:>+12.4f}  {std:>10.4f}  {share*100:>6.1f}%{marker}")

    print()
    dominant = [k for k, _, _, share in rows
                if k.startswith("reward/") and share > args.dominance_threshold]
    if dominant:
        print(f"FAIL: components above {args.dominance_threshold*100:.0f}% share "
              f"under random policy: {dominant}")
        print("      these likely have missing gates or runaway shaping. "
              "investigate before training.")
        raise SystemExit(1)
    print(f"PASS: no reward component exceeds {args.dominance_threshold*100:.0f}% "
          "share of |total| under random policy.")


if __name__ == "__main__":
    main()
