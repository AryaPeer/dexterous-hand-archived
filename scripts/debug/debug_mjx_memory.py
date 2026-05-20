from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx

from dexterous_hand.envs.scene_builder import build_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=16)
    args = ap.parse_args()

    n = args.num_envs
    model, _, _ = build_scene()
    print(f"mj model: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"ngeom={model.ngeom} njnt={model.njnt} nbody={model.nbody}")

    mjx_model = mjx.put_model(model)
    print(f"mjx model: nconmax={getattr(mjx_model, 'nconmax', '?')} "
          f"njmax={getattr(mjx_model, 'njmax', '?')}")

    base = mjx.make_data(mjx_model)
    batch = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n,) + x.shape) if hasattr(x, "shape") else x,
        base,
    )

    rows = []
    total = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(batch):
        if not hasattr(leaf, "shape"):
            continue
        shape = tuple(leaf.shape)
        dtype = leaf.dtype
        nbytes = int(jnp.prod(jnp.array(shape)) * dtype.itemsize)
        per_env = nbytes // max(n, 1) if shape and shape[0] == n else nbytes
        name = ".".join(str(p).strip(".[]") for p in path)
        rows.append((nbytes, per_env, name, shape, str(dtype)))
        total += nbytes

    rows.sort(reverse=True)
    print()
    print(f"{'total (MB)':>12} {'per-env (KB)':>14}  name  shape  dtype")
    print("-" * 100)
    for nbytes, per_env, name, shape, dt in rows[:30]:
        print(f"{nbytes / 1e6:>12.1f} {per_env / 1e3:>14.1f}  {name}  {shape}  {dt}")
    print("-" * 100)
    print(f"TOTAL across batched mjx.Data at n={n}: {total / 1e9:.2f} GB")
    print(f"Projected at n=256: {total * 256 / n / 1e9:.2f} GB")
    print(f"Projected at n=1024: {total * 1024 / n / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
