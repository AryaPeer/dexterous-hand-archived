# Dexterous Hand RL

Training a simulated Shadow Hand to solve manipulation tasks with reinforcement learning. Built on MuJoCo 3, MJX (JAX-backed batched physics), and SBX PPO.

The hand has 24 degrees of freedom and learns three tasks: grasping a cube, pick-and-place to a goal, and peg-in-hole insertion. All policies share a clamped-σ Gaussian actor (see `dexterous_hand/policies/clamped_actor.py`).

## Requirements

* Python 3.11+
* [uv](https://github.com/astral-sh/uv) for dependency management
* For training: an NVIDIA GPU with CUDA 12.4+ (4090 / 5090 / L40S / A100 / H100). MJX runs the batched sim on GPU; CPU-only is not a supported training path.

## Setup

```bash
git clone https://github.com/AryaPeer/Dexterous-Hand.git
cd Dexterous-Hand

# Local (tests, lint, scene-builder probes — no training):
uv sync

# GPU box (training):
uv sync --extra mjx
```

The `mjx` extra pulls `mujoco-mjx`, `sbx-rl`, `jax[cuda12]`, `flax`, `optax`, `chex`.

Sanity-check JAX sees the GPU:

```bash
uv run python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]
```

## Training

All training goes through `main.py`. Each task has its own MJX command:

```bash
# Grasping (PPO, 70M default, 768 envs)
uv run python main.py train-grasp-mjx --num-envs 768 --total-timesteps 70000000

# Peg-in-hole (PPO, 150M default, 768 envs, assembly curriculum)
uv run python main.py train-peg-mjx --num-envs 768 --total-timesteps 150000000

# Pick-and-place (PPO, 70M default, 768 envs, random goal)
uv run python main.py train-pickplace-mjx --num-envs 768 --total-timesteps 70000000
```

Each command also has a `resume-*-mjx` counterpart that reloads `final_model.zip` + `vec_normalize.pkl` from a previous run and continues for `--additional-timesteps`. See §8 of `runpod_peg_full.md` or §7 of `runpod_grasp_full.md`.

For end-to-end RunPod recipes (setup, env vars, tmux, watcher, pass criteria), see:

* `assets/shadow_hand/runpod_sanity_all_tasks.md` — cheap peg sanity recipe
* `assets/shadow_hand/runpod_peg_full.md` — peg 150M full run (with 30M gate)
* `assets/shadow_hand/runpod_grasp_full.md` — grasp 70M full run

## Tasks

### Grasping

Pick a cube off the table and hold it at height. Actuator set: 23 (X/Y/Z slider + 20 hand joints), obs 108.

### Pick-and-place

Pick a cube off the table, carry it to a randomized goal marker, and release it on target. Trained from scratch (no curriculum). Actuator set: 23 (X/Y/Z slider + 20 hand joints), obs 114 (adds goal position + object-to-goal vector).

### Peg-in-hole

Grasp a peg, transport it over an elevated guide tube, engage the tip, and release — gravity finishes the insertion. Actuator set: 23 (X/Y/Z slider + 20 hand joints), obs 134.

## Scope

Simulation-only. Observations include ground-truth object pose and velocities, which a real robot would not have directly. Sim-to-real would need either a pose estimator in front of the policy or an asymmetric actor-critic (full state to the value net, restricted obs to the policy).

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Acknowledgments

The Shadow Hand MJCF model is sourced from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
