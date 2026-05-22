# Dexterous Hand RL

Training a simulated Shadow Hand to solve manipulation tasks with reinforcement learning. Built on MuJoCo 3, MJX (JAX-backed batched physics), and SBX PPO.

The hand has 24 degrees of freedom and learns two tasks: grasping a cube and peg-in-hole insertion. Both policies share a clamped-σ Gaussian actor (see `dexterous_hand/policies/clamped_actor.py`).

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
```

Each command also has a `resume-*-mjx` counterpart that reloads `final_model.zip` + `vec_normalize.pkl` from a previous run and continues for `--additional-timesteps`. See `assets/shadow_hand/runpod_full_runs.md` §6.

For end-to-end RunPod recipes (setup, env vars, tmux, watcher, pass criteria), see:

* `assets/shadow_hand/runpod_sanity_all_tasks.md` — cheap peg sanity recipe
* `assets/shadow_hand/runpod_full_runs.md` — full-budget commands for grasp and peg

## Tasks

### Grasping

Pick a cube off the table. Actuator set: 22 (X/Y slider + 20 hand joints), obs 105.

### Peg-in-hole

Grasp a peg, align with a hole, drive it in. Actuator set: 23 (X/Y/Z slider + 20 hand joints), obs 134. Z slider was added in round-10 because finger flexion alone couldn't lift the peg past ~4cm.

## Scope

Simulation-only. Observations include ground-truth object pose and velocities, which a real robot would not have directly. Sim-to-real would need either a pose estimator in front of the policy or an asymmetric actor-critic (full state to the value net, restricted obs to the policy).

## Scope

This is a simulation-only benchmark. Observations include the ground-truth
object pose and velocities, which would not be directly available on real
hardware. Sim-to-real transfer would need either a pose estimator in front of
the policy or an asymmetric actor-critic setup (full state to the value
network, restricted observations to the policy).

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Acknowledgments

The Shadow Hand MJCF model is sourced from the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
