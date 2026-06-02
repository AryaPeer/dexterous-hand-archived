# RunPod grasp full (70M)

Single 5090 pod, ~62 hr, ~$61 at measured 316 fps for PPO+MJX at 768
envs. Round-13 grasp 50M had `success_hold_steps` plateau at 5.6 due to
adaptive LR collapse; round-14 raises `target_kl` to 0.05 to let LR stay
near 3e-4. The gates are now **automatic** (disable with `--no-gate`):
a 10M grip-health gate and a 50M lift-emergence gate stop the run if it
regresses. Lift via finger-curl only emerges around ~40M (flat
`object_height` at 5M is EXPECTED), so — unlike the old manual gate — the
10M check does NOT require any lift or success-hold; that is deferred to
the 50M gate.

## 1. Pod

CUDA 12.4+, >=24 GB VRAM. RTX 5090 is canonical; 4090 also works but
slower at same cost. Either driver 570.x or 580.x is fine — the JAX
dependency is pinned to <0.5 in `pyproject.toml`, which bundles
cuDNN 9.5/9.6 and works on both driver lines.

## 2. Setup (paste once on a fresh pod)

```
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd ~
git clone -b cleanup-dead-code https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd dexterous_hand
uv sync --extra mjx
mkdir -p runs
```

`uv sync --extra mjx` resolves the locked dependency set including the
pinned JAX. No manual `pip install` follow-ups needed.

Pre-flight (free, CPU-only — run before paying for the GPU run):

```
uv run python scripts/check_reward_gradient.py
# expected: PEG: PASS / GRASP: PASS
```

JAX GPU sanity:

```
uv run python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]

uv run python -c "import jax; x = jax.numpy.ones((4,4)); print((x @ x).sum())"
# expected: 16.0 (no CUDNN_STATUS_NOT_INITIALIZED)
```

If JAX still errors with `CUDNN_STATUS_NOT_INITIALIZED` despite the pin,
the host driver is older than 545. Destroy and redeploy. PyTorch may
print a "CUDA driver too old" warning and fall back to CPU — ignore it,
training runs entirely on JAX/Flax and is unaffected.

## 3. Run

```
tmux new-session -s grasp
```

Inside tmux:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export WANDB_MODE=disabled

uv run python main.py train-grasp-mjx \
    --num-envs 768 \
    --total-timesteps 70000000 \
    2>&1 | tee runs/grasp_full_stdout.log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t grasp`.

## 3b. Troubleshooting: `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` at env.reset()

1. `nvidia-smi` in a new shell. Kill anything holding VRAM.
2. Drop `--num-envs` to 512.
3. Restart on a bigger GPU if still OOM at 512 envs.

## 4. Automatic gates (10M grip-health, 50M lift-emergence)

The run gates itself — you don't run anything. `MilestoneGateCallback`
prints a `===== MILESTONE GATE =====` table and **exits cleanly** if a
floor is breached. Floors (source of truth:
`scripts/training/train_grasp.py::GRASP_GATES`):

10M — grip health (lift NOT expected yet):
- `metrics/num_finger_contacts >= 3.0`  (5M 4.92)
- `reward/grasping >= 0.70`             (5M 0.985)
- `reward/grasp_quality >= 0.70`        (5M 1.0)

50M — lift emergence:
- `metrics/object_height >= 0.437`  (5M 0.4349 at rest; ~11mm lift expected
  by 40M — a flat 0.4349 means the policy never lifts)

The old manual 10M bar `success_hold_steps >= 3.0` was WRONG: grasp has no
success hold until it lifts (~40M), so it would have false-killed every
healthy run. `learning_rate`/`value_loss` aren't in the env infos, so they
are not auto-gated — watch them with the snippet in §4b.

On a gate stop the process saves and exits — preserve and stop the pod:

```
cp -rf ~/dexterous_hand/runs/. /workspace/runs/
runpodctl stop pod "$RUNPOD_POD_ID"
```

A ~500k checkpoint exists under `runs/grasp_mjx_768env_42/checkpoints/`;
resume per §8 if you judge a stop premature.

Expected throughput on a saturated 5090 is ~316 fps (PPO+MJX is
GPU-bound, not env-bound — bumping `--num-envs` won't help once util is
99%).

## 4b. Optional: inspect progress yourself any time

```
cd ~/dexterous_hand
python3 << 'EOF'
import csv
with open("runs/grasp_mjx_768env_42/logs/progress.csv") as f:
    rows = list(csv.DictReader(f))
last = rows[-1]
for k in ["time/total_timesteps", "train/metrics/num_finger_contacts",
          "train/reward/grasping", "train/reward/grasp_quality",
          "train/metrics/object_height", "train/metrics/success_hold_steps",
          "train/learning_rate", "train/value_loss", "rollout/ep_rew_mean",
          "train/std"]:
    print(f"{k:38s} {last.get(k, 'n/a')}")
EOF
```

Watch `learning_rate >= 1e-4` (adaptive-LR collapse = the round-13 failure)
and `value_loss < 100`. Neither is auto-gated.

## 5. Watcher in a second tmux (auto-copy + stop pod when done)

Start this right after launching the run — it copies results and stops
the pod whenever training exits, **whether at an auto-gate stop or after
the full 70M**. So you can leave the run unattended either way.

```
tmux new-session -s watcher
```

Inside:

```
while pgrep -f "main.py train-grasp-mjx" > /dev/null; do sleep 60; done \
  && cp -rf ~/dexterous_hand/runs/. /workspace/runs/ \
  && runpodctl stop pod "$RUNPOD_POD_ID"
```

## 6. Pass criteria (after 70M)

| metric                                  | bar                              |
|-----------------------------------------|----------------------------------|
| `train/std`                             | stays in [0.05, 1.1], never >1.5 |
| `train/metrics/nan_rate`                | < 0.01                           |
| `train/metrics/object_height`           | >= 0.448 sustained               |
| `train/metrics/success_hold_steps`      | > 10 mean (out of 20)            |
| `train/learning_rate`                   | >= 1e-4 throughout               |
| `train/value_loss`                      | < 100, flat or declining         |

If task bars trend positive but aren't fully cleared, resume per §8.

## 7. Cost

| pod      | rate     | wall    | cost |
|----------|----------|---------|------|
| RTX 5090 | $0.99/hr | ~62 hr  | ~$61 |
| RTX 5090 (killed at 10M gate) | $0.99/hr | ~9 hr | ~$8 |
| RTX 4090 | $0.69/hr | ~100 hr | ~$70 |

## 8. Resume

```
uv run python main.py resume-grasp-mjx \
    --model-path runs/<run_name>/final_model.zip \
    --vec-normalize-path runs/<run_name>/vec_normalize.pkl \
    --additional-timesteps 50000000 \
    --num-envs 768 \
    --seed 42
```

`--additional-timesteps` is additional, not cumulative. Output writes
to `runs/<run_name>_resumed/` unless `--output-dir` is set.

**Do not resume from any grasp checkpoint that pre-dates round-14** —
`norm_reward=True` and `target_kl=0.05` both flipped, so VecNormalize
statistics from older runs would be wrong. Start round-14 from scratch.
