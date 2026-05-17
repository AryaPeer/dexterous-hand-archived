# RunPod sanity — all 3 tasks

One block on a fresh pod that runs peg + grasp + reorient sanity
back-to-back into a single log. Primary purpose: verify the
ClampedActor + `ent_coef=0.0` fix actually keeps `train/std` bounded
across every task before paying for full runs.

~10 hr wallclock on a 4090 at 256 envs. ~$7 at $0.69/hr.

## 1. Pod

CUDA 12.4+, ≥24 GB VRAM (4090, L40S, 6000 Ada, A100, H100). 4090 at 256 envs
is the canonical sanity recipe; 5090 if available cuts time roughly in half.

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

Sanity-check JAX sees the GPU:

```
uv run python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]
```

## 3. Run the bundle

```
tmux new-session -s sanity
```

Inside tmux:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
# PREALLOCATE=false + MEM_FRACTION=0.95 is the safe default for ≤32 GB GPUs at 256 envs.
# If you OOM at env.reset(), drop --num-envs to 128.
# If on a 32 GB+ GPU and feeling lucky: PREALLOCATE=true, MEM_FRACTION=0.7.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export WANDB_MODE=disabled

(
  echo "==================== train-peg-mjx ===================="
  uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 1000000 || true

  echo "==================== train-grasp-mjx ===================="
  uv run python main.py train-grasp-mjx --num-envs 256 --total-timesteps 5000000 || true

  echo "==================== train-reorient-mjx ===================="
  uv run python main.py train-reorient-mjx \
      --num-envs 256 \
      --total-timesteps 3000000 \
      --curriculum-reference-timesteps 30000000 || true
) 2>&1 | tee runs/sanity_stdout.log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t sanity`.

Notes:
- `|| true` between tasks means one failure does not skip the others.
- `--curriculum-reference-timesteps 30000000` on reorient pins sanity in
  stage 0 (30° targets). Without it, the 3-stage curriculum compresses
  into the 3M sanity window and reorient spends most of it in stage 2
  (180°), which 3M can't train.
- Order is peg → grasp → reorient. Peg fails fastest if something is
  broken (~1.5 hr), so it's first.

## 3b. Troubleshooting: `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` at env.reset()

JAX's JIT-compiled MJX reset kernel allocates more scratch memory than
fits with `PREALLOCATE=true` on ≤24 GB GPUs. Triage in order:

1. **Check GPU and free memory** — `nvidia-smi` in a new shell. If
   another process is holding memory, kill it.
2. **Already using `PREALLOCATE=false` + `MEM_FRACTION=0.95`** (the
   defaults above) — drop `--num-envs` from 256 to 128 on the failing
   task. Halves the reset-kernel allocation.
3. **Still OOM at 128 envs** — pod is too small for this codebase
   (need ≥16 GB usable VRAM for 128 envs). Restart on a bigger GPU.

## 4. Watcher in a second tmux (auto-copy + stop pod when done)

```
tmux new-session -s watcher
```

Inside:

```
while pgrep -f "main.py train-" > /dev/null; do sleep 60; done \
  && cp -rf ~/dexterous_hand/runs/. /workspace/runs/ \
  && runpodctl stop pod "$RUNPOD_POD_ID"
```

## 5. Pass criteria

The single make-or-break metric is `train/std` — if it climbs past ~1.5
on any task, ClampedActor isn't doing its job and task metrics aren't
worth reading.

| metric                                            | bar                              |
|---------------------------------------------------|----------------------------------|
| **all** `train/std`                               | stays in [0.05, ~1.1], never > 1.5 |
| **all** `train/metrics/nan_rate`                  | < 0.01                            |
| **peg** `train/metrics/num_finger_contacts`       | ≥ 1.0 by 500k, climbing           |
| **peg** `train/metrics/stage`                     | > 2.0 by 1M (gripping + lifting)  |
| **peg** `train/metrics/insertion_depth`           | > 0.001 by 1M (rising above noise)|
| **grasp** `train/metrics/num_finger_contacts`     | ≥ 2.0 sustained                   |
| **grasp** `train/metrics/object_height`           | ≥ 0.448 reached (= 1.2 cm delta)  |
| **grasp** `train/metrics/success_hold_steps`      | > 0 at least once                 |
| **grasp** `train/reward/success`                  | > 0 at least once                 |
| **reorient** `train/metrics/angular_distance`     | trending DOWN, not drifting up    |
| **reorient** `train/metrics/num_finger_contacts`  | ≥ 1.5 sustained (cube not dropped)|

`metrics/object_height = 0.448` is the geometric ceiling for grasp
(7 cm cube + fixed-z wrist caps delta-lift at ~1.2 cm). `lift_target =
0.012` is set to match; `is_success` only fires when the policy reaches
this plateau.

## 6. Cost

| pod      | rate     | wall    | cost  |
|----------|----------|---------|-------|
| RTX 4090 | $0.69/hr | ~10 hr  | ~$7   |
| RTX 5090 | $0.99/hr | ~5 hr   | ~$5   |

## 7. After sanity passes

See `runpod_full_runs.md` for the full-run commands.
