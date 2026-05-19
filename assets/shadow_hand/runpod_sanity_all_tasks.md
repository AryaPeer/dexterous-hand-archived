# RunPod sanity — peg + reorient

One block on a fresh pod that runs peg 10M + reorient 3M back-to-back into a
single log. Two purposes:
1. Verify ClampedActor + `ent_coef=0.0` keeps `train/std` bounded.
2. **Peg 10M:** confirm `metrics/stage` rolling mean clears 2.5 with the
   curriculum fully ramped — earlier 1M sanity was inconclusive because the
   curriculum compressed to 1% of reference and the policy had ~680k steps
   at the hardest setting (clearance=1mm, p_pre_grasped=0.2). 10M leaves
   ~6.8M of real training at max difficulty.
3. **Reorient 3M:** validate the no-drop-termination + smooth `drop_factor`
   change. Cube can fall and the episode keeps going — `num_finger_contacts`
   and `cube_drop` per-step penalty are the signals now, not early-end.

Grasp is no longer in the bundle — the prior 5M sanity already showed clean
learning (`eval/success_rate` 0 → 0.25, `obj_height` climbing toward the
0.448 plateau, std bounded). No code path has changed that affects grasp,
so re-running it would be ~$5 of confirmation.

~17 hr wallclock on a 4090 at 256 envs. ~$12 at $0.69/hr.

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
  uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 10000000 || true

  echo "==================== train-reorient-mjx ===================="
  uv run python main.py train-reorient-mjx \
      --num-envs 256 \
      --total-timesteps 3000000 \
      --curriculum-reference-timesteps 30000000 || true
) 2>&1 | tee runs/sanity_stdout.log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t sanity`.

Notes:
- `|| true` between tasks means peg failure does not skip reorient.
- `--curriculum-reference-timesteps 30000000` on reorient pins sanity in
  stage 0 (30° targets). Without it, the 3-stage curriculum compresses
  into the 3M sanity window and reorient spends most of it in stage 2
  (180°), which 3M can't train.
- Order is peg → reorient. Peg is the longer of the two; reorient is the
  variance-test on the no-drop-termination change.

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

`train/std` is the regression test on ClampedActor — if it climbs past
~1.5 on either task, the policy clamp isn't doing its job and task
metrics aren't worth reading.

**Both tasks:**

| metric                            | bar                                |
|-----------------------------------|------------------------------------|
| `train/std`                       | stays in [0.05, ~1.1], never > 1.5 |
| `train/metrics/nan_rate`          | < 0.01                             |

**Peg (10M) — diagnostic bars on real curriculum:**

The curriculum at 10M scales to stage starts (0, 800k, 1.6M, 2.4M, 3.2M),
so by step 3.2M the policy is training at max difficulty (clearance=1mm,
p_pre_grasped=0.2). Measure final-window (last 1M of training) rolling
means:

| metric                                          | bar                                      |
|-------------------------------------------------|------------------------------------------|
| `train/metrics/stage`                           | ≥ 2.5 (lift sustained, approaching align)|
| `train/metrics/num_finger_contacts`             | ≥ 2.0 sustained                          |
| `train/metrics/peg_height`                      | rising trend, > initial + 0.02 mean       |
| `train/metrics/insertion_depth`                 | > 0.001 mean (above noise floor)          |
| `train/reward/insertion_drive`                  | > 0 occurring (4-gate firing sometimes)   |

If stage stalls < 2.0 at 10M, **don't ship the 150M.** Dig into:
- `lift_target=0.1m` vs stage-2 trigger of 0.02m (mismatch?)
- `p_pre_grasped` ramping too aggressively in the curriculum
- whether `insertion_drive`'s 4-gate is locked out (align_weight sigmoid
  needs peg_clearance > 2cm; if peg never gets there, drive is 0 forever)

**Reorient (3M):**

| metric                                            | bar                                |
|---------------------------------------------------|------------------------------------|
| `train/metrics/angular_distance`                  | trending DOWN, not drifting up     |
| `train/metrics/num_finger_contacts`               | ≥ 1.5 sustained (cube held)        |
| `train/metrics/success_steps`                     | > 0 (any non-zero is signal)       |

No-drop-termination side effect: episodes will hit the 400-step time
limit much more often than before. `rollout/ep_len_mean` should approach
400 even when the cube falls. If `ep_len_mean` stays near 100 (early
truncation pattern), the change didn't land — re-verify the env code.

## 6. Cost

| pod      | rate     | wall    | cost  |
|----------|----------|---------|-------|
| RTX 4090 | $0.69/hr | ~17 hr  | ~$12  |
| RTX 5090 | $0.99/hr | ~9 hr   | ~$9   |

## 7. After sanity passes

See `runpod_full_runs.md` for the full-run commands. Grasp 70M is safe
to ship without re-sanity since no relevant code has changed; bundle it
with peg full and reorient full if you want a single pod.
