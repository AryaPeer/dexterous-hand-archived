# RunPod sanity — round-14 (peg + grasp, 5M each)

Round-13 confirmed that `norm_reward=True` fixed the value-function
divergence (peg `value_loss` 0.30 vs round-12's 61,920) but introduced
two new failures:

- **Peg 67M (round-13)**: smooth `lift_step_bonus` ramp removed the
  discontinuity that round-12 used to escape the grasp-and-sit basin.
  Policy stuck at +43mm lift forever; `stage=1.17`, `idle_stage1_pen=
  -0.073`. The smooth ramp was over-correction — `norm_reward=True`
  already bounds value-function variance, so the binary step bonus was
  safe to keep.
- **Grasp 50M (round-13)**: `target_kl=0.02` plus adaptive LR throttled
  learning rate to 5e-5 by 50M; `success_hold_steps` plateaued at 5.59
  (slight improvement over round-12 peak but never climbing toward the
  10/20 bar). LR collapse, not value-fn collapse.

Round-14 changes (`dexterous_hand/rewards/peg_reward.py`,
`scripts/training/train_{peg,grasp}.py`):

1. **Revert `lift_step_bonus` to binary** in `peg_reward.py`:
   `jnp.where(lift_height > 0.005, 1.0, 0.0)`. Round-12's original design.
   With `norm_reward=True` keeping the value function bounded, the
   discontinuity is no longer a problem.
2. **Raise `target_kl` from 0.02 → 0.05** in both training scripts. The
   0.02 bound was calibrated for unnormalized rewards; normalized rewards
   give larger KL per update at the same learning rate, tripping the
   adaptive-LR throttle prematurely.

## Pre-flight (run BEFORE spending on a pod)

Round-14 ships a free CPU-side reward-gradient check that catches the
exact failure modes of rounds 11–13:

```
uv run python scripts/check_reward_gradient.py
```

Expected output: `PEG: PASS` and `GRASP: PASS`. If either fails, do not
spend on a pod — the reward shape is broken. Gates checked:

- **Peg**: lifting 0mm → 6mm (just past `lift_step_threshold`) must
  produce `delta_total ≥ 1.0` (catches missing step bonus) AND the lift
  component must beat the static grasp reward at the same state (catches
  round-11/13 grasp-and-sit basin).
- **Grasp**: lifting 0mm → `lift_target` must produce `delta_total ≥ 5.0`
  AND be monotonic through the intermediate 6mm range (catches any
  reward shape where the gradient zeros out partway up).

~4.5 hr wallclock per task on a 4090 at 256 envs (~$3 each). Both can
share a pod sequentially or run on two pods in parallel.

## 1. Pod

CUDA 12.4+, ≥24 GB VRAM (4090, L40S, 6000 Ada, A100, H100). 4090 at 256
envs is the canonical sanity recipe; 5090 if available cuts time roughly
in half.

## 2. Setup (paste once on a fresh pod)

```
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd ~
git clone https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd dexterous_hand
uv sync --extra mjx
mkdir -p runs
```

`uv sync --extra mjx` resolves the locked dependency set (JAX pinned to
<0.5 in `pyproject.toml`, which bundles cuDNN 9.5/9.6 and works on both
driver 570.x and 580.x lines). No manual `pip install` follow-ups
needed.

Free pre-flight before the sanity run:

```
uv run python scripts/check_reward_gradient.py
# expected: PEG: PASS / GRASP: PASS
```

If either fails, the reward shape regressed and the sanity will fail —
fix locally before paying for a pod.

JAX GPU sanity:

```
uv run python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]

uv run python -c "import jax; x = jax.numpy.ones((4,4)); print((x @ x).sum())"
# expected: 64.0 (no CUDNN_STATUS_NOT_INITIALIZED)
```

If JAX errors with `CUDNN_STATUS_NOT_INITIALIZED` despite the pin, the
host driver is older than 545 — destroy and redeploy. PyTorch may print
a "CUDA driver too old" warning and fall back to CPU; ignore it,
training runs on JAX/Flax (not torch).

## 3. Run

```
tmux new-session -s sanity
```

Inside tmux (peg first, then grasp on the same pod, or run on two pods):

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export WANDB_MODE=disabled

uv run python main.py train-peg-mjx \
    --num-envs 256 \
    --total-timesteps 5000000 \
    2>&1 | tee runs/sanity_peg_stdout.log

uv run python main.py train-grasp-mjx \
    --num-envs 256 \
    --total-timesteps 5000000 \
    2>&1 | tee runs/sanity_grasp_stdout.log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t sanity`.

## 3b. Troubleshooting: `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` at env.reset()

1. `nvidia-smi` in a new shell. Kill anything holding VRAM.
2. Drop `--num-envs` to 128.
3. Restart on a bigger GPU if still OOM at 128 envs.

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

`train/std` is the regression test on ClampedActor. If it climbs past
~1.5 the policy clamp isn't doing its job and task metrics aren't worth
reading.

### Shared (both tasks)

| metric                            | bar                                |
|-----------------------------------|------------------------------------|
| `train/std`                       | stays in [0.05, ~1.1], never > 1.5 |
| `train/metrics/nan_rate`          | < 0.01                             |

### Value-function health (round-13 regression test)

These bars exist because round-12 only manifested its failure at scale.
With `norm_reward=True`, returns flowing into PPO are std-normalized
(typically std ≈ 1), so value-loss should sit in single-digit to low-tens
territory regardless of how high raw rewards climb.

| metric                            | round-12 observed (grasp 43M) | round-13 bar (5M sanity)             |
|-----------------------------------|-------------------------------|--------------------------------------|
| `train/value_loss`                | 469 → 61,920 (130×, climbing) | < 100, **flat or declining** by 3M   |
| `train/explained_variance`        | 0.97 peak → 0.91 (declining)  | > 0.8 by 3M, **flat or rising**      |
| `train/clip_fraction`             | 0.25 → 0.30 (climbing)        | < 0.25, flat                         |
| `train/approx_kl`                 | 0.018 → 0.022                 | < target_kl=0.02 most epochs         |

If `value_loss` is still climbing quadratically with reward magnitude at
3M, `norm_reward=True` didn't take effect — check `VecNormalize` is
wrapping the env and `config.norm_reward` is `True`.

### Peg (5M) — round-14: binary lift_step_bonus + norm_reward

The curriculum at 5M with reference=100M compresses stage starts to
(0, 400k, 800k, 1.2M, 1.6M). By 1.6M the policy is at max difficulty
(clearance=1mm, p_pre_grasped=0.2). Measure final-window (last 1M)
rolling means:

| metric                                          | round-13 observed (67M cook) | round-14 bar (5M sanity)               |
|-------------------------------------------------|------------------------------|----------------------------------------|
| `train/metrics/peg_height`                      | 0.466 (flat at +43mm)        | ≥ 0.443 (+20mm), **trending up**       |
| `train/metrics/stage`                           | 1.17 (stuck)                 | ≥ 2.0 sustained                        |
| `train/metrics/num_finger_contacts`             | 3.6 ✓                        | ≥ 2.0 sustained                        |
| `train/reward/lift` (raw, post-weight)          | 0.492                        | ≥ 1.0 (step bonus firing)              |
| `train/reward/idle_stage1_penalty`              | -0.073 (worsening)           | less negative than -0.02, **declining**|
| `train/metrics/insertion_depth`                 | 1.34e-4                      | > 5e-4 mean by 5M                      |
| `rollout/ep_rew_mean`                           | flat 4,500-6,500             | trending up, **not regressing**        |

Round-14 specifically expects `reward/lift ≥ 1.0` because the binary
step bonus contributes a +1.0 jump (post-weight: 1.0 × 15 = 15 max).
Round-13's smooth ramp capped at 0.49 post-weight regardless of how
high the policy lifted — that was the round-13 cook in a single number.

### Grasp (5M) — round-14: norm_reward + raised target_kl

| metric                                  | round-13 observed (50M cook) | round-14 bar (5M)              |
|-----------------------------------------|------------------------------|--------------------------------|
| `train/metrics/object_height`           | 0.444 (just under bar)       | rises ≥ 0.438 by 5M            |
| `train/metrics/success_hold_steps`      | 5.59 plateau                 | ≥ 1.5 by 5M, **still climbing**|
| `train/learning_rate`                   | 5e-5 (collapsed)             | **stays ≥ 1e-4** at 5M         |
| `train/reward/total`                    | flat 18.5 (no change)        | trending up                    |
| `train/value_loss`                      | 0.30 ✓                       | < 100, flat or declining       |

The crucial new bar: `learning_rate` must stay above 1e-4. If adaptive
LR is collapsing inside 5M with `target_kl=0.05`, that means even the
relaxed target is still too tight and grasp will plateau again — kill
and raise to 0.1.

## Full-run kill gates (post-sanity)

Sanity confirms reward design + value-fn stability. Full-run kill gates
catch failures sanity is too short to see. These are **hard kill** bars,
not soft warnings.

### Peg (150M target) — kill at 10M if any fail

| metric                          | 10M bar (~$8 in)               |
|---------------------------------|--------------------------------|
| `train/metrics/peg_height`      | ≥ 0.45 (+27mm), **rising**     |
| `train/metrics/stage`           | ≥ 2.0                          |
| `train/reward/lift` (raw)       | ≥ 1.0 (step bonus firing)      |
| `train/value_loss`              | < 100                          |

Round-13's `peg_height` was 0.466 at 13M and falling. If 10M peg shows
< 0.45 or stage < 2.0, this run is heading to round-13's cook — kill.

### Grasp (70M target) — kill at 10M if any fail

| metric                              | 10M bar (~$8 in)                            |
|-------------------------------------|---------------------------------------------|
| `train/metrics/success_hold_steps`  | ≥ 3.0 AND projected ≥ 12 at 70M (linear extrap.) |
| `train/learning_rate`               | ≥ 1e-4                                       |
| `train/value_loss`                  | < 100, not climbing                          |

Round-13 hit succ_hold=4.96 at 24M which linearly extrapolated to ~14
at 70M — but adaptive LR collapsed and it flat-lined at 5.6 instead.
Watch the LR, not just the task metric.

If grasp's `value_loss` is bounded at sanity scale but climbs in the
full run, the issue isn't `norm_reward` — escalate.

## 6. Cost

Two tasks at 5M each. Run sequentially on one pod or in parallel on two.

| pod      | rate     | wall per task | both sequential | both parallel (2 pods) |
|----------|----------|---------------|-----------------|------------------------|
| RTX 4090 | $0.69/hr | ~4.5 hr       | ~$6             | ~$3 each = $6          |
| RTX 5090 | $0.99/hr | ~2.5 hr       | ~$5             | ~$2.5 each = $5        |

## 7. After sanity passes

See `runpod_peg_full.md` and `runpod_grasp_full.md` for the full-run
commands. **Do not resume from any peg or grasp checkpoint that pre-dates
round-13** — `norm_reward` flipped from False to True, so VecNormalize
reward statistics would be uninitialized/wrong on resume, and the peg
reward formula also changed (smooth lift_step_bonus). Round-13 must
start from a fresh run.
