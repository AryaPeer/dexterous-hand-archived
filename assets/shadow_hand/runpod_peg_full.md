# RunPod peg full (150M)

Single 5090 pod, ~120 hr, ~$120 at measured 316 fps for PPO+MJX at 768
envs. The 10M and 30M gates are now **automatic** — the training process
prints a diagnostic and stops itself if the task metrics regress or stall,
so a stuck policy costs ~$8 or ~$24 instead of $120. No manual kill needed
(disable with `--no-gate` if you want to override).

**2026-06-10:** the insertion-depth metric gained a lateral-containment
gate (`get_insertion_depth_jax`) — before it, ANY peg at table level
scored insertion fraction 1.0, so the 2026-06-01 5M sanity numbers
(insertion_depth 0.060, complete 3.24, hold 1.83) measured a
drop-the-peg exploit, not insertion, and were retired as gate baselines.
Current floors are first-principles collapse bars (see
`scripts/training/train_peg.py::PEG_GATES`); **run a fresh 5M sanity
before this full run and re-derive floors from it.** The lift reward
remains capped (weight 10, cap 1.0) so `reward/lift` sits well below
the old `>= 1.0` bar and is deliberately NOT gated.

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
git clone https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
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
# expected: 64.0 (no CUDNN_STATUS_NOT_INITIALIZED)
```

If JAX still errors with `CUDNN_STATUS_NOT_INITIALIZED` despite the pin,
the host driver is older than 545. Destroy and redeploy. PyTorch may
print a "CUDA driver too old" warning and fall back to CPU — ignore it,
training runs entirely on JAX/Flax and is unaffected.

## 3. Run

```
tmux new-session -s peg
```

Inside tmux:

```
cd ~/dexterous_hand

export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export WANDB_MODE=disabled

uv run python main.py train-peg-mjx \
    --num-envs 768 \
    --total-timesteps 150000000 \
    2>&1 | tee runs/peg_full_stdout.log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t peg`.

The run **auto-gates at 10M and 30M** (see §4/§5): it prints a
`===== MILESTONE GATE =====` block and exits cleanly if metrics regress.
Add `--no-gate` to the command above to disable and force a full 150M run.

## 3b. Troubleshooting: `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` at env.reset()

1. `nvidia-smi` in a new shell. Kill anything holding VRAM.
2. Drop `--num-envs` to 512.
3. Restart on a bigger GPU if still OOM at 512 envs.

## 4. Automatic gates (10M and 30M)

The run gates itself — you don't run anything. At ~10M (~9 hr, ~$8) and
~30M (~26 hr, ~$26) `MilestoneGateCallback` prints a
`===== MILESTONE GATE =====` table of the recent-mean task metrics vs
floors and **exits the process cleanly** if any metric is below floor.
Floors (source of truth: `scripts/training/train_peg.py::PEG_GATES`):

10M — vertical lifted grip (real insertion not expected yet):
- `metrics/axis_align >= 0.70`   (round-16 collapse mode was 0.07)
- `metrics/stage >= 1.5`         (past grasp-and-sit)
- `metrics/peg_height >= 0.45`   (peg held lifted, not dropped; round-14 bar)

30M — insertion exists:
- `metrics/axis_align >= 0.80`              (vertical grip held)
- `metrics/insertion_depth >= 0.001`        (in-bore insertion happening at
  all — with the containment-gated metric an exact 0 over the ~1.5M-step
  window means the policy never inserts)
- `metrics/insertion_hold_steps >= 0.05`    (some sustained in-bore holds)

These are deliberately loose collapse bars, not health bars — the old
sanity-derived floors were measured with the pre-containment metric and
would either pass an exploiting run or kill an honest one. Re-derive
real floors from the first post-fix 5M sanity.

`reward/lift` is intentionally NOT gated: the lift weight was capped
(weight 10, cap 1.0), so it sits at ~0.07 and the old `>= 1.0` bar would
false-fail. On a gate stop the process saves and exits — preserve and
stop the pod:

```
# training already exited; just preserve + shut down
cp -rf ~/dexterous_hand/runs/. /workspace/runs/
runpodctl stop pod "$RUNPOD_POD_ID"
```

A ~500k checkpoint always exists under `runs/peg_mjx_768env_42/checkpoints/`.
If you judge a stop premature (e.g. a sigmoidal learner still climbing),
resume per §9 from the latest checkpoint rather than restarting.

## 5. Optional: inspect progress yourself any time

The gate is automatic, but to peek mid-run:

```
cd ~/dexterous_hand
python3 << 'EOF'
import csv
with open("runs/peg_mjx_768env_42/logs/progress.csv") as f:
    rows = list(csv.DictReader(f))
last = rows[-1]
for k in ["time/total_timesteps", "train/metrics/axis_align",
          "train/metrics/insertion_depth", "train/metrics/insertion_hold_steps",
          "train/metrics/stage", "train/reward/complete",
          "train/metrics/num_finger_contacts", "train/value_loss",
          "rollout/ep_rew_mean"]:
    print(f"{k:38s} {last.get(k, 'n/a')}")
EOF
```

`value_loss` should stay < 100 (norm_reward working); it is NOT
auto-gated, so watch it here — a climb past ~100 means investigate.

## 6. Watcher in a second tmux (auto-copy + stop pod when done)

Start this right after launching the run — it copies results and stops
the pod whenever training exits, **whether at an auto-gate stop or after
the full 150M**. So you can leave the run unattended either way.

```
tmux new-session -s watcher
```

Inside:

```
while pgrep -f "main.py train-peg-mjx" > /dev/null; do sleep 60; done \
  && cp -rf ~/dexterous_hand/runs/. /workspace/runs/ \
  && runpodctl stop pod "$RUNPOD_POD_ID"
```

## 7. Pass criteria (after 150M)

| metric                                | bar                              |
|---------------------------------------|----------------------------------|
| `train/std`                           | stays in [0.05, 1.1], never >1.5 |
| `train/metrics/nan_rate`              | < 0.01                           |
| `train/metrics/stage`                 | reaches 4.0 sustained            |
| `train/metrics/insertion_depth`       | > 0.05 (~0.66 frac) sustained    |
| `train/metrics/insertion_hold_steps`  | > 10 (sustained 10-step hold)    |
| `train/value_loss`                    | < 100, flat or declining         |
| `eval/success_rate`                   | > 0.10                           |

If task bars trend positive but aren't fully cleared, resume per §9.

## 8. Cost

| pod      | rate     | wall    | cost |
|----------|----------|---------|------|
| RTX 5090 | $0.99/hr | ~120 hr | ~$120 |
| RTX 5090 (killed at 10M gate) | $0.99/hr | ~9 hr | ~$8 |
| RTX 5090 (killed at 30M gate) | $0.99/hr | ~26 hr | ~$26 |
| RTX 4090 | $0.69/hr | ~190 hr | ~$130 |

## 9. Resume

```
uv run python main.py resume-peg-mjx \
    --model-path runs/<run_name>/final_model.zip \
    --vec-normalize-path runs/<run_name>/vec_normalize.pkl \
    --additional-timesteps 50000000 \
    --num-envs 768 \
    --seed 42
```

`--additional-timesteps` is additional, not cumulative. Output writes
to `runs/<run_name>_resumed/` unless `--output-dir` is set.

**Do not resume from any peg checkpoint that pre-dates round-14** — the
lift reward formula and `norm_reward` setting both changed across
rounds 12, 13, and 14, so VecNormalize statistics from older runs
would be wrong. Start round-14 from scratch.
