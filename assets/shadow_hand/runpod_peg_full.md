# RunPod peg full (150M)

Single 5090 pod, ~120 hr, ~$120 at measured 316 fps for PPO+MJX at 768
envs. The 10M and 30M kill gates exist so a stuck policy costs ~$8 or
~$24 instead of $120 — use them.

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

## 3b. Troubleshooting: `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` at env.reset()

1. `nvidia-smi` in a new shell. Kill anything holding VRAM.
2. Drop `--num-envs` to 512.
3. Restart on a bigger GPU if still OOM at 512 envs.

## 4. 10M gate (run when total_timesteps ~= 10M, ~9 hr in, ~$8 sunk)

```
cd ~/dexterous_hand
python3 << 'EOF'
import csv
with open("runs/peg_mjx_768env_42/logs/progress.csv") as f:
    rows = list(csv.DictReader(f))
last = rows[-1]
print(f"timesteps:       {last['time/total_timesteps']}")
print(f"stage:           {last['train/metrics/stage']:>10}  (bar >= 2.0)")
print(f"peg_height:      {last['train/metrics/peg_height']:>10}  (bar >= 0.45)")
print(f"reward/lift:     {last['train/reward/lift']:>10}  (bar >= 1.0; step bonus firing)")
print(f"value_loss:      {last['train/value_loss']:>10}  (bar < 100)")
print(f"ep_rew_mean:     {last['rollout/ep_rew_mean']:>10}")
EOF
```

Hard kill criteria — any one fails -> kill:
- `peg_height < 0.45` (no lift past +27mm)
- `stage < 2.0`
- `reward/lift < 1.0` (step bonus not firing reliably)
- `value_loss > 100` (norm_reward isn't taking; investigate before continuing)

Kill cleanly:

```
tmux send-keys -t peg C-c
sleep 5
cp -rf ~/dexterous_hand/runs/. /workspace/runs/
runpodctl stop pod "$RUNPOD_POD_ID"
```

## 5. 30M gate (run when total_timesteps ~= 30M, ~26 hr in, ~$26 sunk)

Same script, stricter bars:
- `stage >= 2.5` sustained
- `insertion_depth >= 5e-4` trending up
- `ep_rew_mean` strictly higher than at 10M (not flat)

Pass all -> continue to 150M and start watcher in section 6. Fail any ->
kill per the section 4 commands.

## 6. Watcher in a second tmux (auto-copy + stop pod when done)

Only run this after both 10M and 30M gates pass.

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
| `train/metrics/insertion_depth`       | > 0.01 sustained                 |
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
