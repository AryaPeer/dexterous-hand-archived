# RunPod full runs — peg + grasp + reorient

All three full-run commands in one doc, with resume commands. Run sanity
first (`runpod_sanity_all_tasks.md`) and confirm `train/std` stays
bounded before paying for any of these.

| task     | timesteps | num-envs | wall (5090) | cost (5090 @ $0.99) |
|----------|-----------|----------|-------------|---------------------|
| peg      | 150 M     | 768      | ~42 hr      | ~$42                |
| grasp    | 70 M      | 768      | ~28 hr      | ~$28                |
| reorient | 200 M     | 768      | ~80 hr      | ~$80                |
| **total**|           |          | **~150 hr** | **~$150**           |

GPU: RTX 5090 is the recommended pod for all three (32 GB VRAM, fast,
predictable $/hr). 4090 also works at 512 envs but cost-neutral and slower.
H100 cuts wall time ~3× but costs more total. Below 24 GB VRAM won't fit
512+ envs.

## 1. Pod setup (same for any task)

CUDA 12.4+ image (RunPod PyTorch 2.4 / CUDA 12.4 templates). Verify after ssh:

```
nvidia-smi | head -3
# CUDA Version >= 12.4, Driver Version >= 550
```

Install + clone:

```
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd ~
git clone -b cleanup-dead-code https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd dexterous_hand
uv sync --extra mjx
mkdir -p runs

uv run python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]
```

Common env vars (set inside the tmux session before launching):

```
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
# omit WANDB_MODE if you want W&B logging; set =disabled to skip
```

## 2. Peg full (150 M)

```
tmux new-session -s peg
```

Inside:

```
cd ~/dexterous_hand
# (export the env vars from §1)

uv run python main.py train-peg-mjx \
    --num-envs 768 \
    --total-timesteps 150000000 \
    2>&1 | tee runs/peg_full_stdout.log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t peg`.

## 3. Grasp full (70 M)

```
tmux new-session -s grasp
```

Inside:

```
cd ~/dexterous_hand
# (export the env vars from §1)

uv run python main.py train-grasp-mjx \
    --num-envs 768 \
    --total-timesteps 70000000 \
    2>&1 | tee runs/grasp_full_stdout.log
```

## 4. Reorient full (200 M)

```
tmux new-session -s reorient
```

Inside:

```
cd ~/dexterous_hand
# (export the env vars from §1)

uv run python main.py train-reorient-mjx \
    --num-envs 768 \
    --total-timesteps 200000000 \
    2>&1 | tee runs/reorient_full_stdout.log
```

Curriculum stages are `[0 → 30°, 20M → 90°, 60M → 180°]` scaled by
`total / curriculum_reference_timesteps`. Config default reference is
200 M, so the 200M run uses the literal thresholds above. If you scale
total up (e.g. 300 M for more headroom at the 180° stage), the stages
auto-scale 1.5× unless you pin `--curriculum-reference-timesteps 200000000`.

## 5. Auto-shutdown watcher (optional, second tmux)

```
tmux new-session -s watcher
```

Inside:

```
while pgrep -f "main.py train-" > /dev/null; do sleep 60; done \
  && cp -rf ~/dexterous_hand/runs/. /workspace/runs/ \
  && runpodctl stop pod "$RUNPOD_POD_ID"
```

## 6. Resume from checkpoint

If a run hits its budget but nearly converges, use the resume CLI to
extend without paying the VecNormalize re-stabilization cost. All three
resume commands take the same args:

```
uv run python main.py resume-{peg,grasp,reorient}-mjx \
    --model-path runs/<run_name>/final_model.zip \
    --vec-normalize-path runs/<run_name>/vec_normalize.pkl \
    --additional-timesteps 50000000 \
    --num-envs 768 \
    --seed 42
```

`--additional-timesteps` is *additional*, not cumulative. Output writes
to `runs/<run_name>_resumed/` unless `--output-dir` is set.

Checkpoint paths also work for `--model-path` /
`--vec-normalize-path` — checkpoints save both the model and
VecNormalize state (`save_vecnormalize=True` is enabled in all training
scripts).

## 7. Pass criteria

**peg** (after 150 M):
- `train/metrics/stage` reaches 4.0 (sustained insertion)
- `train/metrics/insertion_depth` > 0.01 sustained
- `eval/success_rate` > 0.10
- `train/std` stayed bounded throughout (regression test on the ClampedActor fix)

**grasp** (after 70 M):
- `train/metrics/object_height` ≥ 0.448 sustained (the geometric plateau)
- `train/metrics/success_hold_steps` > 10 mean (out of `success_hold_steps=20`)
- `train/reward/success` firing regularly
- `train/std` stayed bounded throughout

**reorient** (after 200 M):
- `train/metrics/angular_distance` ≤ 1.0 rad sustained
- `train/metrics/success_steps` ≥ 0.20
- `eval/success_rate` > 0.10
- `train/metrics/num_finger_contacts` ≥ 1.5 (cube held, not dropped)
- `train/std` stayed bounded throughout

If any task nearly clears the bar but doesn't fully converge, resume per
§6 with +50 M timesteps.

## 8. Common ops

- W&B: `wandb login <key>` before launching, and unset `WANDB_MODE`. Each
  run writes to project `dexterous-hand`.
- OOM at 768 envs: drop to 512. Peg is the most VRAM-hungry of the three.
- CUDA fragmentation (`CUDA_ERROR_ILLEGAL_ADDRESS`): confirm
  `XLA_PYTHON_CLIENT_PREALLOCATE=true` and `MEM_FRACTION=0.7` are
  exported. If it still happens, drop `--num-envs` to 512.
- Throughput: 5090 hits ~300–500 fps depending on task. Peg is fastest
  (~470 fps), reorient ~500 fps, grasp ~310 fps (shortest episodes →
  more reset overhead).
