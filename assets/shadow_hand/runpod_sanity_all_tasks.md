# RunPod sanity — peg + reorient (post math-audit fixes)

One block on a fresh pod that runs peg 5M + reorient 5M back-to-back into a
single log. Validates three concrete bug fixes from the round-10 math audit:

1. **Reorient settle uses GRIP_BIAS ctrl** (was `zero_ctrl`, which opened
   the fingers during the 5-step settle and dropped the cube before the
   policy ever acted). Peg env had the same bug; both fixed.
2. **Peg scene has a `slide_z` actuator** (was X/Y only, so the hand could
   not lift the peg vertically at all — max physical lift from finger flex
   was 4.4cm against a lift_target of 10cm). Range [-0.10, +0.15], kp=8000,
   forcerange ±250N. New peg obs_size = 134, new action_size = 23.
3. **Reorient `orientation_contact_alpha = 0`** (was 3/7, which let an
   idle hand earn ~0.15/step of "orientation" reward while the cube sat on
   the floor — a do-nothing local minimum). Now orientation reward is 0
   at zero contacts.

Grasp is not in the bundle — its 5M sanity already passed and no relevant
code path changed.

~9 hr wallclock on a 4090 at 256 envs. ~$6 at $0.69/hr.

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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export WANDB_MODE=disabled

(
  echo "==================== train-peg-mjx ===================="
  uv run python main.py train-peg-mjx --num-envs 256 --total-timesteps 5000000 || true

  echo "==================== train-reorient-mjx ===================="
  uv run python main.py train-reorient-mjx \
      --num-envs 256 \
      --total-timesteps 5000000 \
      --curriculum-reference-timesteps 50000000 || true
) 2>&1 | tee runs/sanity_stdout.log
```

Detach with `Ctrl+b d`. Reattach with `tmux attach -t sanity`.

Notes:
- `|| true` means peg failure does not skip reorient.
- `--curriculum-reference-timesteps 50000000` on reorient locks the 5M
  sanity primarily in stage 0 (30° targets), enough to confirm the settle
  + alpha=0 fixes without testing 180° prematurely.
- Order is peg → reorient.

## 3b. Troubleshooting: `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` at env.reset()

1. `nvidia-smi` in a new shell. Kill anything holding VRAM.
2. Drop `--num-envs` to 128 on the failing task.
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
~1.5 on either task, the policy clamp isn't doing its job and task
metrics aren't worth reading.

**Both tasks:**

| metric                            | bar                                |
|-----------------------------------|------------------------------------|
| `train/std`                       | stays in [0.05, ~1.1], never > 1.5 |
| `train/metrics/nan_rate`          | < 0.01                             |

**Peg (5M) — slide_z + GRIP_BIAS settle test:**

The curriculum at 5M with reference=100M compresses stage starts to
(0, 400k, 800k, 1.2M, 1.6M). By 1.6M the policy is at max difficulty
(clearance=1mm, p_pre_grasped=0.2). Measure final-window (last 1M)
rolling means:

| metric                                          | bar                                      |
|-------------------------------------------------|------------------------------------------|
| `train/metrics/peg_height`                      | rising > initial + 0.04 (slide_z used)   |
| `train/metrics/stage`                           | ≥ 2.0 sustained (lift gate fires)        |
| `train/metrics/num_finger_contacts`             | ≥ 2.0 sustained                          |
| `train/metrics/insertion_depth`                 | > 0.001 mean by 5M                       |
| `train/reward/insertion_drive`                  | > 0 occurring (4-gate finally activating)|
| `eval/mean_reward`                              | trending up, > 500 by 5M (vs 282 peak)   |

If `peg_height` stays flat at initial after 1M, slide_z exists but the
policy hasn't discovered it — possible causes: lift weight too low, or
the policy needs more exploration time. If still flat at 5M, **dig into
the actuator more before scaling.**

**Reorient (5M, locked stage 0):**

| metric                                            | bar                                |
|---------------------------------------------------|------------------------------------|
| `rollout/ep_len_mean`                             | = 400 (no early termination)       |
| `train/metrics/num_finger_contacts`               | ≥ 1.5 sustained by 1M              |
| `train/reward/cube_drop`                          | trending toward 0 (cube held)      |
| `train/metrics/angular_distance`                  | trending DOWN, not drifting up     |
| `train/metrics/success_steps`                     | > 0.1 by 5M                        |

If `nfc` stays < 0.5 and `cube_drop` stays at -10/step, the settle fix
didn't land — re-verify `_grip_ctrl` was actually applied during settle.

## 6. Cost

| pod      | rate     | wall    | cost  |
|----------|----------|---------|-------|
| RTX 4090 | $0.69/hr | ~9 hr   | ~$6   |
| RTX 5090 | $0.99/hr | ~5 hr   | ~$5   |

## 7. After sanity passes

See `runpod_full_runs.md` for the full-run commands. **Do not resume from
any peg checkpoint that pre-dates these fixes** — slide_z changed the
action space, so all old peg checkpoints are incompatible. Reorient
checkpoints from prior runs are still compatible (no action-dim change)
but the env behavior is different now, so a fresh start is recommended.
