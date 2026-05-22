# RunPod sanity — peg only (round-12)

Single peg 5M run on a fresh pod. Round-11 (5M peg + 5M reorient) showed
both fixes (`build_grip_ctrl` tendon-aware, `lift_target=0.10→0.01`)
landed at the physics layer but neither task cleared its bar — peg still
grasp-and-sits at +4mm lift; reorient nfc stayed at 0.19 after 5M and the
no-floor-recovery reward shape can't be fixed without a multi-week
redesign. Reorient is dropped from the roadmap. This round-12 bundle
ships two peg-only changes targeting the grasp-and-sit local minimum.

The two changes are grounded in published prior work (robosuite Lift uses
a step-bonus lift reward, Adroit Relocate uses an explicit
"contact-without-lift" penalty):

1. **Step-bonus lift reward.** Replace
   `lift = min(lift_height/lift_target, 1.5) * contact_scale` with
   `lift = jnp.where(lift_height > 0.005, 1.0, 0.0) +
   min(lift_height/0.05, 1.5) * contact_scale`. The step term flips on
   the moment the peg clears 5mm; the proportional term then continues
   pulling toward 7.5cm. Round-11 stayed at 4mm lift forever because
   the proportional term alone has zero value at zero lift, and PPO
   couldn't find the gradient. The step bonus gives an immediate
   discontinuous reward jump the moment the policy raises the peg, which
   PPO can credit-assign back to whatever action pulled `slide_z`
   upward.
2. **`idle_stage1_penalty`** (new config field, mirror of existing
   `idle_stage0_penalty`). Fires when `nfc ≥ 2 AND peg_z < initial +
   0.02m AND stage == 1`, capped after `idle_grace_steps`. Makes
   grasp-and-sit actively costly instead of merely unrewarded. Robosuite
   Lift uses the same pattern to escape the contact-without-lift
   plateau.

Grasp is not in the bundle — its 5M sanity already passed and no
relevant code path changed.

~4.5 hr wallclock on a 4090 at 256 envs. ~$3 at $0.69/hr.

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

## 3. Run

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

uv run python main.py train-peg-mjx \
    --num-envs 256 \
    --total-timesteps 5000000 \
    2>&1 | tee runs/sanity_stdout.log
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

| metric                            | bar                                |
|-----------------------------------|------------------------------------|
| `train/std`                       | stays in [0.05, ~1.1], never > 1.5 |
| `train/metrics/nan_rate`          | < 0.01                             |

**Peg (5M) — step-bonus lift + idle_stage1 penalty:**

The curriculum at 5M with reference=100M compresses stage starts to
(0, 400k, 800k, 1.2M, 1.6M). By 1.6M the policy is at max difficulty
(clearance=1mm, p_pre_grasped=0.2). Measure final-window (last 1M)
rolling means:

| metric                                          | round-11 observed       | round-12 bar                              |
|-------------------------------------------------|-------------------------|-------------------------------------------|
| `train/metrics/peg_height`                      | +4mm (flat)             | rising > initial + 0.02m by 5M            |
| `train/metrics/stage`                           | 1.0 (stuck)             | ≥ 2.0 sustained (lift gate fires)         |
| `train/metrics/num_finger_contacts`             | 3.86 ✓                  | ≥ 2.0 sustained                           |
| `train/reward/lift` (raw, post-weight)          | 1.2e-4                  | ≥ 5.0 sustained once lifted               |
| `train/reward/idle_stage1_penalty` (new column) | n/a                     | non-zero early, decays to ~0 by 2M        |
| `train/metrics/insertion_depth`                 | 1.4e-4 (flat)           | > 0.001 mean by 5M                        |
| `eval/mean_reward`                              | 909                     | trending up, > 1500 by 5M                 |

If peg_height is *still* flat at +4mm after 1M with both fixes in, the
problem is below the reward layer (slide_z action not getting through,
or curriculum stage 0 not actually putting the peg in the gripper).
Debug by dumping per-env action-component means in `peg_env._step_single`
to see whether the policy is even commanding upward slide_z.

## 6. Cost

| pod      | rate     | wall    | cost  |
|----------|----------|---------|-------|
| RTX 4090 | $0.69/hr | ~4.5 hr | ~$3   |
| RTX 5090 | $0.99/hr | ~2.5 hr | ~$2.5 |

## 7. After sanity passes

See `runpod_full_runs.md` for the full-run commands. **Do not resume
from any peg checkpoint that pre-dates round-12** — the reward formula
changed, so resumed VecNormalize stats would carry the wrong
distribution.
