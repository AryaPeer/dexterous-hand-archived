# RunPod sanity — peg + reorient (post math-audit fixes)

One block on a fresh pod that runs peg 5M + reorient 5M back-to-back into a
single log. The round-10 sanity bundle finished cleanly but failed its pass
criteria — reorient nfc stuck at 0.135 from iter 1 (cube fell at reset),
peg grasped but never lifted (peg_height +5mm over 5M, bar +4cm). This
round-11 bundle validates two further fixes, grounded in published prior
work (robosuite Lift, Adroit Relocate, Menagerie Shadow Hand keyframes):

1. **`build_grip_ctrl` now drives the four tendon actuators**
   (`rh_A_FFJ0 / MFJ0 / RFJ0 / LFJ0`). The previous version skipped any
   actuator whose trntype wasn't `mjTRN_JOINT`, so the four coupled distal
   joint pairs (FFJ1+FFJ2, etc.) drifted open under gravity during the
   5-step settle and the pre-grasped cube fell. The new path sums the
   bias_map across each tendon's wrapped joints and writes the sum as the
   actuator's ctrl (MuJoCo interprets it as desired tendon length). With
   GRIP_BIAS this puts ctrl ≈ 2.8 on each of the four tendons; local audit
   (`scripts/audit_grip_tendon.py`) confirms the cube is held within
   ±0.75cm of spawn across 200 settle steps.
2. **Peg `lift_target` reduced from 0.10m to 0.01m.** Lift reward formula
   is unchanged: `lift = min(lift_height / lift_target, 1.5) * contact_scale`.
   At the policy's actual operating point (2-5mm lift, nfc=3.8), this
   sharpens the gradient 10× and lets the lift contribution saturate near
   the max once the peg clears 1.5cm — matching the lift-vs-grasp reward
   ratios used in robosuite Lift (lift=9× grasp) and Adroit Relocate
   (lift=10× reach). Stage-2 gate (`peg_z > initial + 2cm`) is unchanged.
3. **Reorient `orientation_contact_alpha = 0`** (round-10 fix, retained).
   Was 3/7; an idle hand could earn ~0.15/step of "orientation" reward
   while the cube sat on the floor.

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

**Peg (5M) — sharpened lift gradient (`lift_target=0.01`):**

The curriculum at 5M with reference=100M compresses stage starts to
(0, 400k, 800k, 1.2M, 1.6M). By 1.6M the policy is at max difficulty
(clearance=1mm, p_pre_grasped=0.2). Measure final-window (last 1M)
rolling means:

| metric                                          | round-10 observed       | round-11 bar                              |
|-------------------------------------------------|-------------------------|-------------------------------------------|
| `train/metrics/peg_height`                      | +5mm (flat)             | rising > initial + 0.02m by 5M            |
| `train/metrics/stage`                           | 1.0 (stuck)             | ≥ 2.0 sustained (lift gate fires)         |
| `train/metrics/num_finger_contacts`             | 3.8 ✓                   | ≥ 2.0 sustained                           |
| `train/metrics/insertion_depth`                 | 1e-4 (decreasing)       | > 0.001 mean by 5M                        |
| `train/reward/insertion_drive`                  | 4e-5 (decreasing)       | > 0 occurring                             |
| `train/reward/lift` (raw)                       | 4e-5                    | ≥ 0.2 sustained once lifted               |
| `eval/mean_reward`                              | 882                     | trending up, > 1500 by 5M                 |

If `peg_height` still stays flat at initial after 1M with the new
`lift_target=0.01`, the next step is adding `idle_stage1_penalty`
(symmetric with `idle_stage0_penalty` but gated on nfc≥2 AND peg_z <
initial+2cm) or restructuring to robosuite-style max-over-stages.

**Reorient (5M, locked stage 0) — tendon-aware settle:**

| metric                                            | round-10 observed | round-11 bar                          |
|---------------------------------------------------|-------------------|---------------------------------------|
| `rollout/ep_len_mean`                             | 400 ✓             | = 400                                 |
| `train/metrics/num_finger_contacts` (iter 1)      | 0.135             | ≥ 1.0 (cube held from t=0)            |
| `train/metrics/num_finger_contacts` (by 1M)       | 0.05-0.25         | ≥ 1.5 sustained                       |
| `train/reward/cube_drop`                          | -13 to -17/step   | trending toward 0                     |
| `train/metrics/angular_distance`                  | 1.85 (drifting)   | trending DOWN                         |
| `train/metrics/success_steps`                     | 0                 | > 0.1 by 5M                           |

If `nfc` at iter 1 is still < 0.5, the tendon-length math in
`build_grip_ctrl` is wrong — re-run `scripts/audit_grip_tendon.py` to
verify each of the four tendon actuators reports `ctrl ≈ 2.8` (sum of
FFJ1+FFJ2 GRIP_BIAS), and that the audit's "cube held" line passes.

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
