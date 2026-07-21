# RunPod overnight blind chain — grasp + peg full runs

Runs both full training jobs back to back, evals each checkpoint, and stops
the pod automatically when done, so you can walk away. Reward correctness,
throughput config, and contact-culling values are already committed
(`main`) — this doc is only the pod-side execution steps.

## 0. Prereqs

- A network volume attached at `/workspace` (crash safety — the watcher
  syncs `runs/` there every 10 min, so a pod death costs at most 10 min of
  logs/checkpoints, not the whole night).
- `git pull` on the pod so it has the latest `main`.

## 1. Setup (paste once on a fresh pod)

```bash
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd ~
git clone https://github.com/AryaPeer/Dexterous-Hand.git dexterous_hand
cd dexterous_hand
uv sync --extra mjx
mkdir -p runs /workspace/runs
```

## 2. Host validation (do this on every fresh pod, no exceptions)

Force the CUDA backend for the whole session — some hosts otherwise
silently fall back to CPU and every check below would still "pass" for
the wrong reason:

```bash
export JAX_PLATFORMS=cuda
```

```bash
uv run python -c "import jax; x=jax.numpy.ones((512,512)); print((x@x).sum())"
# MUST print 134217728.0 (512^3). If it errors, segfaults, or the process
# exits with no number printed, this host's driver/CUDA combo is broken —
# do NOT continue, redeploy on a different host.
```

```bash
uv run pytest tests/test_grasp_env.py tests/test_peg_env.py -q
# expect: 7 passed. Timing varies a lot by host load — anywhere from
# ~17 to ~25+ min is normal (XLA compiling the fused vmapped step is
# CPU-bound and sensitive to noisy neighbors; GPU-Util reading 0% in
# nvidia-smi during this is NOT a sign of failure). Only worry if the
# process's %CPU (ps -eo pid,pcpu,etime,cmd | grep python) drops to ~0
# with no new output for 10+ minutes — that's actually stuck.
```

```bash
uv run python scripts/mjx_parity_check.py
# expect: grasp final_lift >= 150mm, peg settled/min_hold >= 0.73/0.70,
# ending in "PARITY OK". Ignore: "Failed to import warp" (optional
# backend, unused), the "hlo_lexer ... Failed to parse int literal"
# line, and "RuntimeWarning: overflow encountered in cast" (benign
# float64->float32 canonicalization noise) — none of these affect the
# result.
```

Both green → proceed. If either fails for real (not the benign noise
above), stop and fix it before spending pod-hours blind.

## 3. Launch the chain — tmux 1

```bash
tmux new-session -s train
```

Paste as one block:

```bash
cd ~/dexterous_hand
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
rm -f runs/ALL_DONE
uv run python main.py train-grasp-mjx --num-envs 768 2>&1 | tee runs/grasp_full_stdout.log ; \
uv run python main.py train-peg-mjx   --num-envs 768 2>&1 | tee runs/peg_full_stdout.log ; \
uv run python scripts/eval_policy.py --task grasp \
  --model-path runs/grasp_mjx_768env_42/final_model.zip \
  --vec-normalize-path runs/grasp_mjx_768env_42/vec_normalize.pkl -n 64 \
  2>&1 | tee runs/eval_grasp.log ; \
uv run python scripts/eval_policy.py --task peg \
  --model-path runs/peg_mjx_768env_42/final_model.zip \
  --vec-normalize-path runs/peg_mjx_768env_42/vec_normalize.pkl -n 64 \
  2>&1 | tee runs/eval_peg.log ; \
touch runs/ALL_DONE
```

`Ctrl+b d` to detach. The `;` chaining means peg still runs even if the
grasp 10M/30M milestone gate stops the grasp job early — a gate stop
exits cleanly and still saves `final_model.zip`, so both evals work
regardless of whether a gate fired.

Expected wall-clock on a 4090 at ~10k steps/s: grasp 70M ≈ 2 h, peg 150M
≈ 4 h, evals a few minutes each. Budget more on a slow/shared host.

## 4. Watcher — tmux 2

```bash
tmux new-session -s watcher
```

```bash
while [ ! -f ~/dexterous_hand/runs/ALL_DONE ]; do
  cp -rf ~/dexterous_hand/runs/. /workspace/runs/ 2>/dev/null
  sleep 600
done
cp -rf ~/dexterous_hand/runs/. /workspace/runs/
runpodctl stop pod "$RUNPOD_POD_ID"
```

`Ctrl+b d` to detach. Sentinel-file gated (not `pgrep main.py`) so it
can't misfire in the gap between the grasp and peg jobs.

## 5. Next morning — what to bring back

From `/workspace/runs/`:

- `grasp_mjx_768env_42/logs/progress.csv`
- `peg_mjx_768env_42/logs/progress.csv`
- `eval_grasp.log`, `eval_peg.log`
- Tail of `grasp_full_stdout.log` and `peg_full_stdout.log` — especially
  any `===== MILESTONE GATE @ ... =====` block, which means a gate
  fired and the run stopped early (still resumable via
  `resume-{grasp,peg}-mjx` from the last ~500k checkpoint if the stop
  looks premature).
