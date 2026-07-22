# RunPod round 18 — post contact-fix sanity, floor re-derivation, full chain

Read this instead of `runpod_sanity_all_tasks.md` for the first run after
the object-filtered contact fix. That doc's numbers were all measured
while the grasp policy was farming table contact, so every gate floor and
every "expected" metric in it is void.

## 0. What changed and why old numbers mean nothing

`get_finger_touch_from_sensors` fired on contact with **any** geom. From
Apr 20 (`e5b833d`, "Refactor") until now, `n_contacts` counted fingers
touching the **table**. The 70M grasp policy pressed five fingers flat on
the table 7cm from the cube and collected `grasping = 0.9866` (2.47/step,
90% of its total reward) for 200 steps without ever touching the cube.

Contacts are now built from `mjx_data.contact` filtered to the object
geom (`get_finger_object_contact_mask`). Under the fix, that same
checkpoint scores `grasping = 0.0000`.

Consequences for this run:

- **Do not resume any checkpoint.** Both `final_model.zip` files are
  optimised for a reward that no longer means the same thing. Start cold.
- **Every gate floor is invalid.** Stage A below re-derives them.
- Expect *lower* absolute `grasping`/`num_finger_contacts` than any
  historical log. That is the fix working, not a regression.

## 1. Local pre-flight (free — do not skip)

```
uv run --no-sync ruff check . && uv run --no-sync mypy dexterous_hand scripts main.py
uv run --no-sync pytest -q
uv run --no-sync python scripts/check_reward_gradient.py   # expect PEG: PASS / GRASP: PASS
uv run --no-sync python scripts/mjx_parity_check.py --backend cpu
```

`tests/test_geometry.py::test_table_press_with_distant_cube_counts_zero_grasp_contacts`
is the regression guard for this whole round. It presses the hand on the
table with the cube 30cm away, asserts the **touch sensors do fire** (4 of
them), and asserts the object-filtered mask counts **0**. If it ever goes
green-by-vacuum (sensors stop firing), it fails loudly rather than
silently proving nothing.

## 2. Pod setup (paste once on a fresh pod)

```
apt-get update && apt-get install -y tmux git
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd ~
git clone https://github.com/AryaPeer/dexterous-hand-archived.git dexterous_hand
cd dexterous_hand
uv sync --extra mjx
mkdir -p runs /workspace/runs
```

`source $HOME/.local/bin/env` is required — without it `uv` is not on
PATH and every later command fails.

## 3. Pod validation (~15 min, before spending on training)

```
uv run python -c "import jax; print(jax.devices())"          # [CudaDevice(id=0)]
uv run python -c "import jax; x=jax.numpy.ones((4,4)); print((x@x).sum())"   # 64.0
echo "EXIT: $?"
uv run python scripts/check_reward_gradient.py               # PEG: PASS / GRASP: PASS
uv run python scripts/mjx_parity_check.py                    # PARITY OK
uv run pytest tests/test_grasp_env.py tests/test_peg_env.py -q
```

A segfault (`EXIT: 139`) on the matmul means the host driver disagrees
with the pinned jaxlib. Destroy the pod and redeploy — do not debug it.

### 3b. Contact-culling headroom (NEW — this fix depends on it)

The reward now reads `mjx_data.contact` directly, so anything MJX culls
is invisible to it. `mjx_max_contact_points` / `mjx_max_geom_pairs` are
96 / 256 (grasp, raised in `903d449`) and 48 / 384 (peg, unchanged).

```
uv run python - <<'EOF'
import numpy as np, jax.numpy as jnp
from dexterous_hand.config import MjxGraspTrainConfig
from dexterous_hand.envs.grasp_env import ShadowHandGraspMjxEnv
c = MjxGraspTrainConfig(); c.num_envs = 64
env = ShadowHandGraspMjxEnv.from_config(c)
env.reset()
mx = 0
for _ in range(300):
    a = np.random.uniform(-1, 1, (64, env.action_space.shape[0])).astype(np.float32)
    env.step_async(a); env.step_wait()
    d = env._mjx_data_batch
    mx = max(mx, int((np.asarray(d.contact.dist) < 0).sum(axis=-1).max()))
print("max simultaneous active contacts per env:", mx)
EOF
```

Measured 2026-07-22 on a 4090, 64 envs x 300 random-action steps:

| task | true max object contacts | max active | old cap | verdict |
|---|---|---|---|---|
| grasp | 57 | 83 | 48 | was clipping, now 96/256 |
| peg | 10 | 22 | 48 | never clipped, unchanged |

`mjx_max_contact_points` is a per-collision-function cap, NOT the array
size, so a total active count above it is normal. The tell for clipping
is the per-object count landing *exactly* on the cap: grasp read exactly
48, and raising the cap moved it to 57. Peg reads 10 against a cap of 48
and needs nothing.

To re-derive after any scene change, run the probe at a generous cap
(192/768) to find the true maximum, then set the cap to ~1.5x it. Never
read the number at the default cap and conclude "fine" — a clipped value
looks like a stable measurement.

## 4. Stage A — gate-free sanity to re-derive floors (~2h)

The peg run died last night because its floors came from a 5M sanity
whose curriculum was **compressed** by `scale_stage_starts`: that sanity
spent its last 3.4M steps in the final stage (peg mostly untouched on the
table, `axis_align` trivially ~1.0), while the 150M run at 10M was still
in stage 0. Incomparable regimes.

`--curriculum-schedule-timesteps` fixes this: it scales the curriculum as
if the run were that long, so a 10M sanity sits at the **same stage** the
long run will be in at 10M.

```
tmux new-session -s sanity
cd ~/dexterous_hand
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export WANDB_MODE=disabled

uv run python main.py train-grasp-mjx \
    --num-envs 768 --total-timesteps 10000000 \
    --curriculum-schedule-timesteps 70000000 --no-gate \
    2>&1 | tee runs/sanityA_grasp.log

uv run python main.py train-peg-mjx \
    --num-envs 768 --total-timesteps 10000000 \
    --curriculum-schedule-timesteps 150000000 --no-gate \
    2>&1 | tee runs/sanityA_peg.log
```

Verify in the first rollout table: `std` ~0.368 (NOT 1.001 — that was the
log_std clamp freeze) and a `[Curriculum]` line at step 0.

Sanity-check the schedule printed at startup: grasp stages must be
`[0, 10000000, 25000000, 40000000, 55000000]` and peg
`[0, 12000000, 24000000, 36000000, 48000000]`. If you see small numbers
like `1428571`, the flag did not take and the floors you derive will be
worthless.

## 5. Re-derive the floors

```
uv run python - <<'EOF'
import csv, sys
for task, run in (("grasp","grasp_mjx_768env_42"), ("peg","peg_mjx_768env_42")):
    rows = list(csv.DictReader(open(f"runs/{run}/logs/progress.csv")))
    rows = [r for r in rows if r.get("time/total_timesteps")]
    tail = rows[-15:]
    def m(k):
        v = [float(r[k]) for r in tail if r.get(k)]
        return sum(v)/len(v) if v else float("nan")
    keys = {
    "grasp": ["train/metrics/num_finger_contacts","train/reward/grasping",
              "train/reward/lifting","train/metrics/success_hold_steps",
              "train/metrics/object_height"],
    "peg":   ["train/metrics/num_finger_contacts","train/reward/axis_in_grip",
              "train/metrics/stage","train/metrics/peg_height",
              "train/metrics/insertion_depth"],
    }[task]
    print(f"\n=== {task} @ {tail[-1]['time/total_timesteps']} steps ===")
    for k in keys:
        print(f"  {k:42s} = {m(k):.4f}   suggested floor {m(k)*0.55:.4f}")
EOF
```

Rules for turning those into gate floors:

- **Floor at ~55% of the observed value.** Tight floors are what killed
  last night's peg run, which was still improving monotonically
  (`axis_align` 0.120 -> 0.707) when the gate stopped it.
- **Never gate on a metric that inaction maximises.** `metrics/axis_align`
  is raw `|dot(peg_axis, hole_axis)|` with no contact gate — an untouched
  peg standing upright scores 1.0. Gate `reward/axis_in_grip` instead
  (already changed). Same reasoning rules out bare `metrics/peg_height`
  above resting (0.438) and `metrics/object_height` above resting (0.4349).
- **Keep exact-zero checks as-is.** `insertion_depth > 0.001` and
  `success_hold_steps > 0.01` are honest "does it ever happen" checks.

Edit `GRASP_GATES` in `scripts/training/train_grasp.py` and `PEG_GATES` in
`scripts/training/train_peg.py`, then commit before the long run so the
config in `runs/*/config.json` matches what actually ran.

## 6. Stage B — full chain (~19.5h on a 4090)

Timings from last night at **steady-state** fps (grasp 7509, peg 2527).
Use steady-state, not the `time/fps` column: that column is cumulative and
still carries the one-time ~760s XLA compile. Peg's cumulative fps read
2121 because the gate killed it at 10M, so the compile was amortised over
only 10M steps — extrapolating from it overstates a 150M run by ~19%.

| leg | steps | time |
|---|---|---|
| grasp | 70M | ~2.8h |
| peg | 150M | ~16.7h |

Peg costs ~3x per step because it runs 20 physics substeps at
`dt=0.002` against grasp's 8 at `dt=0.005` (same 0.04s control period)
plus 3x the collision budget. `dt` cannot be raised — the pinch grip
fails at `dt >= 0.004`.

If you want this inside one sleep, drop peg to 100M (~11h) — it had not
plateaued at 10M, so the extra 50M is speculative anyway.

Clear the previous run's artifacts first, on the pod **and** on the
volume. `/workspace/runs` is never pruned, so a stale 70M checkpoint set
sits alongside the new one and the downloaded zip becomes ambiguous about
which files belong to which run:

```
rm -rf ~/dexterous_hand/runs /workspace/runs/* && mkdir -p ~/dexterous_hand/runs
```

```
tmux new-session -s train
cd ~/dexterous_hand
export CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export WANDB_MODE=disabled

uv run python main.py train-grasp-mjx --num-envs 768 --total-timesteps 70000000 \
    2>&1 | tee runs/grasp_full_stdout.log ; \
uv run python main.py train-peg-mjx --num-envs 768 --total-timesteps 150000000 \
    2>&1 | tee runs/peg_full_stdout.log ; \
uv run python scripts/eval_policy.py --task grasp \
    --model-path runs/grasp_mjx_768env_42/final_model.zip \
    --vec-normalize-path runs/grasp_mjx_768env_42/vec_normalize.pkl \
    --p-pre-grasped 0.0 2>&1 | tee runs/eval_grasp.log ; \
uv run python scripts/eval_policy.py --task peg \
    --model-path runs/peg_mjx_768env_42/final_model.zip \
    --vec-normalize-path runs/peg_mjx_768env_42/vec_normalize.pkl \
    --p-pre-grasped 0.0 2>&1 | tee runs/eval_peg.log ; \
touch runs/ALL_DONE
```

Semicolons (not `&&`) so peg still runs if a grasp gate fires.

`--p-pre-grasped 0.0` is now the default, but pass it explicitly: last
night's eval inherited `p_pre_grasped` from curriculum stage 0 via
`from_config`, so half the grasp eval episodes spawned already gripping
the cube and it reported "50% success" when the true rate was 0%.

## 7. Watcher (second tmux — sync + auto-stop)

```
tmux new-session -s watcher
```

```
while [ ! -f ~/dexterous_hand/runs/ALL_DONE ]; do
  cp -rf ~/dexterous_hand/runs/. /workspace/runs/ 2>/dev/null; sleep 600
done
cp -rf ~/dexterous_hand/runs/. /workspace/runs/
sync
runpodctl stop pod "$RUNPOD_POD_ID" || kill -9 1
```

The `|| kill -9 1` is load-bearing. The pod-injected `RUNPOD_API_KEY` is
not accepted for account operations — `runpodctl get pod` and
`runpodctl config --apiKey` both return `Unauthorized` — so the stop
command fails and the loop would otherwise exit leaving the pod billing.
Killing PID 1 exits the container, which ends GPU billing. The pod shows
`Exited` rather than `Stopped`; that is cosmetic, and `/workspace` is a
network volume so results survive.

Confirm it is actually running before you sleep — a blank pane looks
identical to an idle shell:

```
pgrep -af "sleep 600"     # must print a PID
```

Without a live watcher there is no `/workspace` sync and no auto-stop,
and the pod bills until you notice.

## 8. Morning review

```
tmux capture-pane -pt train | tail -40
grep -n "MILESTONE GATE" -A 12 runs/*_full_stdout.log
cat runs/eval_grasp.log runs/eval_peg.log
```

Read in this order:

1. **`[eval] p_pre_grasped=0.00`** in both eval logs. If it says anything
   else, the success rate is contaminated — discard it.
2. **Grasp `reward/lifting` and `metrics/success_hold_steps`.** Exact 0.0
   means it still never lifts.
3. **Divide grasp metrics by the stage's `p_pre_grasped`.** If
   `lifting / p` is flat across stages (it was 0.857, 0.875, 0.850 last
   night at p = 0.3, 0.2, 0.1), the policy only succeeds when handed the
   cube and has learned nothing about grasping.
4. **`metrics/object_speed`.** Near 0 with high contacts means it is
   holding the cube still rather than lifting it.
5. **Any gate block.** Check whether the metric was still *improving*
   before concluding the run deserved to die.

## 9. Pitfalls this round exists to prevent

- Eval/render inheriting the curriculum's easiest stage from
  `from_config` (fixed; both now default to 0.0 and print the value).
- Gate floors derived from a curriculum-compressed sanity (fixed by
  `--curriculum-schedule-timesteps`).
- Gating on a metric that inaction maximises (fixed for peg).
- Contact detection that ignores *what* is being touched (fixed; guarded
  by a test that would have failed today).
- A 48-test suite passing while the policy earned 98.7% of max `grasping`
  from 7cm away. Green tests did not mean a working reward — check the
  render and the per-step diagnostics too.
