# Review fixes — change summary

Branch: `cleanup-dead-code` (working tree, not committed). WandB logging intentionally left as-is.

**Verification:** `ruff` clean · `mypy` clean (18 files) · 25 tests pass + 2 skipped (mjx) · `check_reward_gradient.py` → PEG **PASS** / GRASP **PASS**.

---

## 1. Peg task — made physically winnable (the headline fix)

The peg-in-hole success bar (0.70 insertion fraction) was **geometrically unreachable**: the hand's middle/ring knuckles bottom out on the table and cap `slide_z` descent, so a closed grip only reached ~0.68. Confirmed three independent ways (physics drop, scene rebuild, grip-descend sim). The config's "~88% insertion" justification was impossible (it silently used `peg_length=0.06` instead of the real `0.076`).

| file | change |
|---|---|
| `dexterous_hand/config.py` | `hole_top_above_table` **0.06 → 0.08**. Measured achievable insertion rises **0.679 → 0.942** (+0.24 margin over the 0.70 threshold). Rewrote the false "88%" comment with real measured numbers + the knuckle-cap explanation. |
| `scripts/check_reward_gradient.py` | Pre-flight now imports **production** geometry (`peg_length=0.076`, `table_height=0.4`, real hole entrance) instead of phantom `pl=0.06`/`table_h=0.82`. Removed dead `_print_components`. |
| `tests/test_geometry.py` (new) | Reachability guard: cheap "success depth fits in tube" invariant + a slow grip-descend sim asserting achievable insertion ≥ `success_threshold + 0.05`. Would have caught the round-16 blocker. |
| `dexterous_hand/utils/mjx_helpers.py` | `get_insertion_depth_jax` now uses the capsule's **geometric lowest point** (`depth_of_center + half·|cos(tilt)| + radius`), removing the `sign(0)=0` artifact that reported spurious depth for a tilted/horizontal peg. |

---

## 2. Reward bugs

| file | change |
|---|---|
| `dexterous_hand/rewards/grasp_reward.py` | **Killed grasp-and-sit subsidy.** Removed the `+0.04` `height_gate` offset (now centers at `lift_target`). At-rest `holding` dropped ~5.8/step → ~0.6/step (pre-flight at-rest total 8.74 → 3.55); lifting still earns full. |
| `dexterous_hand/config.py` | `hold_height_smoothness_k` **50 → 200** so the holding gate is ~0 below `lift_target`. |
| `dexterous_hand/rewards/peg_reward.py` | **Lift no longer dominates insertion.** Capped proportional lift at **1.0×** (was 1.5× — no reward for over-lifting). |
| `dexterous_hand/config.py` | Peg `lift` weight **15 → 10** so depth (~30/step max) out-rewards lift (~20/step max). ⚠️ *the precise lift-vs-depth balance is the key remaining tuning knob — validate with a short sanity run.* |
| `grasp_reward.py` + `peg_reward.py` | Reward `info` dicts made **consistent** — all components logged raw (pre-weight); previously the idle penalties were logged post-weight while everything else was raw. |

---

## 3. RL training pipeline

| file | change |
|---|---|
| `dexterous_hand/envs/mjx_vec_env.py` | **Success is now a true terminal** (no value bootstrap) instead of a truncation — fixes value overestimation at the goal. Only timeouts bootstrap now. |
| `dexterous_hand/envs/peg_env.py` | Curriculum **no longer recompiles the physics step** on `p_pre_grasped`-only stage changes (only `_batched_reset` closes over it). Saves minutes of XLA stall per stage. |
| `dexterous_hand/curriculum/callbacks.py` | **Resume jumps straight to the correct curriculum stage** from `num_timesteps` instead of detouring through stage-0 (which rebuilt to the easiest clearance + reset all envs + recompiled at every boundary). |
| `dexterous_hand/config.py` | `ent_coef` **0 → 1e-3** in both MJX configs to actively maintain exploration (clamp still prevents runaway). Documented the deliberate peg-DR disable. |

---

## 4. Dead code & tooling cleanup

| item | change |
|---|---|
| `dexterous_hand/utils/quaternion.py` | **Deleted** — fully orphaned (reorient-task leftover, zero importers). |
| `dexterous_hand/config.py` | Removed dead `TrainConfig` + `PegTrainConfig` (SAC-era, only referenced by their own tests). |
| `tests/test_config.py` | Removed the two dead-config tests; updated `hole_top_above_table`/`hold_height_smoothness_k` assertions; swapped the instantiation list to the MJX configs. |
| `scene_builder.py` / `peg_scene_builder.py` | Removed 8 unused `NameMap`/`PegNameMap` fields (`fingertip_geom_ids`, `n_actuators`, `hand_actuator_ids`, `ctrl_ranges`, `table_geom_id`, `hole_pos`, `hole_quat`, `hole_wall_geom_ids`) + `SensorMap.n_sensors` and their construction/lookups. Dropped the now-unused `numpy` import + `config` param. |
| `README.md` | Removed the duplicated `## Scope` section. |
| `pyproject.toml` | `vulture` `min_confidence` **70 → 60** with a framework-callback whitelist (70 hid every real positive); added `tensorflow_probability` to the mypy import override. |
| `clamped_actor.py` + scripts | Fixed mypy annotation; `ruff --fix` across the repo (package import-sort + 13 script nits → repo now clean). |

---

## Caveats / not done

- **Reward-weight rebalance (lift vs depth)** is the one change that needs a short sanity run to validate — everything else is structurally or empirically confirmed.
- **`clamped_actor.py` `get_std()` stub + dead discrete-action fields** left untouched — SBX-interface-adjacent and can't be validated without the `mjx` extra installed locally. Clean up on a GPU box where SBX is present.
- **WandB scalar-logging gap** not addressed (per request).
- Nothing committed yet.
