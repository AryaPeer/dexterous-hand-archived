# Dexterous Hand — Full Implementation Review (2026-07-14)

**Scope.** Every tracked source file was read end-to-end (envs, rewards, scene builders, MJX vec-env, policy, curriculum, training/resume scripts, tests, diagnostic scripts, hand XML). Existing repo markdown docs were deliberately ignored per request. Claims below marked **[verified]** were confirmed by running CPU MuJoCo 3.7.0 against the actual compiled scenes on this machine; claims marked **[inspection]** follow from code reading; claims marked **[literature]** were checked against primary sources (MuJoCo/MJX docs, MuJoCo Menagerie, MuJoCo Playground, sbx source).

**Overall verdict.** The implementation is in good shape: the reward machinery is unusually well-instrumented, the historical exploit classes (false insertion metric, success re-arm farming, grasp-and-sit annuities) are genuinely fixed and regression-tested, the winnability of both tasks is proven under real physics, and the pre-flight tooling (reward-gradient gates, CPU↔MJX parity, milestone compute-savers) is better than most published codebases. The problems found are concentrated in three places: **(1) the compiled scenes silently lost the Menagerie contact model**, **(2) the MJX solver runs at CPU-defaults and is likely burning ~1–2 orders of magnitude of GPU throughput**, and **(3) a handful of reward/reset details that are exploitable or off-spec but not fatal.** Nothing found invalidates the round-17 design; several things should be fixed before the next paid run.

---

## 1. System summary (as-built)

| | Grasp | Peg-in-hole |
|---|---|---|
| Scene | Menagerie Shadow Hand on x/y/z slides over table, 7 cm 100 g cube | Same hand, 7.6 cm × 1.6 cm 20 g capsule peg, square-bore tube, entrance 8 cm above table |
| Actions | 23 (20 hand + 3 slides), position servos, EMA-smoothed (α=0.2), 25 Hz (2 ms × 20 substeps) | same |
| Obs | 108 (state-based) | 134 (adds hole pose, insertion depth, wall forces, stage) |
| Reward | reach/grasp/side-ratio/lift/hold + once-latched +250, no success terminal | staged reach/grasp/opposition/axis/lift/align/place/depth + per-step ungated `complete`, no success terminal |
| Termination | fall/launch only; horizon 200 | fall only; horizon 500 |
| RL | SBX PPO, 768 envs, VecNormalize(obs+rew), clamped log-std [−3, 0], ent 1e-3, KL-adaptive LR (target 0.05) | same, γ=0.997, clearance + p_pre_grasped curriculum |
| DR | mass/friction/gain ±30/30/15 % per env | disabled (deliberate) |

The "make the solved state the highest-paying per-step state, remove the success terminal" design adopted in round-17 is internally coherent, and `scripts/check_reward_gradient.py` proves the monotone ordering table < lift < hover < settled and settled > 1.5× the sub-threshold farm state. That is the correct discipline for annuity-style (absolute-level) shaping; the residual risks found are exactly the terms that fall outside that proof (see §3.3, §3.4).

---

## 2. Findings verified by experiment

### F1 — The Menagerie contact model is silently dropped at scene build **[verified — HIGH]**

`assets/shadow_hand/right_hand.xml` carries `<option cone="elliptic" impratio="10"/>` — the contact tuning the Menagerie authors ship specifically for stable grasping. `MjSpec.attach()` does **not** merge the child spec's `<option>`, and neither scene builder sets these on the parent spec. Measured on the compiled scenes:

```
grasp/peg scene:  cone=0 (pyramidal)  impratio=1.0   solver=Newton  iterations=100  ls_iterations=50  integrator=Euler
standalone hand:  cone=1 (elliptic)   impratio=10.0
```

Impact: friction impedance at impratio=1 is 10× softer than the asset intends — grasped objects creep/slip more easily. Several historical failure symptoms ("grip collapses", nfc decay) plausibly had this as a contributing factor. Two honest options exist:

- **Match the asset** (elliptic, impratio 10): highest grip fidelity; MJX supports elliptic cones with the Newton solver (verify under the pinned jax/mujoco-mjx on the pod), at some GPU cost.
- **Match MJX practice** (pyramidal, impratio 1): this is what MuJoCo Playground's LEAP-hand MJX scene effectively runs [literature], so the *current accidental* configuration is also a *defensible deliberate* one.

The bug is not which value is set — it's that the choice was made silently by an API quirk, diverges from the CPU-validated asset, and is guarded by no test. Whichever is chosen: set it explicitly in both builders and add a compiled-model assertion test (`model.opt.cone/impratio`). If the values change, re-run `tune_grip_bias`, the geometry tests, the renders, and the parity check — every grip proof was measured under pyramidal/1.

### F2 — Four of five hole-wall touch sensors are dead **[verified — HIGH]**

MuJoCo `touch` sensors only sum contacts whose point lies **inside the site volume**. The wall sites are 2.5 mm spheres at the center of 6 cm-tall walls. Measured with a peg physically pressed against the +x wall:

```
sensor values: px=0.0 nx=0.0 py=0.0 ny=0.0 bottom=0.195
actual contacts: ('hole_wall_px', 0.0056 N, pos z=0.48) ('hole_bottom', 0.195 N)
```

The wall contact exists; the sensor reads zero. Only `hole_bottom` (whose site is a 12 mm sphere covering the plate) works. Consequences:

- `force_penalty` (−0.01·(F−15)²) can effectively never fire on wall jams — the jamming-detection channel the round-16 design intended is inert.
- 4 of the 6 contact-force observation dims are constants (and after VecNormalize + obs noise they become pure noise channels).

Fix: make each wall site a **box** site matching its wall geom's dimensions (sites support box type), or drop the sensor path and compute per-wall normal forces from contacts. Add a liveness test: wedge a peg against each wall, assert sensor > 0.

### F3 — MJX runs with CPU-default solver settings; likely the dominant throughput loss **[verified locally + literature — HIGH]**

Compiled scenes carry `solver=Newton, iterations=100, ls_iterations=50, integrator=Euler, timestep=0.002` with `frame_skip=20`. On CPU this is cheap (the Newton solver early-exits on tolerance). On MJX/GPU, iterations run to the configured count — the MJX docs' first performance instruction is to reduce `iterations`/`ls_iterations` "to just low enough that the simulation remains stable", and to cull contact pairs. Compare the flagship MJX hand task (MuJoCo Playground LEAP cube reorient) [literature]:

| | this repo | Playground LEAP (MJX) |
|---|---|---|
| timestep | 0.002 | 0.01 |
| substeps / ctrl | 20 | 5 |
| iterations | 100 | 5 |
| ls_iterations | 50 | 8 |
| eulerdamp | on | disabled |
| contact culling | none (implicit pairs) | `max_contact_points=30`, `max_geom_pairs=12` |
| fingertip collision geoms | **meshes** (f_distal_pst) | primitives |

Per control step that is ~20×100 = 2000 solver iterations vs ~5×5 = 25 — nearly two orders of magnitude more solver work, on top of mesh-mesh/mesh-primitive fingertip collisions (MJX docs: keep convex meshes small or replace with primitives; `maxhullvert` helps). If the measured ~316 steps/s at 768 envs stands, a 150 M-step peg run is ~5.5 GPU-days; Playground-style settings put comparable scenes in the 10⁴–10⁵ steps/s range, i.e. hours. Recommended sequence (each step re-validated by `mjx_parity_check.py` + renders, which is exactly what that script exists for):

1. `iterations=8, ls_iterations=8` (then try 4/8), keep Newton.
2. Capsule-ize the two fingertip mesh collision geoms (or `maxhullvert=32`).
3. `timestep 0.002→0.004` with `frame_skip 20→10` (25 Hz preserved); try `integrator=implicitfast` (plays well with the hand's joint damping/armature) and `eulerdamp=disable` if staying on Euler.
4. Explicit `<pair>`/culling for hand–object contacts if still contact-bound.
5. Secondary, host-side: per-step `infos` construction builds 768 Python dicts × ~25 numpy scalars, and `RewardInfoLoggerCallback` appends ~2.4 M floats per rollout; aggregate metrics on-device (means over envs) and emit one small dict per step instead.

Architectural note: the SB3/SBX bridge forces a device→host round-trip per control step. It was a sane choice for tooling continuity, but a Brax/Playground-style fully on-device PPO would remove the ceiling entirely if a rewrite is ever justified. Fixing 1–4 first is much cheaper and probably sufficient.

### F4 — Pre-grasped resets start by commanding the grip open **[verified — MEDIUM]**

At reset `smoothed_actions = 0`, and a zero smoothed action maps to actuator ctrl-range **midpoints**, not to the settle grip. Measured gap: holding the settle grip requires actions ≈ +0.72…+1.0 (e.g. FFJ0 grip ctrl 2.80 vs zero-action ctrl 1.57; THJ5 1.047 vs 0.0). With α=0.2 smoothing, even a policy outputting +1 needs ~10 steps (0.4 s) to re-reach the grip command; meanwhile the servos are told to relax.

Measured consequence: under zero actions the peg **sags but is retained** (z 0.483→0.461 over 25 steps), so this is a systematic perturbation rather than a broken curriculum — but at σ≈1 exploration some episodes will fumble the spawn grip and eat the (now armed) −20 drop penalty for reasons outside the policy's control at t=0.

Fix: at reset, initialize `smoothed_actions` (and `previous_actions`, which feeds the obs) to the action-space inverse of the settle ctrl — `a = 2(c−lo)/(hi−lo)−1` of `grip_ctrl` for pre-grasped spawns (and of the zero-vector-consistent hover for table spawns, which is already ~consistent by design for grasp's slide_z). Note the same inconsistency in miniature: the peg scene's slide_z spawns at qpos 0 but its zero-action ctrl is the range midpoint +0.025, so the hand drifts up 2.5 cm at episode start; grasp got this right by setting `SLIDE_Z_INIT` = ctrl midpoint.

### F5 — Peg reset noise leaves `slide_z` randomized ±5 cm; comment claims otherwise **[verified — LOW/MEDIUM]**

`peg_env._reset_single` zeroes noise on indices 0:2 with a comment saying "qpos[0]=slide_x, qpos[1]=slide_y are linear (meters)" — but index 2 (`slide_z`) is equally linear and keeps the full ±0.05 m noise. Grasp's env zeroes 0:3. At the −5 cm extreme the fingers start in table contact (measured 1.4 mm penetration, 2 contacts); the settle then pops them out. ~17 % of episodes start below the knuckle-contact height. Zero the noise on index 2 (or bound it positive). One-line fix.

---

## 3. Reward-design analysis

### 3.1 Magnitude audit (defaults, weights applied)

Peg, per step: reach ≤0.2 · grasp ≤3.0 · opposition ≤1 · axis_in_grip ≤1 · **lift ≤20** · align ≤2 · place ≤8 · **depth ≤30 (settled ≈22.7)** · **complete ≤250·gates (settled ≈189)** · insertion_drive 15·v_z · idle −0.3/−0.1 · drop −20 once. Gripped-lifted hover ≈27–34/step; settled-in-bore ≈215–220/step. The ordering the design depends on holds with wide margin — consistent with the `check_reward_gradient` gates.

Grasp, per step: reaching ≤0.42 · grasping ≤2.5 · side_ratio ≤1 · lifting ≤6 · holding ≤4.4 · success +250 once-latched. Steady held ≈14/step vs yo-yo ≈11/step: holding wins, matching the round-17 latch analysis and its regression test.

### 3.2 What is now *correct* and worth preserving

- **No success terminal + highest-paying absorbing state** (both tasks): removes termination-farming by construction. This mirrors robosuite/Adroit horizon-based conventions and is the right pairing for annuity-style shaping.
- **Once-per-episode success latch** (grasp): correctly kills the lift/drop cycle; the ManiSkill/robosuite comparison in the comment is accurate, as is the observation that OpenAI's re-firing bonus is sound only because their goal changes (Dactyl's +5 fires per *new* goal).
- **Insertion metric containment** (lateral + axial window + pedestal): the three-layer fix is sound and each layer is independently tested. `get_insertion_depth_jax`'s deepest-point formula (half·|cosθ|+radius) and the sign(0) horizontal-peg edge case are handled correctly.
- **Ungated `complete`**: paying completion without finger contact is required by the release endgame and is regression-tested.
- **Truncation bootstrapping**: `TimeLimit.truncated` + `terminal_observation` only on timeout, physical failures as true terminals — exactly SB3's contract. Many public MJX bridges get this wrong; this one doesn't.

### 3.3 `insertion_drive` is structurally unsound (pumpable) **[inspection — MEDIUM]**

`insertion_drive = gates · max(−v_z, 0) · 5` pays downward peg velocity near the hole while gripped and never charges the matching ascent. Over a closed bob cycle the net is strictly positive (≈ +15·v_z per descending step at weight 3) — a textbook non-potential shaping term (Ng et al., 1999: only *potential differences* are cycle-free). A hovering policy that oscillates z above the bore collects it indefinitely. It predates `place`, which now covers the same gradient gap without the loophole. Recommendation: delete it, or convert to a potential difference (pay Φ(s')−Φ(s) with Φ = −k·‖tip − engaged_target‖ — which is nearly what `place` already is). Historical support: OpenAI Dactyl's dense term was a distance *delta* for exactly this reason. Re-run `check_reward_gradient` after removal (gates 1–4 do not depend on it).

### 3.4 The reward-optimal endgame is probably *gripped* insertion, not release **[inspection — MEDIUM]**

Because hand geoms don't collide with the hole walls (`contype/conaffinity` split, §5.2), fingers can occupy wall space. Kinematics check from the repo's own audit numbers: peg tip can be servoed to z≈0.407 before knuckles hit the table; success depth needs tip ≈0.435 — reachable **while gripped**. A gripped, fully-inserted peg keeps the grasp/axis/lift annuities (~+14 with lift's step bonus at the settled height) *on top of* depth+complete, out-paying the released settled state ≈231 vs ≈217 per step. So the policy is mildly incentivized to keep fingers clamped through the tube walls rather than perform the engaged release the design (and the `place` term's story) intends. This is not a correctness bug — the insertion metric stays honest, `is_success` fires — but:

- if the *demo* should show a release, either enable hand↔wall collision (making gripped insertion physically impossible, as the design comments already assume it is) and re-prove winnability, or accept clip-through visuals;
- if clip-through is acceptable, note that the "release is the only physical way" comments in `peg_env.py`/`peg_reward.py` are not true of the sim as built, and the `complete`-ungating rationale ("fingers cannot fit") is doing less work than stated.

### 3.5 Smaller reward notes **[inspection — LOW]**

- **Grasp double-counts side_ratio**: it appears inside `grasping` (0.3+0.7·side_ratio) *and* as the standalone `opposition`-weighted term. Harmless but muddles ablation reads; fold into one term or rename the info key.
- **Holding's speed gate caps at 0.73**: σ(20·(0.05−v)) = 0.731 at v=0, so a perfectly still held cube gets 73 % of `holding`. If "holding pays comparably to lifting" is the intent, use k≈100 (σ(5)=0.993) or center the gate lower. (The function's own default is 100; the config overrides to 20.)
- **Reach uses fingertip→object-center distance**, so its floor is the object half-size (cube: tanh(5·0.035) ⇒ max ≈0.83). Fine — just don't expect 1.0 in logs.
- **VecNormalize details**: defaults apply `clip_reward=10` (won't bind here given return-scale normalization, but worth knowing it exists) and `gamma=0.99` for the return running-std, mismatching PPO's 0.995/0.997 — pass `gamma=config.gamma` for consistency. Also note the known non-stationarity: when the first insertions appear, the return std inflates and all pre-insertion shaping shrinks in normalized units. The milestone gates read *raw* env infos, so they are immune — good.
- **Stale comment**: grasp reward's side_ratio comment says cube y∈[−0.05,+0.05]; the spawn is now y∈[−0.03,+0.03].
- `tests/test_rewards.py` passes 22-dim actions to the peg reward (env emits 23); only Σa² is computed so nothing breaks, but sync it.

---

## 4. RL stack review

- **PPO hyperparameters** are mainstream for this scale (768 envs × 128 steps = 98 k-sample rollouts, 4096 minibatch, 10 epochs, γ 0.995/0.997, λ 0.95, ELU 3×256). DeXtreme/RL-Games run shorter horizons with more envs, but nothing here is off-distribution.
- **`target_kl=0.05` in sbx is a KL-adaptive LR** (verified in sbx source: `KLAdaptiveLR.update(approx_kl)` then LR override), i.e. the RL-Games/DeXtreme mechanism (they target 0.016). The round-13/14 comments describe this correctly. 0.05 is loose; if policy churn appears late in runs, 0.02–0.03 is the literature range.
- **ClampedActor** (σ∈[0.05,1.0], init at ceiling) is a sound guard against the σ-runaway failure mode of clip-to-box Gaussians; pairing with ent_coef=1e-3 (vs Playground LEAP's 1e-2 cited in the comment) is reasonable given the clamp floor. Note σ init 1.0 over a [−1,1] box clips ~32 % of samples per dim at bounds early on — standard, but expect boundary-biased early exploration (Fujita & Maeda 2018).
- **Curriculum machinery** is well built (resume-aware stage fast-forward, p-only changes avoid re-jit, clearance changes rebuild + re-jit + reset). Two small warts [inspection]: (a) at a clearance switch the envs hard-reset but SB3's `_last_obs`/`episode_starts` don't know — one transition per env crosses the discontinuity with stale obs and a GAE bootstrap across the reset; (b) `MjxVecEnv.reset()` derives env keys from `fold_in(master_key, 0)` every call, so each full reset (training start + each clearance change) replays the identical spawn sequence. Both are cheap to fix (dirty-flag → return fresh obs via a proper reset path; fold in a reset counter).
- **First-rollout p_pre_grasped**: env default is p=0; the callback sets stage-0 p=1.0 *after* `_setup_learn` has already reset envs, so the first episode wave is table-spawned. Cosmetic, but it skews the first gate window's metrics.
- **DR** multiplies `body_mass` without touching `body_inertia` (inertia no longer matches the mass it was computed from) and scales `actuator_gainprm[0]` without `biasprm` (turning a position servo into one with a systematic ~±15 % position scale error — defensible as "gain error" DR, but make it a documented choice). Compared to Dactyl/DeXtreme the DR menu is minimal (no latency/observation-delay/damping randomization) — fine for a sim-only project, insufficient if sim-to-real ever becomes a goal.
- **No deterministic eval**: success-rate comes from exploration rollouts. A periodic σ=0 eval (even 1 rollout per few M steps) would make the milestone gates read policy skill instead of exploration-contaminated skill. Cheap via a second small env batch.
- **Resume scripts** rebuild config from *defaults* + CLI (num_envs/seed) — any non-default reward/scene config in the original run silently reverts on resume. Serialize the config into the run dir and reload it.

---

## 5. Physics & scene review

### 5.1 Hand asset provenance **[verified]**

`right_hand.xml` is Menagerie's Shadow Hand with one local modification: `plastic_collision` gained `contype="1" conaffinity="0"` (upstream: `group="3"` only). Effect: **all hand self-collision is disabled** (measured: 0 hand-hand contacts at full grip bias), and the two upstream `<exclude>` elements are now dead weight. This is a legitimate MJX perf tactic, but it lets the thumb pass through fingers during aggressive grips. Make it a documented decision; if self-collision is ever restored, use explicit fingertip-pair culling to keep MJX cost bounded.

### 5.2 Collision-bit matrix **[verified]**

hand(1,0) · table/floor(1,1) · cube(1,1) · peg(3,3) · walls(2,2): hand↔table ✓, hand↔peg ✓, peg↔walls ✓, peg↔table ✓, hand↔walls ✗ (intended, enables in-tube grips; see §3.4 for the incentive side-effect), walls↔table ✗ (intended), hand↔hand ✗ (§5.1).

### 5.3 Other scene notes

- **Square bore, round peg**: walls form a 24 mm square around a 16 mm capsule; "clearance 4 mm" is face clearance, corner clearance ≈9 mm. Fine for training; just keep the semantics in mind when quoting tightness vs Factory/IndustReal (whose bores are round and clearances sub-mm).
- **Peg↔bore `<pair>` friction (μ=0.2)** matching machined-part practice (Factory/IndustReal assets) with the Whitney jamming citation is correct and empirically load-bearing (fraction 0.55 wedge at μ=1 → 0.757 at μ=0.2). Note MJX support for explicit pairs is exercised only on the pod — the parity script covers it; keep that in the pre-flight.
- **Hole pose is never randomized** (`hole_offset` fixed at (0,0)); with hole pos/quat in the obs, the policy can memorize the station and ignore those dims. Fine for a fixed-station demo; if generalization is wanted, per-env `body_pos` override through the existing `_apply_dr` model-replace path is the cheap route (Factory randomizes socket pose per episode).
- **Dead obs dims**: hole pos (3) + hole quat (4) constants; 4 dead wall-force dims (F2). After VecNormalize+noise these become noise inputs. Harmless, but trimming or fixing them is free signal-to-noise.
- **Obs noise** (σ=0.005) is applied to *every* dim including previous_actions and stage — papers noise physical measurements only; minor.
- Table mass ≈10 kg of the scene's 13.9 kg total is a static body — irrelevant to dynamics, just don't read "hand ≈14 kg" from `body_mass` sums (hand ≈3.9 kg).

---

## 6. Literature cross-check

| Design choice | This repo | Field practice | Assessment |
|---|---|---|---|
| Control rate | 25 Hz, EMA α=0.2 | Dactyl 12 Hz (relative targets), DeXtreme 30 Hz, Playground LEAP 20 Hz | In range. EMA filter instead of action-rate penalty is a valid smoothness mechanism (cf. CAPS-style penalties as the alternative). |
| Success bonus semantics | Once-latched (grasp), per-step gated annuity (peg), no success terminal | robosuite/Adroit: horizon-based, no success terminal; ManiSkill: capped bonuses; Dactyl re-fires only on goal change | Matches; the repo's own comment history documents this correctly. |
| Peg shaping | Keypoint distance to **engaged release** pose (2-keypoint tanh) | Factory (Narang '22): keypoint distance; IndustReal (Tang '23): engagement distinction, SDF rewards, curriculum lowering initial engagement | Correct adaptation; the engaged-vs-inserted target choice (wall-press local minimum avoidance) is a genuine insight consistent with IndustReal's engagement definition. p_pre_grasped decay ≈ IndustReal's sampling-based curriculum / reverse curriculum (Florensa '17) applied to grasp acquisition. |
| Velocity-based drive term | `insertion_drive ∝ max(−v_z,0)` | Dactyl uses potential *differences*; Ng '99 forbids cycle-positive shaping | Off-spec; pumpable (§3.3). Delete or make potential-based. |
| Contact gating of lift | Hard n≥2 touch gate | robosuite grasp checks (contact-based); Isaac dexterity mostly ungated | Fine; touch sensors verified working for fingertips (and on-pod parity re-checks under MJX). |
| KL-adaptive LR | target 0.05 | RL-Games/DeXtreme default 0.016 | Mechanism identical (verified in sbx source); consider tightening if late-run churn. |
| DR breadth | mass/friction/gain, obs noise | Dactyl/DeXtreme add latency, damping, gravity, adr | Adequate for sim-only; document the inertia and servo-bias caveats (§4). |
| Solver/step config | 2 ms × 20, Newton 100/50 | Playground MJX: 10 ms × 5, Newton 5/8, contact culling | Main perf gap (F3). |
| Engine/trainer split | SBX PPO (host) + MJX (device) | Playground/Brax: fully on-device | Known ceiling; acceptable if F3 fixes land. |

One deliberate divergence worth blessing explicitly: **grasp lift_target=0.10 with slide_z** restores a real pick-up in the Adroit-relocate sense, and the geometry test enforcing "51 % of sampled grips can hold 20 cm+" is a stronger winnability guarantee than most published grasp envs ship with.

---

## 7. What is notably good

- The **failure-driven test suite**: every historical exploit has a named regression test that re-states the mechanism in its docstring. `test_geometry.py` proves task winnability under real physics rather than asserting config arithmetic.
- **`mjx_parity_check.py`** converting the CPU-proof/GPU-train solver split "from a doubt into a measurement" is exactly right, and rare.
- **`check_reward_gradient.py`** pins the monotone per-step ordering of the winning trajectory — the correct pre-flight for annuity-style shaping.
- **Milestone gates** with SKIP-on-unknown-keys and always-resumable checkpoints are a disciplined compute-saver design.
- Comment quality: the code documents *why* (with measurements) at nearly every decision point. The few places comments have drifted from code are listed in §3.5/F5.

---

## 8. Prioritized recommendations

**P0 — before the next paid run**
1. Decide and pin contact options explicitly in both scene builders (`spec.option.cone`, `impratio`) + regression test (F1). If values change, re-run grip tuning, geometry tests, renders, parity.
2. Solver/perf pass on the pod: iterations 8/8 → parity+render check → timestep 0.004 × skip 10 → fingertip capsules (F3). Expect order-of-magnitude gains; measure, don't assume.
3. Fix wall touch-sensor sites (box sites sized to walls) + liveness test; this also un-deadens 4 obs dims and re-arms `force_penalty` (F2).

**P1 — cheap correctness/robustness**
4. Zero `slide_z` reset noise in the peg env (F5, one line + comment fix).
5. Initialize `smoothed_actions`/`previous_actions` at reset to the inverse-mapped settle ctrl for pre-grasped spawns; align peg slide_z spawn with its ctrl midpoint (F4).
6. Remove `insertion_drive` or convert it to a potential difference (§3.3); re-run the gradient gates.
7. Decide the endgame stance on hand↔wall collision (§3.4): enable collision + re-prove, or keep clip-through and update the comments/rationale.
8. Pass `gamma=config.gamma` to VecNormalize; serialize/reload full config in resume scripts (§4).

**P2 — quality & future-proofing**
9. Deterministic eval rollouts feeding the gates; on-device info aggregation (one dict of means per step).
10. Fix curriculum-switch reset glitches + reset-key reuse (§4); zero-out or fix dead obs dims (§5.3).
11. If generalization matters later: per-env hole-pose randomization via the DR model-replace path; broaden DR (inertia-consistent mass scaling, servo bias, latency).
12. Consider a Brax/Playground-style on-device PPO only if post-F3 throughput still gates iteration speed.

---

## Appendix — reproduction

Verification scripts live in the session scratchpad (`verify_physics.py`, `verify_reset.py`); each rebuilds the real scenes via `build_scene`/`build_peg_scene` and needs only CPU MuJoCo. Key raw outputs are quoted inline above. Literature sources: MuJoCo MJX docs (performance section), MuJoCo Menagerie `shadow_hand/right_hand.xml` (upstream), MuJoCo Playground `leap_hand` MJX XML + reorient task, sbx `ppo.py` (KLAdaptiveLR), SB3 `VecNormalize` source; plus OpenAI Learning Dexterity (2018), DeXtreme (Handa et al. 2022), Factory (Narang et al. 2022), IndustReal (Tang et al. 2023), robosuite/Adroit/ManiSkill conventions, Ng et al. (1999) on potential-based shaping, Whitney (1982) on jamming.
