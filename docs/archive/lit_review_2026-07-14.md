# Reward & Task Design vs. the Literature — 2026-07-14

Scope: `main @ c39e75a` (post engaged-release redesign + grasp slide_z restore). This doc maps every
load-bearing design decision in `grasp` and `peg` onto published RL-manipulation systems and their open
codebases, and recasts the three pre-flight defects (see `audit` trail / review chat) in literature terms.
Companion to `audit_2026-06-10.md`.

TL;DR: the architecture is squarely inside the published playbook — staged tanh-shaped rewards
(robosuite/ManiSkill), keypoint place term (Factory/IndustReal), no success terminal (IndustReal/robosuite),
reverse-style curriculum (Florensa/IndustReal SBC). The three defects found are all instances of failure
modes the literature has already named: a containment metric one axis short of ManiSkill's insertion check,
a curriculum that violates the reverse-curriculum invariant ("being closer to the goal must never pay
less"), and a re-armable bonus of the exact kind that produced DeepMind's famous Lego-flip hack.

---

## 1. Decision-by-decision comparison

| Our decision | Closest published analogue | Match / delta |
|---|---|---|
| `reach = 1 − tanh(5·d)` (both tasks) | ManiSkill PickCube: `reaching = 1 − tanh(5 · tcp_to_obj_dist)` — same kernel, same constant ([code](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/pick_cube.py)) | Exact match |
| Staged reach → grasp → lift with hard contact gate (`n_contacts ≥ 2`) | robosuite Lift: reaching ∈ [0,1] → grasp {0, 0.25} → lift gated on grasp ([code](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/lift.py)); ManiSkill gates placement on `is_grasped`; Popov et al. composite grasp-then-stack reward ([arXiv:1704.03073](https://arxiv.org/abs/1704.03073)) | Match. Gating later stages on grasp state is the standard defense against premature-stage farming |
| `place` = 2-keypoint distance to target pose, `1 − tanh(4·kd)` | Factory: keypoint-distance reward with squashing kernel, nut-bolt ([paper](https://arxiv.org/pdf/2205.03532), [code](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/factory/factory_task_nut_bolt_place.py)); IndustReal upgraded keypoints → SDF-based dense reward ([paper](https://arxiv.org/pdf/2305.17110)) | Match (Factory-style). If place-shaping ever proves too coarse, IndustReal's SDF reward is the published denser upgrade |
| Insertion success = depth fraction + lateral containment gate | ManiSkill PegInsertionSide `has_peg_inserted`: peg head transformed into **hole frame**, requires axial bound AND \|y\| ≤ r AND \|z\| ≤ r ([code](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/peg_insertion_side.py)); IndustReal `check_plug_inserted_in_socket` = below-opening depth + keypoint clustering | **Delta = defect #1.** Ours checks radial containment only — no axial window — which is exactly the missing third axis of ManiSkill's check, and is what admits the under-tube fraction-1.0 spoof |
| No success terminal; settled-in-bore is the highest-paying absorbing state (~217/step) | IndustReal: **no early success termination**, terminal-step bonuses only; robosuite TwoArmPegInHole: success is a reward signal, episode runs full horizon ([code](https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/environments/manipulation/two_arm_peg_in_hole.py)); OpenAI Learning Dexterity: success ⇒ *new goal*, episode continues ([paper](https://arxiv.org/abs/1808.00177)) | Match in kind; ours is the most aggressive version (large per-step stream). Sound, given the stream is unspoofable — which is why defect #1 must be fixed first |
| Success requires N-step hold (grasp 25, peg 10) | ManiSkill success requires object at goal AND robot static (qvel < 0.2); IndustReal engagement/success evaluated after settling | Match — "stable at goal", not "touched goal" |
| Grasp one-shot `+250` success bonus | ManiSkill +5 (once, capped), robosuite sparse 1.0·scale, OpenAI +5 per *new* goal | **Delta = defect #3.** Every published bonus is either once-per-goal or the goal changes; ours re-arms on the *same* goal after a drop → yo-yo farms it (+24% over steady hold) |
| Drop penalty −20 | OpenAI Learning Dexterity: **−20** on drop ([paper](https://arxiv.org/abs/1808.00177)) | Exact match (same constant) |
| `p_pre_grasped` curriculum 1.0 → 0.2 + clearance 4→1 mm | Florensa reverse curriculum: start at/near goal, expand start states outward ([CoRL'17](https://arxiv.org/abs/1707.05300)); IndustReal SBC: full initial-state range from step 0, lower bound raised with success rate ([docs](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/industreal.md)) | Match in intent. **Delta = defect #2**: the lift term's spawn-height reference makes near-goal (pre-grasped) starts pay *less* at the engaged pose than far ones — the one invariant a reverse curriculum must not break. Also: our easiest start is "in-hand, high"; SBC/Florensa would include "already engaged" starts |
| Force penalty `−0.01·max(0, F−15)²` on bore walls | FORGE: force-threshold penalty during training, policy conditioned on max allowable force, snap-fit insertions ([paper](https://arxiv.org/abs/2408.04587)) | Match in kind (fixed threshold vs. FORGE's conditioned one) |
| Bore friction pair μ=0.2, peg keeps μ=1.0 vs fingers | Factory/IndustReal machined-part asset friction; Whitney's two-point jamming analysis (classic assembly theory) | Match; the measured 0.55-fraction wedge at μ=1.0 is textbook Whitney jamming |
| Elevated guide tube (entrance 8 cm above table) | robosuite NutAssembly/TwoArmPegInHole raised receptacles; mechanical-assembly practice: chamfers/funnels for passive alignment (RCC, Whitney) | Match; the tube is a funnel. **But** floating it created the under-tube slot (defect #1's physical half) — published receptacles are solid to the surface |
| Release-and-let-gravity-finish endgame | No direct RL precedent found. Closest: passive-alignment assembly theory (above); IndustReal's *engagement* concept (engaged ⇒ laterally captured) | Deliberate novelty, forced by hand-scale vs 12 mm bore. CPU proof (0.757 settle, ±4 mm tolerance) is the right kind of evidence; keep `test_geometry.py` + `mjx_parity_check.py` as the guard |
| EMA action smoothing (α=0.2) + `−2e−4·Σa²` | DeXtreme: action + delta-action + joint-velocity penalties ([paper](https://arxiv.org/abs/2210.13702)); MuJoCo Playground: action_rate/energy costs ([report](https://playground.mujoco.org/assets/playground_technical_report.pdf)) | Match (mechanical smoothing instead of penalizing Δa — fine, arguably stronger) |
| DR on for grasp, off for peg until baseline exists | IndustReal trains insert with *moderate* randomization; DeXtreme's ADR ramps DR only as competence grows | Match — staging DR after competence is the published pattern |
| 7 cm cube, 0.1 kg | MuJoCo Playground LEAP reorient: 7 cm cube | Exact match (nice) |

## 2. PPO configuration vs. the field

| Knob | Ours | Field |
|---|---|---|
| Parallel envs | 768 (RTX 5090, MJX) | IsaacGymEnvs Factory/IndustReal: 128–8192; Playground: ~8192 on A100. 768 is small-but-fine for a 5090; throughput is the binding constraint (~316 fps measured) |
| Unroll / horizon | 128 steps × 768 envs, batch 4096, 10 epochs | rl-games configs for Factory-class tasks use shorter horizons (16–32) with more envs; long-unroll + fewer envs is the SB3-style equivalent — same data budget per update, no known pathology |
| γ | 0.995 (grasp, 200-step) / 0.997 (peg, 500-step) | Matches horizon-scaled discounting practice (effective horizon ≈ episode) |
| ent_coef | 1e-3 + σ∈[0.05, 1.0] clamp, init at ceiling | Playground LEAP uses 1e-2; our clamp-at-ceiling + small bonus is a defensible conservative variant. If exploration stalls in sanity, 1e-2 is the literature value to try first |
| lr | 3e-4 constant + target_kl 0.05 | Standard; rl-games uses adaptive-kl lr for these tasks — target_kl early-stop is the SB3 equivalent |
| norm_obs / norm_reward | VecNormalize both | Universal (rl-games value/obs norm, Playground running stats). Note: the ~10× reward jump when `complete` is first reached is *expected*; don't panic-stop on `reward/total` discontinuities |

## 3. The three pre-flight defects, in literature terms

> **Status 2026-07-14 (later the same day): all three FIXED** (local working tree, uncommitted).
> Axial window in `get_insertion_depth_jax` + `hole_pedestal` geom + 2 under-tube metric poses and a
> slot-blocked test; lift-reference clamp in `peg_env._reset_single` + MJX-side regression test;
> sticky success latch in `grasp_reward` + once-per-episode test. Verified after the fix: exploit poses
> measure depth 0.000 (legit settle unchanged at 0.757), the pre-grasped chain equals the monotone
> table chain (26.5 < 30.9 < 38.7), yo-yo now loses to steady holding by 16.5%. CPU parity check
> re-passes untouched (grasp 235.6 mm, peg 0.757); 33/33 CPU tests green;
> `check_reward_gradient.py` PEG PASS / GRASP PASS. MJX-marked tests skip locally (no `mujoco-mjx`
> in the local venv) — run `pytest` + `mjx_parity_check.py --backend both` on the pod.

1. **Under-tube false insertion (CRITICAL).** A table-lying peg with one end slid into the 1.75 cm slot
   under the floating tube measures insertion fraction **1.000** with zero wall contact and pays ~58/step
   ungrasped (verified against the compiled scene + production reward). This is the canonical
   specification-gaming class — DeepMind's Lego agent flipping the block to satisfy a bottom-face-height
   reward ([Popov et al. 2017](https://arxiv.org/abs/1704.03073)) — reborn 13 months later through the one
   axis our containment check doesn't test and ManiSkill's does. *Fix:* axial gate
   (`lower_end depth ≤ hole_depth`; legit max 0.0495, exploit 0.072) + optionally a pedestal geom filling
   table→plate + a 4th pose in `test_insertion_depth_requires_lateral_containment`.
2. **Pre-grasped spawn inverts the endgame gradient (HIGH).** With `initial_peg_height` sampled at the
   in-hand spawn (~0.52), descending to the engaged pose zeroes the 20/step lift term:
   held-high 25.5 > hover 23.5 > engaged 18.7 (table spawns: 26.5 < 30.9 < 38.7 ✓). Florensa/SBC curricula
   work precisely because value flows backward from near-goal starts; a reward that pays near-goal starts
   *less* poisons that flow in 100% of episodes for the first ~12 M steps. *Fix:* clamp the reference to
   the table spawn height (`min(settled, table_h + half_len + r + 1 mm)`), which restores the monotone
   chain; then, per IndustReal SBC, consider an "already-engaged" spawn stage so the release→settle→
   complete payoff appears in rollouts from step 0.
3. **Grasp success re-arm (MODERATE).** `was_success_prev = is_success` is an edge detector, so
   drop → re-hold re-fires +250; lift-hold-drop cycling beats steady holding by +23.9%. OpenAI's +5
   re-fires only because the goal *changes*; ManiSkill/robosuite bonuses are once/capped. *Fix:* sticky
   per-episode latch (`state.was_success_prev | is_success`).

## 4. Residual risks the literature flags (no code change yet)

- **Thin success margin:** settled fraction 0.757 vs threshold 0.70 (+4.3 mm). IndustReal reports 88.6%
  insertion *with* SDF rewards and SBC; our margin leaves little room for MJX/CPU contact-solver deltas —
  keep `mjx_parity_check.py` (MJX side still unrun) as a hard pre-run gate.
- **Sideways-grip attractor** (round-16's axis_align 0.07): mitigations since (THJ5 grip bias, opposition
  index fix, `axis_in_grip`, pre-grasped starts) are plausible but unproven at scale; the 10 M
  `axis_align ≥ 0.70` gate is the right tripwire. DeXtreme/OpenAI solve this class with goal-conditioned
  rotation rewards — overkill here unless the gate trips twice.
- **30 M `axis_align ≥ 0.80` gate** may false-trip once table spawns mix in (toppled pegs drag the mean);
  re-derive floors from the first post-fix 5 M sanity, as already planned.

## 5. Source index

Papers: [Factory (RSS'22)](https://arxiv.org/pdf/2205.03532) · [IndustReal (RSS'23)](https://arxiv.org/pdf/2305.17110) ·
[FORGE](https://arxiv.org/abs/2408.04587) · [DeXtreme](https://arxiv.org/abs/2210.13702) ·
[OpenAI Learning Dexterity](https://arxiv.org/abs/1808.00177) · [Popov et al. Lego stacking](https://arxiv.org/abs/1704.03073) ·
[Florensa reverse curriculum](https://arxiv.org/abs/1707.05300) · [Meta-World](https://arxiv.org/pdf/1910.10897) ·
[MuJoCo Playground](https://arxiv.org/pdf/2502.08844)

Codebases: [ManiSkill tasks](https://github.com/haosulab/ManiSkill/tree/main/mani_skill/envs/tasks/tabletop) ·
[IsaacGymEnvs (Factory + IndustReal)](https://github.com/isaac-sim/IsaacGymEnvs) ·
[industreallib (real-robot side)](https://github.com/NVLabs/industreallib) ·
[robosuite manipulation envs](https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments/manipulation) ·
[Meta-World](https://github.com/Farama-Foundation/Metaworld) ·
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
