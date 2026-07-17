"""
Round-14 pre-flight: verify the reward gradient actually pushes the policy
toward the right behavior at the operating point we care about.

Both round-12 and round-13 cooked at full scale, and both failures were
visible in the reward design at sanity-relevant lift heights. This script
runs the production reward functions on synthetic states and asserts:

  PEG: marginal reward gain from lifting 0 -> 5mm must exceed marginal
       reward gain from improving grasp_quality. Otherwise the policy
       converges to "perfect grip, no lift" and never escapes.

  GRASP: marginal reward gain from lifting 0 -> 12mm (lift_target) must
         exceed gain from reaching/grasping plateau. Same logic.

If either check FAILS, the full run will fail at scale — kill any
planned pod spend before clicking go.
"""

import jax.numpy as jnp

from dexterous_hand.config import PegRewardConfig, PegSceneConfig, RewardConfig
from dexterous_hand.rewards.grasp_reward import grasp_reward, init_grasp_reward_state
from dexterous_hand.rewards.peg_reward import init_peg_reward_state, peg_reward


def check_peg() -> bool:
    cfg = PegRewardConfig()
    scene = PegSceneConfig()
    # Production geometry. Previously hardcoded pl=0.06/table_h=0.82, which did
    # not match the real env (peg_length=0.076, table_height=0.4) and let the
    # pre-flight pass against a phantom scene whose align/insertion gates were
    # active where production's are gated off.
    pl = scene.peg_half_length * 2.0 + scene.peg_radius * 2.0  # 0.076
    table_h = scene.table_height  # 0.4
    hole_z = table_h + scene.hole_top_above_table  # hole entrance
    # On-table grasp height: the grasp-and-sit operating point we must escape.
    initial_z = table_h + scene.peg_half_length + scene.peg_radius + 0.001

    def run(
        peg_z: float,
        peg_xy: tuple[float, float] = (0.0, 0.0),
        gripped: bool = True,
        insertion_depth: float = 0.0,
        stage: int = 1,
        steady_steps: int = 1,
    ) -> dict:
        """Evaluate the reward at a synthetic state. steady_steps > 1 re-runs
        the reward against its own state to ramp counters (hold sigmoid) to
        their steady per-step value."""
        state = init_peg_reward_state(initial_z)
        px, py = peg_xy
        if gripped:
            # 5 fingers actively contacting the peg with thumb opposition
            fp = jnp.array(
                [
                    [px + 0.005, py, peg_z],
                    [px - 0.005, py, peg_z],
                    [px - 0.005, py + 0.005, peg_z],
                    [px - 0.005, py - 0.005, peg_z],
                    [px - 0.005, py - 0.01, peg_z],
                ]
            )
            mask = jnp.array([True, True, True, True, True])
        else:
            # released: hand hovering above, no contacts
            fp = jnp.tile(jnp.array([px, py, peg_z + 0.08]), (5, 1))
            mask = jnp.array([False] * 5)
        info: dict = {}
        for _ in range(steady_steps):
            _, state, info = peg_reward(
                state=state,
                stage=jnp.asarray(stage),
                finger_positions=fp,
                peg_position=jnp.array([px, py, peg_z]),
                peg_axis=jnp.array([0.0, 0.0, 1.0]),
                hole_position=jnp.array([0.0, 0.0, hole_z]),
                hole_axis=jnp.array([0.0, 0.0, 1.0]),
                insertion_depth=jnp.asarray(insertion_depth),
                contact_force_magnitude=jnp.asarray(0.0),
                finger_contact_mask=mask,
                peg_height=jnp.asarray(peg_z),
                peg_linvel=jnp.zeros(3),
                actions=jnp.zeros(23),
                previous_actions=jnp.zeros(23),
                weights=cfg.weights,
                peg_length=pl,
                lift_target=cfg.lift_target,
                table_height=table_h,
                drop_penalty_value=cfg.drop_penalty,
                complete_bonus=cfg.complete_bonus,
                force_threshold=cfg.force_threshold,
                idle_stage0_penalty=cfg.idle_stage0_penalty,
                idle_stage1_penalty=cfg.idle_stage1_penalty,
                idle_stage1_min_contacts=cfg.idle_stage1_min_contacts,
                lift_step_threshold=cfg.lift_step_threshold,
                lateral_gate_k=cfg.lateral_gate_k,
                idle_stage_cutoff=cfg.idle_stage_cutoff,
                success_threshold=cfg.success_threshold,
                peg_hold_steps=cfg.peg_hold_steps,
                reach_tanh_k=cfg.reach_tanh_k,
                fingertip_weights=cfg.fingertip_weights,
                action_penalty_scale=cfg.action_penalty_scale,
                depth_reward_scale=cfg.depth_reward_scale,
                idle_grace_steps=cfg.idle_grace_steps,
                release_height=cfg.release_height,
                place_k=cfg.place_k,
            )
        return info

    print("\n=== PEG reward gradient (round-14) ===\n")
    info_sit = run(initial_z + 0.0)
    info_lift = run(initial_z + 0.006)  # just past lift_step_threshold = 5mm

    total_sit = float(info_sit["reward/total"])
    total_lift = float(info_lift["reward/total"])
    delta_total = total_lift - total_sit
    delta_lift_component = float(info_lift["reward/lift"]) - float(info_sit["reward/lift"])
    grasp_post_weight = float(info_sit["reward/grasp"])

    print("  at lift_height = 0mm (perfect grip, no lift):")
    print(f"    reward/total              = {total_sit:>8.4f}")
    print(f"    reward/grasp              = {grasp_post_weight:>8.4f}  (post-weight)")
    print(f"    reward/lift               = {float(info_sit['reward/lift']):>8.4f}")
    print()
    print("  at lift_height = 6mm (just past lift_step_threshold):")
    print(f"    reward/total              = {total_lift:>8.4f}")
    print(f"    reward/lift               = {float(info_lift['reward/lift']):>8.4f}")
    print()
    print(f"  delta_total (lifting 6mm)  = {delta_total:>+8.4f}")
    print(f"  delta_lift_component        = {delta_lift_component:>+8.4f}")
    print(f"  grasp out-rewards lift?     = {grasp_post_weight > delta_total}")
    print()

    # Bars:
    # 1. Lifting must produce a meaningfully positive delta vs sitting.
    # 2. That delta must exceed the grasp reward currently being banked
    #    (otherwise sitting-with-perfect-grasp wins on a per-step basis).
    bar_delta = 1.0  # lift step bonus is +1.0 post-weight when threshold cleared
    pass_delta = delta_total >= bar_delta
    pass_lift_vs_grasp = delta_total > grasp_post_weight

    print(f"  GATE 1: delta_total >= {bar_delta}     {'PASS' if pass_delta else 'FAIL'}")
    print(f"  GATE 2: lifting beats grasp          {'PASS' if pass_lift_vs_grasp else 'FAIL'}")

    # --- 2026-07-14: full winning-trajectory monotonicity + release dominance.
    # States along the intended demo: gripped on table (at spawn radius) ->
    # gripped lifted -> gripped at the release pose over the bore -> RELEASED,
    # settled in the tube (fraction 0.757, no contacts). Each must out-pay the
    # previous per-step, and the settled state must dominate BOTH the hover
    # and the "grip the peg partially inserted just below threshold" farm
    # state — with the old contact-gated complete + success terminal, farming
    # beat completing and releasing forfeited the completion payment.
    spawn_r = scene.spawn_min_radius
    s_table = run(initial_z, peg_xy=(spawn_r, 0.0))
    s_lift = run(initial_z + 0.05, peg_xy=(spawn_r, 0.0), stage=2)
    # the release pose is ENGAGED (tip -release_height inside the bore), so
    # the gripped hover state carries the matching insertion depth
    hover_z = hole_z + cfg.release_height + pl / 2.0
    s_hover = run(hover_z, stage=3, insertion_depth=max(0.0, -cfg.release_height))
    settled_depth = scene.hole_depth - 0.0025  # peg resting on hole_bottom plate
    s_settled = run(
        hole_z - settled_depth + pl / 2.0,
        gripped=False,
        insertion_depth=settled_depth,
        stage=3,
        steady_steps=30,
    )
    farm_depth = 0.69 * pl  # gripped just below success_threshold=0.7
    s_farm = run(hole_z - farm_depth + pl / 2.0, insertion_depth=farm_depth,
                 stage=3, steady_steps=30)

    t_table = float(s_table["reward/total"])
    t_lift = float(s_lift["reward/total"])
    t_hover = float(s_hover["reward/total"])
    t_settled = float(s_settled["reward/total"])
    t_farm = float(s_farm["reward/total"])
    print()
    print("  winning-trajectory per-step totals:")
    print(f"    gripped on table (r={spawn_r*100:.0f}cm)  = {t_table:>9.3f}")
    print(f"    gripped lifted 5cm             = {t_lift:>9.3f}")
    print(f"    gripped at release pose        = {t_hover:>9.3f}")
    print(f"    RELEASED, settled in bore      = {t_settled:>9.3f}")
    print(f"    farm state (grip @ frac 0.69)  = {t_farm:>9.3f}")
    monotone = t_table < t_lift < t_hover < t_settled
    beats_farm = t_settled > t_farm * 1.5
    print()
    print(f"  GATE 3: monotone table<lift<hover<settled   {'PASS' if monotone else 'FAIL'}")
    print(f"  GATE 4: settled > 1.5x farm state           {'PASS' if beats_farm else 'FAIL'}")

    return pass_delta and pass_lift_vs_grasp and monotone and beats_farm


def check_grasp() -> bool:
    cfg = RewardConfig()
    table_h = 0.4
    initial_z = 0.43  # just above table

    def run(obj_z: float) -> dict:
        state = init_grasp_reward_state(initial_z, table_h)
        fp = jnp.array(
            [
                [+0.005, 0.0, obj_z],
                [-0.005, 0.0, obj_z],
                [-0.005, 0.005, obj_z],
                [-0.005, -0.005, obj_z],
                [-0.005, -0.01, obj_z],
            ]
        )
        _, _, info = grasp_reward(
            state=state,
            finger_positions=fp,
            object_position=jnp.array([0.0, 0.0, obj_z]),
            object_linear_velocity=jnp.zeros(3),
            finger_contact_mask=jnp.array([True, True, True, True, True]),
            actions=jnp.zeros(23),
            previous_actions=jnp.zeros(23),
            table_height=table_h,
            lift_target=cfg.lift_target,
            hold_velocity_threshold=cfg.hold_velocity_threshold,
            drop_penalty_value=cfg.drop_penalty,
            no_contact_idle_penalty=cfg.no_contact_idle_penalty,
            success_bonus_per_step=cfg.success_bonus_per_step,
            success_hold_steps=cfg.success_hold_steps,
            weights=cfg.weights,
            reach_tanh_k=cfg.reach_tanh_k,
            hold_height_k=cfg.hold_height_smoothness_k,
            hold_velocity_k=cfg.hold_velocity_smoothness_k,
            fingertip_weights=cfg.fingertip_weights,
            drop_arm_height=cfg.drop_arm_height,
            action_penalty_scale=cfg.action_penalty_scale,
        )
        return info

    print("\n=== GRASP reward gradient (round-14) ===\n")
    info_sit = run(initial_z + 0.0)
    info_lift = run(initial_z + cfg.lift_target)  # at lift_target

    total_sit = float(info_sit["reward/total"])
    total_lift = float(info_lift["reward/total"])
    delta_total = total_lift - total_sit
    grasp_post_weight = float(info_sit["reward/grasping"])

    print("  at lift_height = 0mm (perfect grip, no lift):")
    print(f"    reward/total              = {total_sit:>8.4f}")
    print(f"    reward/grasping           = {grasp_post_weight:>8.4f}  (post-weight)")
    print(f"    reward/lifting            = {float(info_sit['reward/lifting']):>8.4f}")
    print()
    print(f"  at lift_height = {cfg.lift_target*1000:.0f}mm (= lift_target):")
    print(f"    reward/total              = {total_lift:>8.4f}")
    print(f"    reward/lifting            = {float(info_lift['reward/lifting']):>8.4f}")
    print()
    print(f"  delta_total (lifting to target) = {delta_total:>+8.4f}")
    print(f"  delta_lift_component             = "
          f"{float(info_lift['reward/lifting']) - float(info_sit['reward/lifting']):>+8.4f}")
    print()

    # Bars for grasp: PPO maximizes total, not components. So the gate
    # is "lifting must produce a meaningful total-reward jump" — small
    # per-step lift component is fine because lift pulls success/holding
    # bonuses with it (they fire after lift_target is held). Also check
    # monotonicity through the intermediate range to make sure the
    # gradient is actually pushing up, not flat.
    intermediate = run(initial_z + cfg.lift_target * 0.5)
    monotonic = (
        float(info_sit["reward/total"])
        < float(intermediate["reward/total"])
        < float(info_lift["reward/total"])
    )

    bar_delta = 5.0  # grasp at lift_target should clear sitting by at least this much
    pass_delta = delta_total >= bar_delta

    print(f"  GATE 1: delta_total >= {bar_delta}             {'PASS' if pass_delta else 'FAIL'}")
    print(f"  GATE 2: monotonic 0 -> half -> full target  {'PASS' if monotonic else 'FAIL'}")
    return pass_delta and monotonic


if __name__ == "__main__":
    peg_ok = check_peg()
    grasp_ok = check_grasp()
    print()
    print("=" * 60)
    print(f"PEG:   {'PASS' if peg_ok else 'FAIL'}")
    print(f"GRASP: {'PASS' if grasp_ok else 'FAIL'}")
    print("=" * 60)
    if not (peg_ok and grasp_ok):
        print("\nDO NOT spend on a full run — fix reward shape first.")
        raise SystemExit(1)
    print("\nReward gradient is correctly oriented. Sanity run is safe to launch.")
