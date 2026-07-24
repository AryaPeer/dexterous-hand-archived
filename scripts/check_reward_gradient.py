"""Pre-flight: verify the reward gradient pushes the policy toward the task."""

import jax.numpy as jnp

from dexterous_hand.config import (
    PegRewardConfig,
    PegSceneConfig,
    PickPlaceRewardConfig,
    PickPlaceSceneConfig,
    RewardConfig,
)
from dexterous_hand.rewards.grasp_reward import grasp_reward, init_grasp_reward_state
from dexterous_hand.rewards.peg_reward import init_peg_reward_state, peg_reward
from dexterous_hand.rewards.pickplace_reward import init_pickplace_reward_state, pickplace_reward


def check_peg() -> bool:
    cfg = PegRewardConfig()
    scene = PegSceneConfig()
    pl = scene.peg_half_length * 2.0 + scene.peg_radius * 2.0
    table_h = scene.table_height
    hole_z = table_h + scene.hole_top_above_table
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
                actions=jnp.zeros(23),
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

    print("\n=== PEG reward gradient ===\n")
    info_sit = run(initial_z + 0.0)
    info_lift = run(initial_z + 0.006)

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

    bar_delta = 1.0
    pass_delta = delta_total >= bar_delta
    pass_lift_vs_grasp = delta_total > grasp_post_weight

    print(f"  GATE 1: delta_total >= {bar_delta}     {'PASS' if pass_delta else 'FAIL'}")
    print(f"  GATE 2: lifting beats grasp          {'PASS' if pass_lift_vs_grasp else 'FAIL'}")

    spawn_r = scene.spawn_min_radius
    s_table = run(initial_z, peg_xy=(spawn_r, 0.0))
    s_lift = run(initial_z + 0.05, peg_xy=(spawn_r, 0.0), stage=2)
    hover_z = hole_z + cfg.release_height + pl / 2.0
    s_hover = run(hover_z, stage=3, insertion_depth=max(0.0, -cfg.release_height))
    settled_depth = scene.hole_depth - 0.0025
    s_settled = run(
        hole_z - settled_depth + pl / 2.0,
        gripped=False,
        insertion_depth=settled_depth,
        stage=3,
        steady_steps=30,
    )
    farm_depth = 0.69 * pl
    s_farm = run(hole_z - farm_depth + pl / 2.0, insertion_depth=farm_depth,
                 stage=3, steady_steps=30)
    s_parked = run(initial_z, peg_xy=(0.05, 0.0), gripped=False, stage=0, steady_steps=30)

    t_table = float(s_table["reward/total"])
    t_lift = float(s_lift["reward/total"])
    t_hover = float(s_hover["reward/total"])
    t_settled = float(s_settled["reward/total"])
    t_farm = float(s_farm["reward/total"])
    t_parked = float(s_parked["reward/total"])
    print()
    print("  winning-trajectory per-step totals:")
    print(f"    gripped on table (r={spawn_r*100:.0f}cm)  = {t_table:>9.3f}")
    print(f"    gripped lifted 5cm             = {t_lift:>9.3f}")
    print(f"    gripped at release pose        = {t_hover:>9.3f}")
    print(f"    RELEASED, settled in bore      = {t_settled:>9.3f}")
    print(f"    farm state (grip @ frac 0.69)  = {t_farm:>9.3f}")
    print(f"    parked by tube, NO grip        = {t_parked:>9.3f}")
    monotone = t_table < t_lift < t_hover < t_settled
    beats_farm = t_settled > t_farm * 1.5
    parked_pays_nothing = t_parked < 1.0 and t_parked < t_table
    print()
    print(f"  GATE 3: monotone table<lift<hover<settled   {'PASS' if monotone else 'FAIL'}")
    print(f"  GATE 4: settled > 1.5x farm state           {'PASS' if beats_farm else 'FAIL'}")
    print(f"  GATE 5: parked-ungripped pays ~nothing      {'PASS' if parked_pays_nothing else 'FAIL'}")

    return pass_delta and pass_lift_vs_grasp and monotone and beats_farm and parked_pays_nothing


def check_grasp() -> bool:
    cfg = RewardConfig()
    table_h = 0.4
    initial_z = 0.43

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

    print("\n=== GRASP reward gradient ===\n")
    info_sit = run(initial_z + 0.0)
    info_lift = run(initial_z + cfg.lift_target)

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

    intermediate = run(initial_z + cfg.lift_target * 0.5)
    monotonic = (
        float(info_sit["reward/total"])
        < float(intermediate["reward/total"])
        < float(info_lift["reward/total"])
    )

    bar_delta = 5.0
    pass_delta = delta_total >= bar_delta

    print(f"  GATE 1: delta_total >= {bar_delta}             {'PASS' if pass_delta else 'FAIL'}")
    print(f"  GATE 2: monotonic 0 -> half -> full target  {'PASS' if monotonic else 'FAIL'}")
    return pass_delta and monotonic


def check_pickplace() -> bool:
    cfg = PickPlaceRewardConfig()
    scfg = PickPlaceSceneConfig()
    table_h = scfg.table_height
    half = scfg.object_half_extent
    rest_z = table_h + half
    initial_z = rest_z + 0.001
    source_xy = (0.075, 0.0)
    goal_xy = scfg.goal_nominal_xy

    def run(
        obj_xy: tuple[float, float],
        obj_z: float,
        gripped: bool,
        was_lifted: bool = False,
        steady_steps: int = 1,
    ) -> dict:
        state = init_pickplace_reward_state(initial_z, table_h)
        state = state._replace(was_lifted=jnp.array(was_lifted))
        px, py = obj_xy
        if gripped:
            fp = jnp.array(
                [
                    [px + 0.005, py, obj_z],
                    [px - 0.005, py, obj_z],
                    [px - 0.005, py + 0.005, obj_z],
                    [px - 0.005, py - 0.005, obj_z],
                    [px - 0.005, py - 0.01, obj_z],
                ]
            )
            mask = jnp.array([True, True, True, True, True])
        else:
            fp = jnp.tile(jnp.array([px, py, obj_z + 0.08]), (5, 1))
            mask = jnp.array([False] * 5)
        info: dict = {}
        for _ in range(steady_steps):
            _, state, info = pickplace_reward(
                state=state,
                finger_positions=fp,
                object_position=jnp.array([px, py, obj_z]),
                object_linear_velocity=jnp.zeros(3),
                finger_contact_mask=mask,
                goal_xy=jnp.asarray(goal_xy),
                actions=jnp.zeros(23),
                table_height=table_h,
                object_half_extent=half,
                weights=cfg.weights,
                lift_target=cfg.lift_target,
                carry_clear_height=cfg.carry_clear_height,
                goal_radius=cfg.goal_radius,
                hold_velocity_threshold=cfg.hold_velocity_threshold,
                drop_penalty_value=cfg.drop_penalty,
                no_contact_idle_penalty=cfg.no_contact_idle_penalty,
                success_bonus_per_step=cfg.success_bonus_per_step,
                place_hold_steps=cfg.place_hold_steps,
                reach_tanh_k=cfg.reach_tanh_k,
                transport_tanh_k=cfg.transport_tanh_k,
                goal_tanh_k=cfg.goal_tanh_k,
                on_table_tol=cfg.on_table_tol,
                on_table_k=cfg.on_table_k,
                at_rest_k=cfg.at_rest_k,
                fingertip_weights=cfg.fingertip_weights,
                drop_arm_height=cfg.drop_arm_height,
                action_penalty_scale=cfg.action_penalty_scale,
                idle_grace_steps=cfg.idle_grace_steps,
            )
        return info

    s_reach = run(source_xy, rest_z, gripped=False)
    s_grip = run(source_xy, rest_z, gripped=True)
    s_lift = run(source_xy, rest_z + cfg.lift_target, gripped=True)
    s_hover = run(goal_xy, rest_z + cfg.lift_target, gripped=True)
    s_settled = run(goal_xy, rest_z, gripped=False, was_lifted=True,
                    steady_steps=cfg.place_hold_steps + 5)
    s_settled_nos = run(goal_xy, rest_z, gripped=False, was_lifted=True, steady_steps=1)
    s_bulldoze = run(goal_xy, rest_z, gripped=True, was_lifted=False)
    s_parked = run(source_xy, rest_z, gripped=False, steady_steps=30)

    t_reach = float(s_reach["reward/total"])
    t_grip = float(s_grip["reward/total"])
    t_lift = float(s_lift["reward/total"])
    t_hover = float(s_hover["reward/total"])
    t_settled = float(s_settled["reward/total"])
    t_settled_nos = float(s_settled_nos["reward/total"])
    t_parked = float(s_parked["reward/total"])
    placed_bulldoze = float(s_bulldoze["reward/placed"])

    print("\n=== PICKPLACE reward gradient ===\n")
    print("  winning-trajectory per-step totals:")
    print(f"    reach-only (no grip)           = {t_reach:>9.3f}")
    print(f"    gripped @ source               = {t_grip:>9.3f}")
    print(f"    gripped lifted @ source        = {t_lift:>9.3f}")
    print(f"    carried, lifted over goal      = {t_hover:>9.3f}")
    print(f"    RELEASED, settled @ goal       = {t_settled:>9.3f}")
    print(f"    released @ goal (no annuity)   = {t_settled_nos:>9.3f}")
    print(f"    parked, no grip                = {t_parked:>9.3f}")
    print(f"    bulldozed @ goal (never lifted) placed = {placed_bulldoze:>7.4f}")
    print()

    monotone = t_reach < t_grip < t_lift < t_hover < t_settled
    anti_cliff = t_settled > t_hover and t_settled_nos > t_hover
    parked_pays_nothing = t_parked < 1.0 and t_parked < t_grip
    no_bulldoze = placed_bulldoze < 0.05

    print(f"  GATE 1: monotone reach<grip<lift<carry<settled   {'PASS' if monotone else 'FAIL'}")
    print(f"  GATE 2: settled beats carry (release, no dip)    {'PASS' if anti_cliff else 'FAIL'}")
    print(f"  GATE 3: parked-ungripped pays ~nothing           "
          f"{'PASS' if parked_pays_nothing else 'FAIL'}")
    print(f"  GATE 4: bulldozing (never lifted) pays nothing   {'PASS' if no_bulldoze else 'FAIL'}")

    return monotone and anti_cliff and parked_pays_nothing and no_bulldoze


if __name__ == "__main__":
    peg_ok = check_peg()
    grasp_ok = check_grasp()
    pickplace_ok = check_pickplace()
    print()
    print("=" * 60)
    print(f"PEG:       {'PASS' if peg_ok else 'FAIL'}")
    print(f"GRASP:     {'PASS' if grasp_ok else 'FAIL'}")
    print(f"PICKPLACE: {'PASS' if pickplace_ok else 'FAIL'}")
    print("=" * 60)
    if not (peg_ok and grasp_ok and pickplace_ok):
        print("\nDO NOT spend on a full run — fix reward shape first.")
        raise SystemExit(1)
    print("\nReward gradient is correctly oriented. Sanity run is safe to launch.")
