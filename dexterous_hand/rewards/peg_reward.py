from typing import NamedTuple

import jax.numpy as jnp

from dexterous_hand.config import PegRewardWeights


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


class PegRewardState(NamedTuple):
    was_lifted: jnp.ndarray
    insertion_hold_steps: jnp.ndarray
    initial_peg_height: jnp.ndarray
    idle_steps: jnp.ndarray
    idle_stage1_steps: jnp.ndarray


def init_peg_reward_state(initial_peg_height: float | jnp.ndarray) -> PegRewardState:
    return PegRewardState(
        was_lifted=jnp.array(False),
        insertion_hold_steps=jnp.array(0, dtype=jnp.int32),
        initial_peg_height=jnp.asarray(initial_peg_height),
        idle_steps=jnp.array(0, dtype=jnp.int32),
        idle_stage1_steps=jnp.array(0, dtype=jnp.int32),
    )


def peg_reward(
    state: PegRewardState,
    stage: jnp.ndarray,
    finger_positions: jnp.ndarray,
    peg_position: jnp.ndarray,
    peg_axis: jnp.ndarray,
    hole_position: jnp.ndarray,
    hole_axis: jnp.ndarray,
    insertion_depth: jnp.ndarray,
    contact_force_magnitude: jnp.ndarray,
    finger_contact_mask: jnp.ndarray,
    peg_height: jnp.ndarray,
    peg_linvel: jnp.ndarray,
    actions: jnp.ndarray,
    previous_actions: jnp.ndarray,
    weights: PegRewardWeights,
    peg_length: float,
    lift_target: float,
    table_height: float,
    drop_penalty_value: float,
    complete_bonus: float,
    force_threshold: float,
    idle_stage0_penalty: float,
    idle_stage1_penalty: float = -0.1,
    idle_stage1_min_contacts: int = 2,
    lift_step_threshold: float = 0.005,
    lateral_gate_k: float = 5.0,
    idle_stage_cutoff: int = 3,
    success_threshold: float = 0.7,
    peg_hold_steps: int = 10,
    reach_tanh_k: float = 5.0,
    fingertip_weights: tuple[float, float, float, float, float] = (2.5, 1.0, 1.0, 1.0, 1.0),
    depth_reward_scale: float = 10.0,
    idle_grace_steps: int = 3,
    release_height: float = -0.015,
    place_k: float = 4.0,
) -> tuple[jnp.ndarray, PegRewardState, dict[str, jnp.ndarray]]:
    del previous_actions

    ft_weights = jnp.asarray(fingertip_weights)
    n_contacts = jnp.sum(finger_contact_mask).astype(jnp.float32)

    # reach
    dists = jnp.linalg.norm(finger_positions - peg_position, axis=1)
    weighted_dist = jnp.sum(ft_weights * dists) / jnp.sum(ft_weights)
    reach = 1.0 - jnp.tanh(reach_tanh_k * weighted_dist)

    # Round-16: reverted peg back to thumb-opposition (round-15 swapped it to
    # side_ratio, which rewards fingers at or below peg center — incompatible
    # with the grip needed for insertion, where the peg must extend below the
    # fingers). Grasp keeps side_ratio because the cube task is z-symmetric.
    #
    # The finger ordering is [ff, mf, rf, lf, th] (FINGER_TOUCH_SITE_NAMES /
    # FINGERTIP_SITE_NAMES), so the thumb is index 4 — NOT index 0. The prior
    # code keyed opposition+tripod off index 0 (the first finger), which never
    # contacts the peg in the natural grip, so both terms were dead (always 0).
    THUMB = 4
    thumb_contact = finger_contact_mask[THUMB]
    others_mask = finger_contact_mask.at[THUMB].set(False)
    others_count = jnp.sum(others_mask)

    thumb_vec = finger_positions[THUMB] - peg_position
    other_vecs = (finger_positions - peg_position) * others_mask[:, None]
    mean_other_vec = jnp.where(
        others_count > 0,
        other_vecs.sum(axis=0) / jnp.maximum(others_count, 1.0),
        jnp.zeros(3),
    )
    thumb_n = jnp.linalg.norm(thumb_vec) + 1e-6
    other_n = jnp.linalg.norm(mean_other_vec) + 1e-6
    raw_opposition = -jnp.dot(thumb_vec / thumb_n, mean_other_vec / other_n)
    opposition = jnp.where(
        thumb_contact & (others_count >= 1),
        jnp.maximum(raw_opposition, 0.0),
        0.0,
    )

    contact_scale = jnp.minimum(n_contacts / 3.0, 1.0)
    tripod_bonus = 0.5 * (thumb_contact & (others_count >= 2)).astype(jnp.float32)
    grasp = contact_scale * (0.3 + 0.7 * opposition) + tripod_bonus

    lift_height = jnp.maximum(peg_height - state.initial_peg_height, 0.0)
    # Round-14: back to the binary step bonus. The round-13 smooth ramp
    # removed the discontinuity that round-12 specifically used to escape
    # the grasp-and-sit basin, and the round-13 67M peg run stayed stuck
    # at +43mm lift forever as a result. The original concern (the binary
    # discontinuity drives value-function divergence) is solved by
    # `norm_reward=True` independently — VecNormalize bounds the variance
    # contribution before PPO sees it.
    lift_gate = (n_contacts >= 2).astype(jnp.float32)
    lift_step_bonus = jnp.where(lift_height > lift_step_threshold, 1.0, 0.0) * lift_gate
    # Cap at 1.0 (was 1.5): there is no task reason to reward lifting 50% past
    # lift_target — the peg only needs to clear the table and reach the hole
    # entrance. The 1.5 cap kept paying out to ~75mm of lift, well past the need,
    # feeding the "lift high and hold" attractor.
    lift_proportional = jnp.minimum(lift_height / lift_target, 1.0) * lift_gate
    lift = lift_step_bonus + lift_proportional

    was_lifted_next = state.was_lifted | (lift_height >= lift_target)

    # Round-16: axis_in_grip rewards holding the peg axis-aligned with the
    # hole while *any* finger is in contact, before lift. The previous design
    # only activated axis_align inside the align term, which was gated on
    # peg_clearance > 2cm (i.e., after lift), so the policy had no signal
    # against settling into a tilted grip in stage 1.
    axis_align = jnp.abs(jnp.dot(peg_axis, hole_axis))
    axis_in_grip = axis_align * contact_scale

    # align + insertion drive: gated on peg actually being above the table
    lateral_dist = jnp.linalg.norm(peg_position[:2] - hole_position[:2])
    lateral_factor_align = 1.0 - jnp.tanh(lateral_gate_k * lateral_dist)
    peg_clearance = jnp.maximum(peg_height - table_height - peg_length * 0.5, 0.0)
    align_weight = _sigmoid((peg_clearance - 0.02) * 150.0)
    align = axis_align * lateral_factor_align * align_weight * contact_scale

    insertion_drive = (
        align_weight
        * lateral_factor_align
        * axis_align
        * contact_scale
        * jnp.maximum(-peg_linvel[2], 0.0)
        * 5.0
    )

    lateral_factor_depth = 1.0 - jnp.tanh(lateral_gate_k * lateral_dist)
    insertion_fraction = jnp.clip(insertion_depth / peg_length, 0.0, 1.0)
    depth_reward = depth_reward_scale * insertion_fraction * lateral_factor_depth

    # place: 2-keypoint distance to the ENGAGED-RELEASE pose — peg vertical
    # with its tip |release_height| INSIDE the bore (Factory/IndustReal-style
    # keypoint shaping, Narang'22/Tang'23, with the target moved from the
    # fully-inserted pose to the engaged release pose). Rationale:
    # (1) after `lift` saturates at lift_target there was no meaningful
    # gradient pulling the peg the remaining way up-and-over the bore — the
    # old "descend anywhere" depth gradient that filled this gap was the
    # false-success exploit, and the containment fix correctly zeroed it,
    # leaving a dead zone that ends exactly at this term;
    # (2) targeting the FULLY-inserted pose would create a wall-press local
    # minimum — an on-table peg 5cm out is Euclidean-closer to the inserted
    # keypoints (kd 0.109) than a peg correctly hovering over the entrance
    # (kd 0.135). With the shallow engaged target the hover state stays
    # strictly closer than any on-table pose, and reaching it requires
    # entering from above;
    # (3) the target is ENGAGED (tip ~1.5cm into the bore), not above the
    # entrance, because a capsule released above the entrance topples (4mm
    # clearance over 7.6cm = ~6 deg self-alignment cone; measured), while an
    # engaged tip is laterally guided and slides to the bottom on release.
    # The in-bore endgame is paid by depth+complete, not place. Ungated: its
    # max is the correct behavior and it pays pennies elsewhere.
    axis_dot_ph = jnp.dot(peg_axis, hole_axis)
    tip = peg_position - peg_axis * jnp.sign(axis_dot_ph) * (peg_length / 2.0)
    top = peg_position + peg_axis * jnp.sign(axis_dot_ph) * (peg_length / 2.0)
    target_tip = hole_position + hole_axis * release_height
    target_top = target_tip + hole_axis * peg_length
    keypoint_dist = jnp.linalg.norm(tip - target_tip) + jnp.linalg.norm(top - target_top)
    place = 1.0 - jnp.tanh(place_k * keypoint_dist)

    new_hold = jnp.where(
        insertion_fraction > success_threshold,
        state.insertion_hold_steps + 1,
        jnp.array(0, dtype=jnp.int32),
    )
    # NO contact_scale here: the bore (12mm) cannot admit fingers, so the only
    # physical way to reach success depth is to RELEASE the peg over the bore
    # and let gravity finish (see config hole_top comment + the drop test).
    # Gating complete on contacts forfeited the entire completion payment at
    # the exact moment the policy did the right thing — the settled peg paid
    # ~0 while a gripped hover below threshold farmed shaping forever.
    complete = (
        complete_bonus
        * axis_align
        * lateral_factor_align
        * _sigmoid(20.0 * (insertion_fraction - success_threshold))
        * _sigmoid((new_hold.astype(jnp.float32) - peg_hold_steps) / 2.0)
    )

    force_excess = jnp.maximum(0.0, contact_force_magnitude - force_threshold)
    force_penalty = -0.01 * force_excess**2

    # A peg released INTO the bore ends ~19.5mm above its table spawn height,
    # so it never trips the lift<0.01 test today — but that margin is set by
    # hole geometry, not by intent. Guard explicitly: insertion is never a
    # "drop".
    just_dropped = state.was_lifted & (lift_height < 0.01) & (insertion_fraction < 0.1)
    drop = jnp.where(just_dropped, drop_penalty_value, 0.0)
    was_lifted = jnp.where(just_dropped, False, was_lifted_next)

    action_penalty = -0.0002 * jnp.sum(actions**2)

    # idle penalty only fires in early stages so the policy isn't punished mid-insertion
    idle_active = (n_contacts == 0) & (stage < idle_stage_cutoff)
    new_idle_steps = jnp.where(
        idle_active, state.idle_steps + 1, jnp.array(0, dtype=jnp.int32)
    )
    idle_raw = jnp.where(new_idle_steps >= idle_grace_steps, idle_stage0_penalty, 0.0)
    idle_penalty = weights.idle_stage0 * idle_raw

    # Grasp-and-sit penalty: fires when the peg is grasped (nfc >= min)
    # but never leaves the initial pose. Same grace-period pattern as
    # idle_stage0 so a brief regrasp doesn't trigger it.
    idle_stage1_active = (
        (n_contacts >= idle_stage1_min_contacts)
        & (lift_height < lift_step_threshold)
        & (stage == 1)
    )
    new_idle_stage1_steps = jnp.where(
        idle_stage1_active,
        state.idle_stage1_steps + 1,
        jnp.array(0, dtype=jnp.int32),
    )
    idle_stage1_raw = jnp.where(
        new_idle_stage1_steps >= idle_grace_steps, idle_stage1_penalty, 0.0
    )
    idle_stage1_pen = weights.idle_stage1 * idle_stage1_raw

    total = (
        weights.reach * reach
        + weights.grasp * grasp
        + weights.opposition * opposition
        + weights.axis_in_grip * axis_in_grip
        + weights.lift * lift
        + weights.align * align
        + weights.depth * depth_reward
        + weights.complete * complete
        + weights.force * force_penalty
        + weights.drop * drop
        + weights.action_penalty * action_penalty
        + weights.insertion_drive * insertion_drive
        + weights.place * place
        + idle_penalty
        + idle_stage1_pen
    )

    new_state = PegRewardState(
        was_lifted=was_lifted,
        insertion_hold_steps=new_hold,
        initial_peg_height=state.initial_peg_height,
        idle_steps=new_idle_steps,
        idle_stage1_steps=new_idle_stage1_steps,
    )

    info = {
        "reward/reach": reach,
        "reward/grasp": grasp,
        "reward/grasp_quality": opposition,
        "reward/axis_in_grip": axis_in_grip,
        "reward/lift": lift,
        "reward/align": align,
        "reward/depth": depth_reward,
        "reward/complete": complete,
        "reward/force_penalty": force_penalty,
        "reward/drop": drop,
        "reward/action_penalty": action_penalty,
        # raw (pre-weight) like the other reward/* components, so logged per-term
        # curves share a common scale; total applies weights.idle_stage0/1.
        "reward/idle_stage0_penalty": idle_raw,
        "reward/idle_stage1_penalty": idle_stage1_raw,
        "reward/insertion_drive": insertion_drive,
        "reward/place": place,
        "reward/total": total,
        "metrics/keypoint_dist": keypoint_dist,
        "metrics/stage": stage.astype(jnp.float32),
        "metrics/num_finger_contacts": n_contacts,
        "metrics/peg_height": peg_height,
        "metrics/insertion_depth": insertion_depth,
        "metrics/contact_force": contact_force_magnitude,
        "metrics/lateral_distance": lateral_dist,
        "metrics/insertion_hold_steps": new_hold.astype(jnp.float32),
        "metrics/axis_align": axis_align,
    }

    return total, new_state, info
