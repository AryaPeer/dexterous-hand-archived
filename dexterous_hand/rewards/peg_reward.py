from typing import NamedTuple

import jax.numpy as jnp

from dexterous_hand.config import PegRewardWeights


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


ALIGN_GATE_CENTER = 0.02
ALIGN_GATE_K = 150.0
COMPLETE_FRAC_K = 20.0
COMPLETE_HOLD_SCALE = 2.0
FORCE_PENALTY_SCALE = 0.01


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
    actions: jnp.ndarray,
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
    fingertip_weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 2.5),
    action_penalty_scale: float = 2e-4,
    depth_reward_scale: float = 10.0,
    idle_grace_steps: int = 3,
    release_height: float = -0.015,
    place_k: float = 4.0,
) -> tuple[jnp.ndarray, PegRewardState, dict[str, jnp.ndarray]]:
    ft_weights = jnp.asarray(fingertip_weights)
    n_contacts = jnp.sum(finger_contact_mask).astype(jnp.float32)

    dists = jnp.linalg.norm(finger_positions - peg_position, axis=1)
    weighted_dist = jnp.sum(ft_weights * dists) / jnp.sum(ft_weights)
    reach = 1.0 - jnp.tanh(reach_tanh_k * weighted_dist)

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
    lift_gate = (n_contacts >= 2).astype(jnp.float32)
    lift_step_bonus = jnp.where(lift_height > lift_step_threshold, 1.0, 0.0) * lift_gate
    lift_proportional = jnp.minimum(lift_height / lift_target, 1.0) * lift_gate
    lift = lift_step_bonus + lift_proportional

    was_lifted_next = state.was_lifted | (lift_height >= lift_target)

    axis_align = jnp.abs(jnp.dot(peg_axis, hole_axis))
    axis_in_grip = axis_align * contact_scale

    lateral_dist = jnp.linalg.norm(peg_position[:2] - hole_position[:2])
    lateral_factor_align = 1.0 - jnp.tanh(lateral_gate_k * lateral_dist)
    peg_clearance = jnp.maximum(peg_height - table_height - peg_length * 0.5, 0.0)
    align_weight = _sigmoid((peg_clearance - ALIGN_GATE_CENTER) * ALIGN_GATE_K)
    align = axis_align * lateral_factor_align * align_weight * contact_scale

    lateral_factor_depth = 1.0 - jnp.tanh(lateral_gate_k * lateral_dist)
    insertion_fraction = jnp.clip(insertion_depth / peg_length, 0.0, 1.0)
    depth_reward = depth_reward_scale * insertion_fraction * lateral_factor_depth

    axis_dot_ph = jnp.dot(peg_axis, hole_axis)
    axis_sign = jnp.where(axis_dot_ph >= 0.0, 1.0, -1.0)
    tip = peg_position - peg_axis * axis_sign * (peg_length / 2.0)
    top = peg_position + peg_axis * axis_sign * (peg_length / 2.0)
    target_tip = hole_position + hole_axis * release_height
    target_top = target_tip + hole_axis * peg_length
    keypoint_dist = jnp.linalg.norm(tip - target_tip) + jnp.linalg.norm(top - target_top)
    place_gate = jnp.maximum(contact_scale, jnp.clip(insertion_fraction / 0.05, 0.0, 1.0))
    place = (1.0 - jnp.tanh(place_k * keypoint_dist)) * place_gate

    new_hold = jnp.where(
        insertion_fraction > success_threshold,
        state.insertion_hold_steps + 1,
        jnp.array(0, dtype=jnp.int32),
    )
    complete = (
        complete_bonus
        * axis_align
        * lateral_factor_align
        * _sigmoid(COMPLETE_FRAC_K * (insertion_fraction - success_threshold))
        * _sigmoid((new_hold.astype(jnp.float32) - peg_hold_steps) / COMPLETE_HOLD_SCALE)
    )

    force_excess = jnp.maximum(0.0, contact_force_magnitude - force_threshold)
    force_penalty = -FORCE_PENALTY_SCALE * force_excess**2

    just_dropped = state.was_lifted & (lift_height < 0.01) & (insertion_fraction < 0.1)
    drop = jnp.where(just_dropped, drop_penalty_value, 0.0)
    was_lifted = jnp.where(just_dropped, False, was_lifted_next)

    action_penalty = -action_penalty_scale * jnp.sum(actions**2)

    idle_active = (n_contacts == 0) & (stage < idle_stage_cutoff)
    new_idle_steps = jnp.where(
        idle_active, state.idle_steps + 1, jnp.array(0, dtype=jnp.int32)
    )
    idle_raw = jnp.where(new_idle_steps >= idle_grace_steps, idle_stage0_penalty, 0.0)
    idle_penalty = weights.idle_stage0 * idle_raw

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
        "reward/idle_stage0_penalty": idle_raw,
        "reward/idle_stage1_penalty": idle_stage1_raw,
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
