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
    lateral_gate_k: float = 10.0,
    idle_stage_cutoff: int = 3,
    success_threshold: float = 0.7,
    peg_hold_steps: int = 10,
    reach_tanh_k: float = 5.0,
    fingertip_weights: tuple[float, float, float, float, float] = (2.5, 1.0, 1.0, 1.0, 1.0),
    depth_reward_scale: float = 10.0,
    idle_grace_steps: int = 3,
) -> tuple[jnp.ndarray, PegRewardState, dict[str, jnp.ndarray]]:
    del previous_actions

    ft_weights = jnp.asarray(fingertip_weights)
    n_contacts = jnp.sum(finger_contact_mask).astype(jnp.float32)

    # reach
    dists = jnp.linalg.norm(finger_positions - peg_position, axis=1)
    weighted_dist = jnp.sum(ft_weights * dists) / jnp.sum(ft_weights)
    reach = 1.0 - jnp.tanh(reach_tanh_k * weighted_dist)

    # grasp quality: thumb opposing the rest
    thumb_contact = finger_contact_mask[0]
    others_mask = finger_contact_mask.at[0].set(False)
    others_count = jnp.sum(others_mask)

    thumb_vec = finger_positions[0] - peg_position
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
    # Step bonus breaks the grasp-and-sit local minimum: PPO sees a
    # discontinuous reward jump the moment the peg clears lift_step_threshold,
    # which credit-assigns back to the action that pulled slide_z up. The
    # proportional term then keeps pulling toward lift_target.
    lift_step_bonus = jnp.where(lift_height > lift_step_threshold, 1.0, 0.0)
    lift_proportional = jnp.minimum(lift_height / lift_target, 1.5) * contact_scale
    lift = lift_step_bonus + lift_proportional

    was_lifted_next = state.was_lifted | (lift_height >= lift_target)

    # align + insertion drive: gated on peg actually being above the table
    lateral_dist = jnp.linalg.norm(peg_position[:2] - hole_position[:2])
    axis_align = jnp.abs(jnp.dot(peg_axis, hole_axis))
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

    new_hold = jnp.where(
        insertion_fraction > success_threshold,
        state.insertion_hold_steps + 1,
        jnp.array(0, dtype=jnp.int32),
    )
    complete = (
        complete_bonus
        * axis_align
        * lateral_factor_align
        * contact_scale
        * _sigmoid(20.0 * (insertion_fraction - success_threshold))
        * _sigmoid((new_hold.astype(jnp.float32) - peg_hold_steps) / 2.0)
    )

    force_excess = jnp.maximum(0.0, contact_force_magnitude - force_threshold)
    force_penalty = -0.01 * force_excess**2

    just_dropped = state.was_lifted & (lift_height < 0.01)
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
        + weights.lift * lift
        + weights.align * align
        + weights.depth * depth_reward
        + weights.complete * complete
        + weights.force * force_penalty
        + weights.drop * drop
        + weights.action_penalty * action_penalty
        + weights.insertion_drive * insertion_drive
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
        "reward/lift": lift,
        "reward/align": align,
        "reward/depth": depth_reward,
        "reward/complete": complete,
        "reward/force_penalty": force_penalty,
        "reward/drop": drop,
        "reward/action_penalty": action_penalty,
        "reward/idle_stage0_penalty": idle_penalty,
        "reward/idle_stage1_penalty": idle_stage1_pen,
        "reward/insertion_drive": insertion_drive,
        "reward/total": total,
        "metrics/stage": stage.astype(jnp.float32),
        "metrics/num_finger_contacts": n_contacts,
        "metrics/peg_height": peg_height,
        "metrics/insertion_depth": insertion_depth,
        "metrics/contact_force": contact_force_magnitude,
        "metrics/lateral_distance": lateral_dist,
        "metrics/insertion_hold_steps": new_hold.astype(jnp.float32),
    }

    return total, new_state, info
