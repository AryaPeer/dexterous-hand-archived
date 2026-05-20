from typing import NamedTuple

import jax.numpy as jnp

from dexterous_hand.config import RewardWeights


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))


class GraspRewardState(NamedTuple):
    was_lifted: jnp.ndarray
    initial_height_above_table: jnp.ndarray
    idle_steps: jnp.ndarray
    success_hold_counter: jnp.ndarray
    was_success_prev: jnp.ndarray


def init_grasp_reward_state(
    initial_object_height: float,
    table_height: float,
) -> GraspRewardState:
    return GraspRewardState(
        was_lifted=jnp.array(False),
        initial_height_above_table=jnp.maximum(
            jnp.array(initial_object_height) - jnp.array(table_height), 0.0
        ),
        idle_steps=jnp.array(0, dtype=jnp.int32),
        success_hold_counter=jnp.array(0, dtype=jnp.int32),
        was_success_prev=jnp.array(False),
    )


def grasp_reward(
    state: GraspRewardState,
    finger_positions: jnp.ndarray,
    object_position: jnp.ndarray,
    object_linear_velocity: jnp.ndarray,
    finger_contact_mask: jnp.ndarray,
    actions: jnp.ndarray,
    previous_actions: jnp.ndarray,
    table_height: float,
    lift_target: float,
    hold_velocity_threshold: float,
    drop_penalty_value: float,
    no_contact_idle_penalty: float,
    success_bonus_value: float,
    success_hold_steps: int,
    weights: RewardWeights,
    reach_tanh_k: float = 5.0,
    hold_height_k: float = 50.0,
    hold_velocity_k: float = 100.0,
    fingertip_weights: tuple[float, float, float, float, float] = (2.5, 1.0, 1.0, 1.0, 1.0),
    idle_grace_steps: int = 3,
) -> tuple[jnp.ndarray, GraspRewardState, dict[str, jnp.ndarray]]:
    del previous_actions

    ft_weights = jnp.asarray(fingertip_weights)

    n_contacts = jnp.sum(finger_contact_mask).astype(jnp.float32)
    obj_height = object_position[2]
    height_above_table = obj_height - table_height
    lift_height = jnp.maximum(height_above_table - state.initial_height_above_table, 0.0)

    dists = jnp.linalg.norm(finger_positions - object_position, axis=1)
    weighted_dist = jnp.sum(ft_weights * dists) / jnp.sum(ft_weights)
    reaching = 1.0 - jnp.tanh(reach_tanh_k * weighted_dist)

    thumb_contact = finger_contact_mask[0]
    others_mask = finger_contact_mask.at[0].set(False)
    others_count = jnp.sum(others_mask)

    thumb_vec = finger_positions[0] - object_position
    other_vecs = (finger_positions - object_position) * others_mask[:, None]
    mean_other_vec = jnp.where(
        others_count > 0, other_vecs.sum(axis=0) / jnp.maximum(others_count, 1.0), jnp.zeros(3)
    )

    thumb_n = jnp.linalg.norm(thumb_vec) + 1e-6
    other_n = jnp.linalg.norm(mean_other_vec) + 1e-6
    raw_opposition = -jnp.dot(thumb_vec / thumb_n, mean_other_vec / other_n)
    opposition = jnp.where(
        thumb_contact & (others_count >= 1),
        jnp.maximum(raw_opposition, 0.0),
        0.0,
    )

    contact_scale = jnp.tanh(n_contacts / 2.0)
    tripod_bonus = 0.5 * (thumb_contact & (others_count >= 2)).astype(jnp.float32)
    grasping = contact_scale * (0.3 + 0.7 * opposition) + tripod_bonus

    lifting = jnp.minimum(lift_height / lift_target, 1.5) * contact_scale

    obj_speed = jnp.linalg.norm(object_linear_velocity)
    height_gate = _sigmoid(hold_height_k * (lift_height - lift_target + 0.04))
    speed_gate = _sigmoid(hold_velocity_k * (hold_velocity_threshold - obj_speed))
    holding = height_gate * speed_gate * contact_scale

    was_lifted_next = state.was_lifted | (lift_height >= lift_target)

    just_dropped = state.was_lifted & (lift_height < 0.01)
    drop = jnp.where(just_dropped, drop_penalty_value, 0.0)
    was_lifted = jnp.where(just_dropped, False, was_lifted_next)

    lift_factor = jnp.clip(lift_height / lift_target, 0.0, 1.0)
    contact_factor = jnp.clip(n_contacts / 3.0, 0.0, 1.0)
    speed_factor = _sigmoid(20.0 * (0.2 - obj_speed))
    at_target = (lift_factor * contact_factor * speed_factor) >= 0.85
    new_success_hold = jnp.where(
        at_target, state.success_hold_counter + 1, jnp.array(0, dtype=jnp.int32)
    )
    is_success = new_success_hold >= success_hold_steps
    success = jnp.where(is_success & ~state.was_success_prev, success_bonus_value, 0.0)

    idle_active = n_contacts == 0
    new_idle_steps = jnp.where(
        idle_active, state.idle_steps + 1, jnp.array(0, dtype=jnp.int32)
    )
    idle_raw = jnp.where(new_idle_steps >= idle_grace_steps, no_contact_idle_penalty, 0.0)
    idle_penalty = weights.idle * idle_raw

    action_penalty = -0.0002 * jnp.sum(actions**2)

    total = (
        weights.reaching * reaching
        + weights.grasping * grasping
        + weights.opposition * opposition
        + weights.lifting * lifting
        + weights.holding * holding
        + weights.drop * drop
        + weights.success * success
        + weights.action_penalty * action_penalty
        + idle_penalty
    )

    new_state = GraspRewardState(
        was_lifted=was_lifted,
        initial_height_above_table=state.initial_height_above_table,
        idle_steps=new_idle_steps,
        success_hold_counter=new_success_hold,
        was_success_prev=is_success,
    )

    info = {
        "reward/reaching": reaching,
        "reward/grasping": grasping,
        "reward/grasp_quality": opposition,
        "reward/lifting": lifting,
        "reward/holding": holding,
        "reward/drop": drop,
        "reward/success": success,
        "reward/idle_penalty": idle_penalty,
        "reward/action_penalty": action_penalty,
        "reward/total": total,
        "metrics/num_finger_contacts": n_contacts,
        "metrics/object_height": obj_height,
        "metrics/object_speed": obj_speed,
        "metrics/mean_fingertip_dist": jnp.mean(dists),
        "metrics/success_hold_steps": new_success_hold.astype(jnp.float32),
        "is_success": is_success.astype(jnp.float32),
    }

    return total, new_state, info
