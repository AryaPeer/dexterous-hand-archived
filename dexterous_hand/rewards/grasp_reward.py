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

    # side_ratio: fraction of contacting fingers whose tip sits at or below the
    # object's vertical midpoint (+ a 1.5cm slack). Matches the Apr-10 "working
    # for everything except spheres" design — rewards any wrap-around, not just
    # thumb-vs-fingers opposition. With the cube spawned in y ∈ [-0.05, +0.05]
    # and the thumb sitting at y≈-0.087, the previous thumb-gated opposition
    # term zeroed out on ~half the cube spawns. side_ratio gives partial credit
    # for any palm/finger contact pattern that's actually wrapping.
    finger_z_below = finger_positions[:, 2] <= (object_position[2] + 0.015)
    side_count = jnp.sum(finger_contact_mask & finger_z_below).astype(jnp.float32)
    side_ratio = jnp.where(n_contacts > 0, side_count / jnp.maximum(n_contacts, 1.0), 0.0)

    contact_scale = jnp.tanh(n_contacts / 2.0)
    grasping = contact_scale * (0.3 + 0.7 * side_ratio)

    # Lift gate matches robosuite Lift / Apr-10 grasp: hard 2-contact gate, no
    # contact_scale attenuation past it. Letting tanh(n/2) scale lifting made
    # the policy converge to grasp-and-sit because each marginal contact above
    # 2 still bought additional reward without needing to actually lift.
    # Linear all the way to lift_target (Apr-10 shape); capped at 1.0 — beyond
    # the target the holding term takes over, so there is no incentive to
    # yo-yo the cube above the cap.
    lift_gate = (n_contacts >= 2).astype(jnp.float32)
    lifting = jnp.clip(lift_height / lift_target, 0.0, 1.0) * lift_gate

    obj_speed = jnp.linalg.norm(object_linear_velocity)
    # height_gate is ~0 for a grasped-but-UNLIFTED cube and ramps to ~1 as the
    # cube reaches lift_target. The previous formula added +0.04 inside the
    # sigmoid, centering the gate at lift_height = -28mm, so it was ~80% active
    # at zero lift and paid ~5.8/step (~1157 over a 200-step episode) to hold a
    # SITTING cube — a direct grasp-and-sit subsidy the team fought for rounds.
    # Centering at lift_target makes a sitting cube earn ~0 holding while a
    # lifted-and-held cube still earns ~full (at lift_target=0.10, k=50 puts
    # the at-rest gate at sigmoid(-5) = 0.7%).
    height_gate = _sigmoid(hold_height_k * (lift_height - lift_target))
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
        + weights.opposition * side_ratio
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
        "reward/grasp_quality": side_ratio,
        "reward/lifting": lifting,
        "reward/holding": holding,
        "reward/drop": drop,
        "reward/success": success,
        # raw (pre-weight) like the other reward/* components, so the logged
        # per-term curves are on a common scale; total applies weights.idle.
        "reward/idle_penalty": idle_raw,
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
