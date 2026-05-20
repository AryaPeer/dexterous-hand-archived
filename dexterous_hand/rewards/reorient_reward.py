from typing import NamedTuple

import jax.numpy as jnp

from dexterous_hand.config import ReorientRewardWeights
from dexterous_hand.utils.quaternion import quat_angular_distance


class ReorientRewardState(NamedTuple):
    success_steps: jnp.ndarray
    prev_ang_dist: jnp.ndarray
    initial_cube_pos: jnp.ndarray
    has_prev: jnp.ndarray


def init_reorient_reward_state(initial_cube_pos: jnp.ndarray) -> ReorientRewardState:
    return ReorientRewardState(
        success_steps=jnp.array(0, dtype=jnp.int32),
        prev_ang_dist=jnp.array(0.0),
        initial_cube_pos=initial_cube_pos,
        has_prev=jnp.array(False),
    )


def reorient_reward(
    state: ReorientRewardState,
    cube_quat: jnp.ndarray,
    target_quat: jnp.ndarray,
    cube_pos: jnp.ndarray,
    cube_linvel: jnp.ndarray,
    finger_positions: jnp.ndarray,
    finger_contact_mask: jnp.ndarray,
    actions: jnp.ndarray,
    previous_actions: jnp.ndarray,
    drop_factor: jnp.ndarray,
    weights: ReorientRewardWeights,
    success_threshold: float,
    success_hold_steps: int,
    drop_penalty_value: float,
    contact_bonus_value: float,
    no_contact_penalty_value: float,
    min_contacts_for_rotation: int,
    angular_progress_clip: float = 0.2,
    tracking_k: float = 2.0,
    orientation_contact_alpha: float = 3.0 / 7.0,
) -> tuple[jnp.ndarray, ReorientRewardState, dict[str, jnp.ndarray], jnp.ndarray]:
    # drop_factor: smooth height-gated multiplier in [0, 1]. 0 at safe height,
    # ramps via clamped smoothstep to 1 at/below the drop threshold.
    del previous_actions, cube_pos, cube_linvel, finger_positions

    n_contacts = jnp.sum(finger_contact_mask).astype(jnp.float32)

    ang_dist = quat_angular_distance(cube_quat, target_quat)

    angular_progress = jnp.where(state.has_prev, state.prev_ang_dist - ang_dist, 0.0)
    angular_progress = jnp.clip(angular_progress, -angular_progress_clip, angular_progress_clip)

    soft_contact_scale = jnp.minimum(n_contacts / float(min_contacts_for_rotation), 1.0)
    orientation_gate = orientation_contact_alpha + (1.0 - orientation_contact_alpha) * soft_contact_scale
    orientation = jnp.exp(-tracking_k * ang_dist) * orientation_gate

    at_target = ang_dist < success_threshold
    enough_contacts = n_contacts >= min_contacts_for_rotation
    new_success_steps = jnp.where(
        at_target & enough_contacts,
        state.success_steps + 1,
        jnp.array(0, dtype=jnp.int32),
    )
    target_reached = new_success_steps >= success_hold_steps

    cube_drop = drop_penalty_value * drop_factor

    action_penalty = -0.0002 * jnp.sum(actions**2)

    contact_raw = contact_bonus_value * jnp.minimum(n_contacts / 3.0, 1.0)
    finger_contact_bonus = weights.contact_bonus * contact_raw

    no_contact_ramp = jnp.exp(-2.0 * n_contacts)
    no_contact_raw = no_contact_penalty_value * no_contact_ramp
    no_contact_penalty = weights.no_contact * no_contact_raw

    total = (
        weights.angular_progress * angular_progress
        + weights.orientation * orientation
        + weights.cube_drop * cube_drop
        + weights.action_penalty * action_penalty
        + finger_contact_bonus
        + no_contact_penalty
    )

    new_state = ReorientRewardState(
        success_steps=new_success_steps,
        prev_ang_dist=ang_dist,
        initial_cube_pos=state.initial_cube_pos,
        has_prev=jnp.array(True),
    )

    info = {
        "reward/angular_progress": angular_progress,
        "reward/orientation": orientation,
        "reward/cube_drop": cube_drop,
        "reward/action_penalty": action_penalty,
        "reward/finger_contact_bonus": finger_contact_bonus,
        "reward/no_contact_penalty": no_contact_penalty,
        "reward/total": total,
        "metrics/angular_distance": ang_dist,
        "metrics/num_finger_contacts": n_contacts,
        "metrics/success_steps": new_success_steps.astype(jnp.float32),
    }

    return total, new_state, info, target_reached
