
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from dexterous_hand.config import PickPlaceRewardConfig, PickPlaceSceneConfig  # noqa: E402
from dexterous_hand.rewards.pickplace_reward import (  # noqa: E402
    init_pickplace_reward_state,
    pickplace_reward,
)

_CFG = PickPlaceRewardConfig()
_SCFG = PickPlaceSceneConfig()
_TABLE_H = _SCFG.table_height
_HALF = _SCFG.object_half_extent
_REST_Z = _TABLE_H + _HALF
_GOAL = _SCFG.goal_nominal_xy


def _eval(obj_xy, obj_z, gripped, was_lifted=False, speed=0.0, steady_steps=1):
    state = init_pickplace_reward_state(_REST_Z + 0.001, _TABLE_H)
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
        mask = jnp.array([False, False, False, False, False])
    info: dict = {}
    for _ in range(steady_steps):
        _, state, info = pickplace_reward(
            state=state,
            finger_positions=fp,
            object_position=jnp.array([px, py, obj_z]),
            object_linear_velocity=jnp.array([speed, 0.0, 0.0]),
            finger_contact_mask=mask,
            goal_xy=jnp.asarray(_GOAL),
            actions=jnp.zeros(23),
            table_height=_TABLE_H,
            object_half_extent=_HALF,
            weights=_CFG.weights,
            lift_target=_CFG.lift_target,
            carry_clear_height=_CFG.carry_clear_height,
            goal_radius=_CFG.goal_radius,
            hold_velocity_threshold=_CFG.hold_velocity_threshold,
            drop_penalty_value=_CFG.drop_penalty,
            no_contact_idle_penalty=_CFG.no_contact_idle_penalty,
            success_bonus_per_step=_CFG.success_bonus_per_step,
            place_hold_steps=_CFG.place_hold_steps,
            reach_tanh_k=_CFG.reach_tanh_k,
            transport_tanh_k=_CFG.transport_tanh_k,
            goal_tanh_k=_CFG.goal_tanh_k,
            on_table_tol=_CFG.on_table_tol,
            on_table_k=_CFG.on_table_k,
            at_rest_k=_CFG.at_rest_k,
            fingertip_weights=_CFG.fingertip_weights,
            drop_arm_height=_CFG.drop_arm_height,
            action_penalty_scale=_CFG.action_penalty_scale,
            idle_grace_steps=_CFG.idle_grace_steps,
        )
    return info


def test_release_at_goal_beats_hover():
    hover = _eval(_GOAL, _REST_Z + _CFG.lift_target, gripped=True)
    settled = _eval(
        _GOAL, _REST_Z, gripped=False, was_lifted=True, steady_steps=_CFG.place_hold_steps + 5
    )
    assert float(settled["reward/total"]) > float(hover["reward/total"])


def test_release_no_dip_before_annuity():
    hover = _eval(_GOAL, _REST_Z + _CFG.lift_target, gripped=True)
    settled_nos = _eval(_GOAL, _REST_Z, gripped=False, was_lifted=True, steady_steps=1)
    assert float(settled_nos["reward/total"]) > float(hover["reward/total"])


def test_bulldoze_pays_no_placed():
    info = _eval(_GOAL, _REST_Z, gripped=True, was_lifted=False)
    assert float(info["reward/placed"]) < 0.05


def test_success_annuity_requires_release():
    steps = _CFG.place_hold_steps + 5
    released = _eval(_GOAL, _REST_Z, gripped=False, was_lifted=True, steady_steps=steps)
    gripped = _eval(_GOAL, _REST_Z, gripped=True, was_lifted=True, steady_steps=steps)
    assert float(released["is_success"]) == 1.0
    assert float(gripped["is_success"]) == 0.0


def test_drop_penalty_only_away_from_goal():
    far_xy = (_GOAL[0] + 0.15, _GOAL[1])
    info_far = _eval(far_xy, _REST_Z, gripped=False, was_lifted=True)
    assert float(info_far["reward/drop"]) < 0.0
    info_goal = _eval(_GOAL, _REST_Z, gripped=False, was_lifted=True)
    assert float(info_goal["reward/drop"]) == 0.0
