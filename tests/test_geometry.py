"""Geometry invariants for the peg task.

These pin the physical-reachability relationship the project repeatedly broke
(round-16 FAIL: the success threshold sat above what the hand could mechanically
achieve). See the project memory note `peg_insertion_geometric_ceiling`.
"""
import pytest

mujoco = pytest.importorskip("mujoco")

from dexterous_hand.config import PegRewardConfig, PegSceneConfig  # noqa: E402
from dexterous_hand.envs.peg_scene_builder import build_peg_scene  # noqa: E402
from dexterous_hand.envs.scene_builder import (  # noqa: E402
    GRIP_BIAS,
    apply_flexion_bias,
    build_grip_ctrl,
)


def _peg_length(cfg: PegSceneConfig) -> float:
    return cfg.peg_half_length * 2.0 + cfg.peg_radius * 2.0


def test_success_depth_fits_in_tube():
    """Cheap necessary condition: the success depth cannot exceed the distance
    from the hole entrance to the table top (the peg collides with the solid
    table — there is no bore through it — so the tip can't descend past it)."""
    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    required = rcfg.success_threshold * _peg_length(cfg)
    max_possible = cfg.hole_top_above_table  # entrance is this far above table top
    assert required < max_possible, (
        f"success needs {required * 1000:.1f}mm insertion but the hole entrance is "
        f"only {max_possible * 1000:.1f}mm above the table top — raise "
        f"hole_top_above_table or lower success_threshold"
    )


@pytest.mark.slow
def test_peg_insertion_physically_reachable():
    """Sufficient condition: a closed grip driving slide_z to its limit must
    achieve insertion >= success_threshold (with margin). This catches the
    round-16 blocker, where the hand's knuckles bottom out on the table and cap
    descent below the success depth regardless of reward tuning."""
    import jax.numpy as jnp

    from dexterous_hand.utils.mjx_helpers import get_insertion_depth_jax

    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, data, nm = build_peg_scene(cfg)
    peg_len = _peg_length(cfg)

    peg_qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    ]
    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    sz_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide_z_act")

    init = data.qpos.copy()
    apply_flexion_bias(init, model, bias_map=GRIP_BIAS)
    grip = build_grip_ctrl(model)

    # pre-grasped reset: peg teleported into the grip, settle
    data.qpos[:] = init
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.qpos[peg_qadr : peg_qadr + 3] = data.site_xpos[grasp_site].copy()
    data.qpos[peg_qadr + 3 : peg_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.ctrl[:] = grip
    for _ in range(5):
        mujoco.mj_step(model, data)

    # drive slide_z to its lower limit (full descent), hold the grip
    ctrl = grip.copy()
    ctrl[sz_act] = model.actuator_ctrlrange[sz_act, 0]
    data.ctrl[:] = ctrl
    for _ in range(800):
        mujoco.mj_step(model, data)

    depth = float(
        get_insertion_depth_jax(
            jnp.array(data.xpos),
            jnp.array(data.xmat),
            nm.peg_body_id,
            nm.hole_body_id,
            cfg.peg_half_length,
            cfg.peg_radius,
        )
    )
    frac = depth / peg_len
    assert frac >= rcfg.success_threshold + 0.05, (
        f"grip-and-descend only reaches insertion_fraction={frac:.3f}, below "
        f"success_threshold={rcfg.success_threshold} (+0.05 margin). The hand "
        f"cannot mechanically push the peg deep enough; raise hole_top_above_table "
        f"(currently {cfg.hole_top_above_table}) or lower success_threshold."
    )
