"""Geometry invariants for the peg and grasp tasks.

These pin the physical-reachability relationships the project repeatedly broke:

- round-16 FAIL: the peg success threshold sat above what the hand could
  mechanically achieve (see memory note `peg_insertion_geometric_ceiling`);
- 2026-06-10 audit: the insertion-depth metric had no lateral containment, so
  with the entrance raised 8cm above the table ANY peg at table level scored
  insertion_fraction 1.0 — a peg lying on the table out-scored a fully
  inserted one (0.757) and "drop the peg" became a free terminal success
  (see memory note `peg_false_insertion_success`). The earlier version of this
  file validated that broken metric: its grip-descend ran ~10cm laterally from
  the hole, so its "0.94 achievable" was open-air descent next to the tube.
- 2026-07-14: grasp `lift_target` had eroded 0.1 -> 0.012 across rounds 11-13
  because the scene had no vertical arm DOF (finger curl caps lift at ~1cm).
  slide_z was added and lift_target restored to 0.10; the grasp test below
  guards that the target stays physically reachable from a formed grip
  (measured: 51% of sampled grip poses hold a 20cm+ lift; best ~0.235m).
"""
import pytest

mujoco = pytest.importorskip("mujoco")
jnp = pytest.importorskip("jax.numpy")

import numpy as np  # noqa: E402

from dexterous_hand.config import (  # noqa: E402
    PegRewardConfig,
    PegSceneConfig,
    RewardConfig,
    SceneConfig,
)
from dexterous_hand.envs.peg_scene_builder import build_peg_scene  # noqa: E402
from dexterous_hand.envs.scene_builder import (  # noqa: E402
    OBJECT_TYPES,
    build_scene,
    get_object_half_height,
)
from dexterous_hand.utils.mjx_helpers import get_insertion_depth_jax  # noqa: E402


def _peg_length(cfg: PegSceneConfig) -> float:
    return cfg.peg_half_length * 2.0 + cfg.peg_radius * 2.0


def _measure_depth(cfg: PegSceneConfig, model, data, nm) -> float:
    return float(
        get_insertion_depth_jax(
            jnp.array(data.xpos),
            jnp.array(data.xmat),
            nm.peg_body_id,
            nm.hole_body_id,
            cfg.peg_half_length,
            cfg.peg_radius,
            cfg.peg_radius + cfg.clearance,
            cfg.hole_depth,
        )
    )


def _set_peg_pose(model, data, pos, quat) -> None:
    peg_qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    ]
    data.qpos[peg_qadr : peg_qadr + 3] = pos
    data.qpos[peg_qadr + 3 : peg_qadr + 7] = quat
    data.qvel[:] = 0.0


def test_success_depth_fits_in_tube():
    """Cheap necessary condition: the success depth must be physically reachable
    INSIDE the tube. The binding bound is whichever the descending tip hits
    first — the top surface of the hole_bottom plate, or the solid table (there
    is no bore through it). Read both from the compiled model, not from config
    arithmetic, so a scene-builder change can't silently invalidate this."""
    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, _, _ = build_peg_scene(cfg)

    entrance_z = float(model.body("hole").pos[2])
    bottom_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hole_bottom")
    bottom_body = model.geom_bodyid[bottom_gid]
    bottom_top_z = float(
        model.body_pos[bottom_body][2] + model.geom_pos[bottom_gid][2] + model.geom_size[bottom_gid][2]
    )
    max_in_tube = min(entrance_z - bottom_top_z, cfg.hole_top_above_table)

    required = rcfg.success_threshold * _peg_length(cfg)
    assert required < max_in_tube, (
        f"success needs {required * 1000:.1f}mm insertion but the tube only "
        f"admits {max_in_tube * 1000:.1f}mm (entrance z={entrance_z:.4f}, hole "
        f"bottom top z={bottom_top_z:.4f}) — deepen hole_depth / raise "
        f"hole_top_above_table or lower success_threshold"
    )


def test_insertion_depth_requires_lateral_containment():
    """Regression test for the 2026-06-10 false-success bug: a peg at table
    level OUTSIDE the bore must measure zero insertion depth, while a peg
    inside the tube must clear the success depth. Pure kinematics (mj_forward),
    no stepping."""
    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, data, nm = build_peg_scene(cfg)
    peg_len = _peg_length(cfg)
    required = rcfg.success_threshold * peg_len

    upright = [1.0, 0.0, 0.0, 0.0]
    lying = [0.7071068, 0.7071068, 0.0, 0.0]
    spawn_z = cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001

    # pre-fix these all measured depth ~0.080 (fraction 1.0 > threshold 0.7)
    outside_poses = [
        ("upright at spawn_min_radius", [cfg.spawn_min_radius, 0.0, spawn_z], upright),
        ("upright in table corner", [0.20, 0.10, spawn_z], upright),
        ("lying flat 10cm out", [0.10, 0.0, cfg.table_height + cfg.peg_radius + 0.001], lying),
    ]
    # Round-17 exploit poses: a table-lying peg with one END slid into the
    # slot under the floating tube. The lower end sits ON the hole axis, so
    # the lateral gate passes; only the AXIAL window (lower-end depth 0.072 >
    # hole_depth 0.06) zeroes it. Kinematic pose — the pedestal now blocks it
    # physically, but the metric must be honest on its own (pre-axial-window
    # these measured depth 0.080, fraction 1.0, with zero wall contact).
    for deg in (3.0, 8.0):
        theta = np.deg2rad(90.0 - deg)
        axis = np.array([np.sin(theta), 0.0, np.cos(theta)])
        center = (
            np.array([0.0, 0.0, cfg.table_height + cfg.peg_radius])
            + axis * cfg.peg_half_length
        )
        quat_about_y = [np.cos(theta / 2.0), 0.0, np.sin(theta / 2.0), 0.0]
        outside_poses.append(
            (f"under-tube, {deg:.0f}deg tilt, end below bore", center.tolist(), quat_about_y)
        )
    for label, pos, quat in outside_poses:
        _set_peg_pose(model, data, pos, quat)
        mujoco.mj_forward(model, data)
        depth = _measure_depth(cfg, model, data, nm)
        assert depth == 0.0, (
            f"{label}: never-inserted peg measured depth {depth:.4f} — the "
            f"lateral containment gate is not working"
        )

    # a peg resting on the hole bottom, centred in the bore, must register
    in_tube_z = (
        float(model.body("hole").pos[2])
        - cfg.hole_depth
        + 0.0025  # hole_bottom plate half-thickness (wt/2)
        + cfg.peg_half_length
        + cfg.peg_radius
    )
    _set_peg_pose(model, data, [0.0, 0.0, in_tube_z], upright)
    mujoco.mj_forward(model, data)
    depth = _measure_depth(cfg, model, data, nm)
    assert depth > required, (
        f"bottomed-out in-tube peg measured depth {depth:.4f} <= required "
        f"{required:.4f} — either the containment gate wrongly zeroes real "
        f"insertions or the tube is too shallow for success_threshold"
    )


def test_under_tube_slot_is_blocked():
    """Round-17: the tube floats (entrance 8cm up, walls only 6cm deep), so a
    pedestal geom must fill table-top -> plate-underside; otherwise a lying peg
    (1.6cm dia) fits in the ~1.75cm slot and gets lost under the receptacle.
    Read the bounds from the compiled model so builder changes can't silently
    reopen the gap."""
    cfg = PegSceneConfig()
    model, _, _ = build_peg_scene(cfg)

    ped_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hole_pedestal")
    assert ped_gid >= 0, "hole_pedestal geom missing — the under-tube slot is open"

    body = model.geom_bodyid[ped_gid]
    body_z = float(model.body_pos[body][2])
    ped_top = body_z + float(model.geom_pos[ped_gid][2]) + float(model.geom_size[ped_gid][2])
    ped_bottom = body_z + float(model.geom_pos[ped_gid][2]) - float(model.geom_size[ped_gid][2])

    plate_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hole_bottom")
    plate_bottom = (
        body_z + float(model.geom_pos[plate_gid][2]) - float(model.geom_size[plate_gid][2])
    )

    assert ped_top >= plate_bottom - 1e-9, (
        f"pedestal top {ped_top:.4f} leaves a gap below the plate underside "
        f"{plate_bottom:.4f}"
    )
    assert ped_bottom <= cfg.table_height + 1e-9, (
        f"pedestal bottom {ped_bottom:.4f} floats above the table top "
        f"{cfg.table_height:.4f}"
    )
    # footprint must cover the walls' footprint so nothing can slide under them
    wall_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hole_wall_px")
    wall_outer_x = float(model.geom_pos[wall_gid][0]) + float(model.geom_size[wall_gid][0])
    assert float(model.geom_size[ped_gid][0]) >= wall_outer_x - 1e-9
    assert float(model.geom_size[ped_gid][1]) >= wall_outer_x - 1e-9


@pytest.mark.slow
def test_peg_drop_insertion_reaches_success_depth():
    """Sufficient condition under real physics: a peg released just above the
    hole entrance must fall into the tube, settle on the bottom, and HOLD an
    insertion fraction >= success_threshold (with margin). This is the honest
    reachability bound: with the entrance hole_top_above_table above the table,
    the hand never needs to descend to table level (the round-16 knuckle-cap
    constraint) — it needs to release the peg over the bore. The measured
    in-tube ceiling is ~0.757; success_threshold=0.70 leaves ~+4mm."""
    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, data, nm = build_peg_scene(cfg)
    peg_len = _peg_length(cfg)

    entrance_z = float(model.body("hole").pos[2])
    _set_peg_pose(model, data, [0.0, 0.0, entrance_z + 0.02], [1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    for _ in range(400):
        mujoco.mj_step(model, data)

    frac = _measure_depth(cfg, model, data, nm) / peg_len
    assert frac >= rcfg.success_threshold + 0.03, (
        f"dropped-in peg settles at insertion_fraction={frac:.3f}, below "
        f"success_threshold={rcfg.success_threshold} (+0.03 margin). The tube "
        f"cannot physically admit the success depth; deepen hole_depth, raise "
        f"hole_top_above_table, or lower success_threshold."
    )

    # success additionally requires holding the depth for peg_hold_steps; the
    # settled pose must be stable, not a bounce artifact
    fracs = []
    for _ in range(rcfg.peg_hold_steps * 5):
        mujoco.mj_step(model, data)
        fracs.append(_measure_depth(cfg, model, data, nm) / peg_len)
    assert np.min(fracs) >= rcfg.success_threshold, (
        f"settled peg does not HOLD the success depth (min fraction over "
        f"{len(fracs)} steps = {np.min(fracs):.3f})"
    )


@pytest.mark.slow
def test_peg_transport_release_insertion():
    """Full winning-trajectory proof under real physics: from the env's
    pre-grasped reset (GRIP_BIAS + peg at grasp_site + 5-step settle), raise
    the peg, servo it over the bore, ENGAGE the tip ~2cm into the bore, then
    release — the peg must slide to the bottom (fraction 0.757) and hold
    above success_threshold. Guards the 2026-07-14 endgame redesign:
    - releasing above the entrance topples the peg (~6 deg self-alignment
      cone at 4mm clearance), so the reward's place target is the ENGAGED
      pose — this test breaks if that geometry regresses;
    - peg<->bore friction pairs (mu=0.2): at the default mu=1.0 the released
      peg two-point-wedged at fraction ~0.55 and never reached success depth.
    Keep the trajectory in sync with scripts/render_peg_transport.py."""
    import numpy as np

    from dexterous_hand.envs.scene_builder import (
        GRIP_BIAS,
        apply_flexion_bias,
        build_grip_ctrl,
    )

    cfg = PegSceneConfig()
    rcfg = PegRewardConfig()
    model, data, nm = build_peg_scene(cfg)
    peg_len = _peg_length(cfg)

    qpos = data.qpos.copy()
    apply_flexion_bias(qpos, model, bias_map=GRIP_BIAS)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    _set_peg_pose(model, data, data.site_xpos[sid].copy(), [1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    grip = build_grip_ctrl(model)
    data.ctrl[:] = grip
    for _ in range(5):
        mujoco.mj_step(model, data)

    hole_pos = data.xpos[nm.hole_body_id].copy()
    entrance_z = hole_pos[2]

    def act(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def slide_xy() -> np.ndarray:
        out = []
        for n in ("slide_x", "slide_y"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            out.append(float(data.qpos[model.jnt_qposadr[jid]]))
        return np.array(out)

    z_cmd = 0.0
    xy_cmd = np.zeros(2)
    open_frac = 0.0

    def do_steps(n: int, open_fingers: bool = False, servo: bool = False) -> None:
        nonlocal xy_cmd, open_frac
        for _ in range(n):
            if open_fingers:
                open_frac = min(open_frac + 0.15, 1.0)
                c = grip * (1.0 - open_frac)
            else:
                c = grip.copy()
            if servo:
                err = hole_pos[:2] - data.xpos[nm.peg_body_id][:2]
                desired = slide_xy() + 0.8 * err
                xy_cmd = xy_cmd + np.clip(desired - xy_cmd, -0.003, 0.003)
            c[act("slide_x_act")] = xy_cmd[0]
            c[act("slide_y_act")] = xy_cmd[1]
            lo, hi = model.actuator_ctrlrange[act("slide_z_act")]
            c[act("slide_z_act")] = float(np.clip(z_cmd, lo, hi))
            data.ctrl[:] = c
            mujoco.mj_step(model, data, nstep=cfg.frame_skip)

    z_cmd = 0.06
    do_steps(15)
    xy_cmd = slide_xy() + (hole_pos[:2] - data.xpos[nm.peg_body_id][:2])
    do_steps(25)
    do_steps(15, servo=True)
    tip_z = data.xpos[nm.peg_body_id][2] - peg_len / 2.0
    z_cmd += entrance_z + 0.01 - tip_z
    do_steps(15, servo=True)
    for _ in range(10):  # gradual engagement descent, servoing
        tip_z = data.xpos[nm.peg_body_id][2] - peg_len / 2.0
        z_cmd += float(np.clip((entrance_z - 0.020) - tip_z, -0.004, 0.004))
        do_steps(2, servo=True)
    do_steps(15, open_fingers=True)
    z_cmd += 0.06
    do_steps(25, open_fingers=True)

    frac = _measure_depth(cfg, model, data, nm) / peg_len
    assert frac >= rcfg.success_threshold + 0.03, (
        f"transport+engaged-release settles at fraction {frac:.3f} < "
        f"success_threshold+0.03 — the winning trajectory is no longer "
        f"physically achievable (check bore friction pairs, release geometry, "
        f"grip bias)"
    )
    fracs = []
    for _ in range(50):
        do_steps(1, open_fingers=True)
        fracs.append(_measure_depth(cfg, model, data, nm) / peg_len)
    assert np.min(fracs) >= rcfg.success_threshold, (
        f"released peg does not HOLD success depth (min {np.min(fracs):.3f})"
    )


# --- grasp task -------------------------------------------------------------

# A grip seed measured to hold the cube through a 20cm+ slide_z lift (from a
# 432-combo CPU sweep, 2026-07-14: 222/432 combos held final lift > 10cm; this
# one held 0.235m with contact on every one of the last 40 steps). It is a
# REPRESENTATIVE reachable grip, not an optimum — if this regresses, the task
# geometry (cube size / mount / slide_z / friction) broke, not the policy.
CUBE_GRIP_SEED = {
    "sx": 0.115, "sy": -0.017, "z0": -0.02,
    "j3": 1.0, "j12": 0.5, "thj5": 0.5, "th1": 0.7, "squeeze": 0.4,
}
CUBE_SPAWN_XY = (0.075, 0.0)  # centre of the env's spawn band


def _grasp_set_joint(model, qpos, name: str, val: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    qpos[model.jnt_qposadr[jid]] = float(np.clip(val, lo, hi))


def _grasp_seta(model, ctrl, act_name: str, target: float) -> None:
    ai = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    if ai < 0:
        return
    lo, hi = model.actuator_ctrlrange[ai]
    ctrl[ai] = float(np.clip(target, lo, hi))


def _cube_grip_ctrl(model, p: dict, squeeze: float, z: float) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=np.float64)
    _grasp_seta(model, ctrl, "slide_x_act", p["sx"])
    _grasp_seta(model, ctrl, "slide_y_act", p["sy"])
    _grasp_seta(model, ctrl, "slide_z_act", z)
    for an in ("rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"):
        _grasp_seta(model, ctrl, an, p["j3"] + squeeze)
    for an in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
        _grasp_seta(model, ctrl, an, p["j12"] * 2 + squeeze)
    _grasp_seta(model, ctrl, "rh_A_THJ5", p["thj5"])
    _grasp_seta(model, ctrl, "rh_A_THJ4", 1.2)
    _grasp_seta(model, ctrl, "rh_A_THJ2", 0.3)
    _grasp_seta(model, ctrl, "rh_A_THJ1", p["th1"] + squeeze)
    return ctrl


@pytest.mark.slow
def test_grasp_lift_reaches_target_height():
    """Sufficient condition under real physics: from a formed grip around the
    cube, raising slide_z must lift the cube past lift_target (with margin)
    and HOLD it there with hand contact. Guards the 2026-07-14 winnability
    restoration: without slide_z this ceiling was ~1cm and lift_target=0.10
    would be unreachable (the round-11-13 failure mode, hidden then by
    lowering the bar to 0.012 instead of fixing the scene)."""
    scfg = SceneConfig()
    rcfg = RewardConfig()
    model, data, nm = build_scene(scfg)
    p = CUBE_GRIP_SEED

    qpos = data.qpos.copy()
    _grasp_set_joint(model, qpos, "slide_x", p["sx"])
    _grasp_set_joint(model, qpos, "slide_y", p["sy"])
    _grasp_set_joint(model, qpos, "slide_z", p["z0"])
    for j in ("FF", "MF", "RF", "LF"):
        _grasp_set_joint(model, qpos, f"rh_{j}J3", p["j3"])
        _grasp_set_joint(model, qpos, f"rh_{j}J2", p["j12"])
        _grasp_set_joint(model, qpos, f"rh_{j}J1", p["j12"])
    _grasp_set_joint(model, qpos, "rh_THJ5", p["thj5"])
    _grasp_set_joint(model, qpos, "rh_THJ4", 1.2)
    _grasp_set_joint(model, qpos, "rh_THJ2", 0.3)
    _grasp_set_joint(model, qpos, "rh_THJ1", p["th1"])

    gt, gs = OBJECT_TYPES["large_cube"]
    obj_z0 = scfg.table_height + get_object_half_height(gt, gs) + 0.001
    s = nm.obj_qpos_start
    qpos[s : s + 3] = [CUBE_SPAWN_XY[0], CUBE_SPAWN_XY[1], obj_z0]
    qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    hand_geoms: set[int] = set()
    for gset in nm.finger_geom_ids_per_finger:
        hand_geoms |= gset

    lift_z = 0.18  # slide_z command during the lift phase (range tops at 0.20)
    n_settle, n_lift, n_hold = 30, 40, 80
    total = n_settle + n_lift + n_hold
    final_lift = 0.0
    held = 0
    for step in range(total):
        if step < n_settle:
            squeeze, z = p["squeeze"] * min(step / 10.0, 1.0), p["z0"]
        elif step < n_settle + n_lift:
            t = (step - n_settle) / n_lift
            squeeze, z = p["squeeze"], p["z0"] + (lift_z - p["z0"]) * t
        else:
            squeeze, z = p["squeeze"], lift_z
        data.ctrl[: model.nu] = _cube_grip_ctrl(model, p, squeeze, z)
        mujoco.mj_step(model, data, nstep=scfg.frame_skip)
        final_lift = float(data.xpos[nm.object_body_id][2] - obj_z0)
        if step >= total - 40:
            ncon = sum(
                1
                for ci in range(data.ncon)
                if (data.contact[ci].geom1 == nm.object_geom_id
                    and data.contact[ci].geom2 in hand_geoms)
                or (data.contact[ci].geom2 == nm.object_geom_id
                    and data.contact[ci].geom1 in hand_geoms)
            )
            held += int(ncon > 0)

    assert final_lift >= rcfg.lift_target + 0.05, (
        f"gripped cube only lifted {final_lift * 1000:.1f}mm; lift_target="
        f"{rcfg.lift_target * 1000:.0f}mm (+50mm margin) is not physically "
        f"reachable — slide_z range/gains, cube size or friction broke"
    )
    assert held >= 39, (
        f"cube did not stay held at height (contact on {held}/40 final steps) "
        f"— the sustained hold that `holding` pays for is not achievable"
    )
