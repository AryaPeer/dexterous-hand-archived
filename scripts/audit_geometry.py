"""Math audit of grasp/peg/reorient. Loads each scene and prints actual values
of every claim made in the analysis (positions, gradients, settle behavior).

Run from repo root:  uv run python scripts/audit_geometry.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dexterous_hand.config import (
    PegRewardConfig,
    PegSceneConfig,
    ReorientRewardConfig,
    ReorientSceneConfig,
    RewardConfig,
    SceneConfig,
)
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.reorient_scene_builder import build_reorient_scene
from dexterous_hand.envs.scene_builder import (
    GRIP_BIAS,
    TABLE_TASK_FLEXION_BIAS,
    apply_flexion_bias,
    build_scene,
)


def hr(title):
    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def body_pos(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        return None
    return np.array(data.xpos[bid])


def site_pos(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        return None
    return np.array(data.site_xpos[sid])


def actuator_summary(model):
    print(f"  num actuators: {model.nu}")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        mid = (lo + hi) / 2
        zero_inside = lo <= 0.0 <= hi
        marker = " <-- ctrl=0 is %.2f rad from low end" % (0.0 - lo) if zero_inside else " <-- ctrl=0 OUTSIDE range"
        print(f"    {i:2d} {name:24s} range=[{lo:+.3f}, {hi:+.3f}] mid={mid:+.3f}{marker}")


def settle_test(model, data, label, n_steps=5, ctrl=None):
    """Run n_steps mj_step with given ctrl, return cube/object trajectory."""
    if ctrl is None:
        ctrl = np.zeros(model.nu)
    data.ctrl[:] = ctrl
    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    if obj_jid < 0:
        obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    if obj_jid < 0:
        obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    qadr = model.jnt_qposadr[obj_jid]
    traj = []
    n_sub = 20  # match env's frame_skip
    for step in range(n_steps):
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        z = float(data.qpos[qadr + 2])
        traj.append(z)
    print(f"  [{label}] object z trajectory over {n_steps} settle steps: " + " -> ".join(f"{z:.4f}" for z in traj))
    return traj


def contact_count(model, data, finger_touch_names):
    n = 0
    forces = []
    for name in finger_touch_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_{name}")
        if sid < 0:
            continue
        adr = model.sensor_adr[sid]
        f = float(data.sensordata[adr])
        forces.append(f)
        if f > 0:
            n += 1
    return n, forces


# ============================================================================
# GRASP
# ============================================================================
def audit_grasp():
    hr("GRASP")
    cfg = SceneConfig()
    model, data, nm = build_scene(cfg)
    apply_flexion_bias(data.qpos, model)
    mujoco.mj_forward(model, data)

    print(f"  table_height       = {cfg.table_height:.4f}")
    print(f"  mount_height       = {cfg.mount_height:.4f}")
    print(f"  object_half_height = 0.035 (large_cube)")

    palm = body_pos(model, data, "rh_palm")
    print(f"  palm world pos     = {palm}")
    print(f"  palm Z above table = {palm[2] - cfg.table_height:.4f}m")

    for name in ["fftip", "mftip", "rftip", "lftip", "thtip"]:
        p = site_pos(model, data, name)
        print(f"  fingertip {name:6s} world pos = {p}  z_above_table = {p[2] - cfg.table_height:+.4f}")

    obj_z_rest = cfg.table_height + 0.035 + 0.001
    print(f"  object spawn z     = {obj_z_rest:.4f} (on table)")

    finger_to_obj = []
    obj_xy = np.array([0.0, 0.0, obj_z_rest])
    for name in ["fftip", "mftip", "rftip", "lftip", "thtip"]:
        p = site_pos(model, data, name)
        finger_to_obj.append(np.linalg.norm(p - obj_xy))
    print(f"  fingertip distances to object (at center, on table):")
    for n, d in zip(["ff", "mf", "rf", "lf", "th"], finger_to_obj):
        print(f"    {n}: {d:.4f}m")
    print(f"  -> mean fingertip-to-object distance: {np.mean(finger_to_obj):.4f}m")

    # The lift_target check
    rc = RewardConfig()
    print(f"\n  lift_target = {rc.lift_target:.4f}m ({rc.lift_target*100:.1f}cm)")
    print(f"  success_bonus = {rc.success_bonus}")
    print(f"  drop_penalty  = {rc.drop_penalty}")

    print(f"\n  ACTUATOR SUMMARY (does ctrl=0 mean open or closed?):")
    actuator_summary(model)


# ============================================================================
# REORIENT
# ============================================================================
def audit_reorient():
    hr("REORIENT")
    cfg = ReorientSceneConfig()
    model, data, nm = build_reorient_scene(cfg)
    apply_flexion_bias(data.qpos, model, bias_map=GRIP_BIAS)
    mujoco.mj_forward(model, data)

    print(f"  mount_height = {cfg.mount_height:.4f}")
    print(f"  cube_size    = {cfg.cube_size:.4f} (half-edge)")
    print(f"  cube_mass    = {cfg.cube_mass:.4f}")
    print(f"  friction     = {cfg.cube_friction}")
    print(f"  drop_height_offset = {ReorientRewardConfig().drop_height_offset}")

    palm = body_pos(model, data, "rh_palm")
    print(f"\n  palm world pos = {palm}")
    grasp = site_pos(model, data, "grasp_site")
    print(f"  grasp_site world pos = {grasp}")
    print(f"  grasp_site relative to palm = {grasp - palm}")

    for name in ["fftip", "mftip", "rftip", "lftip", "thtip"]:
        p = site_pos(model, data, name)
        delta = p - grasp
        print(f"  fingertip {name:6s} pos = {p}  dist to grasp_site = {np.linalg.norm(delta):.4f}m")

    # Place cube at grasp_site, settle for 5 steps WITH zero ctrl (matches env)
    print()
    print("  --- Reproducing _reset_single behavior ---")
    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    qadr = model.jnt_qposadr[cube_jid]
    vadr = model.jnt_dofadr[cube_jid]
    data.qpos[qadr:qadr + 3] = grasp
    data.qpos[qadr + 3:qadr + 7] = np.array([1, 0, 0, 0])
    data.qvel[vadr:vadr + 6] = 0
    mujoco.mj_forward(model, data)

    print(f"  cube placed at grasp_site = ({grasp[0]:.3f}, {grasp[1]:.3f}, {grasp[2]:.3f})")
    print(f"  palm_z = {palm[2]:.4f}, drop threshold = palm_z - 0.05 = {palm[2] - 0.05:.4f}")

    # Hypothesis 1: zero ctrl opens the fingers (drops cube)
    settle_test(model, data, "zero ctrl (env current behavior)")
    n, forces = contact_count(model, data, ["ff_touch", "mf_touch", "rf_touch", "lf_touch", "th_touch"])
    print(f"    -> finger contacts after settle: {n} / 5, forces = {[f'{f:.3f}' for f in forces]}")
    print(f"    -> cube z after settle: {float(data.qpos[qadr + 2]):.4f}")
    print(f"    -> drop offset from palm: {float(data.qpos[qadr + 2]) - palm[2]:.4f}m")

    # Hypothesis 2: ctrl that matches GRIP_BIAS (keeps fingers closed)
    print()
    model2, data2, _ = build_reorient_scene(cfg)
    apply_flexion_bias(data2.qpos, model2, bias_map=GRIP_BIAS)
    mujoco.mj_forward(model2, data2)
    palm2 = body_pos(model2, data2, "rh_palm")
    grasp2 = site_pos(model2, data2, "grasp_site")
    data2.qpos[qadr:qadr + 3] = grasp2
    data2.qpos[qadr + 3:qadr + 7] = np.array([1, 0, 0, 0])
    data2.qvel[vadr:vadr + 6] = 0
    mujoco.mj_forward(model2, data2)

    grip_ctrl = np.zeros(model2.nu)
    for jname, target in GRIP_BIAS.items():
        # find actuator for this joint
        for ai in range(model2.nu):
            aname = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_ACTUATOR, ai)
            if aname and aname.startswith(jname.replace("rh_", "rh_") + "_") or aname == jname or (aname and jname.replace("J", "A") in aname):
                lo, hi = model2.actuator_ctrlrange[ai]
                grip_ctrl[ai] = float(np.clip(target, lo, hi))
                break
    # second attempt: match actuator-to-joint via trntype/trnid for position actuators
    for ai in range(model2.nu):
        trntype = int(model2.actuator_trntype[ai])
        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model2.actuator_trnid[ai, 0])
            jname = mujoco.mj_id2name(model2, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname in GRIP_BIAS:
                lo, hi = model2.actuator_ctrlrange[ai]
                grip_ctrl[ai] = float(np.clip(GRIP_BIAS[jname], lo, hi))
    settle_test(model2, data2, "GRIP_BIAS ctrl (proposed fix)", ctrl=grip_ctrl)
    n, forces = contact_count(model2, data2, ["ff_touch", "mf_touch", "rf_touch", "lf_touch", "th_touch"])
    print(f"    -> finger contacts after settle: {n} / 5, forces = {[f'{f:.3f}' for f in forces]}")
    print(f"    -> cube z after settle: {float(data2.qpos[qadr + 2]):.4f}")
    print(f"    -> drop offset from palm: {float(data2.qpos[qadr + 2]) - palm2[2]:.4f}m")

    # Reward shape near initial position
    print()
    print("  --- REWARD GRADIENT MATH (orientation reward) ---")
    rc = ReorientRewardConfig()
    print(f"  tracking_k = {rc.tracking_k}, orientation_contact_alpha = {rc.orientation_contact_alpha}")
    print(f"  orientation_gate at nfc=0: {rc.orientation_contact_alpha:.4f}")
    print(f"  orientation_gate at nfc>=1 (min_contacts=1): 1.0")
    print(f"  orientation reward at ang_dist (in rad) and nfc=0 (gate={rc.orientation_contact_alpha:.3f}):")
    for ang in [0.1, 0.3, 0.5, 1.0, math.pi/2, math.pi]:
        o = math.exp(-rc.tracking_k * ang) * rc.orientation_contact_alpha
        print(f"    ang={ang:.3f} rad ({math.degrees(ang):5.1f}°): orientation = {o:.6f}")

    # cube_drop reward at full drop
    print()
    print("  --- DROP PENALTY MATH ---")
    print(f"  drop_penalty_value = {rc.drop_penalty}, weight = 5")
    print(f"  per-step cube_drop at drop_factor=1.0: {rc.drop_penalty * 1.0 * 5:.2f}")
    print(f"  per-step cube_drop at drop_factor=0.5: {rc.drop_penalty * 0.5 * 5:.2f}")
    print(f"  observed cube_drop reward in CSV: -13 -> drop_factor = {-13 / (rc.drop_penalty * 5):.3f}")

    # action penalty
    print()
    print("  --- ACTION PENALTY ---")
    print(f"  action_penalty = -0.0002 * sum(actions^2), weight = 1")
    n_act = model.nu
    print(f"  n_act = {n_act}, max |action|=1, max sum = {n_act}")
    print(f"  per-step action_penalty at max: {-0.0002 * n_act * 1:.4f}")
    print(f"  -> too small to influence the policy")


# ============================================================================
# PEG
# ============================================================================
def audit_peg():
    hr("PEG")
    cfg = PegSceneConfig()
    rc = PegRewardConfig()
    model, data, nm = build_peg_scene(cfg)
    apply_flexion_bias(data.qpos, model)
    mujoco.mj_forward(model, data)

    print(f"  table_height        = {cfg.table_height:.4f}")
    print(f"  peg_half_length     = {cfg.peg_half_length:.4f} (full length = {cfg.peg_half_length*2:.3f})")
    print(f"  peg_radius          = {cfg.peg_radius:.4f}")
    print(f"  hole_depth          = {cfg.hole_depth:.4f}")
    print(f"  clearance           = {cfg.clearance:.4f}")
    print(f"  lift_target         = {rc.lift_target:.4f}m ({rc.lift_target*100:.1f}cm)")

    palm = body_pos(model, data, "rh_palm")
    print(f"\n  palm world pos = {palm}")
    print(f"  palm Z above table = {palm[2] - cfg.table_height:.4f}m")

    peg_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    hole_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hole")
    if peg_body_id >= 0:
        print(f"  peg world pos (default xml) = {data.xpos[peg_body_id]}")
    if hole_body_id >= 0:
        print(f"  hole world pos (default xml) = {data.xpos[hole_body_id]}")

    # Initial peg height (where reset places it)
    initial_z = cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001
    print(f"\n  computed initial_peg_height = {initial_z:.4f}")

    # Stage thresholds in _step_single
    print(f"\n  Stage check thresholds (from peg_env._step_single):")
    print(f"    stage 1 (fingers_on_peg)    : n_contacts >= 2")
    print(f"    stage 2 (peg_lifted)        : peg_z > initial + 0.02  (= {initial_z + 0.02:.4f})")
    print(f"    stage 3 (near hole+aligned) : lateral < 0.03 AND axis_dot > 0.95")

    # Lift reward gradient
    print(f"\n  --- LIFT REWARD GRADIENT ---")
    print(f"  lift = min(lift_height / lift_target, 1.5) * contact_scale")
    print(f"  lift_target = {rc.lift_target:.4f}m (10cm) — POLICY MUST LIFT 10CM FOR FULL CREDIT")
    print(f"  stage 2 check fires at lift_height = 2cm (= 0.02m)")
    print(f"  reward at lift_height=2cm: lift = 0.02/0.10 = 0.2 (only 20%% of max)")
    print(f"  weight = {15} -> contribution at 2cm = 0.2 * 15 = 3.0/step (with nfc>=3)")
    print(f"  contribution at 10cm = 1.0 * 15 = 15.0/step")
    print(f"  contribution at 0cm  = 0.0 * 15 = 0.0/step")
    print(f"  per-cm reward slope  = 15 / 10 = 1.5 reward/cm at the table")

    # Align gate
    print(f"\n  --- ALIGN GATE ---")
    peg_length = cfg.peg_half_length * 2 + cfg.peg_radius * 2
    print(f"  peg_length = 2*peg_half + 2*radius = {peg_length:.4f}m")
    print(f"  peg_clearance = peg_z - table_height - 0.5*peg_length")
    print(f"  align_weight = sigmoid((peg_clearance - 0.02) * 150)")
    initial_clearance = initial_z - cfg.table_height - 0.5 * peg_length
    print(f"  at rest:        peg_clearance = {initial_clearance:.4f}m  -> align_weight = {1/(1+math.exp(-(initial_clearance - 0.02)*150)):.4f}")
    print(f"  at 2cm lift:    peg_clearance = {initial_clearance + 0.02:.4f}m  -> align_weight = {1/(1+math.exp(-(initial_clearance + 0.02 - 0.02)*150)):.4f}")
    print(f"  at 4cm lift:    peg_clearance = {initial_clearance + 0.04:.4f}m  -> align_weight = {1/(1+math.exp(-(initial_clearance + 0.04 - 0.02)*150)):.4f}")
    print(f"  at 10cm lift:   peg_clearance = {initial_clearance + 0.10:.4f}m  -> align_weight = {1/(1+math.exp(-(initial_clearance + 0.10 - 0.02)*150)):.4f}")
    print(f"  -> align contributes ZERO until peg is lifted ~4cm (full at ~4cm).")

    # Drop trap
    print(f"\n  --- DROP TRAP ---")
    print(f"  was_lifted toggles True when lift_height >= lift_target ({rc.lift_target:.2f}m)")
    print(f"  drop fires when (was_lifted AND lift_height < 0.01)")
    print(f"  drop_penalty = {rc.drop_penalty}")
    print(f"  Because lift_target=10cm, was_lifted never fires unless policy actually achieves 10cm lift.")
    print(f"  Observed peg_height ~= initial -> was_lifted never True -> drop never fires.")
    print(f"  But: insertion_drive, align, depth, complete ALL need peg_clearance > 2cm.")
    print(f"  Policy is stuck at stage 0/1 (grasp without lift).")

    # Peg vs hand reach
    print(f"\n  --- HAND REACH ---")
    print(f"  palm Z = {palm[2]:.4f}, peg Z at rest = {initial_z:.4f}")
    print(f"  palm-to-peg vertical distance = {palm[2] - initial_z:.4f}m")
    for name in ["fftip", "mftip", "rftip", "lftip", "thtip"]:
        p = site_pos(model, data, name)
        print(f"  fingertip {name:6s} z = {p[2]:.4f}  (= {p[2] - cfg.table_height:.4f} above table)")

    print(f"\n  ACTUATOR SUMMARY (peg scene):")
    actuator_summary(model)


if __name__ == "__main__":
    audit_grasp()
    audit_reorient()
    audit_peg()
