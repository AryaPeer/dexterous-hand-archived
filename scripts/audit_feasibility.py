"""Feasibility tests: can the hand actually do what we're asking?

1. Reorient: can fingers hold a cube against gravity at ALL?
2. Peg: can the hand lift the peg 10cm WITHOUT a Z-prismatic actuator?
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dexterous_hand.config import PegSceneConfig, ReorientSceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.reorient_scene_builder import build_reorient_scene
from dexterous_hand.envs.scene_builder import apply_flexion_bias, build_grip_ctrl, GRIP_BIAS


def hr(t):
    print("\n" + "=" * 78 + f"\n {t}\n" + "=" * 78)


# ============================================================================
# REORIENT — can the hand hold the cube against gravity?
# ============================================================================
def reorient_grip_test():
    hr("REORIENT — Can hand hold cube against gravity at all?")
    cfg = ReorientSceneConfig()
    model, data, _ = build_reorient_scene(cfg)
    apply_flexion_bias(data.qpos, model, bias_map=GRIP_BIAS)
    mujoco.mj_forward(model, data)

    cube_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    qadr = model.jnt_qposadr[cube_jid]
    vadr = model.jnt_dofadr[cube_jid]
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
    grasp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

    palm = np.array(data.xpos[palm_bid])
    grasp = np.array(data.site_xpos[grasp_sid])
    print(f"palm Z = {palm[2]:.4f}, grasp_site Z = {grasp[2]:.4f}")
    print(f"finger tips (after GRIP_BIAS qpos):")
    for name in ["fftip", "mftip", "rftip", "lftip", "thtip"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        p = np.array(data.site_xpos[sid])
        print(f"  {name}: {p}  (palm-rel = {p - palm})")

    # Strategy: scan cube spawn position in a grid around the fingers,
    # apply GRIP_BIAS ctrl for 100 steps, and see which spawn lets the
    # cube stay in hand.
    grip_ctrl = build_grip_ctrl(model)
    print(f"\nGRIP_BIAS ctrl summary: {np.count_nonzero(grip_ctrl)} non-zero of {model.nu}")

    print(f"\nScanning cube spawn positions (offset from grasp_site, +Z = away from palm)...")
    print(f"  hold = cube_z > palm_z - 0.05 after 100 steps with GRIP_BIAS ctrl")
    best_offset = None
    best_z = -np.inf

    # X is "out from palm" in this scene, Z is "perpendicular"
    for dx in [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02]:
        for dy in [0.0]:
            for dz in [-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02]:
                model_, data_, _ = build_reorient_scene(cfg)
                apply_flexion_bias(data_.qpos, model_, bias_map=GRIP_BIAS)
                mujoco.mj_forward(model_, data_)
                grasp_ = np.array(data_.site_xpos[grasp_sid])
                spawn = grasp_ + np.array([dx, dy, dz])
                data_.qpos[qadr:qadr + 3] = spawn
                data_.qpos[qadr + 3:qadr + 7] = np.array([1, 0, 0, 0])
                data_.qvel[vadr:vadr + 6] = 0
                data_.ctrl[:] = grip_ctrl
                for _ in range(100):
                    mujoco.mj_step(model_, data_)
                final_z = float(data_.qpos[qadr + 2])
                held = final_z > palm[2] - 0.05
                marker = "HELD" if held else "----"
                print(f"  offset=({dx:+.2f}, {dy:+.2f}, {dz:+.2f}) spawn_z={spawn[2]:.3f} -> final_z={final_z:.3f}  [{marker}]")
                if held and final_z > best_z:
                    best_z = final_z
                    best_offset = (dx, dy, dz)

    if best_offset:
        print(f"\nBest spawn offset = {best_offset}  -> cube held at z={best_z:.3f}")
    else:
        print(f"\nNO spawn position allowed the cube to be held by GRIP_BIAS over 100 steps.")
        print(f"This means the closed-finger geometry CANNOT cradle the cube under gravity in this hand orientation.")


# ============================================================================
# PEG — can the hand lift the peg 10cm without a Z-prismatic actuator?
# ============================================================================
def peg_lift_test():
    hr("PEG — Maximum achievable lift with current actuator set?")
    cfg = PegSceneConfig()
    model, data, _ = build_peg_scene(cfg)
    apply_flexion_bias(data.qpos, model)
    mujoco.mj_forward(model, data)

    peg_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    qadr = model.jnt_qposadr[peg_jid]
    vadr = model.jnt_dofadr[peg_jid]

    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
    peg_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    grasp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

    palm = np.array(data.xpos[palm_bid])
    print(f"palm Z = {palm[2]:.4f}")
    print(f"peg initial Z (rest on table) = {data.xpos[peg_bid][2]:.4f}")
    print(f"lift_target = 0.10m -> peg target Z = {data.xpos[peg_bid][2] + 0.10:.4f}")

    # Test: teleport peg into pre-grasp pose, then close fingers and see how
    # high we can get the peg through finger flexion alone (no Z motion).
    grasp = np.array(data.site_xpos[grasp_sid])
    print(f"\ngrasp_site Z = {grasp[2]:.4f}  (this is the 'pre-grasp peg height')")
    print(f"lift achievable from pre-grasp = {grasp[2] - cfg.table_height - cfg.peg_half_length - cfg.peg_radius - 0.001:.4f}m")
    # That's grasp_site - initial. If positive, lifting peg from table TO grasp_site is a real lift.

    # Place peg in grasp_site, settle with GRIP_BIAS for 200 steps, see peg z
    print(f"\nTest 1: peg starts at grasp_site, GRIP_BIAS ctrl. Can the hand hold it there?")
    data2 = mujoco.MjData(model)
    data2.qpos = data.qpos.copy()  # restore initial qpos including GRIP_BIAS hand
    apply_flexion_bias(data2.qpos, model)
    data2.qpos[qadr:qadr + 3] = grasp
    data2.qpos[qadr + 3:qadr + 7] = np.array([1, 0, 0, 0])
    data2.qvel[:] = 0
    grip_ctrl = build_grip_ctrl(model)
    data2.ctrl[:] = grip_ctrl
    mujoco.mj_forward(model, data2)
    zs = []
    for step in range(200):
        mujoco.mj_step(model, data2)
        zs.append(float(data2.qpos[qadr + 2]))
    print(f"  peg z trajectory (10/50/100/200): {zs[9]:.3f} / {zs[49]:.3f} / {zs[99]:.3f} / {zs[199]:.3f}")
    held = zs[199] > 0.4 + 0.05  # not on table
    print(f"  -> {'HELD above table' if held else 'FELL to table'}")

    # Test 2: peg starts on table, force-close fingers + apply max upward wrist torque.
    # Find max achievable Z just by curling fingers (no slider Z, which doesn't exist).
    print(f"\nTest 2: peg on table, apply GRIP_BIAS ctrl with hand near peg. Max lift?")
    model3, data3, _ = build_peg_scene(cfg)
    apply_flexion_bias(data3.qpos, model3)
    mujoco.mj_forward(model3, data3)

    # spawn peg directly under the fingers
    spawn_x = -0.005  # roughly under fingertips
    spawn_y = 0.013
    spawn_z = cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001
    data3.qpos[qadr:qadr + 3] = np.array([spawn_x, spawn_y, spawn_z])
    data3.qpos[qadr + 3:qadr + 7] = np.array([1, 0, 0, 0])
    data3.qvel[:] = 0
    data3.ctrl[:] = grip_ctrl
    mujoco.mj_forward(model3, data3)
    zs = []
    for step in range(300):
        mujoco.mj_step(model3, data3)
        zs.append(float(data3.qpos[qadr + 2]))
    print(f"  peg z trajectory: start={zs[0]:.4f}, t=50: {zs[49]:.4f}, t=100: {zs[99]:.4f}, t=200: {zs[199]:.4f}, t=300: {zs[299]:.4f}")
    max_z = max(zs)
    initial_z = cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001
    max_lift = max_z - initial_z
    print(f"  max peg Z achieved = {max_z:.4f}, max lift height = {max_lift:.4f}m ({max_lift*100:.1f}cm)")
    print(f"  lift_target = 10.0cm. Achievable = {max_lift*100:.1f}cm. ", end="")
    print("REACHABLE" if max_lift >= 0.10 else f"UNREACHABLE (short by {(0.10 - max_lift)*100:.1f}cm)")

    # Test 3: what if we also use wrist flexion?
    print(f"\nTest 3: GRIP_BIAS + wrist flexion (WRJ1=0.49 max, WRJ2=0.17 max). Max lift?")
    model4, data4, _ = build_peg_scene(cfg)
    apply_flexion_bias(data4.qpos, model4)
    mujoco.mj_forward(model4, data4)
    data4.qpos[qadr:qadr + 3] = np.array([spawn_x, spawn_y, spawn_z])
    data4.qpos[qadr + 3:qadr + 7] = np.array([1, 0, 0, 0])
    data4.qvel[:] = 0

    flex_ctrl = grip_ctrl.copy()
    for ai in range(model4.nu):
        if int(model4.actuator_trntype[ai]) != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = int(model4.actuator_trnid[ai, 0])
        jname = mujoco.mj_id2name(model4, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname in ("rh_WRJ1", "rh_WRJ2"):
            hi = float(model4.actuator_ctrlrange[ai, 1])
            flex_ctrl[ai] = hi
            print(f"  {jname} driven to {hi:+.3f}")
    data4.ctrl[:] = flex_ctrl
    mujoco.mj_forward(model4, data4)
    zs = []
    for step in range(300):
        mujoco.mj_step(model4, data4)
        zs.append(float(data4.qpos[qadr + 2]))
    max_z = max(zs)
    max_lift = max_z - initial_z
    print(f"  peg max Z achieved = {max_z:.4f}, max lift = {max_lift:.4f}m ({max_lift*100:.1f}cm)")
    print(f"  vs lift_target = 10.0cm. ", end="")
    print("REACHABLE" if max_lift >= 0.10 else f"STILL UNREACHABLE (short by {(0.10 - max_lift)*100:.1f}cm)")


if __name__ == "__main__":
    reorient_grip_test()
    peg_lift_test()
