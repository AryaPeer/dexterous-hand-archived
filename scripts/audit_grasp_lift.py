"""Test what lift height the grasp scene's hand can ACTUALLY achieve.
Grasp uses lift_target=1.2cm — if we can verify that's reachable in the same
geometry, we triangulate that peg's 10cm is the bug, not the hand."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dexterous_hand.config import SceneConfig
from dexterous_hand.envs.scene_builder import GRIP_BIAS, apply_flexion_bias, build_scene


def joint_actuator_map(model):
    out = {}
    for ai in range(model.nu):
        if int(model.actuator_trntype[ai]) == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model.actuator_trnid[ai, 0])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname:
                lo, hi = model.actuator_ctrlrange[ai]
                out[jname] = (ai, float(lo), float(hi))
    return out


def make_grip_ctrl(model):
    j2a = joint_actuator_map(model)
    ctrl = np.zeros(model.nu)
    for jname, target in GRIP_BIAS.items():
        if jname in j2a:
            ai, lo, hi = j2a[jname]
            ctrl[ai] = float(np.clip(target, lo, hi))
    return ctrl


cfg = SceneConfig()
model, data, _ = build_scene(cfg)
apply_flexion_bias(data.qpos, model)
mujoco.mj_forward(model, data)

obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
qadr = model.jnt_qposadr[obj_jid]
palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
palm_z = float(data.xpos[palm_bid][2])

obj_z_initial = cfg.table_height + 0.035 + 0.001
print(f"Object initial z = {obj_z_initial:.4f}  (on table)")
print(f"Palm z           = {palm_z:.4f}")
print(f"Object can be lifted up to ~palm-1cm  ~ {palm_z - 0.01:.4f}")
print(f"Max possible lift = palm_z - obj_z_initial = {palm_z - obj_z_initial:.4f}m = {(palm_z - obj_z_initial)*100:.1f}cm")
print()

# Same test: spawn obj on table, close fingers, see max lift.
data.qpos[qadr:qadr + 3] = np.array([0.0, 0.0, obj_z_initial])
data.qpos[qadr + 3:qadr + 7] = np.array([1, 0, 0, 0])
data.qvel[:] = 0
grip_ctrl = make_grip_ctrl(model)
data.ctrl[:] = grip_ctrl
mujoco.mj_forward(model, data)
zs = []
for step in range(400):
    mujoco.mj_step(model, data)
    zs.append(float(data.qpos[qadr + 2]))

max_z = max(zs)
max_lift = max_z - obj_z_initial
final_z = zs[-1]
print(f"Closing fingers around object on table:")
print(f"  z trajectory: t=50: {zs[49]:.4f}, t=100: {zs[99]:.4f}, t=200: {zs[199]:.4f}, t=400: {zs[399]:.4f}")
print(f"  max object z = {max_z:.4f}, max lift = {max_lift:.4f}m ({max_lift*100:.1f}cm)")
print(f"  final object z = {final_z:.4f}, final lift = {final_z - obj_z_initial:.4f}m")
print()
print(f"Grasp lift_target = 0.012m (1.2cm). Achieved: {max_lift*100:.1f}cm. ", end="")
print("REACHABLE" if max_lift >= 0.012 else "UNREACHABLE")
