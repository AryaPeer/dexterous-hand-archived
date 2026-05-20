"""Verify slide_z is wired correctly: (1) hand doesn't sag at rest under
gravity, (2) commanding slide_z up lifts the held peg, (3) lift_target=10cm
is now reachable."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dexterous_hand.config import PegSceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import (
    GRIP_BIAS,
    apply_flexion_bias,
    build_grip_ctrl,
)


def hr(t):
    print("\n" + "=" * 78 + f"\n {t}\n" + "=" * 78)


def find_actuator(model, joint_name):
    for ai in range(model.nu):
        if int(model.actuator_trntype[ai]) != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = int(model.actuator_trnid[ai, 0])
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) == joint_name:
            lo, hi = model.actuator_ctrlrange[ai]
            return ai, float(lo), float(hi)
    raise KeyError(joint_name)


hr("Peg slide_z verification")
cfg = PegSceneConfig()
model, data, _ = build_peg_scene(cfg)
print(f"model.nu = {model.nu}  (expected 23)")

sz_ai, sz_lo, sz_hi = find_actuator(model, "slide_z")
print(f"slide_z actuator: idx={sz_ai}, range=[{sz_lo}, {sz_hi}]")

palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
peg_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
peg_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
peg_qadr = model.jnt_qposadr[peg_jid]
slide_z_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_z")
sz_qadr = model.jnt_qposadr[slide_z_jid]
grasp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

# ============================================================
# TEST 1: hand at rest, no peg, ctrl=0. Does slide_z sag?
# ============================================================
apply_flexion_bias(data.qpos, model)
mujoco.mj_forward(model, data)
palm_z_start = float(data.xpos[palm_bid][2])
sz_start = float(data.qpos[sz_qadr])
print(f"\nTest 1: rest without peg.")
print(f"  initial palm z = {palm_z_start:.4f}, slide_z qpos = {sz_start:.4f}")
data.ctrl[:] = build_grip_ctrl(model)
data.ctrl[sz_ai] = 0.0  # explicit zero on slide_z
for _ in range(300):
    mujoco.mj_step(model, data)
palm_z_end = float(data.xpos[palm_bid][2])
sz_end = float(data.qpos[sz_qadr])
sag = palm_z_start - palm_z_end
print(f"  after 300 steps: palm z = {palm_z_end:.4f}, slide_z qpos = {sz_end:.4f}")
print(f"  sag = {sag:.4f}m ({sag*100:.2f}cm)")
print(f"  -> {'GOOD: < 1cm sag' if abs(sag) < 0.01 else 'BAD: hand sags too much'}")

# ============================================================
# TEST 2: peg pre-grasped, ctrl_slide_z=0. Hand+peg held in place?
# ============================================================
hr("Test 2: peg pre-grasped, slide_z=0. Held?")
model2, data2, _ = build_peg_scene(cfg)
apply_flexion_bias(data2.qpos, model2)
mujoco.mj_forward(model2, data2)
grasp = np.array(data2.site_xpos[grasp_sid])
data2.qpos[peg_qadr:peg_qadr + 3] = grasp
data2.qpos[peg_qadr + 3:peg_qadr + 7] = np.array([1, 0, 0, 0])
data2.qvel[:] = 0
data2.ctrl[:] = build_grip_ctrl(model2)
mujoco.mj_forward(model2, data2)
peg_z_start = float(data2.qpos[peg_qadr + 2])
zs = []
for _ in range(200):
    mujoco.mj_step(model2, data2)
    zs.append(float(data2.qpos[peg_qadr + 2]))
print(f"  peg z trajectory: start={peg_z_start:.4f}, t=50: {zs[49]:.4f}, t=100: {zs[99]:.4f}, t=200: {zs[199]:.4f}")
print(f"  -> {'HELD' if zs[-1] > 0.45 else 'FELL'}")

# ============================================================
# TEST 3: command slide_z up to lift the peg
# ============================================================
hr("Test 3: command slide_z up — can the policy lift the peg?")
model3, data3, _ = build_peg_scene(cfg)
apply_flexion_bias(data3.qpos, model3)
mujoco.mj_forward(model3, data3)
grasp = np.array(data3.site_xpos[grasp_sid])
data3.qpos[peg_qadr:peg_qadr + 3] = grasp
data3.qpos[peg_qadr + 3:peg_qadr + 7] = np.array([1, 0, 0, 0])
data3.qvel[:] = 0
data3.ctrl[:] = build_grip_ctrl(model3)
mujoco.mj_forward(model3, data3)

# settle for 50 steps to get a stable grip
for _ in range(50):
    mujoco.mj_step(model3, data3)
peg_z_after_grip = float(data3.qpos[peg_qadr + 2])
print(f"  peg z after grip settle = {peg_z_after_grip:.4f}")

# now command slide_z to its UPPER bound (+0.05)
data3.ctrl[sz_ai] = sz_hi
zs = [peg_z_after_grip]
for _ in range(500):
    mujoco.mj_step(model3, data3)
    zs.append(float(data3.qpos[peg_qadr + 2]))

peg_z_final = zs[-1]
peg_initial = cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001
lift_height = peg_z_final - peg_initial
print(f"  command: ctrl_slide_z = +{sz_hi}")
print(f"  peg z trajectory: start={peg_z_after_grip:.4f}, t=100: {zs[100]:.4f}, t=300: {zs[300]:.4f}, final={peg_z_final:.4f}")
print(f"  peg initial z (on table) = {peg_initial:.4f}")
print(f"  lift_height = peg_final - peg_initial = {lift_height:.4f}m ({lift_height*100:.1f}cm)")
print(f"  vs lift_target = 10.0cm. {'REACHED ' if lift_height >= 0.10 else 'STILL SHORT'}")

# ============================================================
# TEST 4: peg starts on table, hand descends, grips, lifts.
# ============================================================
hr("Test 4: full pick-up motion from peg on table")
model4, data4, _ = build_peg_scene(cfg)
apply_flexion_bias(data4.qpos, model4)
mujoco.mj_forward(model4, data4)
peg_spawn = np.array([0.0, 0.0, peg_initial])
data4.qpos[peg_qadr:peg_qadr + 3] = peg_spawn
data4.qpos[peg_qadr + 3:peg_qadr + 7] = np.array([1, 0, 0, 0])
data4.qvel[:] = 0
grip_ctrl = build_grip_ctrl(model4)

# Phase 1: fingers OPEN (ctrl=0 for fingers), slide_z DOWN to peg level
data4.ctrl[:] = 0.0  # all open
data4.ctrl[sz_ai] = sz_lo  # slide_z to min = -0.10
mujoco.mj_forward(model4, data4)
for _ in range(100):
    mujoco.mj_step(model4, data4)
print(f"  after descent: palm z = {float(data4.xpos[palm_bid][2]):.4f}, peg z = {float(data4.qpos[peg_qadr + 2]):.4f}")

# Phase 2: close fingers
data4.ctrl[:] = grip_ctrl
data4.ctrl[sz_ai] = sz_lo
for _ in range(100):
    mujoco.mj_step(model4, data4)
print(f"  after grip: palm z = {float(data4.xpos[palm_bid][2]):.4f}, peg z = {float(data4.qpos[peg_qadr + 2]):.4f}")

# Phase 3: slide_z UP to lift
data4.ctrl[sz_ai] = sz_hi
zs = []
for _ in range(300):
    mujoco.mj_step(model4, data4)
    zs.append(float(data4.qpos[peg_qadr + 2]))
peg_final = zs[-1]
lift = peg_final - peg_initial
print(f"  after lift cmd: peg z = {peg_final:.4f}, lift = {lift:.4f}m ({lift*100:.1f}cm)")
print(f"  {'FULL PICK-UP WORKS' if lift >= 0.05 else 'pick-up failed'}")
