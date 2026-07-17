"""Render the peg winning trajectory: pre-grasped -> raise -> transport ->"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import jax.numpy as jnp
import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import (
    GRIP_BIAS,
    apply_flexion_bias,
    build_grip_ctrl,
)
from dexterous_hand.utils.mjx_helpers import get_insertion_depth_jax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/render/peg_transport_release.mp4"))
    args = ap.parse_args()

    cfg = PegSceneConfig()
    model, data, nm = build_peg_scene(cfg)
    peg_len = cfg.peg_half_length * 2.0 + cfg.peg_radius * 2.0

    qpos = data.qpos.copy()
    apply_flexion_bias(qpos, model, bias_map=GRIP_BIAS)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    peg_qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    ]
    data.qpos[peg_qadr : peg_qadr + 3] = data.site_xpos[sid]
    data.qpos[peg_qadr + 3 : peg_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
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

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []
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
            renderer.update_scene(data, camera="track_cam")
            frames.append(renderer.render().copy())

    z_cmd = 0.06
    do_steps(15)
    xy_cmd = slide_xy() + (hole_pos[:2] - data.xpos[nm.peg_body_id][:2])
    do_steps(25)
    do_steps(15, servo=True)
    tip_z = data.xpos[nm.peg_body_id][2] - peg_len / 2.0
    z_cmd += entrance_z + 0.01 - tip_z
    do_steps(15, servo=True)
    for _ in range(10):
        tip_z = data.xpos[nm.peg_body_id][2] - peg_len / 2.0
        z_cmd += float(np.clip((entrance_z - 0.020) - tip_z, -0.004, 0.004))
        do_steps(2, servo=True)
    do_steps(15, open_fingers=True)
    z_cmd += 0.06
    do_steps(25, open_fingers=True)
    do_steps(75, open_fingers=True)

    depth = float(
        get_insertion_depth_jax(
            jnp.array(data.xpos), jnp.array(data.xmat),
            nm.peg_body_id, nm.hole_body_id,
            cfg.peg_half_length, cfg.peg_radius,
            cfg.peg_radius + cfg.clearance,
            cfg.hole_depth,
        )
    )
    frac = depth / peg_len

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(args.out), fps=25, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    renderer.close()
    print(f"settled insertion fraction: {frac:.3f} (success_threshold 0.70)")
    print(f"video: {args.out}")


if __name__ == "__main__":
    main()
