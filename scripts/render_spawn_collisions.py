"""Render an mp4 showing 12 consecutive spawn samples (grasp env) and 12 for
peg env (mixed p_pre_grasped). Each spawn freezes for 15 frames showing the
spawn state, then runs 35 settle frames to show what happens. So 50 frames
per spawn × 12 spawns = 600 frames per env (~24 sec at 25fps).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig, SceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import (
    GRIP_BIAS,
    OBJECT_TYPES,
    TABLE_TASK_FLEXION_BIAS,
    apply_flexion_bias,
    build_grip_ctrl,
    build_scene,
    get_object_half_height,
)


def grasp_render(out_path: Path, n_spawns: int = 12, seed: int = 0) -> None:
    cfg = SceneConfig()
    model, data, nm = build_scene(cfg)
    gt, gs = OBJECT_TYPES["large_cube"]
    half_h = get_object_half_height(gt, gs)
    obj_z = cfg.table_height + half_h + 0.001
    # settle_ctrl matches the spawn-pose qpos so position actuators don't
    # drive the bent fingers back toward fully open (which was making the hand
    # swing into the cube on the first sim step).
    settle_ctrl = build_grip_ctrl(model, bias_map=TABLE_TASK_FLEXION_BIAS)

    init_qpos = data.qpos.copy()
    apply_flexion_bias(init_qpos, model)
    rng = np.random.default_rng(seed)

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []

    cube_g = nm.object_geom_id
    hand_geoms: set[int] = set()
    for gset in nm.finger_geom_ids_per_finger:
        hand_geoms |= gset

    for i in range(n_spawns):
        qpos = init_qpos.copy()
        # Mirror env reset post-fix: ±0.05 noise on rotational joints, 0 on sliders.
        hand_noise = rng.uniform(-0.05, 0.05, size=nm.hand_qpos_end - nm.hand_qpos_start)
        hand_noise[0:2] = 0.0
        qpos[nm.hand_qpos_start : nm.hand_qpos_end] += hand_noise
        ox, oy = rng.uniform(0.05, 0.10), rng.uniform(-0.03, 0.03)
        qpos[nm.obj_qpos_start : nm.obj_qpos_start + 3] = [ox, oy, obj_z]
        qpos[nm.obj_qpos_start + 3 : nm.obj_qpos_start + 7] = [1, 0, 0, 0]
        data.qpos[:] = qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        # measure penetration at spawn
        penetrating = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            if (c.geom1 == cube_g and c.geom2 in hand_geoms) or (
                c.geom2 == cube_g and c.geom1 in hand_geoms
            ):
                if c.dist < -1e-4:
                    penetrating += 1
        slide_x = float(qpos[0])
        slide_y = float(qpos[1])
        # freeze spawn for 15 frames
        renderer.update_scene(data, camera="track_cam")
        spawn_frame = renderer.render().copy()
        for _ in range(15):
            frames.append(spawn_frame)

        # settle 35 frames at ctrl=settle_ctrl
        data.ctrl[: model.nu] = settle_ctrl
        for k in range(35):
            for _ in range(cfg.frame_skip):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera="track_cam")
            frames.append(renderer.render().copy())

        post_obj = data.xpos[nm.object_body_id].copy()
        spawn_obj = np.array([ox, oy, obj_z])
        disp = float(np.linalg.norm(post_obj - spawn_obj))
        print(f"  grasp spawn {i+1}/{n_spawns}: cube=({ox:+.3f},{oy:+.3f}) slider=({slide_x:+.3f},{slide_y:+.3f}) penetrating_geoms={penetrating} post_settle_disp={disp*1000:.2f}mm")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=25, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    renderer.close()


def peg_render(out_path: Path, n_spawns: int = 12, seed: int = 0, p_pre_grasped: float = 0.5) -> None:
    cfg = PegSceneConfig()
    model, data, nm = build_peg_scene(cfg)
    # peg env switches qpos bias based on p_pre_grasped, so the matching ctrl
    # depends on the per-spawn pose. Compute both upfront and pick at spawn time.
    grip_ctrl = build_grip_ctrl(model, bias_map=GRIP_BIAS)
    open_ctrl = build_grip_ctrl(model, bias_map=TABLE_TASK_FLEXION_BIAS)

    grip_qpos = data.qpos.copy()
    apply_flexion_bias(grip_qpos, model, bias_map=GRIP_BIAS)
    open_qpos = data.qpos.copy()
    apply_flexion_bias(open_qpos, model)

    grasp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    peg_z_table = cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001
    rng = np.random.default_rng(seed)

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []

    peg_g = nm.peg_geom_id
    hand_geoms: set[int] = set()
    for gset in nm.finger_geom_ids_per_finger:
        hand_geoms |= gset

    for i in range(n_spawns):
        pre_grasped = rng.uniform() < p_pre_grasped
        base = grip_qpos if pre_grasped else open_qpos
        qpos = base.copy()
        # Mirror env reset post-fix: ±0.05 noise on rotational joints, 0 on sliders.
        hand_noise = rng.uniform(-0.05, 0.05, size=nm.hand_qpos_end - nm.hand_qpos_start)
        hand_noise[0:2] = 0.0
        qpos[nm.hand_qpos_start : nm.hand_qpos_end] += hand_noise

        if pre_grasped:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            gs = data.site_xpos[grasp_site_id].copy()
            qpos[nm.peg_qpos_start : nm.peg_qpos_start + 3] = gs
            qpos[nm.peg_qpos_start + 3 : nm.peg_qpos_start + 7] = [1, 0, 0, 0]
            obj_pos = gs.copy()
        else:
            theta = rng.uniform(-0.5 * np.pi, 0.5 * np.pi)
            r = rng.uniform(cfg.spawn_min_radius, cfg.spawn_max_radius)
            px, py = float(np.cos(theta) * r), float(np.sin(theta) * r)
            qpos[nm.peg_qpos_start : nm.peg_qpos_start + 3] = [px, py, peg_z_table]
            qpos[nm.peg_qpos_start + 3 : nm.peg_qpos_start + 7] = [1, 0, 0, 0]
            obj_pos = np.array([px, py, peg_z_table])

        data.qpos[:] = qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        penetrating = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            if (c.geom1 == peg_g and c.geom2 in hand_geoms) or (
                c.geom2 == peg_g and c.geom1 in hand_geoms
            ):
                if c.dist < -1e-4:
                    penetrating += 1
        slide_x, slide_y = float(qpos[0]), float(qpos[1])
        renderer.update_scene(data, camera="track_cam")
        spawn_frame = renderer.render().copy()
        for _ in range(15):
            frames.append(spawn_frame)
        data.ctrl[: model.nu] = grip_ctrl if pre_grasped else open_ctrl
        for k in range(35):
            for _ in range(cfg.frame_skip):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera="track_cam")
            frames.append(renderer.render().copy())

        post = data.xpos[nm.peg_body_id].copy()
        disp = float(np.linalg.norm(post - obj_pos))
        tag = "pre_grasped" if pre_grasped else "on_table"
        print(f"  peg spawn {i+1}/{n_spawns} [{tag}]: peg=({obj_pos[0]:+.3f},{obj_pos[1]:+.3f},{obj_pos[2]:.3f}) slider=({slide_x:+.3f},{slide_y:+.3f}) pen={penetrating} disp={disp*1000:.2f}mm")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=25, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    renderer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("runs/render"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"=== Grasp spawn-collision render ({args.n} spawns) ===")
    grasp_render(args.out_dir / "spawn_grasp.mp4", n_spawns=args.n, seed=args.seed)

    print(f"\n=== Peg spawn-collision render ({args.n} spawns, p_pre_grasped=0.5) ===")
    peg_render(args.out_dir / "spawn_peg.mp4", n_spawns=args.n, seed=args.seed + 100, p_pre_grasped=0.5)


if __name__ == "__main__":
    main()
