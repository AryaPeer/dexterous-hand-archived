"""Audit: when grasp / peg envs reset, do the hand and object spawn in
penetrating contact?

For each env type we sample many (cube, slider) random offsets from the
real reset distribution and:
  1) compute fingertip & palm positions at reset
  2) measure minimum distance to the object surface
  3) count cases where a hand geom is *inside* the object volume
  4) settle 5 mj_steps with ctrl=settle_ctrl and report object displacement +
     fingertip-cube contact count

Same probe is run at peg curriculum p_pre_grasped ∈ {0.0, 1.0} so we cover
both the "peg on table" and "peg in hand" stages.
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig, SceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import (
    GRIP_BIAS,
    OBJECT_TYPES,
    apply_flexion_bias,
    build_grip_ctrl,
    build_scene,
    get_object_half_height,
)


def grasp_sample(rng: np.random.Generator, n: int = 256) -> dict:
    cfg = SceneConfig()
    model, data, nm = build_scene(cfg)
    gt, gs = OBJECT_TYPES["large_cube"]
    half_h = get_object_half_height(gt, gs)
    obj_z = cfg.table_height + half_h + 0.001
    settle_ctrl = build_grip_ctrl(model, bias_map={})  # no grip during settle for grasp

    init_qpos = data.qpos.copy()
    apply_flexion_bias(init_qpos, model)

    results = []
    overlap_count = 0
    for _i in range(n):
        qpos = init_qpos.copy()
        # Mirror env reset: ±0.05 noise on rotational joints, 0 on sliders.
        hand_noise = rng.uniform(-0.05, 0.05, size=nm.hand_qpos_end - nm.hand_qpos_start)
        hand_noise[0:2] = 0.0
        qpos[nm.hand_qpos_start : nm.hand_qpos_end] += hand_noise
        ox, oy = rng.uniform(0.05, 0.10), rng.uniform(-0.03, 0.03)
        qpos[nm.obj_qpos_start : nm.obj_qpos_start + 3] = [ox, oy, obj_z]
        qpos[nm.obj_qpos_start + 3 : nm.obj_qpos_start + 7] = [1, 0, 0, 0]
        data.qpos[:] = qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        # min finger-cube center distance
        obj_pos = np.array([ox, oy, obj_z])
        ft = np.stack([data.site_xpos[sid] for sid in nm.fingertip_site_ids])
        dists = np.linalg.norm(ft - obj_pos[None, :], axis=1)
        min_dist = float(dists.min())
        min_idx = int(dists.argmin())

        # count any hand geom currently in penetration with cube
        cube_g = nm.object_geom_id
        hand_geoms: set[int] = set()
        for gset in nm.finger_geom_ids_per_finger:
            hand_geoms |= gset
        penetrating = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            if ((c.geom1 == cube_g and c.geom2 in hand_geoms) or (
                c.geom2 == cube_g and c.geom1 in hand_geoms
            )) and c.dist < -1e-4:  # actual interpenetration
                penetrating += 1
        if penetrating > 0:
            overlap_count += 1

        # settle 5 steps with no-grip ctrl
        data.ctrl[: model.nu] = settle_ctrl
        for _ in range(5):
            mujoco.mj_step(model, data)
        post_obj = data.xpos[nm.object_body_id].copy()
        disp = float(np.linalg.norm(post_obj - obj_pos))

        results.append(
            {
                "min_dist": min_dist,
                "closest_finger": ["ff", "mf", "rf", "lf", "th"][min_idx],
                "penetrating_geoms": penetrating,
                "settle_obj_displacement": disp,
            }
        )

    md = np.array([r["min_dist"] for r in results])
    disp = np.array([r["settle_obj_displacement"] for r in results])
    return {
        "n_samples": n,
        "n_overlapping": overlap_count,
        "min_dist_p5": float(np.percentile(md, 5)),
        "min_dist_p50": float(np.percentile(md, 50)),
        "min_dist_p95": float(np.percentile(md, 95)),
        "settle_disp_p50": float(np.percentile(disp, 50)),
        "settle_disp_p95": float(np.percentile(disp, 95)),
        "settle_disp_max": float(np.max(disp)),
    }


def peg_sample(rng: np.random.Generator, n: int = 256, p_pre_grasped: float = 0.0) -> dict:
    cfg = PegSceneConfig()
    model, data, nm = build_peg_scene(cfg)

    grip_bias_qpos = data.qpos.copy()
    apply_flexion_bias(grip_bias_qpos, model, bias_map=GRIP_BIAS)
    open_bias_qpos = data.qpos.copy()
    apply_flexion_bias(open_bias_qpos, model)
    settle_ctrl = build_grip_ctrl(model)  # closed grip ctrl

    peg_z_table = cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001
    grasp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")

    results = []
    overlap_count = 0
    for _i in range(n):
        pre_grasped = rng.uniform() < p_pre_grasped
        base_qpos = grip_bias_qpos.copy() if pre_grasped else open_bias_qpos.copy()

        qpos = base_qpos.copy()
        # Mirror env reset: ±0.05 noise on rotational joints, 0 on sliders.
        hand_noise = rng.uniform(-0.05, 0.05, size=nm.hand_qpos_end - nm.hand_qpos_start)
        hand_noise[0:2] = 0.0
        qpos[nm.hand_qpos_start : nm.hand_qpos_end] += hand_noise

        if pre_grasped:
            # need forward kinematics to find grasp_site, peg gets placed there
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

        ft = np.stack([data.site_xpos[sid] for sid in nm.fingertip_site_ids])
        dists = np.linalg.norm(ft - obj_pos[None, :], axis=1)
        min_dist = float(dists.min())
        min_idx = int(dists.argmin())

        peg_g = nm.peg_geom_id
        hand_geoms: set[int] = set()
        for gset in nm.finger_geom_ids_per_finger:
            hand_geoms |= gset
        penetrating = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            if ((c.geom1 == peg_g and c.geom2 in hand_geoms) or (
                c.geom2 == peg_g and c.geom1 in hand_geoms
            )) and c.dist < -1e-4:
                penetrating += 1
        if penetrating > 0:
            overlap_count += 1

        data.ctrl[: model.nu] = settle_ctrl
        for _ in range(5):
            mujoco.mj_step(model, data)
        post_obj = data.xpos[nm.peg_body_id].copy()
        disp = float(np.linalg.norm(post_obj - obj_pos))

        results.append(
            {
                "min_dist": min_dist,
                "closest_finger": ["ff", "mf", "rf", "lf", "th"][min_idx],
                "penetrating_geoms": penetrating,
                "settle_obj_displacement": disp,
                "pre_grasped": pre_grasped,
            }
        )

    md = np.array([r["min_dist"] for r in results])
    disp = np.array([r["settle_obj_displacement"] for r in results])
    return {
        "n_samples": n,
        "n_overlapping": overlap_count,
        "min_dist_p5": float(np.percentile(md, 5)),
        "min_dist_p50": float(np.percentile(md, 50)),
        "min_dist_p95": float(np.percentile(md, 95)),
        "settle_disp_p50": float(np.percentile(disp, 50)),
        "settle_disp_p95": float(np.percentile(disp, 95)),
        "settle_disp_max": float(np.max(disp)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"=== GRASP env ({args.n} seeds) ===")
    g = grasp_sample(rng, n=args.n)
    print(f"  n_overlapping (penetration > 0.1mm): {g['n_overlapping']} / {g['n_samples']} ({100*g['n_overlapping']/g['n_samples']:.1f}%)")
    print(f"  min finger-cube distance:  p5={g['min_dist_p5']*1000:.1f}mm  p50={g['min_dist_p50']*1000:.1f}mm  p95={g['min_dist_p95']*1000:.1f}mm")
    print(f"  settle obj displacement:   p50={g['settle_disp_p50']*1000:.2f}mm  p95={g['settle_disp_p95']*1000:.2f}mm  max={g['settle_disp_max']*1000:.2f}mm")

    print("\n=== PEG env, p_pre_grasped=0.0 (peg on table) ===")
    rng2 = np.random.default_rng(args.seed + 1)
    p0 = peg_sample(rng2, n=args.n, p_pre_grasped=0.0)
    print(f"  n_overlapping: {p0['n_overlapping']} / {p0['n_samples']} ({100*p0['n_overlapping']/p0['n_samples']:.1f}%)")
    print(f"  min finger-peg distance:  p5={p0['min_dist_p5']*1000:.1f}mm  p50={p0['min_dist_p50']*1000:.1f}mm  p95={p0['min_dist_p95']*1000:.1f}mm")
    print(f"  settle peg displacement:   p50={p0['settle_disp_p50']*1000:.2f}mm  p95={p0['settle_disp_p95']*1000:.2f}mm  max={p0['settle_disp_max']*1000:.2f}mm")

    print("\n=== PEG env, p_pre_grasped=1.0 (peg in grip) ===")
    rng3 = np.random.default_rng(args.seed + 2)
    p1 = peg_sample(rng3, n=args.n, p_pre_grasped=1.0)
    print(f"  n_overlapping: {p1['n_overlapping']} / {p1['n_samples']} ({100*p1['n_overlapping']/p1['n_samples']:.1f}%)")
    print(f"  min finger-peg distance:  p5={p1['min_dist_p5']*1000:.1f}mm  p50={p1['min_dist_p50']*1000:.1f}mm  p95={p1['min_dist_p95']*1000:.1f}mm")
    print(f"  settle peg displacement:   p50={p1['settle_disp_p50']*1000:.2f}mm  p95={p1['settle_disp_p95']*1000:.2f}mm  max={p1['settle_disp_max']*1000:.2f}mm")


if __name__ == "__main__":
    main()
