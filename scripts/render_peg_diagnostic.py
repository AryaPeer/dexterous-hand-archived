from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import (
    GRIP_BIAS,
    apply_flexion_bias,
    build_grip_ctrl,
)


def make_data(cfg: PegSceneConfig, seed: int = 0, pre_grasped: bool = False):
    model, data, nm = build_peg_scene(cfg)
    rng = np.random.default_rng(seed)

    qpos = data.qpos.copy()
    if pre_grasped:
        apply_flexion_bias(qpos, model, bias_map=GRIP_BIAS)
        # peg spawn at grasp site for pre-grasped case
        grasp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        if grasp_site_id >= 0:
            # we need data.site_xpos which requires forward kinematics first
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            gs = data.site_xpos[grasp_site_id]
            qpos[nm.peg_qpos_start : nm.peg_qpos_start + 3] = gs
            # peg-axis upright (capsule axis is local-z by default)
            qpos[nm.peg_qpos_start + 3 : nm.peg_qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]
    else:
        apply_flexion_bias(qpos, model)
        theta = rng.uniform(0.0, 2 * np.pi)
        r = 0.045
        px = float(np.cos(theta) * r)
        py = float(np.sin(theta) * r)
        peg_z = cfg.table_height + cfg.peg_half_length + cfg.peg_radius + 0.001
        qpos[nm.peg_qpos_start : nm.peg_qpos_start + 3] = [px, py, peg_z]
        qpos[nm.peg_qpos_start + 3 : nm.peg_qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]

    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    return model, data, nm


def scripted_grip_and_lift(
    model: mujoco.MjModel,
    step: int,
    grip_ctrl: np.ndarray,
) -> np.ndarray:
    """Open-loop: hold the closed-grip ctrl always; slowly slide_y to nudge peg."""
    ctrl = grip_ctrl.copy()
    # at no point do we override slide_x/slide_y — let them stay at 0
    return ctrl


def scripted_reach_and_grip(
    model: mujoco.MjModel, step: int, grip_ctrl: np.ndarray, peg_init_xy: tuple[float, float]
) -> np.ndarray:
    """Reach toward the peg's XY, then close fingers."""
    ctrl = np.zeros(model.nu, dtype=np.float64)
    sx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide_x_act")
    sy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide_y_act")

    px, py = peg_init_xy
    target_x = float(np.clip(px, -0.15, 0.15))
    target_y = float(np.clip(py, -0.15, 0.15))

    if sx >= 0:
        ctrl[sx] = target_x
    if sy >= 0:
        ctrl[sy] = target_y

    # ramp closure
    if step < 30:
        alpha = 0.0
    elif step < 80:
        alpha = (step - 30) / 50.0
    else:
        alpha = 1.0
    for ai in range(model.nu):
        if ai in (sx, sy):
            continue
        ctrl[ai] = alpha * grip_ctrl[ai]
    return ctrl


def render_episode(
    cfg: PegSceneConfig,
    label: str,
    out_path: Path,
    mode: str,
    steps: int = 200,
    fps: int = 25,
    seed: int = 0,
) -> dict[str, float]:
    pre_grasped = mode == "pregrasp_hold"
    model, data, nm = make_data(cfg, seed=seed, pre_grasped=pre_grasped)

    grip_ctrl = build_grip_ctrl(model)
    peg_xy = (float(data.qpos[nm.peg_qpos_start]), float(data.qpos[nm.peg_qpos_start + 1]))
    peg_z0 = float(data.qpos[nm.peg_qpos_start + 2])

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames: list[np.ndarray] = []

    max_lift = 0.0
    n_contact_steps = 0
    peg_geom_id = nm.peg_geom_id
    hand_geoms: set[int] = set()
    for gset in nm.finger_geom_ids_per_finger:
        hand_geoms |= gset

    for step in range(steps):
        if mode == "pregrasp_hold":
            ctrl = scripted_grip_and_lift(model, step, grip_ctrl)
        else:
            ctrl = scripted_reach_and_grip(model, step, grip_ctrl, peg_xy)

        data.ctrl[: model.nu] = ctrl
        for _ in range(cfg.frame_skip):
            mujoco.mj_step(model, data)

        peg_z = float(data.xpos[nm.peg_body_id][2])
        lift = peg_z - peg_z0
        if lift > max_lift:
            max_lift = lift

        contacts_this = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            if (c.geom1 == peg_geom_id and c.geom2 in hand_geoms) or (
                c.geom2 == peg_geom_id and c.geom1 in hand_geoms
            ):
                contacts_this += 1
        if contacts_this > 0:
            n_contact_steps += 1

        renderer.update_scene(data, camera="track_cam")
        frames.append(renderer.render().copy())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    renderer.close()

    return {
        "max_lift_m": max_lift,
        "n_contact_steps": float(n_contact_steps),
        "n_steps": float(steps),
        "peg_xy": peg_xy,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("runs/render"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results: list[tuple[str, dict[str, float]]] = []

    # 1. current geometry + pre-grasped hold (curriculum stage 0)
    cfg1 = PegSceneConfig()
    out1 = args.out_dir / "peg_1_current_pregrasp.mp4"
    print(f"[1/4] current mount={cfg1.mount_height} + pre-grasped hold -> {out1}")
    r1 = render_episode(cfg1, "current_pregrasp", out1, mode="pregrasp_hold", steps=args.steps, seed=args.seed)
    results.append(("current+pregrasp_hold", r1))

    # 2. current geometry + reach-and-grip
    out2 = args.out_dir / "peg_2_current_reach.mp4"
    print(f"[2/4] current mount + reach-and-grip -> {out2}")
    r2 = render_episode(cfg1, "current_reach", out2, mode="reach_and_grip", steps=args.steps, seed=args.seed)
    results.append(("current+reach_grip", r2))

    # 3. lowered mount + pre-grasped
    cfg2 = PegSceneConfig(mount_height=0.78)
    out3 = args.out_dir / "peg_3_lowered_pregrasp.mp4"
    print(f"[3/4] mount=0.78 + pre-grasped hold -> {out3}")
    r3 = render_episode(cfg2, "lowered_pregrasp", out3, mode="pregrasp_hold", steps=args.steps, seed=args.seed)
    results.append(("mount=0.78+pregrasp_hold", r3))

    # 4. lowered mount + reach-and-grip
    out4 = args.out_dir / "peg_4_lowered_reach.mp4"
    print(f"[4/4] mount=0.78 + reach-and-grip -> {out4}")
    r4 = render_episode(cfg2, "lowered_reach", out4, mode="reach_and_grip", steps=args.steps, seed=args.seed)
    results.append(("mount=0.78+reach_grip", r4))

    print("\n=== summary ===")
    print(f"{'condition':<28}  {'max_lift_m':>10}  {'contact_steps':>14}  {'peg_xy':>16}")
    for name, r in results:
        peg_xy = r["peg_xy"]
        print(f"{name:<28}  {r['max_lift_m']:>10.4f}  {int(r['n_contact_steps']):>10}/{int(r['n_steps'])}  ({peg_xy[0]:+.3f},{peg_xy[1]:+.3f})")


if __name__ == "__main__":
    main()
