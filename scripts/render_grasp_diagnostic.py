"""Render grasp-scene diagnostics.

Three conditions:
  1. current geometry + random policy (sanity: scene loads, nothing explodes)
  2. grip+lift proof at the spawn-band centre — a seeded grip around the cube,
     settle, then slide_z lift to ~0.18 and hold. This is the physical
     winnability demo for lift_target=0.10 (see
     tests/test_geometry.py::test_grasp_lift_reaches_target_height).
  3. same proof at the far corner of the spawn band (grip seed shifted with
     the cube) — checks the lift works across the band, not just at centre.

The old scripted open-loop "slide+curl" policy was removed 2026-07-14: it
never actually captured the cube (it swatted it away during the approach and
closed on air), so its metrics measured nothing. The teleported-grip proof
isolates the question that matters for the reward chain — "does a formed grip
plus slide_z yield a stable lift?" — the approach itself is RL's job (the 5M
sanity already showed nfc ~4.9 grips form reliably).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from dexterous_hand.config import SceneConfig
from dexterous_hand.envs.scene_builder import (
    OBJECT_TYPES,
    build_scene,
    get_object_half_height,
)

# Grip seed measured to hold a 20cm+ lift (2026-07-14 sweep: 222/432 combos
# held >10cm; this one 0.235m). Keep in sync with
# tests/test_geometry.py::CUBE_GRIP_SEED.
CUBE_GRIP_SEED = {
    "sx": 0.115, "sy": -0.017, "z0": -0.02,
    "j3": 1.0, "j12": 0.5, "thj5": 0.5, "th1": 0.7, "squeeze": 0.4,
}
LIFT_Z = 0.18
N_SETTLE, N_LIFT, N_HOLD = 30, 40, 80


def _set_joint(model: mujoco.MjModel, qpos: np.ndarray, name: str, val: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    qpos[model.jnt_qposadr[jid]] = float(np.clip(val, lo, hi))


def _seta(model: mujoco.MjModel, ctrl: np.ndarray, act_name: str, target: float) -> None:
    ai = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    if ai < 0:
        return
    lo, hi = model.actuator_ctrlrange[ai]
    ctrl[ai] = float(np.clip(target, lo, hi))


def _grip_ctrl(model: mujoco.MjModel, p: dict, squeeze: float, z: float) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=np.float64)
    _seta(model, ctrl, "slide_x_act", p["sx"])
    _seta(model, ctrl, "slide_y_act", p["sy"])
    _seta(model, ctrl, "slide_z_act", z)
    for an in ("rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"):
        _seta(model, ctrl, an, p["j3"] + squeeze)
    for an in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
        _seta(model, ctrl, an, p["j12"] * 2 + squeeze)
    _seta(model, ctrl, "rh_A_THJ5", p["thj5"])
    _seta(model, ctrl, "rh_A_THJ4", 1.2)
    _seta(model, ctrl, "rh_A_THJ2", 0.3)
    _seta(model, ctrl, "rh_A_THJ1", p["th1"] + squeeze)
    return ctrl


def _hand_geoms(nm) -> set[int]:
    geoms: set[int] = set()
    for gset in nm.finger_geom_ids_per_finger:
        geoms |= gset
    return geoms


def render_random(cfg: SceneConfig, out_path: Path, steps: int, seed: int) -> dict[str, float]:
    model, data, nm = build_scene(cfg)
    rng = np.random.default_rng(seed)

    gt, gs = OBJECT_TYPES["large_cube"]
    obj_z0 = cfg.table_height + get_object_half_height(gt, gs) + 0.001
    s = nm.obj_qpos_start
    data.qpos[s : s + 3] = [0.075, 0.0, obj_z0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    max_lift = 0.0
    for _ in range(steps):
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        a = rng.uniform(-1.0, 1.0, size=model.nu)
        data.ctrl[: model.nu] = lo + (a + 1.0) / 2.0 * (hi - lo)
        mujoco.mj_step(model, data, nstep=cfg.frame_skip)
        max_lift = max(max_lift, float(data.xpos[nm.object_body_id][2] - obj_z0))
        renderer.update_scene(data, camera="track_cam")
        frames.append(renderer.render().copy())
    _write_video(out_path, frames)
    renderer.close()
    return {"max_lift_m": max_lift}


def render_grip_lift(
    cfg: SceneConfig, cube_xy: tuple[float, float], out_path: Path
) -> dict[str, float]:
    model, data, nm = build_scene(cfg)
    # shift the grip seed's slides with the cube (seed was measured at x=0.075)
    p = dict(CUBE_GRIP_SEED)
    p["sx"] = CUBE_GRIP_SEED["sx"] + (cube_xy[0] - 0.075)
    p["sy"] = CUBE_GRIP_SEED["sy"] + cube_xy[1]

    qpos = data.qpos.copy()
    _set_joint(model, qpos, "slide_x", p["sx"])
    _set_joint(model, qpos, "slide_y", p["sy"])
    _set_joint(model, qpos, "slide_z", p["z0"])
    for j in ("FF", "MF", "RF", "LF"):
        _set_joint(model, qpos, f"rh_{j}J3", p["j3"])
        _set_joint(model, qpos, f"rh_{j}J2", p["j12"])
        _set_joint(model, qpos, f"rh_{j}J1", p["j12"])
    _set_joint(model, qpos, "rh_THJ5", p["thj5"])
    _set_joint(model, qpos, "rh_THJ4", 1.2)
    _set_joint(model, qpos, "rh_THJ2", 0.3)
    _set_joint(model, qpos, "rh_THJ1", p["th1"])

    gt, gs = OBJECT_TYPES["large_cube"]
    obj_z0 = cfg.table_height + get_object_half_height(gt, gs) + 0.001
    s = nm.obj_qpos_start
    qpos[s : s + 3] = [cube_xy[0], cube_xy[1], obj_z0]
    qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    hand_geoms = _hand_geoms(nm)
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    total = N_SETTLE + N_LIFT + N_HOLD
    final_lift = max_lift = 0.0
    held = 0
    for step in range(total):
        if step < N_SETTLE:
            squeeze, z = p["squeeze"] * min(step / 10.0, 1.0), p["z0"]
        elif step < N_SETTLE + N_LIFT:
            t = (step - N_SETTLE) / N_LIFT
            squeeze, z = p["squeeze"], p["z0"] + (LIFT_Z - p["z0"]) * t
        else:
            squeeze, z = p["squeeze"], LIFT_Z
        data.ctrl[: model.nu] = _grip_ctrl(model, p, squeeze, z)
        mujoco.mj_step(model, data, nstep=cfg.frame_skip)
        final_lift = float(data.xpos[nm.object_body_id][2] - obj_z0)
        max_lift = max(max_lift, final_lift)
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
        renderer.update_scene(data, camera="track_cam")
        frames.append(renderer.render().copy())
    _write_video(out_path, frames)
    renderer.close()
    return {"final_lift_m": final_lift, "max_lift_m": max_lift, "held_last40": float(held)}


def _write_video(out_path: Path, frames: list[np.ndarray], fps: int = 25) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("runs/render"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SceneConfig()
    results: list[tuple[str, dict[str, float]]] = []

    out1 = args.out_dir / "1_random.mp4"
    print(f"[1/3] random policy -> {out1}")
    results.append(("random", render_random(cfg, out1, args.steps, args.seed)))

    out2 = args.out_dir / "2_grip_lift_centre.mp4"
    print(f"[2/3] grip+lift proof, spawn centre -> {out2}")
    results.append(("grip+lift centre", render_grip_lift(cfg, (0.075, 0.0), out2)))

    out3 = args.out_dir / "3_grip_lift_corner.mp4"
    print(f"[3/3] grip+lift proof, spawn corner -> {out3}")
    results.append(("grip+lift corner", render_grip_lift(cfg, (0.095, 0.025), out3)))

    print("\n=== summary ===")
    for name, r in results:
        parts = "  ".join(f"{k}={v:.4f}" for k, v in r.items())
        print(f"{name:<20} {parts}")


if __name__ == "__main__":
    main()
