from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from dexterous_hand.config import SceneConfig
from dexterous_hand.envs.scene_builder import (
    OBJECT_TYPES,
    apply_flexion_bias,
    build_scene,
    get_object_half_height,
    TABLE_TASK_FLEXION_BIAS,
)


def make_data(cfg: SceneConfig, seed: int = 0):
    model, data, nm = build_scene(cfg)
    rng = np.random.default_rng(seed)

    qpos = data.qpos.copy()
    apply_flexion_bias(qpos, model)

    gt, gs = OBJECT_TYPES["large_cube"]
    half_h = get_object_half_height(gt, gs)
    obj_x = rng.uniform(0.05, 0.10)
    obj_y = rng.uniform(-0.03, 0.03)
    obj_z = cfg.table_height + half_h + 0.001
    s = nm.obj_qpos_start
    qpos[s : s + 3] = [obj_x, obj_y, obj_z]
    qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]

    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    return model, data, nm, (obj_x, obj_y, obj_z)


def _act_idx(model: mujoco.MjModel, name: str) -> int | None:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def random_actions(model, data, nm, rng, frame_skip: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=model.nu).astype(np.float64)


def scripted_grasp(model, data, nm, step, frame_skip: int) -> np.ndarray:
    """Open-loop policy: slide forward, descend by curling, close fingers, lift.

    Phases (each phase is N control steps; control step = frame_skip * sim_step):
      0-30:    slide to (+0.07, target_y)
      30-60:   curl fingers to ~0.7 and bring thumb opposed
      60-90:   close grip
      90-200:  hold + try to lift by raising slide_x slightly (no z slider, so
               lift comes from finger curl pulling cube up against palm)
    """
    ctrl = np.zeros(model.nu, dtype=np.float64)

    sx = _act_idx(model, "slide_x_act")
    sy = _act_idx(model, "slide_y_act")
    # cube now spawns forward at x≈0.075, so slider needs to extend further.
    if sx is not None:
        ctrl[sx] = 0.13
    if sy is not None:
        ctrl[sy] = -0.02  # shift toward thumb side so thumb opposes cube

    # mapping helpers (Shadow actuators are named rh_A_<JOINT>)
    def seta(act_name: str, target: float) -> None:
        ai = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if ai < 0:
            return
        lo, hi = model.actuator_ctrlrange[ai]
        ctrl[ai] = float(np.clip(target, lo, hi))

    # phase ramps
    if step < 30:
        curl = 0.0
    elif step < 60:
        curl = 0.7 * (step - 30) / 30.0
    elif step < 90:
        curl = 0.7 + 0.6 * (step - 60) / 30.0  # ramp to 1.3
    else:
        curl = 1.3

    # knuckle (J3) per finger
    for an in ("rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"):
        seta(an, curl)
    # FFJ0 actuator drives the FFJ1+FFJ2 coupled tendon (range [0, pi])
    for an in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
        seta(an, curl * 2.0)

    seta("rh_A_THJ4", 1.2)
    seta("rh_A_THJ2", 0.5)
    seta("rh_A_THJ1", curl)

    return ctrl


def render_episode(
    cfg: SceneConfig,
    label: str,
    policy_name: str,
    out_path: Path,
    steps: int = 200,
    fps: int = 25,
    seed: int = 0,
) -> dict[str, float]:
    model, data, nm, (obj_x, obj_y, obj_z) = make_data(cfg, seed=seed)
    renderer = mujoco.Renderer(model, height=480, width=640)
    rng = np.random.default_rng(seed)
    frames: list[np.ndarray] = []

    cube_top_z0 = obj_z + get_object_half_height(*OBJECT_TYPES["large_cube"])
    max_lift = 0.0
    n_contact_steps = 0

    for step in range(steps):
        if policy_name == "random":
            action = rng.uniform(-1.0, 1.0, size=model.nu).astype(np.float64)
            lo = model.actuator_ctrlrange[: model.nu, 0]
            hi = model.actuator_ctrlrange[: model.nu, 1]
            ctrl = lo + (action + 1.0) / 2.0 * (hi - lo)
            data.ctrl[: model.nu] = ctrl
        elif policy_name == "scripted":
            data.ctrl[: model.nu] = scripted_grasp(model, data, nm, step, frame_skip=cfg.frame_skip)
        else:
            raise ValueError(policy_name)

        for _ in range(cfg.frame_skip):
            mujoco.mj_step(model, data)

        obj_pos = data.xpos[nm.object_body_id]
        lift = float(obj_pos[2] - obj_z)
        if lift > max_lift:
            max_lift = lift

        # count contacts between hand geoms and cube geom
        cube_g = nm.object_geom_id
        hand_geoms: set[int] = set()
        for gset in nm.finger_geom_ids_per_finger:
            hand_geoms |= gset
        contacts_this = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            if (c.geom1 == cube_g and c.geom2 in hand_geoms) or (
                c.geom2 == cube_g and c.geom1 in hand_geoms
            ):
                contacts_this += 1
        if contacts_this > 0:
            n_contact_steps += 1

        renderer.update_scene(data, camera="track_cam")
        frame = renderer.render().copy()
        # overlay simple HUD: step number + lift
        # (skipping overlay to keep dependencies thin — mp4 will show the cube)
        frames.append(frame)

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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("runs/render"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results: list[tuple[str, dict[str, float]]] = []

    # 1. current config + random policy
    cfg1 = SceneConfig()  # current: mount_height=0.82
    out1 = args.out_dir / "1_current_random.mp4"
    print(f"[1/3] current geometry (mount_height={cfg1.mount_height}) + random policy -> {out1}")
    r1 = render_episode(cfg1, "current_random", "random", out1, steps=args.steps, seed=args.seed)
    results.append(("current+random", r1))

    # 2. current config + scripted grasp (open-loop "ideal-attempt")
    cfg2 = SceneConfig()
    out2 = args.out_dir / "2_current_scripted.mp4"
    print(f"[2/3] current geometry + scripted grasp -> {out2}")
    r2 = render_episode(cfg2, "current_scripted", "scripted", out2, steps=args.steps, seed=args.seed)
    results.append(("current+scripted", r2))

    # 3. lowered mount + scripted grasp
    cfg3 = SceneConfig(mount_height=0.78)
    out3 = args.out_dir / "3_lowered_scripted.mp4"
    print(f"[3/3] mount_height=0.78 + scripted grasp -> {out3}")
    r3 = render_episode(cfg3, "lowered_scripted", "scripted", out3, steps=args.steps, seed=args.seed)
    results.append(("mount=0.78+scripted", r3))

    print("\n=== summary ===")
    print(f"{'condition':<25}  {'max_lift_m':>10}  {'contact_steps':>14}")
    for name, r in results:
        print(f"{name:<25}  {r['max_lift_m']:>10.4f}  {int(r['n_contact_steps']):>10}/{int(r['n_steps'])}")


if __name__ == "__main__":
    main()
