"""Local CPU harness to tune GRIP_BIAS so the closed grip holds the peg vertical."""
from __future__ import annotations

import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import GRIP_BIAS, apply_flexion_bias, build_grip_ctrl


def _place_pregrasped(model, data, nm, bias, z_offset=0.0):
    qpos = data.qpos.copy()
    apply_flexion_bias(qpos, model, bias_map=bias)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    gs_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    gs = data.site_xpos[gs_id].copy()
    gs[2] += z_offset  # shift peg center up so the grip catches its middle, not the top cap
    qpos[nm.peg_qpos_start : nm.peg_qpos_start + 3] = gs
    qpos[nm.peg_qpos_start + 3 : nm.peg_qpos_start + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _peg_axis(data, bid):
    return data.xmat[bid].reshape(3, 3)[:, 2]


def _contact_report(model, data, nm):
    """Per-finger: in-contact?, mean contact-point z relative to peg center."""
    peg_gid = nm.peg_geom_id
    peg_c = data.xpos[nm.peg_body_id]
    names = ["ff", "mf", "rf", "lf", "th"]
    out = {n: [] for n in names}
    for ci in range(data.ncon):
        c = data.contact[ci]
        other = None
        if c.geom1 == peg_gid:
            other = c.geom2
        elif c.geom2 == peg_gid:
            other = c.geom1
        if other is None:
            continue
        for fi, gset in enumerate(nm.finger_geom_ids_per_finger):
            if other in gset:
                out[names[fi]].append(c.pos.copy())
                break
    summary = {}
    for n in names:
        pts = out[n]
        if pts:
            mean = np.mean(pts, axis=0)
            summary[n] = (len(pts), float(mean[2] - peg_c[2]), float(mean[0] - peg_c[0]), float(mean[1] - peg_c[1]))
        else:
            summary[n] = (0, None, None, None)
    return summary


def _fingertip_report(model, data, nm):
    """At placement: each fingertip's offset from the peg center (who can reach)."""
    peg_c = data.xpos[nm.peg_body_id]
    names = ["ff", "mf", "rf", "lf", "th"]
    print("  fingertip offsets from peg center (dx,dy,dz | dist):")
    for n, sid in zip(names, nm.fingertip_site_ids, strict=False):
        p = data.site_xpos[sid] - peg_c
        print(f"    {n}: ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) | {np.linalg.norm(p):.3f}")


def hold_test(bias, steps=200, verbose=True, label="bias", show_place=False, z_offset=0.0):
    cfg = PegSceneConfig()
    model, data, nm = build_peg_scene(cfg)
    _place_pregrasped(model, data, nm, bias, z_offset=z_offset)
    if verbose and show_place:
        print(f"\n--- {label}: placement geometry ---")
        _fingertip_report(model, data, nm)
    grip = build_grip_ctrl(model, bias_map=bias)
    data.ctrl[: model.nu] = grip

    bid = nm.peg_body_id
    traj = {}
    checkpoints = [0, 5, 20, 50, 100, steps]
    min_aa = 1.0
    for step in range(steps + 1):
        pa = _peg_axis(data, bid)
        aa = abs(float(pa[2]))
        if step > 0:
            min_aa = min(min_aa, aa)
        if step in checkpoints:
            traj[step] = (aa, float(data.xpos[bid][2]), pa.copy())
        if step == steps:
            break
        for _ in range(cfg.frame_skip):
            mujoco.mj_step(model, data)

    if verbose:
        print(f"\n=== {label} ===")
        for s in checkpoints:
            aa, pz, pa = traj[s]
            print(f"  step{s:>3}: axis_align={aa:.3f}  peg_z={pz:.4f}  peg_axis=({pa[0]:+.2f},{pa[1]:+.2f},{pa[2]:+.2f})")
        cr = _contact_report(model, data, nm)
        print(f"  min axis_align over hold = {min_aa:.3f}")
        print("  final contacts (n, dz_from_center, dx, dy):")
        for n, (cnt, dz, dx, dy) in cr.items():
            if cnt:
                print(f"    {n}: n={cnt} dz={dz:+.3f} dx={dx:+.3f} dy={dy:+.3f}")
            else:
                print(f"    {n}: --")
    return traj[steps][0], min_aa  # (final, min) axis_align


def _with(**over):
    b = dict(GRIP_BIAS)
    b.update(over)
    return b


def _without(*keys):
    b = dict(GRIP_BIAS)
    for k in keys:
        b.pop(k, None)
    return b


# Verification: the committed GRIP_BIAS (with THJ5) vs the old pose (THJ5 removed).
CANDIDATES = {
    "committed GRIP_BIAS": GRIP_BIAS,
    "old (no THJ5)": _without("rh_THJ5"),
}


def main():
    results = {}
    for label, bias in CANDIDATES.items():
        results[label] = hold_test(bias, label=label, show_place=False)
    print("\n=== peg pre-grasp hold: axis_align @200 (final / min; 1.0 = vertical) ===")
    for label, (fin, mn) in sorted(results.items(), key=lambda kv: -kv[1][1]):
        print(f"  {label:<24} final={fin:.3f}  min={mn:.3f}")


if __name__ == "__main__":
    main()
