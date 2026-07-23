"""MJX <-> CPU-MuJoCo physics parity check."""
from __future__ import annotations

import argparse
import time
from typing import Any

import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig, SceneConfig
from dexterous_hand.envs.peg_scene_builder import build_peg_scene
from dexterous_hand.envs.scene_builder import (
    GRIP_BIAS,
    apply_flexion_bias,
    build_grip_ctrl,
    build_scene,
)

GRASP_LIFT_BAR = 0.15
PEG_SETTLE_BAR = 0.73
PEG_HOLD_BAR = 0.70




class CpuEngine:
    name = "cpu"

    def __init__(self, model: mujoco.MjModel, frame_skip: int) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.frame_skip = frame_skip

    def set_state(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def ctrl_step(self, ctrl: np.ndarray, n: int = 1) -> None:
        for _ in range(n):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def xpos(self, body_id: int) -> np.ndarray:
        return np.array(self.data.xpos[body_id])

    def xpos_all(self) -> np.ndarray:
        return np.array(self.data.xpos)

    def xmat_all(self) -> np.ndarray:
        return np.array(self.data.xmat).reshape(self.model.nbody, 9)

    def site_xpos(self, site_id: int) -> np.ndarray:
        return np.array(self.data.site_xpos[site_id])

    def qpos_at(self, adr: int) -> float:
        return float(self.data.qpos[adr])

    def sensordata(self) -> np.ndarray:
        return np.array(self.data.sensordata)


class MjxEngine:
    name = "mjx"

    def __init__(self, model: mujoco.MjModel, frame_skip: int) -> None:
        import jax
        import jax.numpy as jnp
        import mujoco.mjx as mjx

        self._jnp = jnp
        self._mjx = mjx
        self.model = model
        self.frame_skip = frame_skip
        self.mjx_model = mjx.put_model(model)
        self.data = mjx.make_data(self.mjx_model)

        mjx_model = self.mjx_model

        @jax.jit
        def _run(data, ctrl):
            data = data.replace(ctrl=ctrl)

            def sub(d, _):
                return mjx.step(mjx_model, d), None

            data, _ = jax.lax.scan(sub, data, None, length=frame_skip)
            return data

        self._stepper = _run

    def set_state(self, qpos: np.ndarray) -> None:
        jnp = self._jnp
        self.data = self.data.replace(
            qpos=jnp.asarray(qpos), qvel=jnp.zeros(self.model.nv)
        )
        self.data = self._mjx.forward(self.mjx_model, self.data)

    def ctrl_step(self, ctrl: np.ndarray, n: int = 1) -> None:
        c = self._jnp.asarray(ctrl)
        for _ in range(n):
            self.data = self._stepper(self.data, c)

    def xpos(self, body_id: int) -> np.ndarray:
        return np.asarray(self.data.xpos[body_id])

    def xpos_all(self) -> np.ndarray:
        return np.asarray(self.data.xpos)

    def xmat_all(self) -> np.ndarray:
        return np.asarray(self.data.xmat).reshape(self.model.nbody, 9)

    def site_xpos(self, site_id: int) -> np.ndarray:
        return np.asarray(self.data.site_xpos[site_id])

    def qpos_at(self, adr: int) -> float:
        return float(self.data.qpos[adr])

    def sensordata(self) -> np.ndarray:
        return np.asarray(self.data.sensordata)




def _act(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def _set_ctrl(model: mujoco.MjModel, ctrl: np.ndarray, name: str, target: float) -> None:
    ai = _act(model, name)
    if ai < 0:
        return
    lo, hi = model.actuator_ctrlrange[ai]
    ctrl[ai] = float(np.clip(target, lo, hi))


def _set_qpos_joint(model: mujoco.MjModel, qpos: np.ndarray, name: str, val: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    qpos[model.jnt_qposadr[jid]] = float(np.clip(val, lo, hi))



CUBE_GRIP_SEED = {
    "sx": 0.115, "sy": -0.017, "z0": -0.02,
    "j3": 1.0, "j12": 0.5, "thj5": 0.5, "th1": 0.7, "squeeze": 1.0,
}


def run_grasp(engine_cls) -> dict[str, float]:
    cfg = SceneConfig()
    model, data, nm = build_scene(cfg)
    eng = engine_cls(model, cfg.frame_skip)
    p = CUBE_GRIP_SEED

    qpos = np.array(data.qpos)
    _set_qpos_joint(model, qpos, "slide_x", p["sx"])
    _set_qpos_joint(model, qpos, "slide_y", p["sy"])
    _set_qpos_joint(model, qpos, "slide_z", p["z0"])
    for j in ("FF", "MF", "RF", "LF"):
        _set_qpos_joint(model, qpos, f"rh_{j}J3", p["j3"])
        _set_qpos_joint(model, qpos, f"rh_{j}J2", p["j12"])
        _set_qpos_joint(model, qpos, f"rh_{j}J1", p["j12"])
    _set_qpos_joint(model, qpos, "rh_THJ5", p["thj5"])
    _set_qpos_joint(model, qpos, "rh_THJ4", 1.2)
    _set_qpos_joint(model, qpos, "rh_THJ2", 0.3)
    _set_qpos_joint(model, qpos, "rh_THJ1", p["th1"])
    obj_z0 = cfg.table_height + cfg.object_half_extent + 0.001
    s = nm.obj_qpos_start
    qpos[s : s + 3] = [0.075, 0.0, obj_z0]
    qpos[s + 3 : s + 7] = [1.0, 0.0, 0.0, 0.0]
    eng.set_state(qpos)

    def grip_ctrl(squeeze: float, z: float) -> np.ndarray:
        ctrl = np.zeros(model.nu)
        _set_ctrl(model, ctrl, "slide_x_act", p["sx"])
        _set_ctrl(model, ctrl, "slide_y_act", p["sy"])
        _set_ctrl(model, ctrl, "slide_z_act", z)
        for an in ("rh_A_FFJ3", "rh_A_MFJ3", "rh_A_RFJ3", "rh_A_LFJ3"):
            _set_ctrl(model, ctrl, an, p["j3"] + squeeze)
        for an in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
            _set_ctrl(model, ctrl, an, p["j12"] * 2 + squeeze)
        _set_ctrl(model, ctrl, "rh_A_THJ5", p["thj5"])
        _set_ctrl(model, ctrl, "rh_A_THJ4", 1.2)
        _set_ctrl(model, ctrl, "rh_A_THJ2", 0.3)
        _set_ctrl(model, ctrl, "rh_A_THJ1", p["th1"] + squeeze)
        return ctrl

    for step in range(30):
        eng.ctrl_step(grip_ctrl(p["squeeze"] * min(step / 10.0, 1.0), p["z0"]))
    for step in range(40):
        t = step / 40.0
        eng.ctrl_step(grip_ctrl(p["squeeze"], p["z0"] + (0.18 - p["z0"]) * t))
    for _ in range(80):
        eng.ctrl_step(grip_ctrl(p["squeeze"], 0.18))

    final_lift = float(eng.xpos(nm.object_body_id)[2] - obj_z0)
    touch = eng.sensordata()[np.asarray(nm.sensor_map.finger_touch_adr)]
    nfc = int(np.sum(touch > 0.0))
    return {"final_lift": final_lift, "nfc_end": float(nfc)}




def run_peg(engine_cls) -> dict[str, float]:
    import jax.numpy as jnp

    from dexterous_hand.utils.mjx_helpers import get_insertion_depth_jax

    cfg = PegSceneConfig()
    model, data, nm = build_peg_scene(cfg)
    eng = engine_cls(model, cfg.frame_skip)
    peg_len = cfg.peg_half_length * 2.0 + cfg.peg_radius * 2.0

    qpos = np.array(data.qpos)
    apply_flexion_bias(qpos, model, bias_map=GRIP_BIAS)
    eng.set_state(qpos)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    grasp_xyz = eng.site_xpos(sid)
    peg_qadr = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    ]
    qpos[peg_qadr : peg_qadr + 3] = grasp_xyz
    qpos[peg_qadr + 3 : peg_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    eng.set_state(qpos)

    grip = build_grip_ctrl(model)
    eng.ctrl_step(grip, n=1)

    hole_pos = eng.xpos(nm.hole_body_id)
    entrance_z = hole_pos[2]

    sx_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")]
    sy_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")]

    state: dict[str, Any] = {"z": 0.0, "xy": np.zeros(2), "open": 0.0}

    def do_steps(n: int, open_fingers: bool = False, servo: bool = False) -> None:
        for _ in range(n):
            if open_fingers:
                state["open"] = min(state["open"] + 0.15, 1.0)
                c = grip * (1.0 - state["open"])
            else:
                c = grip.copy()
            if servo:
                err = hole_pos[:2] - eng.xpos(nm.peg_body_id)[:2]
                slide_now = np.array([eng.qpos_at(sx_adr), eng.qpos_at(sy_adr)])
                desired = slide_now + 0.8 * err
                state["xy"] = state["xy"] + np.clip(desired - state["xy"], -0.003, 0.003)
            _set_ctrl(model, c, "slide_x_act", state["xy"][0])
            _set_ctrl(model, c, "slide_y_act", state["xy"][1])
            _set_ctrl(model, c, "slide_z_act", state["z"])
            eng.ctrl_step(c)

    def depth() -> float:
        return float(
            get_insertion_depth_jax(
                jnp.asarray(eng.xpos_all()), jnp.asarray(eng.xmat_all()),
                nm.peg_body_id, nm.hole_body_id,
                cfg.peg_half_length, cfg.peg_radius,
                cfg.peg_radius + cfg.clearance,
                cfg.hole_depth,
            )
        )

    state["z"] = 0.06
    do_steps(15)
    peg = eng.xpos(nm.peg_body_id)
    state["xy"] = np.array([eng.qpos_at(sx_adr), eng.qpos_at(sy_adr)]) + (
        hole_pos[:2] - peg[:2]
    )
    do_steps(25)
    do_steps(15, servo=True)
    tip_z = eng.xpos(nm.peg_body_id)[2] - peg_len / 2.0
    state["z"] += entrance_z + 0.01 - tip_z
    do_steps(15, servo=True)
    for _ in range(10):
        tip_z = eng.xpos(nm.peg_body_id)[2] - peg_len / 2.0
        state["z"] += float(np.clip((entrance_z - 0.020) - tip_z, -0.004, 0.004))
        do_steps(2, servo=True)
    do_steps(15, open_fingers=True)
    state["z"] += 0.06
    do_steps(25, open_fingers=True)
    do_steps(75, open_fingers=True)

    settled = depth() / peg_len
    fracs = []
    for _ in range(50):
        do_steps(1, open_fingers=True)
        fracs.append(depth() / peg_len)
    return {"settled_frac": settled, "min_hold_frac": float(np.min(fracs))}




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend", choices=["both", "cpu", "mjx"], default="both",
        help="which engine(s) to run (default both — the parity A/B)",
    )
    args = ap.parse_args()

    engines: list[type[CpuEngine] | type[MjxEngine]] = []
    if args.backend in ("both", "cpu"):
        engines.append(CpuEngine)
    if args.backend in ("both", "mjx"):
        try:
            import mujoco.mjx  # noqa: F401
        except ImportError:
            print("ERROR: mujoco.mjx not importable — install the mjx extra "
                  "(uv sync --extra mjx). Refusing to 'pass' without testing MJX.")
            raise SystemExit(2) from None
        engines.append(MjxEngine)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for eng_cls in engines:
        for task, fn in (("grasp", run_grasp), ("peg", run_peg)):
            t0 = time.time()
            print(f"[{eng_cls.name}] {task} trajectory ...", flush=True)
            r = fn(eng_cls)
            r["seconds"] = time.time() - t0
            results.setdefault(task, {})[eng_cls.name] = r

    print("\n=== parity results ===")
    ok = True
    for task, per_engine in results.items():
        for name, r in per_engine.items():
            if task == "grasp":
                passed = r["final_lift"] >= GRASP_LIFT_BAR and r["nfc_end"] >= 2
                print(f"  grasp [{name}]: final_lift={r['final_lift']*1000:6.1f}mm "
                      f"(bar {GRASP_LIFT_BAR*1000:.0f}) nfc_end={int(r['nfc_end'])} "
                      f"({r['seconds']:.0f}s)  {'PASS' if passed else 'FAIL'}")
            else:
                passed = (r["settled_frac"] >= PEG_SETTLE_BAR
                          and r["min_hold_frac"] >= PEG_HOLD_BAR)
                print(f"  peg   [{name}]: settled={r['settled_frac']:.3f} "
                      f"(bar {PEG_SETTLE_BAR}) min_hold={r['min_hold_frac']:.3f} "
                      f"(bar {PEG_HOLD_BAR}) ({r['seconds']:.0f}s)  "
                      f"{'PASS' if passed else 'FAIL'}")
            ok &= passed

    print()
    if ok:
        print("PARITY OK — MJX reproduces the CPU-proven winning trajectories."
              if args.backend == "both" else "All trajectories PASS.")
    else:
        print("PARITY FAILURE — do NOT launch a sanity/full run until resolved.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
