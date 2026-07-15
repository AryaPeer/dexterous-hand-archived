from dataclasses import dataclass, field
import math
from pathlib import Path

import mujoco
import numpy as np

from dexterous_hand.config import SceneConfig
from dexterous_hand.utils.mujoco_helpers import get_joint_qpos_qvel_range

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "shadow_hand"

OBJECT_TYPES: dict[str, tuple[int, list[float]]] = {
    "large_cube": (mujoco.mjtGeom.mjGEOM_BOX, [0.035, 0.035, 0.035]),
}

# Distal-link body names: first/middle/ring/little/thumb
FINGERTIP_BODIES = [
    "rh_ffdistal",
    "rh_mfdistal",
    "rh_rfdistal",
    "rh_lfdistal",
    "rh_thdistal",
]

FINGERTIP_SITE_NAMES = ["fftip", "mftip", "rftip", "lftip", "thtip"]

FINGERTIP_OFFSETS: dict[str, list[float]] = {
    "rh_ffdistal": [0.0, 0.0, 0.026],
    "rh_mfdistal": [0.0, 0.0, 0.026],
    "rh_rfdistal": [0.0, 0.0, 0.026],
    "rh_lfdistal": [0.0, 0.0, 0.026],
    "rh_thdistal": [0.0, 0.0, 0.032],
}

FINGER_BODY_PREFIXES = ["rh_ff", "rh_mf", "rh_rf", "rh_lf", "rh_th"]
FINGER_TOUCH_SITE_NAMES = ["ff_touch", "mf_touch", "rf_touch", "lf_touch", "th_touch"]

# Grasp-scene vertical slide: joint range and the reset/hover position. The
# reset qpos is the ctrl midpoint so a zero (smoothed) action holds the reset
# height instead of jerking the hand at episode start.
SLIDE_Z_RANGE: tuple[float, float] = (-0.05, 0.20)
SLIDE_Z_INIT: float = (SLIDE_Z_RANGE[0] + SLIDE_Z_RANGE[1]) / 2.0

# Pre-curl applied at reset for table-top tasks (no object pre-grip)
TABLE_TASK_FLEXION_BIAS: dict[str, float] = {
    "rh_FFJ3": 1.2,
    "rh_MFJ3": 1.2,
    "rh_RFJ3": 1.2,
    "rh_LFJ3": 1.2,
    "rh_FFJ2": 1.0,
    "rh_MFJ2": 1.0,
    "rh_RFJ2": 1.0,
    "rh_LFJ2": 1.0,
    "rh_THJ4": 1.2,
    "rh_THJ1": 1.0,
}


# Heavier pre-curl used when an object is teleported into the closed grip
GRIP_BIAS: dict[str, float] = {
    "rh_FFJ3": 1.5,
    "rh_MFJ3": 1.5,
    "rh_RFJ3": 1.5,
    "rh_LFJ3": 1.5,
    "rh_FFJ2": 1.4,
    "rh_MFJ2": 1.4,
    "rh_RFJ2": 1.4,
    "rh_LFJ2": 1.4,
    "rh_FFJ1": 1.4,
    "rh_MFJ1": 1.4,
    "rh_RFJ1": 1.4,
    "rh_LFJ1": 1.4,
    # THJ5 (thumb-base opposition rotation) was missing. Without it the thumb
    # never swung across the palm to oppose the fingers, so the closed grip
    # touched the peg with only the middle+ring fingers from one side and the
    # capsule pivoted to ~45° off vertical (axis_align 0.71) even when placed
    # perfectly upright and held with a constant grip. Driving THJ5 to its
    # limit brings the thumb into opposition; the CPU hold-render
    # (scripts/tune_grip_bias.py) measures axis_align 0.71 -> 0.94, stable over
    # 200 steps, with the thumb now in contact. The residual ~20° is the policy's
    # to refine (it has full finger + slide_z control during the episode).
    "rh_THJ5": 1.047,
    "rh_THJ4": 1.2,
    "rh_THJ2": 0.5,
    "rh_THJ1": 1.4,
}


def apply_flexion_bias(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    bias_map: dict[str, float] = TABLE_TASK_FLEXION_BIAS,
) -> np.ndarray:
    """Set the joints listed in `bias_map` to their target angles, clipped to range."""
    for jname, bias in bias_map.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        adr = int(model.jnt_qposadr[jid])
        low, high = model.jnt_range[jid]
        qpos[adr] = float(np.clip(bias, low, high))
    return qpos


def build_grip_ctrl(
    model: mujoco.MjModel,
    bias_map: dict[str, float] = GRIP_BIAS,
) -> np.ndarray:
    """Return a ctrl vector that drives each `bias_map` joint to its target
    angle (clipped to the actuator ctrlrange). Non-bias actuators get 0.
    Used during reset-settle so closed fingers stay closed instead of
    snapping to ctrl=0 (= fully open for the flexion joints).

    Handles both joint and tendon actuators. For a tendon actuator (e.g.
    the Shadow Hand's FFJ0 driving FFJ1+FFJ2), MuJoCo interprets ctrl as
    the desired tendon length — the linear combination of the coupled
    joint angles — so we sum bias_map across the joints wrapped by the
    tendon.
    """
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for ai in range(model.nu):
        trntype = int(model.actuator_trntype[ai])
        lo, hi = model.actuator_ctrlrange[ai]
        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model.actuator_trnid[ai, 0])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname in bias_map:
                ctrl[ai] = float(np.clip(bias_map[jname], lo, hi))
        elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
            tid = int(model.actuator_trnid[ai, 0])
            wrap_start = int(model.tendon_adr[tid])
            wrap_count = int(model.tendon_num[tid])
            target = 0.0
            for wi in range(wrap_start, wrap_start + wrap_count):
                if int(model.wrap_type[wi]) != mujoco.mjtWrap.mjWRAP_JOINT:
                    continue
                jid = int(model.wrap_objid[wi])
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if jname in bias_map:
                    target += bias_map[jname]
            if target > 0.0:
                ctrl[ai] = float(np.clip(target, lo, hi))
    return ctrl


@dataclass
class SensorMap:
    finger_touch_adr: list[int]
    wall_force_adr: list[int] = field(default_factory=list)

    @staticmethod
    def empty() -> "SensorMap":
        return SensorMap(finger_touch_adr=[], wall_force_adr=[])


@dataclass
class NameMap:
    hand_joint_ids: list[int]
    hand_qpos_start: int
    hand_qpos_end: int
    hand_qvel_start: int
    hand_qvel_end: int

    palm_body_id: int
    fingertip_site_ids: list[int]
    finger_geom_ids_per_finger: list[set[int]]
    object_body_id: int
    object_geom_id: int
    obj_qpos_start: int
    obj_qvel_start: int
    sensor_map: SensorMap = field(default_factory=SensorMap.empty)


def build_scene(
    config: SceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, NameMap]:
    """Compile the table+hand+object grasp scene."""

    if config is None:
        config = SceneConfig()

    spec = mujoco.MjSpec()
    spec.option.timestep = config.sim_timestep
    spec.option.gravity = [0.0, 0.0, -9.81]
    # Contact model pinned EXPLICITLY: MjSpec.attach() does NOT merge the hand
    # XML's <option cone="elliptic" impratio="10"/>, so the compiled scene was
    # silently running MuJoCo defaults (pyramidal, impratio=1). Every grip
    # proof, grip-bias tuning and parity trajectory was measured under those
    # defaults — they are now the deliberate choice (also the usual MJX
    # hand-task configuration, e.g. MuJoCo Playground). Switching to the
    # Menagerie-intended elliptic/10 requires re-running tune_grip_bias, the
    # geometry tests, the renders and mjx_parity_check. Guarded by
    # tests/test_geometry.py::test_compiled_scene_contact_options.
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    spec.option.iterations = config.solver_iterations
    spec.option.ls_iterations = config.ls_iterations
    spec.stat.extent = 1.0
    spec.stat.center = [0.0, 0.0, config.table_height]

    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.01],
        rgba=[0.3, 0.3, 0.3, 1.0],
        conaffinity=1,
        condim=3,
    )

    table_half_h = 0.02
    table_body = spec.worldbody.add_body(
        name="table",
        pos=[0.0, 0.0, config.table_height - table_half_h],
    )
    table_body.add_geom(
        name="table_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[config.table_half_size, config.table_half_size, table_half_h],
        rgba=[0.5, 0.35, 0.2, 1.0],
        conaffinity=1,
        condim=3,
    )

    spec.worldbody.add_light(
        name="top_light",
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        specular=[0.3, 0.3, 0.3],
    )
    spec.worldbody.add_camera(
        name="track_cam",
        pos=[0.8, -0.8, 0.8],
        xyaxes=[0.707, 0.707, 0.0, -0.354, 0.354, 0.866],
    )

    slider = spec.worldbody.add_body(
        name="hand_slider",
        pos=[config.mount_x, config.mount_y, config.mount_height],
    )
    slider.add_joint(
        name="slide_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[-0.15, 0.15],
    )
    slider.add_joint(
        name="slide_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0],
        range=[-0.15, 0.15],
    )
    # Vertical arm DOF. Without it the hand is bolted at mount_height and the
    # only lift mechanism is finger curl (~1cm ceiling) — which is why grasp
    # lift_target eroded 0.1 -> 0.012 over rounds 11-13 while every reference
    # implementation (Adroit relocate: 6-DOF arm; robosuite Lift: full arm;
    # peg task in this repo: slide_z) gives the hand vertical mobility.
    # Range floor -0.05 lets fingers reach the table; +0.20 gives a visible
    # pick-up-and-hold. SLIDE_Z_INIT (ctrl midpoint) is the reset hover.
    slider.add_joint(
        name="slide_z",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[SLIDE_Z_RANGE[0], SLIDE_Z_RANGE[1]],
    )

    mount = slider.add_body(
        name="hand_mount",
        euler=[math.pi, 0.0, 0.0],
    )
    mount_site = mount.add_site(
        name="hand_attach",
        pos=[0.0, 0.0, 0.0],
    )

    spec.add_actuator(
        name="slide_x_act",
        target="slide_x",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -100, -10, 0, 0, 0, 0, 0, 0, 0],
        ctrlrange=[-0.15, 0.15],
        forcerange=[-30, 30],
    )
    spec.add_actuator(
        name="slide_y_act",
        target="slide_y",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -100, -10, 0, 0, 0, 0, 0, 0, 0],
        ctrlrange=[-0.15, 0.15],
        forcerange=[-30, 30],
    )
    # Same gains as the peg scene's slide_z: it fights gravity on the whole
    # ~4 kg hand, so kp=100 (the x/y gain) would sag to the joint floor under
    # load. kp=8000 + matching damping + 250 N holds position with sub-cm sag
    # while lifting hand + object.
    spec.add_actuator(
        name="slide_z_act",
        target="slide_z",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[8000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -8000, -250, 0, 0, 0, 0, 0, 0, 0],
        ctrlrange=[SLIDE_Z_RANGE[0], SLIDE_Z_RANGE[1]],
        forcerange=[-250, 250],
    )

    hand_xml = str(ASSETS_DIR / "right_hand.xml")
    child_spec = mujoco.MjSpec.from_file(hand_xml)
    spec.attach(child_spec, site=mount_site, prefix="")
    spec.body("rh_forearm").quat = [0.0, 1.0, 0.0, 0.0]

    for body_name, site_name in zip(FINGERTIP_BODIES, FINGERTIP_SITE_NAMES, strict=True):
        body = spec.body(body_name)
        offset = FINGERTIP_OFFSETS[body_name]
        body.add_site(
            name=site_name,
            pos=offset,
            size=[0.005],
            rgba=[1.0, 0.0, 0.0, 1.0],
        )

    # touch sensor sites: spheres co-located with the fingertip
    for body_name, touch_site in zip(FINGERTIP_BODIES, FINGER_TOUCH_SITE_NAMES, strict=True):
        body = spec.body(body_name)
        offset = FINGERTIP_OFFSETS[body_name]
        body.add_site(
            name=touch_site,
            pos=offset,
            size=[0.012],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            group=4,
        )

    for touch_site in FINGER_TOUCH_SITE_NAMES:
        spec.add_sensor(
            name=f"sensor_{touch_site}",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=touch_site,
        )

    default_type, default_size = OBJECT_TYPES["large_cube"]
    obj_body = spec.worldbody.add_body(
        name="object",
        pos=[0.0, 0.0, config.table_height + default_size[2]],
    )
    obj_body.add_freejoint(name="object_freejoint")
    obj_body.add_geom(
        name="object_geom",
        type=default_type,
        size=default_size,
        mass=config.object_mass,
        friction=list(config.object_friction),
        rgba=[0.2, 0.6, 0.9, 1.0],
        contype=1,
        conaffinity=1,
        condim=4,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    name_map = _resolve_names(model, spec)

    return model, data, name_map


def _resolve_names(model: mujoco.MjModel, spec: mujoco.MjSpec) -> NameMap:
    # object freejoint
    obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    obj_qpos_start = model.jnt_qposadr[obj_jnt_id]
    obj_qvel_start = model.jnt_dofadr[obj_jnt_id]

    # hand joints (everything except the object freejoint)
    hand_joint_ids = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name != "object_freejoint":
            hand_joint_ids.append(jid)

    if hand_joint_ids:
        hand_qpos_start, hand_qpos_end, hand_qvel_start, hand_qvel_end = get_joint_qpos_qvel_range(
            model, hand_joint_ids
        )
    else:
        hand_qpos_start = hand_qpos_end = 0
        hand_qvel_start = hand_qvel_end = 0

    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")

    # fingertips
    fingertip_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FINGERTIP_SITE_NAMES
    ]

    finger_geom_ids_per_finger: list[set[int]] = [set() for _ in FINGER_BODY_PREFIXES]
    for gid in range(model.ngeom):
        body_id = model.geom_bodyid[gid]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue

        for finger_idx, prefix in enumerate(FINGER_BODY_PREFIXES):
            if body_name.startswith(prefix):
                finger_geom_ids_per_finger[finger_idx].add(gid)
                break

    # object
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")

    # sensors
    finger_touch_adr = []
    for touch_site in FINGER_TOUCH_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_{touch_site}")
        if sid >= 0:
            finger_touch_adr.append(int(model.sensor_adr[sid]))
    sensor_map = SensorMap(finger_touch_adr=finger_touch_adr)

    return NameMap(
        hand_joint_ids=hand_joint_ids,
        hand_qpos_start=hand_qpos_start,
        hand_qpos_end=hand_qpos_end,
        hand_qvel_start=hand_qvel_start,
        hand_qvel_end=hand_qvel_end,
        palm_body_id=palm_body_id,
        fingertip_site_ids=fingertip_site_ids,
        finger_geom_ids_per_finger=finger_geom_ids_per_finger,
        object_body_id=object_body_id,
        object_geom_id=object_geom_id,
        obj_qpos_start=obj_qpos_start,
        obj_qvel_start=obj_qvel_start,
        sensor_map=sensor_map,
    )


def get_object_half_height(geom_type: int, geom_size: list[float]) -> float:
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return geom_size[2]
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return geom_size[0]
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return geom_size[1]  # cylinder size = (radius, half_length)
    else:
        return 0.03  # safe default for unknown primitives
