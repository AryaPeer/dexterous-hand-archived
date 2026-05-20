from dataclasses import dataclass, field
import math
from pathlib import Path

import mujoco
import numpy as np

from dexterous_hand.config import SceneConfig
from dexterous_hand.utils.cpu.mujoco_helpers import get_joint_qpos_qvel_range

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
    "rh_THJ4": 1.2,
    "rh_THJ2": 0.5,
    "rh_THJ1": 1.4,
}


def apply_flexion_bias(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    bias_map: dict[str, float] = TABLE_TASK_FLEXION_BIAS,
) -> np.ndarray:
    """Clamp `bias_map` joint targets into qpos in place.

    @param qpos: full qpos buffer
    @type qpos: np.ndarray
    @param model: compiled mujoco model
    @type model: mujoco.MjModel
    @param bias_map: joint name -> desired angle (radians)
    @type bias_map: dict[str, float]
    """

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
    """
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for ai in range(model.nu):
        if int(model.actuator_trntype[ai]) != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = int(model.actuator_trnid[ai, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname in bias_map:
            lo, hi = model.actuator_ctrlrange[ai]
            ctrl[ai] = float(np.clip(bias_map[jname], lo, hi))
    return ctrl


@dataclass
class SensorMap:
    finger_touch_adr: list[int]
    n_sensors: int = 0
    wall_force_adr: list[int] = field(default_factory=list)

    @staticmethod
    def empty() -> "SensorMap":
        return SensorMap(finger_touch_adr=[], n_sensors=0, wall_force_adr=[])


@dataclass
class NameMap:
    hand_joint_ids: list[int]
    hand_actuator_ids: list[int]
    hand_qpos_start: int
    hand_qpos_end: int
    hand_qvel_start: int
    hand_qvel_end: int
    n_actuators: int
    ctrl_ranges: np.ndarray

    palm_body_id: int
    fingertip_site_ids: list[int]
    fingertip_geom_ids: set[int]
    finger_geom_ids_per_finger: list[set[int]]
    object_body_id: int
    object_geom_id: int
    obj_qpos_start: int
    obj_qvel_start: int
    table_geom_id: int
    sensor_map: SensorMap = field(default_factory=SensorMap.empty)


def build_scene(
    config: SceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, NameMap]:
    """Compile the table+hand+object grasp scene.

    @param config: scene physics + layout
    @type config: SceneConfig | None
    @return: (model, data, name_map)
    @rtype: tuple[mujoco.MjModel, mujoco.MjData, NameMap]
    """

    if config is None:
        config = SceneConfig()

    spec = mujoco.MjSpec()
    spec.option.timestep = config.sim_timestep
    spec.option.gravity = [0.0, 0.0, -9.81]
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

    # actuators
    hand_actuator_ids = list(range(model.nu))
    ctrl_ranges = model.actuator_ctrlrange[: model.nu].copy()
    n_actuators = model.nu

    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")

    # fingertips
    fingertip_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in FINGERTIP_SITE_NAMES
    ]

    fingertip_geom_ids: set[int] = set()
    fingertip_body_ids = set()

    for body_name in FINGERTIP_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            fingertip_body_ids.add(bid)

    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in fingertip_body_ids:
            fingertip_geom_ids.add(gid)

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

    # object + table
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")

    # sensors
    finger_touch_adr = []
    for touch_site in FINGER_TOUCH_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_{touch_site}")
        if sid >= 0:
            finger_touch_adr.append(int(model.sensor_adr[sid]))
    sensor_map = SensorMap(finger_touch_adr=finger_touch_adr, n_sensors=model.nsensor)

    return NameMap(
        hand_joint_ids=hand_joint_ids,
        hand_actuator_ids=hand_actuator_ids,
        hand_qpos_start=hand_qpos_start,
        hand_qpos_end=hand_qpos_end,
        hand_qvel_start=hand_qvel_start,
        hand_qvel_end=hand_qvel_end,
        n_actuators=n_actuators,
        ctrl_ranges=ctrl_ranges,
        palm_body_id=palm_body_id,
        fingertip_site_ids=fingertip_site_ids,
        fingertip_geom_ids=fingertip_geom_ids,
        finger_geom_ids_per_finger=finger_geom_ids_per_finger,
        object_body_id=object_body_id,
        object_geom_id=object_geom_id,
        obj_qpos_start=obj_qpos_start,
        obj_qvel_start=obj_qvel_start,
        table_geom_id=table_geom_id,
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
