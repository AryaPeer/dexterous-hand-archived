from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

from dexterous_hand.config import PegSceneConfig
from dexterous_hand.envs.scene_builder import (
    ASSETS_DIR,
    FINGER_BODY_PREFIXES,
    FINGER_TOUCH_SITE_NAMES,
    FINGERTIP_BODIES,
    FINGERTIP_OFFSETS,
    FINGERTIP_SITE_NAMES,
    SensorMap,
)
from dexterous_hand.utils.cpu.mujoco_helpers import get_joint_qpos_qvel_range


@dataclass
class PegNameMap:
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
    table_geom_id: int

    peg_body_id: int
    peg_geom_id: int
    peg_qpos_start: int
    peg_qvel_start: int
    hole_body_id: int
    hole_wall_geom_ids: list[int] = field(default_factory=list)
    hole_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hole_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    sensor_map: SensorMap = field(default_factory=SensorMap.empty)


def build_peg_scene(
    config: PegSceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, PegNameMap]:
    """Compile the table+hand+peg+hole scene."""

    if config is None:
        config = PegSceneConfig()

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
    slider.add_joint(
        name="slide_z",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[-0.10, 0.15],
    )

    mount = slider.add_body(
        name="hand_mount",
        euler=[math.pi, 0.0, 0.0],
    )
    mount_site = mount.add_site(name="hand_attach", pos=[0.0, 0.0, 0.0])

    spec.add_actuator(
        name="slide_x_act",
        target="slide_x",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -100, -10, 0, 0, 0, 0, 0, 0, 0],
        ctrlrange=[-0.15, 0.15],
        forcerange=[-15, 15],
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
        forcerange=[-15, 15],
    )
    # slide_z fights gravity on the entire hand (~4 kg). The slide_x/y actuator
    # gains (kp=100) leave the hand sagging to its lower bound under load. Use
    # kp=8000 + matching damping + 250 N force range to hold position with
    # sub-cm sag while still letting the policy command lifts up to 15cm.
    spec.add_actuator(
        name="slide_z_act",
        target="slide_z",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        gainprm=[8000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -8000, -250, 0, 0, 0, 0, 0, 0, 0],
        ctrlrange=[-0.10, 0.15],
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

    # per-wall touch sensors so the reward can read individual hole-wall forces
    wall_sensor_names = [
        "hole_wall_px",
        "hole_wall_nx",
        "hole_wall_py",
        "hole_wall_ny",
        "hole_bottom",
    ]
    for wall_name in wall_sensor_names:
        spec.add_sensor(
            name=f"sensor_force_{wall_name}",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"site_{wall_name}",
        )

    # peg + hole walls live on disjoint contype/conaffinity bits so they only
    # interact with the hand and each other, not with the table or floor.
    peg_kwargs = dict(
        contype=3,
        conaffinity=3,
        condim=4,
        solref=[0.005, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
    )
    wall_kwargs = dict(
        contype=2,
        conaffinity=2,
        condim=4,
        solref=[0.005, 1.0],
        solimp=[0.9, 0.95, 0.001, 0.5, 2.0],
    )

    peg_z = config.table_height + config.peg_half_length + config.peg_radius + 0.001
    peg_body = spec.worldbody.add_body(
        name="peg",
        pos=[0.0, 0.0, peg_z],
    )
    peg_body.add_freejoint(name="peg_freejoint")
    peg_body.add_geom(
        name="peg_geom",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[config.peg_radius, config.peg_half_length, 0.0],
        mass=config.peg_mass,
        friction=list(config.peg_friction),
        rgba=[0.8, 0.2, 0.2, 1.0],
        **peg_kwargs,
    )

    hole_x = config.hole_offset[0]
    hole_y = config.hole_offset[1]
    hole_z = config.table_height
    hole_body = spec.worldbody.add_body(
        name="hole",
        pos=[hole_x, hole_y, hole_z],
    )

    cr = config.peg_radius + config.clearance
    wt = 0.005
    wh = config.hole_depth / 2

    wall_rgba = [0.4, 0.4, 0.5, 1.0]

    hole_body.add_geom(
        name="hole_wall_px",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[wt / 2, cr + wt, wh],
        pos=[cr + wt / 2, 0.0, -wh],
        rgba=wall_rgba,
        **wall_kwargs,
    )

    hole_body.add_geom(
        name="hole_wall_nx",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[wt / 2, cr + wt, wh],
        pos=[-(cr + wt / 2), 0.0, -wh],
        rgba=wall_rgba,
        **wall_kwargs,
    )

    hole_body.add_geom(
        name="hole_wall_py",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cr + wt, wt / 2, wh],
        pos=[0.0, cr + wt / 2, -wh],
        rgba=wall_rgba,
        **wall_kwargs,
    )

    hole_body.add_geom(
        name="hole_wall_ny",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cr + wt, wt / 2, wh],
        pos=[0.0, -(cr + wt / 2), -wh],
        rgba=wall_rgba,
        **wall_kwargs,
    )

    hole_body.add_geom(
        name="hole_bottom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cr + wt, cr + wt, wt / 2],
        pos=[0.0, 0.0, -config.hole_depth],
        rgba=wall_rgba,
        **wall_kwargs,
    )

    # touch-sensor sites on each wall, sized to match the wall thickness
    wall_positions = {
        "hole_wall_px": [cr + wt / 2, 0.0, -wh],
        "hole_wall_nx": [-(cr + wt / 2), 0.0, -wh],
        "hole_wall_py": [0.0, cr + wt / 2, -wh],
        "hole_wall_ny": [0.0, -(cr + wt / 2), -wh],
        "hole_bottom": [0.0, 0.0, -config.hole_depth],
    }
    for wall_name, pos in wall_positions.items():
        site_size = cr if wall_name == "hole_bottom" else wt / 2
        hole_body.add_site(
            name=f"site_{wall_name}",
            pos=pos,
            size=[site_size],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            group=4,
        )

    model = spec.compile()
    data = mujoco.MjData(model)
    name_map = _resolve_peg_names(model, config)

    return model, data, name_map


def _resolve_peg_names(model: mujoco.MjModel, config: PegSceneConfig) -> PegNameMap:
    # peg freejoint
    peg_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint")
    peg_qpos_start = model.jnt_qposadr[peg_jnt_id]
    peg_qvel_start = model.jnt_dofadr[peg_jnt_id]

    # hand joints (everything except the peg freejoint)
    hand_joint_ids = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name != "peg_freejoint":
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

    # peg + hole + table
    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")
    peg_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    peg_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
    hole_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hole")

    wall_names = ["hole_wall_px", "hole_wall_nx", "hole_wall_py", "hole_wall_ny", "hole_bottom"]
    hole_wall_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in wall_names]

    hole_pos = np.array([config.hole_offset[0], config.hole_offset[1], config.table_height])
    hole_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # sensors
    finger_touch_adr = []
    for touch_site in FINGER_TOUCH_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_{touch_site}")
        if sid >= 0:
            finger_touch_adr.append(int(model.sensor_adr[sid]))

    wall_sensor_site_names = [
        "hole_wall_px",
        "hole_wall_nx",
        "hole_wall_py",
        "hole_wall_ny",
        "hole_bottom",
    ]
    wall_force_adr = []
    for wall_name in wall_sensor_site_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_force_{wall_name}")
        if sid >= 0:
            wall_force_adr.append(int(model.sensor_adr[sid]))

    sensor_map = SensorMap(
        finger_touch_adr=finger_touch_adr,
        n_sensors=model.nsensor,
        wall_force_adr=wall_force_adr,
    )

    return PegNameMap(
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
        table_geom_id=table_geom_id,
        peg_body_id=peg_body_id,
        peg_geom_id=peg_geom_id,
        peg_qpos_start=peg_qpos_start,
        peg_qvel_start=peg_qvel_start,
        hole_body_id=hole_body_id,
        hole_wall_geom_ids=hole_wall_geom_ids,
        hole_pos=hole_pos,
        hole_quat=hole_quat,
        sensor_map=sensor_map,
    )
