from dataclasses import dataclass, field

import mujoco
import numpy as np

from dexterous_hand.config import ReorientSceneConfig
from dexterous_hand.envs.scene_builder import (
    ASSETS_DIR,
    FINGER_BODY_PREFIXES,
    FINGER_TOUCH_SITE_NAMES,
    FINGERTIP_BODIES,
    FINGERTIP_OFFSETS,
    FINGERTIP_SITE_NAMES,
    SensorMap,
)
from dexterous_hand.utils.mujoco_helpers import get_joint_qpos_qvel_range


@dataclass
class ReorientNameMap:
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
    cube_body_id: int
    cube_geom_id: int
    cube_qpos_start: int
    cube_qvel_start: int
    sensor_map: SensorMap = field(default_factory=SensorMap.empty)


def build_reorient_scene(
    config: ReorientSceneConfig | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, ReorientNameMap]:
    """Compile the hand+cube reorient scene (no table, hand-mounted)."""

    if config is None:
        config = ReorientSceneConfig()

    spec = mujoco.MjSpec()
    spec.option.timestep = config.sim_timestep
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.stat.extent = 1.0
    spec.stat.center = [0.0, 0.0, config.mount_height]

    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.01],
        rgba=[0.3, 0.3, 0.3, 1.0],
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
        pos=[0.6, -0.6, 0.6],
        xyaxes=[0.707, 0.707, 0.0, -0.354, 0.354, 0.866],
    )

    mount = spec.worldbody.add_body(
        name="hand_mount",
        pos=[0.0, 0.0, config.mount_height],
    )
    mount_site = mount.add_site(name="hand_attach", pos=[0.0, 0.0, 0.0])

    hand_xml = str(ASSETS_DIR / "right_hand.xml")
    child_spec = mujoco.MjSpec.from_file(hand_xml)
    spec.attach(child_spec, site=mount_site, prefix="")

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

    hs = config.cube_size
    cube_body = spec.worldbody.add_body(
        name="cube",
        pos=[0.0, 0.0, config.mount_height + 0.12],
    )
    cube_body.add_freejoint(name="cube_freejoint")
    cube_body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[hs, hs, hs],
        mass=config.cube_mass,
        friction=list(config.cube_friction),
        rgba=[0.2, 0.6, 0.9, 1.0],
        contype=1,
        conaffinity=1,
        condim=4,
    )

    model = spec.compile()
    data = mujoco.MjData(model)
    name_map = _resolve_reorient_names(model)

    return model, data, name_map


def _resolve_reorient_names(model: mujoco.MjModel) -> ReorientNameMap:
    # cube freejoint
    cube_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    cube_qpos_start = model.jnt_qposadr[cube_jnt_id]
    cube_qvel_start = model.jnt_dofadr[cube_jnt_id]

    # hand joints (everything except the cube freejoint)
    hand_joint_ids = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name != "cube_freejoint":
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

    # cube
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

    # sensors
    finger_touch_adr = []
    for touch_site in FINGER_TOUCH_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"sensor_{touch_site}")
        if sid >= 0:
            finger_touch_adr.append(int(model.sensor_adr[sid]))
    sensor_map = SensorMap(finger_touch_adr=finger_touch_adr, n_sensors=model.nsensor)

    return ReorientNameMap(
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
        cube_body_id=cube_body_id,
        cube_geom_id=cube_geom_id,
        cube_qpos_start=cube_qpos_start,
        cube_qvel_start=cube_qvel_start,
        sensor_map=sensor_map,
    )
