import jax.numpy as jnp


def get_finger_touch_from_sensors(
    sensordata: jnp.ndarray,
    finger_touch_adr: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-finger touch values and boolean contact mask from the sensor buffer."""
    touch_vals = sensordata[finger_touch_adr]
    contact_mask = touch_vals > 0.0
    return touch_vals, contact_mask


def get_object_state_jax(
    qpos: jnp.ndarray,
    qvel: jnp.ndarray,
    xpos: jnp.ndarray,
    obj_body_id: int,
    obj_qpos_start: int,
    obj_qvel_start: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Position, orientation, linear and angular velocity of a free-joint object."""
    position = xpos[obj_body_id]
    orientation = qpos[obj_qpos_start + 3 : obj_qpos_start + 7]
    linear_vel = qvel[obj_qvel_start : obj_qvel_start + 3]
    angular_vel = qvel[obj_qvel_start + 3 : obj_qvel_start + 6]
    return position, orientation, linear_vel, angular_vel


def get_fingertip_positions_jax(
    site_xpos: jnp.ndarray,
    fingertip_site_ids: jnp.ndarray,
) -> jnp.ndarray:
    """World positions of all fingertip sites -> (N, 3)."""

    return site_xpos[fingertip_site_ids]


def get_palm_position_jax(
    xpos: jnp.ndarray,
    palm_body_id: int,
) -> jnp.ndarray:
    """Palm world position."""

    return xpos[palm_body_id]


def get_body_axis_jax(
    xmat: jnp.ndarray,
    body_id: int,
    axis: int = 2,
) -> jnp.ndarray:
    """Body's local axis in world frame (default Z). Returns (3,)."""

    return xmat[body_id].reshape(3, 3)[:, axis]


def get_insertion_depth_jax(
    xpos: jnp.ndarray,
    xmat: jnp.ndarray,
    peg_body_id: int,
    hole_body_id: int,
    peg_half_length: float,
    peg_radius: float = 0.0,
) -> jnp.ndarray:
    """Depth of the peg's deepest point below the hole entrance, measured along
    the hole axis (0 if the peg is at/above the entrance).

    Uses the capsule's geometric lowest point rather than ``sign(dot)`` to pick a
    tip: the previous formula used ``jnp.sign(dot(peg_axis, hole_axis))``, which
    is 0 for a horizontal peg (dot==0) and collapsed the tip to the peg CENTRE,
    and for tilted pegs projected the full (half+radius) offset along the peg
    axis instead of taking the true lowest point. The deepest point of the
    capsule along -hole_axis sits ``peg_half_length*|cos(tilt)| + peg_radius``
    below the centre, for either peg orientation (|dot| is sign-agnostic)."""

    peg_pos = xpos[peg_body_id]
    hole_pos = xpos[hole_body_id]
    hole_axis = xmat[hole_body_id].reshape(3, 3)[:, 2]
    peg_axis = xmat[peg_body_id].reshape(3, 3)[:, 2]

    depth_of_center = jnp.dot(hole_pos - peg_pos, hole_axis)
    lowest_point_extent = peg_half_length * jnp.abs(jnp.dot(peg_axis, hole_axis)) + peg_radius
    depth = depth_of_center + lowest_point_extent
    return jnp.maximum(depth, 0.0)


def get_peg_hole_relative_jax(
    xpos: jnp.ndarray,
    xmat: jnp.ndarray,
    peg_body_id: int,
    hole_body_id: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Relative position and angular error between peg and hole."""
    peg_pos = xpos[peg_body_id]
    hole_pos = xpos[hole_body_id]
    rel_pos = peg_pos - hole_pos

    peg_rot = xmat[peg_body_id].reshape(3, 3)
    hole_rot = xmat[hole_body_id].reshape(3, 3)

    peg_axis = peg_rot[:, 2]
    hole_axis = hole_rot[:, 2]

    cross = jnp.cross(peg_axis, hole_axis)
    dot = jnp.clip(jnp.dot(peg_axis, hole_axis), -1.0, 1.0)
    angle = jnp.arccos(jnp.abs(dot))
    norm = jnp.linalg.norm(cross)
    angular_error = jnp.where(norm > 1e-8, cross / norm * angle, jnp.zeros(3))

    return rel_pos, angular_error
