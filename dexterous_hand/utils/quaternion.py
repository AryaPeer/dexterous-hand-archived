import jax
import jax.numpy as jnp


def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product. Quats are [w, x, y, z] (MuJoCo convention)."""

    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Conjugate (= inverse for unit quats)."""

    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quat_angular_distance(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Geodesic angle between two orientations [0, pi]. Handles double-cover."""

    dot = jnp.clip(jnp.abs(jnp.dot(q1, q2)), 0.0, 1.0)
    return 2.0 * jnp.arccos(dot)


def random_quaternion_within_angle(
    key: jax.Array,
    max_angle_rad: float | jax.Array,
    min_angle_rad: float | jax.Array = 0.0,
) -> jnp.ndarray:
    """Random rotation within [min_angle_rad, max_angle_rad]. Axis sampled uniformly on S^2."""

    k1, k2, k3 = jax.random.split(key, 3)
    z = jax.random.uniform(k1, minval=-1.0, maxval=1.0)
    phi = jax.random.uniform(k2, minval=0.0, maxval=2.0 * jnp.pi)
    r = jnp.sqrt(1.0 - z * z)
    axis = jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])

    lo = jnp.minimum(jnp.asarray(min_angle_rad), jnp.asarray(max_angle_rad))
    angle = jax.random.uniform(k3, minval=lo, maxval=max_angle_rad)
    half = angle / 2.0
    s = jnp.sin(half)
    return jnp.array([jnp.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


def sample_target_quat_rel_to_cube(
    key: jax.Array,
    cube_quat: jnp.ndarray,
    max_angle_rad: float | jax.Array,
    min_angle_rad: float | jax.Array = 0.0,
    n_candidates: int = 8,
) -> jnp.ndarray:
    """Pick a target quat at least `min_angle_rad` away from `cube_quat`.

    Vmap-friendly: samples `n_candidates`, returns the first one past the
    threshold (or the farthest if none pass) without Python branching.
    """
    keys = jax.random.split(key, n_candidates)
    sample_one = lambda k: random_quaternion_within_angle(k, max_angle_rad)
    candidates = jax.vmap(sample_one)(keys)
    dists = jax.vmap(lambda q: quat_angular_distance(q, cube_quat))(candidates)

    lo = jnp.minimum(jnp.asarray(min_angle_rad), jnp.asarray(max_angle_rad))
    acceptable = dists >= lo
    any_acceptable = jnp.any(acceptable)
    first_ok_idx = jnp.argmax(acceptable.astype(jnp.int32))
    farthest_idx = jnp.argmax(dists)
    chosen_idx = jnp.where(any_acceptable, first_ok_idx, farthest_idx)
    return candidates[chosen_idx]
